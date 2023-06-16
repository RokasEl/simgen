import abc

import ase
import networkx as nx
import numpy as np
import torch
from pydantic import BaseModel, Field, PrivateAttr

from moldiff.atoms_cleanup import (
    attach_calculator,
    relax_hydrogens,
    run_dynamics,
)
from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
from moldiff.hydrogenation import (
    NATURAL_VALENCES,
    add_hydrogens_to_atoms,
)
from moldiff.manifolds import PointCloudPrior
from moldiff.particle_filtering import ParticleFilterGenerator
from moldiff.utils import get_mace_similarity_calculator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UpdateScene(BaseModel, abc.ABC):
    @abc.abstractmethod
    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        pass


def load_mace_calc(model_path, reference_data_path):
    pretrained_mace_path = model_path
    rng = np.random.default_rng(0)
    data_path = reference_data_path
    mace_calc = get_mace_similarity_calculator(
        pretrained_mace_path,
        data_path,
        num_reference_mols=-1,
        device=DEVICE,
        rng=rng,
    )
    return mace_calc


def load_hydrogenation_model(model_path):
    try:
        from hydromace.interface import HydroMaceCalculator

        model = torch.load(model_path)
        model = model.to(torch.float)
        hydrogenation_model = HydroMaceCalculator(model, device=DEVICE)
        return hydrogenation_model
    except Exception as e:
        print(f"Could not load hydrogenation model due to exception: {e}")
        return None


noise_params = SamplerNoiseParameters(
    sigma_max=10, sigma_min=2e-3, S_churn=1.3, S_min=2e-3, S_noise=0.5
)
swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])


def hydrogenate_by_bond_lengths(atoms, graph_representation):
    edge_array = nx.adjacency_matrix(graph_representation).todense()  # type: ignore
    current_neighbours = edge_array.sum(axis=0)
    max_valence = np.array(
        [
            NATURAL_VALENCES[atomic_number]
            for atomic_number in atoms.get_atomic_numbers()
        ]
    )
    num_hs_to_add_per_atom = max_valence - current_neighbours
    atoms_with_hs = add_hydrogens_to_atoms(atoms, num_hs_to_add_per_atom)
    return atoms_with_hs


def hydrogenate_by_model(atoms, model):
    num_hs_to_add_per_atom = model.predict_missing_hydrogens(atoms)
    num_hs_to_add_per_atom = np.round(num_hs_to_add_per_atom).astype(int)
    atoms_with_hs = add_hydrogens_to_atoms(atoms, num_hs_to_add_per_atom)
    return atoms_with_hs


class MoldiffGeneration(UpdateScene):
    model_path: str = Field(
        "/home/rokas/Programming/Generative_model_energy/models/SPICE_sm_inv_neut_E0_swa.model"
    )
    reference_data_path: str = Field(
        "/home/rokas/Programming/data/qm9_reference_data.xyz"
    )
    hydrogenation_model_path: str = Field(
        "/home/rokas/Programming/hydromace/spice_hydrogenation.model"
    )
    num_atoms_to_add: int = Field(
        5, ge=1, le=30, description="Number of atoms to generate"
    )
    relaxation_steps: int = Field(
        10, ge=1, le=100, description="Number of relaxation steps"
    )
    run_type: str = Field("generate", description="Type of operation to do")
    _calc = PrivateAttr(None)
    _hydro_model = PrivateAttr(None)

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        if self._calc is None:
            print("Initializing MACE calculator")
            self._calc = load_mace_calc(
                kwargs["model_path"], kwargs["reference_data_path"]
            )
        else:
            print("Using existing MACE calculator")

        if "run_type" in kwargs:
            run_type = kwargs["run_type"].strip().lower()
        else:
            run_type = self.run_type.strip().lower()

        if run_type == "generate":
            return self._generate(atom_ids, atoms, **kwargs)
        elif run_type == "hydrogenate":
            return self._hydrogenate(atom_ids, atoms, **kwargs)
        elif run_type == "relax":
            return self._relax(atom_ids, atoms, **kwargs)
        else:
            print(
                f"Unrecognized `run_type`:f{run_type}\nShould be one of: `generate`, `hydrogenate`, `relax`"
            )
            return [atoms]

    def _generate(
        self, atom_ids: list[int], atoms: ase.Atoms, **kwargs
    ) -> list[ase.Atoms]:
        calc = self._calc
        prior = PointCloudPrior(kwargs["points"], beta=5.0)
        num_atoms_to_add = (
            kwargs["num_atoms_to_add"]
            if "num_atoms_to_add" in kwargs
            else self.num_atoms_to_add
        )
        num_atoms_to_add = int(num_atoms_to_add)
        restorative_force_strength = calculate_restorative_force_strength(
            num_atoms_to_add
        )
        generator = ParticleFilterGenerator(
            calc,
            prior,
            noise_params=noise_params,
            device=DEVICE,
            restorative_force_strength=restorative_force_strength,
        )
        mol = ase.Atoms(f"C{num_atoms_to_add}")
        mol = prior.initialise_positions(mol, scale=0.5)
        molecule, mask, torch_mask = generator._merge_scaffold_and_create_mask(
            mol, atoms, num_particles=10, device=DEVICE
        )
        trajectories = generator._maximise_log_similarity(
            molecule,
            particle_swap_frequency=4,
            num_particles=10,
            swapping_z_table=swapping_z_table,
            mask=mask,
            torch_mask=torch_mask,
        )
        return trajectories

    def _hydrogenate(
        self, atom_ids: list[int], atoms: ase.Atoms, **kwargs
    ) -> list[ase.Atoms]:
        calc = self._calc
        relaxation_steps = (
            kwargs["relaxation_steps"]
            if "relaxation_steps" in kwargs
            else self.relaxation_steps
        )
        relaxation_steps = int(relaxation_steps)
        hydro_model = self._get_hydrogenation_model(kwargs["hydrogenation_model_path"])
        if hydro_model is None:
            connectivity_graph, found, error = self._try_to_get_graph_representation(
                atoms
            )
            if not found:
                print(
                    "No graph representation found, try resetting the scene or clicking `Save` in the `Bonds` tab"
                )
                print(f"Error: {error}")
                return [atoms]
            hydrogenated_atoms = hydrogenate_by_bond_lengths(atoms, connectivity_graph)
        else:
            hydrogenated_atoms = hydrogenate_by_model(atoms, hydro_model)

        relaxed_hydrogenated_atoms = hydrogenated_atoms.copy()
        relaxed_hydrogenated_atoms = relax_hydrogens(
            [relaxed_hydrogenated_atoms], calc, num_steps=relaxation_steps, max_step=0.1
        )[0]
        return [hydrogenated_atoms, relaxed_hydrogenated_atoms]

    def _get_hydrogenation_model(self, hydrogenation_model_path):
        if self._hydro_model is None:
            print("Initializing hydrogenation model")
            self._hydro_model = load_hydrogenation_model(hydrogenation_model_path)
        else:
            print("Using loaded hydrogenation model")
        return self._hydro_model

    @staticmethod
    def _try_to_get_graph_representation(atoms):
        try:
            connectivity_graph = atoms.info["graph_representation"]
            found = True
            error = None
        except Exception as e:
            connectivity_graph = None
            found = False
            error = e
        return connectivity_graph, found, error

    def _relax(
        self, atom_ids: list[int], atoms: ase.Atoms, **kwargs
    ) -> list[ase.Atoms]:
        calc = self._calc
        relaxation_steps = (
            kwargs["relaxation_steps"]
            if "relaxation_steps" in kwargs
            else self.relaxation_steps
        )
        relaxation_steps = int(relaxation_steps)
        original_atoms = atoms.copy()
        if len(atom_ids) != 0:
            print("Will relax only the selected atoms")
            mask = np.ones(len(atoms)).astype(bool)
            mask[atom_ids] = False
        else:
            print("Will relax all atoms")
            mask = np.zeros(len(atoms)).astype(bool)

        relaxed_atoms = attach_calculator(
            [atoms], calc, calculation_type="mace", mask=mask
        )
        relaxed_atoms = run_dynamics(relaxed_atoms, num_steps=relaxation_steps)
        return [original_atoms, *relaxed_atoms]
