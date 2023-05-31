import abc

import ase
import networkx as nx
import numpy as np
import torch
from pydantic import BaseModel, Field

from moldiff.atoms_cleanup import (
    attach_calculator,
    relax_hydrogens,
    run_dynamics,
)
from moldiff.calculators import MaceSimilarityCalculator
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


def load_mace_calc():
    pretrained_mace_path = "/home/rokas/Programming/Generative_model_energy/models/SPICE_sm_inv_neut_E0_swa.model"
    rng = np.random.default_rng(0)
    data_path = "/home/rokas/Programming/data/qm9_reference_data.xyz"
    mace_calc = get_mace_similarity_calculator(
        pretrained_mace_path,
        data_path,
        num_reference_mols=-1,
        device=DEVICE,
        rng=rng,
    )
    return mace_calc


def try_to_get_calc(atoms):
    if atoms.calc is not None and isinstance(atoms.calc, MaceSimilarityCalculator):
        calc = atoms.calc
        print("Using preloaded MACE calculator")
    else:
        calc = load_mace_calc()
        print("Loaded MACE calculator")
    return calc


noise_params = SamplerNoiseParameters(
    sigma_max=10, sigma_min=2e-3, S_churn=1.3, S_min=2e-3, S_noise=0.5
)
swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])


class ConstrainedGeneration(UpdateScene):
    num_atoms: int = Field(5, ge=1, le=30, description="Number of atoms to generate")

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        mace_calc = try_to_get_calc(atoms)
        prior = PointCloudPrior(kwargs["points"], beta=5.0)
        restorative_force_strength = calculate_restorative_force_strength(
            self.num_atoms
        )
        generator = ParticleFilterGenerator(
            mace_calc,
            prior,
            noise_params=noise_params,
            device=DEVICE,
            restorative_force_strength=restorative_force_strength,
        )
        mol = ase.Atoms(f"C{self.num_atoms}")
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
        trajectories[-1].calc = mace_calc
        return trajectories


def hydrogenate(atoms, graph_representation):
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


class Hydrogenate(UpdateScene):
    num_steps: int = Field(
        10, ge=1, le=100, description="Number of hydrogen relaxtion steps"
    )

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        calc = try_to_get_calc(atoms)
        try:
            connectivity_graph = atoms.info["graph_representation"]
        except KeyError:
            print(
                "No graph representation found, try resetting the scene of clicking `Save` in the `Bonds` tab"
            )
            return [atoms]
        hydrogenated_atoms = hydrogenate(atoms, connectivity_graph)
        relaxed_hydrogenated_atoms = hydrogenated_atoms.copy()
        relaxed_hydrogenated_atoms = relax_hydrogens(
            [relaxed_hydrogenated_atoms], calc, num_steps=self.num_steps, max_step=0.1
        )[0]
        relaxed_hydrogenated_atoms.calc = calc
        return [hydrogenated_atoms, relaxed_hydrogenated_atoms]


class Relax(UpdateScene):
    num_steps: int = Field(10, ge=1, le=100, description="Number of calculation steps")

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        calc = try_to_get_calc(atoms)
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
        relaxed_atoms = run_dynamics(relaxed_atoms, num_steps=self.num_steps)
        relaxed_atoms[-1].calc = calc
        return [original_atoms, *relaxed_atoms]
