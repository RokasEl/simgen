import abc

import ase
import numpy as np
import torch
from pydantic import BaseModel, Field

from moldiff.atoms_cleanup import relax_hydrogens
from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
from moldiff.hydrogenation import hydrogenate_deterministically
from moldiff.manifolds import PointCloudPrior
from moldiff.particle_filtering import ParticleFilterGenerator
from moldiff.utils import get_mace_similarity_calculator, initialize_mol

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UpdateScene(BaseModel, abc.ABC):
    @abc.abstractmethod
    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        pass


def load_mace_calc():
    pretrained_mace_path = "/home/rokas/Programming/Generative_model_energy/models/SPICE_sm_inv_neut_E0_swa.model"
    rng = np.random.default_rng(0)
    data_path = "/home/rokas/Programming/data/qm9_full_data.xyz"
    mace_calc = get_mace_similarity_calculator(
        pretrained_mace_path,
        data_path,
        num_reference_mols=64,
        num_to_sample_uniformly_per_size=2,
        device=DEVICE,
        rng=rng,
    )
    return mace_calc


noise_params = SamplerNoiseParameters(
    sigma_max=10, sigma_min=2e-3, S_churn=1.3, S_min=2e-3, S_noise=0.5
)
swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])


class ConstrainedGeneration(UpdateScene):
    num_atoms: int = Field(5, ge=1, le=30, description="Number of atoms to generate")

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        mace_calc = load_mace_calc()
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
        mol = initialize_mol(f"C{self.num_atoms}")
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


class Hydrogenate(UpdateScene):
    num_steps: int = Field(
        10, ge=1, le=100, description="Number of hydrogen relaxtion steps"
    )

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        calc = load_mace_calc()
        print(atoms.info)
        print(kwargs)
        hydrogenated_atoms = hydrogenate_deterministically(atoms)
        relaxed_hydrogenated_atoms = hydrogenated_atoms.copy()
        relaxed_hydrogenated_atoms = relax_hydrogens(
            [relaxed_hydrogenated_atoms], calc, num_steps=self.num_steps, max_step=0.1
        )[0]
        return [hydrogenated_atoms, relaxed_hydrogenated_atoms]