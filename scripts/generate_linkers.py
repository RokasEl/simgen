import os

import numpy as np
import torch
from e3nn import o3

from moldiff.particle_filtering import ParticleFilterGenerator

torch.set_default_dtype(torch.float64)
from ase import Atoms
from mace.data.atomic_data import AtomicData

from moldiff.utils import initialize_mol, setup_logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import logging

import ase.io as ase_io
from mace.modules.blocks import (
    RadialDistanceTransformBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import ScaleShiftMACE

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import remove_elements
from moldiff.manifolds import MultivariateGaussianPrior


def calculate_restorative_force_strength(num_atoms: int | float) -> float:
    sqrt_prefactor = 1.5664519
    bounding_sphere_diameter = sqrt_prefactor * np.sqrt(num_atoms)
    force_strength = 1 / (0.2 + 0.1 * bounding_sphere_diameter) ** 2
    return force_strength


def main():
    setup_logger(level=logging.DEBUG, tag="particle_filter", directory="./logs")
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0_swa.model"
    pretrained_model = torch.load(pretrained_mace)
    print(pretrained_model)
    model = ScaleShiftMACE(
        r_max=4.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=2,
        num_elements=10,
        atomic_energies=np.zeros(10),
        avg_num_neighbors=15.653135299682617,
        correlation=3,
        interaction_cls_first=RealAgnosticInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        hidden_irreps=o3.Irreps("96x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.088502,
        atomic_inter_shift=0.0,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.radial_embedding = RadialDistanceTransformBlock(
        r_min=0.75, **dict(r_max=4.5, num_bessel=8, num_polynomial_cutoff=5)
    )
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    rng = np.random.default_rng(0)
    data_path = "../data/qm9_full_data.xyz"
    all_data = ase_io.read(data_path, index=":", format="extxyz")
    # Make sure we have a reference atom from molecules of all sizes
    molecule_sizes = {len(mol) for mol in all_data}
    training_data = []
    for size in molecule_sizes:
        size_mols = list(filter(lambda mol: len(mol) == size, all_data))
        idx = rng.integers(0, len(size_mols))
        training_data.append(size_mols[idx])
    # now add further random molecules
    too_add = 256 - len(training_data)
    rand_mols = [x for x in rng.choice(all_data, size=too_add)]
    training_data.extend(rand_mols)
    training_data = [remove_elements(mol, [1, 9]) for mol in training_data]
    score_model = MaceSimilarityCalculator(
        model, reference_data=training_data, device=DEVICE
    )
    energies = np.empty(len(training_data))
    for i, mol in enumerate(training_data):
        mol.calc = score_model
        mol.info["calculation_type"] = "similarity"
        mol.info["time"] = 1e-2
        energies[i] = mol.get_potential_energy()
    logging.debug(f"Energies of training data: {energies}")
    noise_params = SamplerNoiseParameters(
        sigma_max=10, sigma_min=2e-3, S_churn=1.3, S_min=2e-3, S_noise=0.5
    )

    linkers_data_path = "../data/zinc_fragments_difflinker.xyz"
    all_fragments = ase_io.read(linkers_data_path, index=":", format="extxyz")

    destination = "./scripts/Generated_trajectories/linkers/"
    # create destination folder if it does not exist
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 100])
    for i in range(100):
        logging.debug(f"Generating molecule {i}")

        scaffold = all_fragments[i].copy()
        linker_positions = scaffold.info["avg_linker_position"]
        scaffold.set_positions(scaffold.get_positions() - linker_positions)
        size = scaffold.info["num_linker_atoms"]

        mol = initialize_mol(f"C{size}")
        restorative_force_strength = calculate_restorative_force_strength(size)
        logging.debug(
            f"Mol size {size}, restorative force strength {restorative_force_strength:.2f}"
        )
        particle_filter = ParticleFilterGenerator(
            score_model,
            num_steps=150,
            guiding_manifold=MultivariateGaussianPrior(
                covariance_matrix=np.diag([1.0, 1.0, 1.0])
            ),
            noise_params=noise_params,
            restorative_force_strength=restorative_force_strength,
        )
        trajectories = particle_filter.generate(
            mol,
            swapping_z_table,
            num_particles=10,
            particle_swap_frequency=4,
            scaffold=scaffold,
        )
        ase_io.write(
            f"{destination}/CHONF_{i}_{size}.xyz",
            trajectories,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    main()