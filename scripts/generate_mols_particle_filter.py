import os

import numpy as np
import torch

from moldiff.particle_filtering import ParticleFilterGenerator

torch.set_default_dtype(torch.float64)
from ase import Atoms

from moldiff.utils import (
    get_mace_similarity_calculator,
    initialize_mol,
    setup_logger,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import logging

import ase.io as ase_io
from hydromace.interface import HydroMaceCalculator

from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
from moldiff.integrators import IntegrationParameters
from moldiff.manifolds import (
    HeartPointCloudPrior,
    MultivariateGaussianPrior,
    PointCloudPrior,
)


def main():
    setup_logger(level=logging.DEBUG, tag="particle_filter", directory="./logs")
    pretrained_mace_path = "./models/SPICE_sm_inv_neut_E0_swa.model"
    rng = np.random.default_rng(0)
    data_path = "../data/qm9_full_data.xyz"
    score_model = get_mace_similarity_calculator(
        pretrained_mace_path,
        data_path,
        num_reference_mols=256,
        num_to_sample_uniformly_per_size=2,
        device=DEVICE,
        rng=rng,
    )
    hydromace_model = torch.load(
        "./models/qm9_and_spice_hydrogenation.model", map_location=DEVICE
    )
    hydromace_calc = HydroMaceCalculator(hydromace_model, device=DEVICE)
    destination = "./scripts/Generated_trajectories/hydromace_new_swaps/"
    noise_params = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
    destination = "./scripts/Generated_trajectories/size_one/"

    # create destination folder if it does not exist
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    for i in range(100):
        logging.debug(f"Generating molecule {i}")
        size = rng.integers(3, 29)
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = calculate_restorative_force_strength(size)
        logging.debug(
            f"Mol size {size}, restorative force strength {restorative_force_strength:.2f}"
        )
        particle_filter = ParticleFilterGenerator(
            score_model,
            guiding_manifold=MultivariateGaussianPrior(np.eye(3)),
            integration_parameters=noise_params,
            restorative_force_strength=restorative_force_strength,
        )
        trajectories = particle_filter.generate(
            mol,
            swapping_z_table,
            num_particles=10,
            particle_swap_frequency=2,
            hydrogenation_type="hydromace",
            hydrogenation_calc=hydromace_calc,
        )
        ase_io.write(
            f"{destination}/CHONF_{i}_{size}.xyz",
            trajectories,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    main()
