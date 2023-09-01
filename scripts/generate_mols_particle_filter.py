import os

import numpy as np
import torch

from moldiff.particle_filtering import ParticleFilterGenerator

torch.set_default_dtype(torch.float64)
from ase import Atoms

from moldiff.utils import (
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

DEVICE = get_system_torch_device_str()

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
from moldiff.utils import get_hydromace_calculator


def main():
    setup_logger(level=logging.DEBUG, tag="particle_filter", directory="./logs")
    rng = np.random.default_rng(0)
    model_repo_path = "/home/rokas/Programming/MACE-Models"
    score_model = get_mace_similarity_calculator(
        model_repo_path,
        num_reference_mols=-1,
        device=DEVICE,
        rng=rng,
    )
    hydromace_calc = get_hydromace_calculator(
        model_repo_path=model_repo_path, device=DEVICE
    )
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
