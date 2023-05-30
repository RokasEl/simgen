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

from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
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
    noise_params = SamplerNoiseParameters(
        sigma_max=10, sigma_min=2e-3, S_churn=1.3, S_min=2e-3, S_noise=0.5
    )
    destination = "./scripts/Generated_trajectories/size_one/"
    # theta = np.linspace(0, 2 * np.pi, 100)
    # x, y = np.cos(theta), np.sin(theta)
    # points = np.stack([x, y, np.zeros_like(x)], axis=1) * 4
    # prior = PointCloudPrior(
    #     points=points,
    #     beta=3,
    #     point_shape=MultivariateGaussianPrior(
    #         covariance_matrix=np.diag([1.5, 1.5, 0.5])
    #     ),
    # )
    # create destination folder if it does not exist
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    for i in range(100):
        logging.debug(f"Generating molecule {i}")
        size = 4
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = calculate_restorative_force_strength(size)
        logging.debug(
            f"Mol size {size}, restorative force strength {restorative_force_strength:.2f}"
        )
        particle_filter = ParticleFilterGenerator(
            score_model,
            num_steps=150,
            guiding_manifold=MultivariateGaussianPrior(np.eye(3)),
            noise_params=noise_params,
            restorative_force_strength=restorative_force_strength,
        )
        scaffold = initialize_mol("C6H6")
        scaffold = scaffold[:6]
        trajectories = particle_filter.generate(
            mol,
            swapping_z_table,
            num_particles=10,
            particle_swap_frequency=3,
            # scaffold=scaffold,
        )
        ase_io.write(
            f"{destination}/scaffolded_CHONF_{i}_{size}.xyz",
            trajectories,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    main()
