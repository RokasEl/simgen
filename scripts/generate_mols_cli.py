import logging
import os

import ase.io as ase_io
import numpy as np

from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
from moldiff.integrators import IntegrationParameters
from moldiff.manifolds import MultivariateGaussianPrior
from moldiff.particle_filtering import ParticleFilterGenerator
from moldiff.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

DEVICE = get_system_torch_device_str()


def main(
    model_repo_path,
    save_path,
    prior_gaussian_covariance,
    num_molecules,
    num_heavy_atoms,
    num_integration_steps,
):
    setup_logger(level=logging.INFO, tag="particle_filter", directory="./logs")
    rng = np.random.default_rng(0)
    score_model = get_mace_similarity_calculator(
        model_repo_path,
        num_reference_mols=-1,
        device=DEVICE,
        rng=rng,
    )
    hydromace_calc = get_hydromace_calculator(
        model_repo_path=model_repo_path, device=DEVICE
    )
    integration_params = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
    destination = save_path
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    for i in range(num_molecules):
        logging.info(f"Generating molecule {i}")
        size = num_heavy_atoms
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = calculate_restorative_force_strength(size)
        particle_filter = ParticleFilterGenerator(
            score_model,
            guiding_manifold=MultivariateGaussianPrior(prior_gaussian_covariance),
            integration_parameters=integration_params,
            restorative_force_strength=restorative_force_strength,
            num_steps=num_integration_steps,
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
            f"{destination}/qm9_like_{i}_{size}.xyz",
            trajectories,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_repo_path",
        help="Path to MACE model repository",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        help="Path to save generated molecules",
        type=str,
        default="./scripts/Generated_trajectories/",
    )
    parser.add_argument(
        "--prior_gaussian_covariance",
        help="Covariance matrix for prior Gaussian",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 2.0],
    )
    parser.add_argument(
        "--num_molecules", help="Number of molecules to generate", type=int, default=100
    )
    parser.add_argument(
        "--num_heavy_atoms",
        help="Number of heavy atoms in generated molecules",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_integration_steps",
        help="Number of integration steps for particle filter",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    covariance_matrix = np.diag(args.prior_gaussian_covariance)
    main(
        args.model_repo_path,
        args.save_path,
        covariance_matrix,
        args.num_molecules,
        args.num_heavy_atoms,
        args.num_integration_steps,
    )
