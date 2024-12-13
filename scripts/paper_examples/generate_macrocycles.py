import logging
import os

import ase.io as ase_io
import numpy as np
import torch

from simgen.element_swapping import SwappingAtomicNumberTable
from simgen.generation_utils import calculate_restorative_force_strength
from simgen.integrators import IntegrationParameters
from simgen.manifolds import CirclePrior, MultivariateGaussianPrior
from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

torch.set_default_dtype(torch.float64)
DEVICE = get_system_torch_device_str()


def main(
    model_repo_path,
    num_reference_mols,
    save_path,
    prior_gaussian_covariance,
    num_molecules,
    circle_radius,
    atoms_per_length,
    restorative_force_multiplier,
):
    setup_logger(level=logging.INFO, tag="particle_filter", directory="./logs")
    rng = np.random.default_rng(0)
    score_model = get_mace_similarity_calculator(
        model_repo_path,
        model_name="medium",
        data_name="simgen_reference_data_medium",
        num_reference_mols=num_reference_mols,
        num_to_sample_uniformly_per_size=2,
        device=DEVICE,
        rng=rng,
    )
    hydromace_calc = get_hydromace_calculator(
        model_repo_path=model_repo_path, device=DEVICE
    )
    integration_parameters = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
    destination = save_path
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    point_shape = MultivariateGaussianPrior(
        prior_gaussian_covariance, normalise_covariance=False
    )
    prior = CirclePrior(circle_radius, num_points=100, beta=8, point_shape=point_shape)
    num_atoms = np.ceil(prior.curve_length * atoms_per_length).astype(int)

    for i in range(num_molecules):
        logging.info(f"Generating molecule {i}")
        size = num_atoms
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = (
            calculate_restorative_force_strength(size) * restorative_force_multiplier
        )
        logging.debug(
            f"Mol size {size}, restorative force strength {restorative_force_strength:.2f}"
        )
        particle_filter = ParticleFilterGenerator(
            score_model,
            guiding_manifold=prior,
            integration_parameters=integration_parameters,
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
            f"{destination}/macrocycle_{i}_{size}.xyz",
            trajectories,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_repo_path",
        help="Path to MACE-models repo",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_reference_mols",
        help="Number of reference molecules to use",
        type=int,
        default=256,
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
        "--circle_radius",
        help="Radius of circle prior in Angstroms",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--atoms_per_length",
        type=float,
        default=1.5,
        help="Atoms per Angstrom of circle length",
    )
    parser.add_argument(
        "--restorative_force_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for restorative force strength",
    )
    args = parser.parse_args()
    covariance_matrix = np.diag(args.prior_gaussian_covariance).astype(np.float64)
    main(
        model_repo_path=args.model_repo_path,
        num_reference_mols=args.num_reference_mols,
        save_path=args.save_path,
        prior_gaussian_covariance=covariance_matrix,
        num_molecules=args.num_molecules,
        circle_radius=args.circle_radius,
        atoms_per_length=args.atoms_per_length,
        restorative_force_multiplier=args.restorative_force_multiplier,
    )
