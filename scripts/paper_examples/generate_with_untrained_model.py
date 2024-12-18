import logging
import os

import ase.io as ase_io
import numpy as np
import torch
from hydromace.interface import HydroMaceCalculator

from simgen.atoms_cleanup import cleanup_atoms
from simgen.element_swapping import SwappingAtomicNumberTable
from simgen.generation_utils import calculate_restorative_force_strength
from simgen.integrators import IntegrationParameters
from simgen.manifolds import MultivariateGaussianPrior
from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import (
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

"""
To generate with an untrained model, we'll have to hack into the ParticleFilterGenerator class and do the
generation in multiple steps, whereas usually it would all be handled by the class.
"""
torch.set_default_dtype(torch.float64)
DEVICE = get_system_torch_device_str()


def main(
    mace_model_path,
    reference_data_path,
    num_reference_mols,
    hydromace_model_path,
    save_path,
    prior_gaussian_covariance,
    num_molecules,
    num_atoms,
    restorative_force_multiplier,
    save_name=None,
    track_trajectory=False,
):
    setup_logger(level=logging.INFO, tag="particle_filter", directory="./logs")
    untrained_mace_path = "./models/untrained_two_layer_mace.model"
    pretrained_mace_path = mace_model_path
    rng = np.random.default_rng(0)
    data_path = reference_data_path
    score_model = get_mace_similarity_calculator(
        untrained_mace_path,
        data_path,
        num_reference_mols=num_reference_mols,
        num_to_sample_uniformly_per_size=2,
        device=DEVICE,
        rng=rng,
    )
    pretrained_score_model = get_mace_similarity_calculator(
        pretrained_mace_path,
        data_path,
        num_reference_mols=1,
        num_to_sample_uniformly_per_size=0,
        device=DEVICE,
    )
    hydromace_model = torch.load(hydromace_model_path, map_location=DEVICE)
    hydromace_calc = HydroMaceCalculator(hydromace_model, device=DEVICE)
    integration_parameters = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
    destination = save_path
    os.makedirs(destination, exist_ok=True)
    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    prior = MultivariateGaussianPrior(
        prior_gaussian_covariance, normalise_covariance=False
    )
    save_file = f"{destination}/covariance_{prior_gaussian_covariance[0,0]:.1f}_{prior_gaussian_covariance[1,1]:.1f}_{prior_gaussian_covariance[2,2]:.1f}.xyz"
    if save_name is not None:
        save_file = f"{destination}/{save_name}.xyz"
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
        mol = prior.initialise_positions(mol, scale=0.5)
        molecule, mask, torch_mask = particle_filter._merge_scaffold_and_create_mask(
            mol, scaffold=None, num_particles=10, device=DEVICE
        )
        trajectories = particle_filter._maximise_log_similarity(
            molecule,
            particle_swap_frequency=2,
            num_particles=10,
            swapping_z_table=swapping_z_table,
            mask=mask,
            torch_mask=torch_mask,
        )
        atoms = trajectories[-1].copy()
        atoms.calc = pretrained_score_model
        cleaned = cleanup_atoms(
            atoms,
            hydrogenation_type="hydromace",
            hydrogenation_calc=hydromace_calc,
            z_table=swapping_z_table,
            num_element_sweeps="all",
            mask=mask,
        )
        trajectories.extend(cleaned)
        to_write = cleaned[-1] if not track_trajectory else trajectories
        ase_io.write(
            save_file,
            to_write,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mace_model_path",
        help="Path to MACE model",
        type=str,
        default="./models/SPICE_sm_inv_neut_E0_swa.model",
    )
    parser.add_argument(
        "--reference_data_path",
        help="Path to reference data",
        type=str,
        default="../data/qm9_full_data.xyz",
    )
    parser.add_argument(
        "--num_reference_mols",
        help="Number of reference molecules to use",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hydromace_path",
        help="Path to hydrogenation model",
        type=str,
        default="./models/qm9_and_spice_hydrogenation.model",
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
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--num_molecules", help="Number of molecules to generate", type=int, default=100
    )
    parser.add_argument(
        "--num_atoms",
        help="Number of heavy atoms per molecule",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--restorative_force_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for restorative force strength",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Name of file to save generated molecules to",
    )
    parser.add_argument("--track_trajectory", action="store_true", default=False)
    args = parser.parse_args()
    covariance_matrix = np.diag(args.prior_gaussian_covariance).astype(np.float64)
    main(
        mace_model_path=args.mace_model_path,
        reference_data_path=args.reference_data_path,
        num_reference_mols=args.num_reference_mols,
        hydromace_model_path=args.hydromace_path,
        save_path=args.save_path,
        prior_gaussian_covariance=covariance_matrix,
        num_molecules=args.num_molecules,
        num_atoms=args.num_atoms,
        restorative_force_multiplier=args.restorative_force_multiplier,
        save_name=args.save_name,
        track_trajectory=args.track_trajectory,
    )
