import logging
from pathlib import Path

import ase.io as ase_io
import numpy as np
import typer

from simgen.element_swapping import SwappingAtomicNumberTable
from simgen.integrators import IntegrationParameters
from simgen.manifolds import LinePrior, MultivariateGaussianPrior
from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

DEVICE = get_system_torch_device_str()


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_repo_path: str = typer.Option(
        "https://github.com/RokasEl/MACE-Models", help="Path to MACE model repository"
    ),
    model_name: str = typer.Option("medium", help="Name of the MACE-OFF model to use"),
    reference_data_name: str = typer.Option(
        "simgen_reference_data_medium", help="Name of reference data to use"
    ),
    save_path: str = typer.Option(
        ..., help="Path to save generated molecules, can be file or directory"
    ),
    hydrogenation_model_name: str = typer.Option(
        "hydromace", help="Name of hydrogenation model to use"
    ),
    line_length: float = typer.Option(
        default=3.0, help="Length of the line to generate"
    ),
    atoms_per_angstrom: float = typer.Option(
        default=1.2, help="Lengthwise atom density"
    ),
    num_molecules: int = typer.Option(
        default=100, help="Number of molecules to generate"
    ),
    num_integration_steps: int = typer.Option(
        default=50, help="Number of integration steps for particle filter"
    ),
    track_trajectories: bool = typer.Option(
        default=False,
        help="If true, save all trajectory configurations instead of just the last",
    ),
    do_final_cleanup: bool = typer.Option(
        default=True,
        help="If true, clean up generated molecules by hydrogenating them and relaxing them. False if you want to save the raw output of the particle filter.",
    ),
    cleanup_scheme: str = typer.Option(
        default="add_hs_once",
    ),
):
    setup_logger(level=logging.INFO, tag="particle_filter", directory="./logs")
    rng = np.random.default_rng(0)
    score_model = get_mace_similarity_calculator(
        model_repo_path,
        model_name=model_name,
        data_name=reference_data_name,
        num_reference_mols=-1,
        device=DEVICE,
        rng=rng,
    )
    hydromace_calc = get_hydromace_calculator(
        model_repo_path=model_repo_path,
        model_name=hydrogenation_model_name,
        device=DEVICE,
    )
    integration_params = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)

    save_path: Path = Path(save_path)
    if save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    num_points = int(line_length * 5)
    point_shape = MultivariateGaussianPrior(
        covariance_matrix=np.diag([1.0, 0.5, 0.5]), normalise_covariance=False
    )
    prior = LinePrior(
        line_length, num_points=num_points, beta=1.0, point_shape=point_shape
    )

    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    for i in range(num_molecules):
        logging.info(f"Generating molecule {i}")
        size = int(prior.curve_length * atoms_per_angstrom)
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = 1.0
        particle_filter = ParticleFilterGenerator(
            score_model,
            guiding_manifold=prior,
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
            do_final_cleanup=do_final_cleanup,
            cleanup_scheme=cleanup_scheme,
        )

        if save_path.is_dir():
            outfile = save_path / f"{i}_{size}.xyz"
        else:
            outfile = save_path
        to_write = trajectories[-1] if not track_trajectories else trajectories
        ase_io.write(
            outfile,
            to_write,
            format="extxyz",
            append=True,
        )


if __name__ == "__main__":
    app()
