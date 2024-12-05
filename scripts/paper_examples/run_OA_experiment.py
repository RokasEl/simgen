import logging
from pathlib import Path

import ase
import ase.io as ase_io
import numpy as np
import typer
import zntrack

from simgen.element_swapping import SwappingAtomicNumberTable
from simgen.generation_utils import calculate_restorative_force_strength
from simgen.integrators import IntegrationParameters
from simgen.manifolds import PointCloudPrior
from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
    get_system_torch_device_str,
    initialize_mol,
    setup_logger,
)

DEVICE = get_system_torch_device_str()

app = typer.Typer()


def construct_prior_from_atoms(
    ligand_atoms: list[ase.Atoms], beta: float, include_h: bool = False
) -> PointCloudPrior:
    """
    Generate a PointCloudPrior from a list of ase.Atoms objects.
    Beta controls the size of the attractive region around each point. Low beta -> fuzzy cloud, high beta -> tight cloud.
    """
    if not include_h:
        ligand_positions = np.concatenate(
            [mol[mol.numbers != 1].get_positions() for mol in ligand_atoms]
        )
    else:
        ligand_positions = np.concatenate([mol.get_positions() for mol in ligand_atoms])
    return PointCloudPrior(ligand_positions, beta)


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
    num_molecules: int = typer.Option(
        default=100, help="Number of molecules to generate"
    ),
    num_heavy_atoms: int = typer.Option(
        default=4, help="Number of heavy atoms in generated molecules"
    ),
    beta: float = typer.Option(default=1.0, help="Beta for the prior"),
    additional_force_multiplier: float = typer.Option(
        1.0,
        "--additional-force-multiplier",
        "-mult",
        help="Additional multiplier for the restorative force strength",
    ),
    num_integration_steps: int = typer.Option(
        default=50, help="Number of integration steps for particle filter"
    ),
    track_trajectories: bool = typer.Option(
        default=False,
        help="If true, save all trajectory configurations instead of just the last",
    ),
    use_scaffold: bool = typer.Option(
        default=False, help="If true, use the OA as a scaffold"
    ),
):
    setup_logger(level=logging.INFO, tag="OA_generation", directory="./logs")

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
        model_repo_path=model_repo_path, device=DEVICE
    )
    integration_params = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)

    save_path: Path = Path(save_path)
    if save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    data_loader = zntrack.from_rev("OA_ligands", remote=model_repo_path)
    OA_ligands = data_loader.get_atoms()
    prior = construct_prior_from_atoms(OA_ligands, beta=beta)

    if use_scaffold:
        oa_structure = zntrack.from_rev("OA_parent", remote=model_repo_path)
        oa_structure = oa_structure.get_atoms()
    else:
        oa_structure = None

    swapping_z_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    for i in range(num_molecules):
        logging.info(f"Generating molecule {i}")
        size = num_heavy_atoms
        mol = initialize_mol(f"C{size}")
        restorative_force_strength = (
            additional_force_multiplier * calculate_restorative_force_strength(size)
        )
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
            scaffold=oa_structure,
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
