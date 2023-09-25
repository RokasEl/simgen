import logging

import ase
import numpy as np
from hydromace.interface import HydroMaceCalculator

from moldiff.atoms_cleanup import (
    attach_calculator,
    relax_hydrogens,
    run_dynamics,
)
from moldiff.calculators import MaceSimilarityCalculator
from moldiff.generation_utils import (
    calculate_restorative_force_strength,
)
from moldiff.hydrogenation import (
    hydrogenate_deterministically,
    hydrogenate_hydromace,
)
from moldiff.manifolds import PointCloudPrior
from moldiff.particle_filtering import ParticleFilterGenerator
from moldiff_server import (
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_INTEGRATION_PARAMS,
)

from .data import RequestAtoms, jsonify_atoms
from .utils import get_edge_array


def generate(request: RequestAtoms, moldiff_calc: MaceSimilarityCalculator, *args):
    prior = PointCloudPrior(request.points, beta=DEFAULT_GENERATION_PARAMS.prior_beta)
    restorative_force = (
        calculate_restorative_force_strength(request.num_atoms_to_add)
        * request.restorative_force_multiplier
    )
    generator = ParticleFilterGenerator(
        moldiff_calc,
        prior,
        integration_parameters=DEFAULT_INTEGRATION_PARAMS,
        device=moldiff_calc.device,
        restorative_force_strength=restorative_force,
        num_steps=request.max_steps,
    )
    mol = ase.Atoms(f"C{request.num_atoms_to_add}")
    trajectory = generator.generate(
        mol,
        swapping_z_table=DEFAULT_GENERATION_PARAMS.swapping_table,
        num_particles=DEFAULT_GENERATION_PARAMS.num_particles,
        particle_swap_frequency=DEFAULT_GENERATION_PARAMS.particle_swap_frequency,
        do_final_cleanup=False,
        scaffold=request.atoms,
    )
    return jsonify_atoms(*trajectory)


def hydrogenate(
    request: RequestAtoms,
    moldiff_calc: MaceSimilarityCalculator,
    hydromace_calc: HydroMaceCalculator | None,
):
    if hydromace_calc is not None:
        hydrogenated = hydrogenate_hydromace(
            request.atoms, hydromace_calc=hydromace_calc
        )
    else:
        edge_array = get_edge_array(request.atoms)
        hydrogenated = hydrogenate_deterministically(
            request.atoms, edge_array=edge_array
        )
    to_relax = hydrogenated.copy()
    relaxed_atoms_with_h = relax_hydrogens(
        [to_relax], moldiff_calc, num_steps=request.max_steps, max_step=0.1
    )[0]
    return jsonify_atoms(hydrogenated, relaxed_atoms_with_h)


def relax(request: RequestAtoms, moldiff_calc: MaceSimilarityCalculator, *args):
    atoms = request.atoms
    atom_ids = request.atom_ids
    if len(atom_ids) != 0:
        logging.info("Will relax only the selected atoms")
        mask = np.ones(len(atoms)).astype(bool)
        mask[atom_ids] = False
    else:
        logging.info("Will relax all atoms")
        mask = np.zeros(len(atoms)).astype(bool)
    relaxed_atoms = attach_calculator(
        [atoms], moldiff_calc, calculation_type="mace", mask=mask
    )
    logging.info("Relaxing structure")
    relaxed_atoms = run_dynamics(relaxed_atoms, num_steps=request.max_steps)
    logging.info("Finished relaxation")
    return jsonify_atoms(*relaxed_atoms)
