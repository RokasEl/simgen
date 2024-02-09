import logging
from time import monotonic

import ase
import numpy as np
from ase.optimize import LBFGS
from hydromace.interface import HydroMaceCalculator

from simgen.atoms_cleanup import attach_calculator, relax_hydrogens
from simgen.calculators import MaceSimilarityCalculator
from simgen.generation_utils import calculate_restorative_force_strength
from simgen.hydrogenation import (
    hydrogenate_deterministically,
    hydrogenate_hydromace,
)
from simgen.manifolds import PointCloudPrior
from simgen.particle_filtering import ParticleFilterGenerator
from simgen_zndraw import (
    DEFAULT_GENERATION_PARAMS,
    DEFAULT_INTEGRATION_PARAMS,
)

from .data import RequestAtoms
from .utils import get_edge_array


def generate(request: RequestAtoms, simgen_calc: MaceSimilarityCalculator, *args):
    prior = PointCloudPrior(request.points, beta=DEFAULT_GENERATION_PARAMS.prior_beta)
    restorative_force = (
        calculate_restorative_force_strength(request.num_atoms_to_add)
        * request.restorative_force_multiplier
    )
    generator = ParticleFilterGenerator(
        simgen_calc,
        prior,
        integration_parameters=DEFAULT_INTEGRATION_PARAMS,
        device=simgen_calc.device,
        restorative_force_strength=restorative_force,
        num_steps=request.max_steps,
    )
    mol = ase.Atoms(f"C{request.num_atoms_to_add}")
    try:
        trajectory = generator.generate(
            mol,
            swapping_z_table=DEFAULT_GENERATION_PARAMS.swapping_table,
            num_particles=DEFAULT_GENERATION_PARAMS.num_particles,
            particle_swap_frequency=DEFAULT_GENERATION_PARAMS.particle_swap_frequency,
            do_final_cleanup=False,
            scaffold=request.atoms,
            timeout=request.timeout,
        )
        return trajectory, None
    except (RuntimeError, ValueError) as e:
        logging.error(f"Error generating molecule: {e}")
        trajectory = []
        return trajectory, e


def hydrogenate(
    request: RequestAtoms,
    simgen_calc: MaceSimilarityCalculator,
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
    mask = np.where(hydrogenated.get_atomic_numbers() != 1)[0]
    to_relax = attach_calculator(
        [hydrogenated.copy()], simgen_calc, calculation_type="mace", mask=mask
    )[0]
    relaxation_trajectory = [to_relax.copy()]
    dyn = LBFGS(to_relax, maxstep=0.2)
    start = monotonic()
    for _ in dyn.irun(fmax=0.01, steps=request.max_steps):
        if monotonic() - start > request.timeout:
            logging.info("Relaxation taking too long, stopping")
            break
        relaxation_trajectory.append(to_relax.copy())
    return relaxation_trajectory, None


def relax(request: RequestAtoms, simgen_calc: MaceSimilarityCalculator, *args):
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
        [atoms], simgen_calc, calculation_type="mace", mask=mask
    )[0]
    logging.info("Relaxing structure")
    relaxation_trajectory = [relaxed_atoms.copy()]
    dyn = LBFGS(relaxed_atoms, maxstep=0.2)
    start = monotonic()
    for _ in dyn.irun(fmax=0.01, steps=request.max_steps):
        if monotonic() - start > request.timeout:
            logging.info("Relaxation taking too long, stopping")
            break
        relaxation_trajectory.append(relaxed_atoms.copy())
    logging.info("Finished relaxation")
    return relaxation_trajectory, None
