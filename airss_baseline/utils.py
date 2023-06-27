from collections import Counter
from typing import List

import ase
import ase.io as aio
import numpy as np
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.morse import MorsePotential
from ase.optimize import LBFGS
from calculators import MopacCalculator, RestorativeCalculator

from moldiff.manifolds import StandardGaussianPrior


def build_mol(composition: str, prior=None, max_steps=1000):
    """A basic analogy of `builcell` in AIRSS. Get random starting positions that minimise
    overlap between atoms.

    Args:
        composition (str): A string representing the composition of the molecule. E.g. 'C2H6'.
    """
    atoms = ase.Atoms(composition)
    if prior is None:
        prior = StandardGaussianPrior()
    atoms = prior.initialise_positions(atoms, scale=1.0)
    morse_calc = MorsePotential(epsilon=3.0, r0=1.2, rho0=6.0)
    restorative_calc = RestorativeCalculator(prior_manifold=prior)
    calc = LinearCombinationCalculator([morse_calc, restorative_calc], [1.0, 1.0])
    atoms.calc = calc
    dynamics = LBFGS(atoms, trajectory="morse.traj")
    dynamics.run(fmax=0.01, steps=max_steps)
    return atoms


def do_mopac_relaxation(initialised_atoms):
    calc = MopacCalculator()
    atoms = calc.do_full_relaxation(initialised_atoms.copy())
    return atoms


def do_hot_airss_relaxation(
    initialised_atoms,
    prior=None,
    num_steps=100,
    step_size=0.2,
    rattle_sigma: float = 1,
) -> List[ase.Atoms]:
    restorative_calc = RestorativeCalculator(
        prior_manifold=prior, zero_energy_radius=1.5
    )
    mopac_calc = MopacCalculator(f_max=1.0)
    calc = LinearCombinationCalculator([mopac_calc, restorative_calc], [1.0, 1.0])
    cosine_schedule = np.linspace(0, np.pi / 2, num_steps)
    cosine_schedule = np.cos(cosine_schedule) ** 2

    atoms = initialised_atoms.copy()
    atoms.calc = calc

    traj = [atoms.copy()]

    for weight in cosine_schedule:
        mopac_weight = 1 - weight
        restorative_weight = weight
        atoms.calc.weights = [mopac_weight, restorative_weight]
        forces = atoms.get_forces()
        noise = np.random.normal(size=atoms.positions.shape) * rattle_sigma * weight
        total_force = forces + noise
        atoms.set_positions(atoms.get_positions() + total_force * step_size)
        traj.append(atoms.copy())
    final_atoms = mopac_calc.do_full_relaxation(atoms.copy())
    traj.append(final_atoms)
    return traj


def get_composition(sybmol_list):
    counts = Counter(sybmol_list)
    list_representation = [f"{k}{v}" for k, v in counts.items()]
    list_representation.sort()
    return "".join(list_representation)


def get_composition_counter(qm9_path):
    all_atoms = aio.read(qm9_path, index=":")
    compositions = [
        get_composition(atoms.get_chemical_symbols()) for atoms in all_atoms  # type: ignore
    ]
    counts = Counter(compositions)
    return counts
