from collections import Counter
from typing import Dict, List, Tuple

import ase
import ase.io as aio
import numpy as np
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.morse import MorsePotential
from ase.optimize import LBFGS

from simgen.manifolds import StandardGaussianPrior

from .calculators import MopacCalculator, RestorativeCalculator


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


def do_mopac_relaxation(initialised_atoms, num_steps=100):
    calc = MopacCalculator()
    atoms = calc.do_full_relaxation(initialised_atoms.copy())
    mopac_calc = MopacCalculator(f_max=50.0)
    atoms = initialised_atoms.copy()
    atoms.calc = mopac_calc
    dyn = LBFGS(atoms, memory=10, maxstep=0.2)
    traj = []
    energies = []
    for i, _ in enumerate(dyn.irun(fmax=0.01, steps=num_steps)):
        traj.append(atoms.copy())
        energies.append(mopac_calc.results["energy"])
    return traj, energies


def do_hot_airss_relaxation(
    initialised_atoms,
    prior=None,
    num_steps=100,
    step_size=0.2,
) -> Tuple[List[ase.Atoms], Dict[str, List[float]]]:
    restorative_calc = RestorativeCalculator(prior_manifold=prior, zero_energy_radius=0)
    mopac_calc = MopacCalculator(f_max=50.0)
    calc = LinearCombinationCalculator([mopac_calc, restorative_calc], [1.0, 0.0])
    schedule = np.linspace(0, 1, num_steps)

    atoms = initialised_atoms.copy()
    atoms.calc = calc

    traj = [atoms.copy()]
    energies_dict = {"mopac": [], "restorative": [], "total": []}
    dyn = LBFGS(atoms, memory=10, maxstep=step_size)

    for i, _ in enumerate(dyn.irun(fmax=0.01, steps=num_steps)):
        weight = schedule[i]
        mopac_weight = weight
        restorative_weight = 1 - weight
        print(f"Step {i} of {num_steps}, weight {weight}")
        print(f"Restorative weight: {restorative_weight}")
        print(f"Mopac weight: {mopac_weight}")
        atoms.calc.weights = [mopac_weight, restorative_weight]
        traj.append(atoms.copy())
        _update_energy_dict(
            energies_dict, mopac_calc, restorative_calc, restorative_weight
        )
        if i == num_steps - 1:  # ase does one extra step
            break

    return traj, energies_dict


def _update_energy_dict(
    dict_to_update, mopac_calc, restorative_calc, restorative_weight
):
    dict_to_update["mopac"].append(mopac_calc.results["energy"])
    dict_to_update["restorative"].append(restorative_calc.results["energy"])
    combined = (
        mopac_calc.results["energy"] * (1 - restorative_weight)
        + restorative_calc.results["energy"] * restorative_weight
    )
    dict_to_update["total"].append(combined)


def get_composition(sybmol_list):
    counts = Counter(sybmol_list)
    list_representation = [f"{k}{v}" for k, v in counts.items()]
    list_representation.sort()
    return "".join(list_representation)


def get_composition_counter(qm9_path):
    all_atoms = aio.read(qm9_path, index=":", format="extxyz")
    compositions = [
        get_composition(atoms.get_chemical_symbols()) for atoms in all_atoms  # type: ignore
    ]
    counts = Counter(compositions)
    return counts
