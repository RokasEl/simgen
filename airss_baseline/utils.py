from collections import Counter

import ase
import ase.io as aio
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.morse import MorsePotential
from ase.optimize import LBFGS
from calculators import MOPACLight, RestorativeCalculator

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
    calc = MOPACLight()
    calc.calculate(initialised_atoms)
    return calc.atoms


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
