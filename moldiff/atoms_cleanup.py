import ase
import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import BFGS
from mace.tools import AtomicNumberTable

from moldiff.element_swapping import (
    collect_particles,
    sweep_all_elements,
)
from moldiff.generation_utils import duplicate_atoms


def run_dynamics(atoms_list):
    for atom in atoms_list:
        dyn = BFGS(atom, maxstep=0.1 / 5)
        dyn.run(fmax=0.01, steps=5)
    return atoms_list


def remove_isolated_atoms_fixed_cutoff(atoms: ase.Atoms, cutoff: float) -> ase.Atoms:
    """
    Remove unconnected atoms from the final atoms object.
    """
    distances = atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    per_atom_min_distances = np.min(distances, axis=1)
    connected_atom_indices = np.where(per_atom_min_distances <= cutoff)[0]
    stripped_atoms = duplicate_atoms(atoms)
    stripped_atoms = stripped_atoms[connected_atom_indices]
    return stripped_atoms  # type: ignore


def remove_isolated_atoms_using_covalent_radii(
    atoms: ase.Atoms, multiplier: float = 1.2
) -> ase.Atoms:
    """
    Remove unconnected atoms from the final atoms object.
    """
    cutoffs = natural_cutoffs(atoms, mult=multiplier)  # type: ignore
    indices_of_connected_atoms = neighbor_list("i", atoms, cutoffs)
    unique_indices = np.unique(indices_of_connected_atoms)
    stripped_atoms = atoms.copy()
    stripped_atoms = stripped_atoms[unique_indices]
    return stripped_atoms  # type: ignore


def get_higest_energy_unswapped_idx(swapped_indices, energies) -> int:
    """
    Get the index of the atom with the highest energy that has not been swapped.
    """
    energies = energies.copy()
    energies[swapped_indices] = -np.inf
    return np.argmax(energies).astype(int)


def attach_calculator(atoms_list, calculator, calculation_type="similarity"):
    for atoms in atoms_list:
        atoms.info["calculation_type"] = calculation_type
        atoms.calc = calculator
    return atoms_list


def relax_elements(
    atoms: ase.Atoms, z_table: AtomicNumberTable, should_run_dynamics: bool = True
) -> ase.Atoms:
    assert atoms.calc is not None
    atoms.info["time"] = 0.0
    atoms.info["calculation_type"] = "mace"
    already_switched = []
    mol = duplicate_atoms(atoms)
    for _ in range(len(mol)):
        mol.calc = atoms.calc
        energies = mol.get_potential_energies()
        idx = get_higest_energy_unswapped_idx(already_switched, energies)
        already_switched.append(idx)
        ensemble = sweep_all_elements(mol, idx, z_table)
        ensemble = [mol, *ensemble]
        ensemble = attach_calculator(ensemble, mol.calc, calculation_type="mace")
        if should_run_dynamics:
            ensemble = run_dynamics(ensemble)
        mol = collect_particles(ensemble, beta=100.0)
    return mol


def cleanup_atoms(
    atoms: ase.Atoms, cutoff: float, z_table: AtomicNumberTable
) -> ase.Atoms:
    """
    Wrapper function to allow easy extension with other cleanup functions if needed.
    """
    pruned_atoms = remove_isolated_atoms_using_covalent_radii(atoms)
    pruned_atoms.calc = atoms.calc
    element_relaxed_atoms = relax_elements(pruned_atoms, z_table)
    return element_relaxed_atoms
