import ase
import numpy as np
from mace.tools import AtomicNumberTable

from moldiff.element_swapping import (
    collect_particles,
    sweep_all_elements,
)


def remove_isolated_atoms(atoms: ase.Atoms, cutoff: float) -> ase.Atoms:
    """
    Remove unconnected atoms from the final atoms object.
    """
    distances = atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    per_atom_min_distances = np.min(distances, axis=1)
    connected_atom_indices = np.where(per_atom_min_distances <= cutoff)[0]
    stripped_atoms = atoms.copy()
    stripped_atoms = stripped_atoms[connected_atom_indices]
    return stripped_atoms


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


def relax_elements(atoms: ase.Atoms, z_table: AtomicNumberTable) -> ase.Atoms:
    assert atoms.calc is not None
    atoms.info["time"] = 0.0
    atoms.info["calculation_type"] = "mace"
    already_switched = []
    mol = atoms.copy()
    for _ in range(len(mol)):
        mol.calc = atoms.calc
        energies = mol.get_potential_energies()
        idx = get_higest_energy_unswapped_idx(already_switched, energies)
        already_switched.append(idx)
        ensemble = sweep_all_elements(mol, idx, z_table)
        ensemble = [mol, *ensemble]
        ensemble = attach_calculator(ensemble, mol.calc, calculation_type="mace")
        mol = collect_particles(ensemble, beta=100.0)
    return mol


def cleanup_atoms(
    atoms: ase.Atoms, cutoff: float, z_table: AtomicNumberTable
) -> ase.Atoms:
    """
    Wrapper function to allow easy extension with other cleanup functions if needed.
    """
    pruned_atoms = remove_isolated_atoms(atoms, cutoff)
    pruned_atoms.calc = atoms.calc
    element_relaxed_atoms = relax_elements(pruned_atoms, z_table)
    return element_relaxed_atoms
