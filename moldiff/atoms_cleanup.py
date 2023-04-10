from typing import List, Literal

import ase
import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import BFGS
from mace.tools import AtomicNumberTable
from scipy.stats import betabinom

from moldiff.element_swapping import (
    collect_particles,
    sweep_all_elements,
)
from moldiff.generation_utils import (
    duplicate_atoms,
    get_edge_array_and_neighbour_numbers,
)
from moldiff.hydrogenation import hydrogenate_stochastically


def run_dynamics(atoms_list: List[ase.Atoms], num_steps=5, max_step=0.2):
    for atom in atoms_list:
        dyn = BFGS(atom, maxstep=max_step)
        dyn.run(fmax=0.01, steps=num_steps)
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


def get_higest_energy_unswapped_idx(swapped_indices: list, energies: np.ndarray) -> int:
    """
    Get the index of the atom with the highest energy that has not been swapped.
    """
    if len(swapped_indices) == len(energies):
        raise ValueError("Attempting to swap once all atoms have been swapped.")
    energies = energies.copy()
    energies[swapped_indices] = -np.inf
    return np.argmax(energies).astype(int)


def attach_calculator(
    atoms_list, calculator: Calculator, calculation_type="similarity", mask=None
):
    for atoms in atoms_list:
        atoms.info["calculation_type"] = calculation_type
        if mask is not None:
            atoms.info["mask"] = mask
        atoms.calc = calculator
    return atoms_list


def relax_hydrogens(atoms_list: List[ase.Atoms], calculator, num_steps=5, max_step=0.2):
    for atoms in atoms_list:
        atoms.info["calculation_type"] = "mace"
        atoms.info["mask"] = np.where(atoms.get_atomic_numbers() != 1)[0]
        atoms.calc = calculator
    atoms_list = run_dynamics(atoms_list, num_steps=num_steps, max_step=max_step)
    return atoms_list


def determine_number_of_element_swaps(num_element_sweeps, already_switched, mol):
    if num_element_sweeps == "all":
        num_element_sweeps = len(mol)
    num_element_sweeps = min(num_element_sweeps, len(mol) - len(already_switched))
    return num_element_sweeps


def relax_elements(
    atoms: ase.Atoms,
    z_table: AtomicNumberTable,
    should_run_dynamics: bool = True,
    num_element_sweeps: int | Literal["all"] = "all",
) -> ase.Atoms:
    """should_run_dynamics: should only be False for testing"""
    assert atoms.calc is not None
    atoms.info["time"] = 0.0
    atoms.info["calculation_type"] = "mace"
    already_switched = [idx for idx in range(len(atoms)) if atoms.numbers[idx] == 1]
    mol = duplicate_atoms(atoms)
    edge_array, _ = get_edge_array_and_neighbour_numbers(mol, mult=1.2)
    num_element_sweeps = determine_number_of_element_swaps(
        num_element_sweeps, already_switched, mol
    )
    calc = atoms.calc
    for _ in range(num_element_sweeps):
        mol.calc = calc
        if "mask" in mol.info:
            del mol.info["mask"]
        calc.calculate(mol)
        energies = mol.get_potential_energies()
        idx = get_higest_energy_unswapped_idx(already_switched, energies)
        already_switched.append(idx)
        neighbours = edge_array[edge_array[:, 0] == idx][:, 1]
        mask = np.append(neighbours, idx)
        ensemble = sweep_all_elements(mol, idx, z_table)
        ensemble = [mol, *ensemble]
        ensemble = attach_calculator(
            ensemble, mol.calc, calculation_type="mace", mask=mask
        )
        if should_run_dynamics:
            ensemble = run_dynamics(ensemble)
        mol = collect_particles(ensemble, beta=100.0)
    return mol


def cleanup_atoms(
    atoms: ase.Atoms,
    z_table: AtomicNumberTable,
    num_hydrogenations: int = 10,
    num_element_sweeps: int | Literal["all"] = "all",
) -> ase.Atoms:
    """
    Wrapper function to allow easy extension with other cleanup functions if needed.
    """
    assert atoms.calc is not None
    calc: Calculator = atoms.calc
    pruned_atoms = remove_isolated_atoms_using_covalent_radii(atoms)
    hydrogenated_atoms_ensemble = [
        hydrogenate_stochastically(pruned_atoms) for _ in range(num_hydrogenations)
    ]
    hydrogenated_atoms_ensemble = relax_hydrogens(
        hydrogenated_atoms_ensemble, calc, num_steps=20, max_step=0.1
    )
    element_relaxed_atoms = [
        relax_elements(
            hydrogenated_atoms, z_table, num_element_sweeps=num_element_sweeps
        )
        for hydrogenated_atoms in hydrogenated_atoms_ensemble
    ]
    element_relaxed_atoms = attach_calculator(
        element_relaxed_atoms, calc, calculation_type="mace"
    )
    lowest_energy_atoms = collect_particles(element_relaxed_atoms, beta=100.0)
    return lowest_energy_atoms
