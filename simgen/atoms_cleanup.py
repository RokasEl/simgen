from typing import List, Literal, Tuple

import ase
import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import LBFGS
from mace.tools import AtomicNumberTable

from simgen.element_swapping import (
    collect_particles,
    sweep_all_elements,
)
from simgen.generation_utils import (
    duplicate_atoms,
    get_edge_array_and_neighbour_numbers,
)
from simgen.hydrogenation import (
    hydrogenate_deterministically,
    hydrogenate_hydromace,
)


def run_dynamics(atoms_list: List[ase.Atoms], num_steps=5, max_step=0.2):
    for atom in atoms_list:
        dyn = LBFGS(atom, maxstep=max_step)
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


def add_hydrogens(atoms: ase.Atoms, hydrogenation_type: str, hydrogenation_calc):
    if hydrogenation_type.lower().strip() == "valence":
        atoms = hydrogenate_deterministically(atoms)
    elif hydrogenation_type.lower().strip() == "hydromace":
        if hydrogenation_calc is None:
            raise ValueError(
                "Trying to use model to hydrogenate, but no model provided."
            )
        atoms = hydrogenate_hydromace(atoms, hydrogenation_calc)
    else:
        raise NotImplementedError(
            f"Hydrogenation type {hydrogenation_type} not implemented."
        )

    return atoms


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


def get_swapping_candidates(
    mol, idx, neighbours, already_switched, z_table
) -> Tuple[List[ase.Atoms], List[int]]:
    """
    Generate ensemble of molecules with one element swapped.
    We construct the ensemble by swapping the highest energy atom and its neighbours.
    """
    candidates = [idx] + [idx for idx in neighbours if idx not in already_switched]
    ensemble = []
    swapped_indices = []
    for candidate_idx in candidates:
        sweep = sweep_all_elements(mol, candidate_idx, z_table)
        ensemble.extend(sweep)
        swapped_indices += [candidate_idx] * len(sweep)
    return ensemble, swapped_indices


def relax_elements(
    atoms: ase.Atoms,
    z_table: AtomicNumberTable,
    num_element_sweeps: int | Literal["all"] = "all",
    mask: np.ndarray | None = None,
) -> ase.Atoms:
    assert atoms.calc is not None
    atoms.info["time"] = 0.0
    atoms.info["calculation_type"] = "mace"
    already_switched = [idx for idx in range(len(atoms)) if atoms.numbers[idx] == 1]
    if mask is not None:
        masked_atoms = np.where(mask == 0)[0]
        already_switched += list(masked_atoms)
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
        neighbours = edge_array[edge_array[:, 0] == idx][:, 1]
        ensemble, swapped_indices = get_swapping_candidates(
            mol, idx, neighbours, already_switched, z_table
        )
        ensemble = [mol, *ensemble]
        ensemble = attach_calculator(ensemble, calc, calculation_type="mace")
        swapped_indices = [idx, *swapped_indices]
        mol, lowest_energy_index = collect_particles(ensemble, beta=100.0)
        already_switched.append(swapped_indices[lowest_energy_index])
    return mol


def cleanup_atoms(
    atoms: ase.Atoms,
    hydrogenation_type: str,
    hydrogenation_calc,
    z_table: AtomicNumberTable,
    num_element_sweeps: int | Literal["all"] = "all",
    mask=None,
) -> list[ase.Atoms]:
    """
    Wrapper function to allow easy extension with other cleanup functions if needed.
    """
    assert atoms.calc is not None
    calc: Calculator = atoms.calc
    pruned_atoms = remove_isolated_atoms_using_covalent_radii(atoms)
    hydrogenated_atoms = add_hydrogens(
        pruned_atoms.copy(), hydrogenation_type, hydrogenation_calc
    )
    relaxed_hydrogenated_atoms = relax_hydrogens(
        [hydrogenated_atoms.copy()], calc, num_steps=30, max_step=0.05
    )[0]
    element_relaxed_atoms = relax_elements(
        relaxed_hydrogenated_atoms,
        z_table,
        num_element_sweeps=num_element_sweeps,
        mask=mask,
    )
    final_relaxed_atoms = attach_calculator(
        [element_relaxed_atoms.copy()], calc, calculation_type="mace"
    )
    final_relaxed_atoms = run_dynamics(final_relaxed_atoms, num_steps=30, max_step=0.1)[
        0
    ]
    final_relaxed_atoms = remove_isolated_atoms_using_covalent_radii(
        final_relaxed_atoms
    )
    return [
        pruned_atoms,
        hydrogenated_atoms,
        relaxed_hydrogenated_atoms,
        element_relaxed_atoms,
        final_relaxed_atoms,
    ]
