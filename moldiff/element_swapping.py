import logging
from typing import List

import numpy as np
from ase import Atoms
from mace.tools import AtomicNumberTable
from scipy.special import softmax

from moldiff.generation_utils import duplicate_atoms


def replace_array_elements_at_indices_with_other_elements_in_z_table(
    atomic_numbers: np.ndarray,
    indices_to_replace: np.ndarray,
    z_table: AtomicNumberTable,
) -> np.ndarray:
    x = atomic_numbers.copy()
    for idx in indices_to_replace:
        new_zs = np.setdiff1d(z_table.zs, x[idx])
        x[idx] = np.random.choice(new_zs)
    return x


def swap_single_particle(
    mol: Atoms, probabilities: np.ndarray, num_change: int, z_table: AtomicNumberTable
) -> Atoms:
    swapped_mol = duplicate_atoms(mol)
    to_change = np.random.choice(
        len(mol), size=num_change, replace=False, p=probabilities
    )
    new_zs = replace_array_elements_at_indices_with_other_elements_in_z_table(
        mol.get_atomic_numbers(), to_change, z_table
    )
    swapped_mol.set_atomic_numbers(new_zs)
    return swapped_mol


def sweep_all_elements(mol: Atoms, idx: int, z_table: AtomicNumberTable) -> list[Atoms]:
    mol_copy = duplicate_atoms(mol)
    original_atomic_numbers = mol_copy.get_atomic_numbers()
    original_z = original_atomic_numbers[idx]
    other_elements = np.setdiff1d(z_table.zs, original_z)
    swapped_mols = []
    for z in other_elements:
        new_atomic_numbers = original_atomic_numbers.copy()
        new_atomic_numbers[idx] = z
        mol_copy.set_atomic_numbers(new_atomic_numbers)
        swapped_mols.append(duplicate_atoms(mol_copy))
    return swapped_mols


def get_how_many_to_change(num_particles: int, beta: float) -> int:
    num_change = np.ceil(0.1 * num_particles).astype(int)
    return num_change


def create_element_swapped_particles(
    atoms: Atoms, beta: float, num_particles: int, z_table: AtomicNumberTable, mask=None
) -> List[Atoms]:
    assert atoms.calc is not None
    if mask is None:
        mask = np.ones(len(atoms))
    energies = (
        atoms.get_potential_energies() * beta
    )  # no minus sign since we want to swap the highest energy atom
    energies[mask == 0] = -np.inf
    probabilities = apply_mask_to_probabilities(softmax(energies), mask)
    logging.debug(f"Probabilities: {probabilities}, beta: {beta}")
    ensemble = [duplicate_atoms(atoms)]
    to_generate = num_particles - 1
    if np.count_nonzero(probabilities) == 1:
        idx = np.argmax(probabilities).astype(int)
        sweep_over_max_energy_atom = sweep_all_elements(atoms, idx, z_table)
        ensemble.extend(sweep_over_max_energy_atom)
        already_generated_num = len(ensemble)
        to_generate = num_particles - already_generated_num
        logging.debug(f"Sweeping particle {idx} with {to_generate} particles")
        logging.debug(f"Generating additional {to_generate} particles")
        probabilities = apply_mask_to_probabilities(np.ones(len(atoms)), mask)

    if to_generate > 0:
        non_zero_ps = np.count_nonzero(probabilities)
        for _ in range(to_generate):
            num_change = min(get_how_many_to_change(sum(mask), beta), non_zero_ps)
            logging.debug(f"Num change: {num_change}")
            ensemble.append(
                swap_single_particle(atoms, probabilities, num_change, z_table)
            )
    return ensemble


def apply_mask_to_probabilities(probibilities, mask):
    masked_probs = probibilities * mask
    masked_probs /= masked_probs.sum()
    return masked_probs


def collect_particles(
    ensemble: List[Atoms],
    beta: float,
) -> Atoms:
    energies = np.array([mol.get_potential_energy() for mol in ensemble])
    energies = energies.flatten()
    num_atoms = np.mean([len(mol) for mol in ensemble])
    energies = catch_diverged_energies(energies, num_atoms) * beta * -1
    probabilities = softmax(energies)
    collect_idx = np.random.choice(len(ensemble), p=probabilities)
    return duplicate_atoms(ensemble[collect_idx])


def catch_diverged_energies(
    energies: np.ndarray, num_atoms: float, energy_divergence_threshold: float = -5.0
) -> np.ndarray:
    """
    ML potentials can have energy 'holes' where energy becomes super low, even though the structure contains wierd geometries.
    This function catches these cases and sets the energy to a high value, so that the particle is not selected.
    """
    energy_holes_indices = np.where(energies < energy_divergence_threshold * num_atoms)[
        0
    ]
    energies[energy_holes_indices] = 1e3
    return energies
