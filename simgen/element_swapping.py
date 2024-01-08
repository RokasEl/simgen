import logging
from typing import List, Sequence, Tuple

import numpy as np
from ase import Atoms
from mace.tools import AtomicNumberTable
from scipy.special import softmax  # type: ignore

from simgen.generation_utils import duplicate_atoms


class SwappingAtomicNumberTable(AtomicNumberTable):
    def __init__(
        self, zs: Sequence[int], swap_frequencies: None | Sequence[int | float] = None
    ):
        self.zs = zs
        if swap_frequencies is None:
            swap_frequencies = [1 for _ in zs]
        if len(zs) != len(swap_frequencies):
            raise ValueError(
                "Swap frequencies must have the same number of elements as `zs`"
            )
        self.swap_frequencies = np.array(swap_frequencies)


def replace_array_elements_at_indices_with_other_elements_in_z_table(
    atomic_numbers: np.ndarray,
    indices_to_replace: np.ndarray | list[int],
    z_table: AtomicNumberTable,
) -> np.ndarray:
    x = atomic_numbers.copy()
    swapping_dictionary = get_element_and_swap_frequency_dictionary(z_table)
    for idx in indices_to_replace:
        new_z = get_new_element_from_swapping_dictionary(x[idx], swapping_dictionary)
        x[idx] = new_z
    return x


def get_element_and_swap_frequency_dictionary(z_table):
    if not isinstance(z_table, SwappingAtomicNumberTable):
        swap_table = SwappingAtomicNumberTable(z_table.zs)
    else:
        swap_table = z_table
    return dict(zip(swap_table.zs, swap_table.swap_frequencies))


def get_new_element_from_swapping_dictionary(current_element, swap_dictionary):
    temp_dict = swap_dictionary.copy()
    temp_dict[current_element] = 0
    keys, values = list(temp_dict.keys()), list(temp_dict.values())
    ps = np.array(values) / np.sum(values)
    return np.random.choice(keys, p=ps)


def swap_single_particle(
    mol: Atoms, to_change: list[int], z_table: AtomicNumberTable
) -> Atoms:
    swapped_mol = duplicate_atoms(mol)
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
    num_change = np.ceil(0.2 * num_particles).astype(int)
    return num_change


def choose_indices_to_change(energies: np.ndarray, beta: float, num_change: int):
    probabilities = softmax(energies * beta)
    energies_copy = energies.copy()
    remaining_to_change = num_change
    indices_to_change = []
    while remaining_to_change > 0:
        if np.count_nonzero(probabilities) < remaining_to_change:
            highest_e_index = np.argsort(energies_copy)[-1]
            energies_copy[highest_e_index] = -np.inf
            probabilities = softmax(energies_copy * beta)
            indices_to_change.append(highest_e_index)
            remaining_to_change -= 1
        else:
            indices_to_change += list(
                np.random.choice(
                    len(energies),
                    size=remaining_to_change,
                    replace=False,
                    p=probabilities,
                )
            )
            remaining_to_change = 0
    return indices_to_change


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
    num_change = get_how_many_to_change(sum(mask), beta)
    ensemble = [duplicate_atoms(atoms)]
    to_generate = num_particles - 1

    for _ in range(to_generate):
        to_change = choose_indices_to_change(energies, beta, num_change)
        ensemble.append(swap_single_particle(atoms, to_change, z_table))
    return ensemble


def apply_mask_to_probabilities(probibilities, mask):
    masked_probs = probibilities * mask
    masked_probs /= masked_probs.sum()
    return masked_probs


def collect_particles(
    ensemble: List[Atoms],
    beta: float,
) -> Tuple[Atoms, int]:
    energies = np.array([mol.get_potential_energy() for mol in ensemble])
    energies = energies.flatten()
    num_atoms = np.mean([len(mol) for mol in ensemble])
    energies = catch_diverged_energies(energies, num_atoms) * beta * -1
    probabilities = softmax(energies)
    collect_idx = np.random.choice(len(ensemble), p=probabilities)
    return duplicate_atoms(ensemble[collect_idx]), collect_idx


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
