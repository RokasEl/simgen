from typing import List

import numpy as np
from ase import Atoms
from mace.tools import AtomicNumberTable
from scipy.special import softmax


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
    swapped_mol = mol.copy()
    to_change = np.random.choice(
        len(mol), size=num_change, replace=False, p=probabilities
    )
    new_zs = replace_array_elements_at_indices_with_other_elements_in_z_table(
        mol.get_atomic_numbers(), to_change, z_table
    )
    swapped_mol.set_atomic_numbers(new_zs)
    return swapped_mol


def sweep_all_elements(mol: Atoms, idx: int, z_table: AtomicNumberTable) -> list[Atoms]:
    original_atomic_numbers = mol.get_atomic_numbers()
    original_z = original_atomic_numbers[idx]
    other_elements = np.setdiff1d(z_table.zs, original_z)
    swapped_mols = []
    for z in other_elements:
        new_atomic_numbers = original_atomic_numbers.copy()
        new_atomic_numbers[idx] = z
        mol.set_atomic_numbers(new_atomic_numbers)
        swapped_mols.append(mol.copy())
    return swapped_mols


def get_how_many_to_change(num_particles: int, beta: float) -> int:
    p_change = np.exp(-beta)
    num_change = np.random.binomial(num_particles, p_change)
    return num_change


def create_element_swapped_particles(
    atoms: Atoms, beta: float, num_particles: int, z_table: AtomicNumberTable
) -> List[Atoms]:
    assert atoms.calc is not None
    energies = (
        atoms.get_potential_energies() * beta
    )  # no minus sign since we want to swap the highest energy atom
    probabilities = softmax(energies)
    probabilities[probabilities < 1e-3] = 0

    ensemble = [atoms.copy()]
    to_generate = num_particles - 1
    if np.count_nonzero(probabilities) == 1:
        idx = np.argmax(probabilities).astype(int)
        sweep_over_max_energy_atom = sweep_all_elements(atoms, idx, z_table)
        ensemble.extend(sweep_over_max_energy_atom)
        already_generated_num = len(ensemble)
        to_generate = num_particles - already_generated_num
        probabilities = np.ones(len(atoms)) / len(atoms)

    if to_generate > 0:
        non_zero_ps = np.count_nonzero(probabilities)
        for _ in range(to_generate):
            num_change = min(get_how_many_to_change(len(atoms), beta), non_zero_ps)
            ensemble.append(
                swap_single_particle(atoms, probabilities, num_change, z_table)
            )
    return ensemble


def collect_particles(
    ensemble: List[Atoms],
    beta: float,
) -> Atoms:
    energies = np.array([mol.get_potential_energy() for mol in ensemble]) * beta * -1
    probabilities = softmax(energies)
    collect_idx = np.random.choice(len(ensemble), p=probabilities)
    return ensemble[collect_idx]
