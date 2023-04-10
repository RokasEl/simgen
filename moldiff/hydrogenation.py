from itertools import product

import ase
import numpy as np
from ase.data import covalent_radii
from frozendict import frozendict
from scipy.stats import binom

from moldiff.generation_utils import (
    get_edge_array_and_neighbour_numbers,
)

NATURAL_VALENCES = frozendict(
    {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
)


def hydrogenate_stochastically(atoms):
    _, num_neighbours = get_edge_array_and_neighbour_numbers(atoms)
    num_hs_to_add_per_atom = np.zeros(len(atoms)).astype(int)
    atomic_numbers = atoms.get_atomic_numbers()
    for idx in range(len(atoms)):
        current_valence = num_neighbours[idx]
        max_valence = NATURAL_VALENCES[atomic_numbers[idx]]
        num_hs_to_add_per_atom[idx] = sample_number_of_hs_to_add(
            current_valence, max_valence
        )

    atoms_with_hs = add_hydrogens_to_atoms(atoms, num_hs_to_add_per_atom)
    return atoms_with_hs


def add_hydrogens_to_atoms(
    atoms: ase.Atoms,
    number_of_hs_to_add: np.ndarray,
) -> ase.Atoms:
    """
    Add hydrogens to the atoms object
    """
    atoms_copy = atoms.copy()
    atomic_numbers = atoms_copy.get_atomic_numbers()
    exclusion_radii_dict = get_exclusion_radii_dict(atomic_numbers)
    h_radius = covalent_radii[1]
    bond_lengths = np.asarray([h_radius + covalent_radii[z] for z in atomic_numbers])

    for idx in range(len(atoms)):
        for _ in range(number_of_hs_to_add[idx]):
            valid, h_pos = find_valid_h_position(
                bond_lengths[idx], idx, atoms_copy, exclusion_radii_dict
            )
            if valid:
                atoms_copy += ase.Atoms("H", positions=[h_pos])
            else:
                break
    return atoms_copy


def sample_number_of_hs_to_add(current_valence, max_valence):
    if current_valence >= max_valence:
        return 0
    else:
        num_missing_bonds = max_valence - current_valence
        # bias sampling towards fully saturating the atom
        p = num_missing_bonds / (num_missing_bonds + 1)
        to_add = binom.rvs(num_missing_bonds, p)
        return to_add


def find_valid_h_position(
    radius, central_atom_idx, atoms, exclusion_radii_dict, max_tries=20
):
    """Find a valid position for a hydrogen using rejection sampling"""
    directions = get_random_unit_vectors(num=max_tries)
    possible_h_positions = atoms.positions[central_atom_idx] + radius * directions
    exclusion_radii = (
        get_exclusion_radii(atoms, central_atom_idx, exclusion_radii_dict) * 1.4
    )
    for pos in possible_h_positions:
        if check_hydrogen_position_is_valid(
            pos, exclusion_radii, atoms.get_positions()
        ):
            return True, pos
    return False, None


def get_exclusion_radii_dict(atomic_numbers: np.ndarray) -> dict:
    exclusion_radii_dict = {}
    possible_elements = np.unique(atomic_numbers).tolist()
    possible_elements.append(1)
    for i, j in product(possible_elements, possible_elements):
        if i not in exclusion_radii_dict:
            exclusion_radii_dict[i] = {}
        exclusion_radii_dict[i][j] = compute_exclusion_radius(
            covalent_radii[i], covalent_radii[j], covalent_radii[1]
        )
    return exclusion_radii_dict


def get_random_unit_vectors(num: int) -> np.ndarray:
    vectors = np.random.normal(size=(num, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def compute_exclusion_radius(
    r_to_which_h_added: float, r_other_atom: float, h_radius: float
) -> float:
    cylinder_radius = max(r_to_which_h_added, r_other_atom)
    length_to_intersection_with_cylinder = r_to_which_h_added + h_radius
    cylinder_length = r_to_which_h_added + r_other_atom
    if length_to_intersection_with_cylinder < cylinder_radius:
        exclusion_radius_squared = (
            cylinder_length**2 + length_to_intersection_with_cylinder**2
        )
        exclusion_radius = np.sqrt(exclusion_radius_squared)
    else:
        angle = np.arcsin(cylinder_radius / (r_to_which_h_added + h_radius))
        cosine = np.cos(angle)
        exclusion_radius_squared = (
            cylinder_length**2
            + length_to_intersection_with_cylinder**2
            - 2 * cosine * cylinder_length * length_to_intersection_with_cylinder
        )
        exclusion_radius = np.sqrt(exclusion_radius_squared)
    return exclusion_radius


def get_exclusion_radii(
    atoms: ase.Atoms, idx: int, exclusion_radius_dict: dict
) -> np.ndarray:
    exclusion_radii = np.zeros(len(atoms))
    atomic_numbers = atoms.get_atomic_numbers()
    this_element = atomic_numbers[idx]
    for i in range(len(atoms)):
        if i == idx:
            continue
        other_element = atomic_numbers[i]
        exclusion_radii[i] = exclusion_radius_dict[this_element][other_element]
    return exclusion_radii


def check_hydrogen_position_is_valid(
    h_position: np.ndarray,  # 1x3
    exclusion_radii: np.ndarray,  # 1xN
    other_atom_positions: np.ndarray,  # Nx3
) -> bool:
    """
    Check if the position of the hydrogen is valid
    """
    distances = np.linalg.norm(other_atom_positions - h_position, axis=1)  # type: ignore
    if np.any(distances < exclusion_radii):
        return False
    return True
