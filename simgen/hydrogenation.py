from itertools import product

import ase
import numpy as np
from ase.data import covalent_radii
from frozendict import frozendict
from hydromace.interface import HydroMaceCalculator
from scipy.stats import binom

from simgen.hydrogenation_deterministic import build_xae_molecule

NATURAL_VALENCES = frozendict(
    {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
)


def hydrogenate_deterministically(
    atoms: ase.Atoms,
    single_bond_stretch_factor: float = 1.1,
    multi_bond_stretch_factor: float = 1.05,
    edge_array: np.ndarray | None = None,
):
    if edge_array is None:
        edge_array = get_edge_array_from_atoms(
            atoms, single_bond_stretch_factor, multi_bond_stretch_factor
        )
    num_hs_to_add_per_atom = calculate_missing_hydrogens_by_valence(atoms, edge_array)
    atoms_with_hs = add_hydrogens_to_atoms(atoms, num_hs_to_add_per_atom)
    return atoms_with_hs


def calculate_missing_hydrogens_by_valence(
    atoms: ase.Atoms, edge_array: np.ndarray
) -> np.ndarray:
    max_valence = np.array(
        [
            NATURAL_VALENCES[atomic_number]
            for atomic_number in atoms.get_atomic_numbers()
        ]
    )
    current_neighbours = edge_array.sum(axis=0)
    return max_valence - current_neighbours


def get_edge_array_from_atoms(
    atoms: ase.Atoms,
    single_bond_stretch_factor: float = 1.1,
    multi_bond_stretch_factor: float = 1.05,
) -> np.ndarray:
    positions, atomic_symbols = atoms.get_positions(), atoms.get_chemical_symbols()
    _, _, edge_array = build_xae_molecule(
        positions,
        atomic_symbols,
        single_bond_stretch_factor=single_bond_stretch_factor,
        multi_bond_stretch_factor=multi_bond_stretch_factor,
    )
    return edge_array


def hydrogenate_hydromace(atoms: ase.Atoms, hydromace_calc: HydroMaceCalculator):
    num_hs_to_add = hydromace_calc.predict_missing_hydrogens(atoms)
    num_hs_to_add_per_atom = np.round(num_hs_to_add).astype(int)
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
    for idx in range(len(atoms)):
        for _ in range(number_of_hs_to_add[idx]):
            exclusion_radii = get_exclusion_radii(atoms_copy, idx)
            valid, h_pos = find_valid_h_position(
                exclusion_radii, idx, atoms_copy, max_tries=100
            )
            if valid:
                atoms_copy += ase.Atoms("H", positions=[h_pos])
            else:
                break
    return atoms_copy


def get_exclusion_radii(atoms, central_atom_idx):
    atomic_numbers = atoms.get_atomic_numbers()
    central_atom_radius = covalent_radii[atomic_numbers[central_atom_idx]]
    exclusion_radii = np.asarray(
        [central_atom_radius + covalent_radii[z] for z in atomic_numbers]
    )
    exclusion_radii[central_atom_idx] = 0
    return exclusion_radii


def find_valid_h_position(exclusion_radii, central_atom_idx, atoms, max_tries=20):
    """Find a valid position for a hydrogen using rejection sampling"""
    directions = get_random_unit_vectors(num=max_tries)
    possible_h_positions = atoms.positions[central_atom_idx] + 0.85 * directions
    for pos in possible_h_positions:
        if check_hydrogen_position_is_valid(
            pos, exclusion_radii, atoms.get_positions()
        ):
            return True, pos
    return False, None


def get_random_unit_vectors(num: int) -> np.ndarray:
    vectors = np.random.normal(size=(num, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


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
