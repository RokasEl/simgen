import ase
import numpy as np


def hydrogenate(atoms, sampling_function, *sampling_function_args):
    num_hs_to_add_per_atom = sampling_function(*sampling_function_args, size=len(atoms))
    return add_hydrogens_to_atoms(atoms, num_hs_to_add_per_atom)


def add_hydrogens_to_atoms(
    atoms: ase.Atoms,
    number_of_hs_to_add: np.ndarray,
    h_min_distance: float = 0.7,
    h_max_distance: float = 0.9,
) -> ase.Atoms:
    """
    Add hydrogens to the atoms object
    """
    atoms_copy = atoms.copy()
    distance_from_central_atoms = np.random.uniform(
        h_min_distance, h_max_distance, size=sum(number_of_hs_to_add)
    )
    directions = get_random_unit_vectors(num=sum(number_of_hs_to_add))
    h_offset_vectors = distance_from_central_atoms[:, np.newaxis] * directions
    h_positions = add_h_offsets_to_original_positions(
        atoms_copy.positions, h_offset_vectors, number_of_hs_to_add
    )
    atoms_copy += ase.Atoms("H" * sum(number_of_hs_to_add), positions=h_positions)
    return atoms_copy


def add_h_offsets_to_original_positions(
    original_positions, offset_vectors, number_of_hs_to_add
):
    h_positions = np.zeros((sum(number_of_hs_to_add), 3))
    start_index = 0
    for i, num_hs in enumerate(number_of_hs_to_add):
        h_positions[start_index : start_index + num_hs] = (
            original_positions[i] + offset_vectors[start_index : start_index + num_hs]
        )
        start_index += num_hs
    return h_positions


def get_random_unit_vectors(num: int) -> np.ndarray:
    vectors = np.random.normal(size=(num, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors
