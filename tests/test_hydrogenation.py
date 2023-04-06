import numpy as np
import pytest

from moldiff.hydrogenation import (
    add_h_offsets_to_original_positions,
    add_hydrogens_to_atoms,
)
from moldiff.utils import initialize_mol


@pytest.mark.parametrize(
    "original_positions, offset_vectors, number_of_hs_to_add, expected_h_positions",
    [
        (
            np.array([[0, 0, 0], [1, 1, 1]]),
            np.array([[0, 0, 1], [0, 0, 1]]),
            np.array([1, 1]),
            np.array([[0, 0, 1], [1, 1, 2]]),
        ),
        (
            np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]]),
            np.array([[0, 0, 1], [0, 0, 1]]),
            np.array([1, 0, 1]),
            np.array([[0, 0, 1], [-1, -1, 0]]),
        ),
        (
            np.array([[0, 0, 0], [1, 1, 1]]),
            np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
            np.array([1, 2]),
            np.array([[0, 0, 1], [1, 1, 2], [1, 1, 2]]),
        ),
    ],
)
def test_add_h_offsets_to_original_positions(
    original_positions, offset_vectors, number_of_hs_to_add, expected_h_positions
):
    h_positions = add_h_offsets_to_original_positions(
        original_positions, offset_vectors, number_of_hs_to_add
    )
    assert np.allclose(h_positions, expected_h_positions)


def test_add_hydrogen_to_atoms():
    mol = initialize_mol("CC")
    mol_copy = mol.copy()
    mol_with_hs = add_hydrogens_to_atoms(mol, np.array([1, 1]))
    assert mol_copy == mol
    assert len(mol_with_hs) == 4
    assert mol_with_hs.get_atomic_numbers().tolist() == [6, 6, 1, 1]
    assert (mol_with_hs.get_positions()[:2] == mol.get_positions()).all()

    mol_with_no_added_hs = add_hydrogens_to_atoms(mol, np.array([0, 0]))
    assert mol_with_no_added_hs == mol
