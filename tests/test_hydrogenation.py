import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii

from simgen.hydrogenation import (
    add_hydrogens_to_atoms,
    check_hydrogen_position_is_valid,
    get_exclusion_radii,
)
from simgen.utils import initialize_mol


@pytest.mark.parametrize(
    "atoms, idx, expected",
    [
        (Atoms("CC"), 0, np.array([0.0, covalent_radii[6] * 2])),
        (Atoms("OC"), 1, np.array([covalent_radii[6] + covalent_radii[8], 0.0])),
    ],
)
def test_get_exclusion_radii(atoms, idx, expected):
    exclusion_radii = get_exclusion_radii(atoms, idx)
    np.testing.assert_array_almost_equal(exclusion_radii, expected)


def test_add_hydrogen_to_atoms():
    mol = Atoms("CC", positions=[[0, 0, 0], [0, 0, 1.4]])
    mol_copy = mol.copy()
    mol_with_hs = add_hydrogens_to_atoms(mol, np.array([1, 1]))
    assert mol_copy == mol
    assert len(mol_with_hs) == 4
    assert mol_with_hs.get_atomic_numbers().tolist() == [6, 6, 1, 1]
    assert (mol_with_hs.get_positions()[:2] == mol.get_positions()).all()

    mol_with_no_added_hs = add_hydrogens_to_atoms(mol, np.array([0, 0]))
    assert mol_with_no_added_hs == mol


@pytest.mark.parametrize(
    "proposed_position, idx, atoms, expected",
    [
        (
            np.array([0.0, 0.0, 2]),
            1,
            Atoms("CC", positions=np.array([[0, 0, 0], [0, 0, 1.4]])),
            True,
        ),
        (
            np.array([0.0, 0.0, 0.7]),
            1,
            Atoms("CC", positions=np.array([[0, 0, 0], [0, 0, 1.4]])),
            False,
        ),
        (
            np.array([0.0, 0.0, 2]),
            1,
            Atoms("CCC", positions=np.array([[0, 0, 0], [0, 0, 1.4], [0, 0, 2.8]])),
            False,
        ),
        (
            np.array([0.0, 1.2, 1.4]),
            1,
            Atoms("CCC", positions=np.array([[0, 0, 0], [0, 0, 1.4], [0, 0, 2.8]])),
            True,
        ),
    ],
)
def test_check_hydrogen_position_is_valid(proposed_position, idx, atoms, expected):
    exclusion_radii = get_exclusion_radii(atoms, idx)
    assert (
        check_hydrogen_position_is_valid(
            proposed_position, exclusion_radii, atoms.get_positions()
        )
        == expected
    )
