import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii

from moldiff.hydrogenation import (
    add_hydrogens_to_atoms,
    check_hydrogen_position_is_valid,
    get_exclusion_radii,
    sample_number_of_hs_to_add,
)
from moldiff.utils import initialize_mol


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


@pytest.mark.parametrize(
    "current_valance, max_valance, expected_mean, expected_var",
    [
        (1, 4, 0.75 * 3, 3 * 0.75 * 0.25),
        (2, 4, 4 / 3, 4 / 9),
        (3, 4, 0.5, 0.25),
    ],
)
def test_sample_number_of_hs_to_add_has_correct_mean_and_var(
    current_valance, max_valance, expected_mean, expected_var
):
    samples = [
        sample_number_of_hs_to_add(current_valance, max_valance) for _ in range(10000)
    ]
    assert np.mean(samples) == pytest.approx(expected_mean, 0.01)
    assert np.var(samples) == pytest.approx(expected_var, 0.01)
