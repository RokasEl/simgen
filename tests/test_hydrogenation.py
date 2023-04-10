import numpy as np
import pytest
from ase import Atoms

from moldiff.hydrogenation import (
    add_hydrogens_to_atoms,
    check_hydrogen_position_is_valid,
    compute_exclusion_radius,
    get_exclusion_radii,
    sample_number_of_hs_to_add,
)
from moldiff.utils import initialize_mol


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
    "r1, r2, h_radius, expected",
    [
        (1, 1, 0, np.sqrt(5)),
        (1, 1, 0.5, 1.3334),
        (1, 2, 0.5, 1.5 * np.sqrt(5)),
        (1, 1.2, 0.5, np.sqrt(313) / 10),
        (1.2, 1.0, 0.5, 1.5594),
    ],
)
def test_compute_exclusion_radius(r1, r2, h_radius, expected):
    assert compute_exclusion_radius(r1, r2, h_radius) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize(
    "atom_1_idx, atoms, expected",
    [
        (0, initialize_mol("CC"), np.array([0, 1.0])),
        (1, initialize_mol("CC"), np.array([1.0, 0])),
        (0, Atoms("CO"), np.array([0, 2.0])),
        (1, Atoms("CO"), np.array([-1, 0])),
    ],
)
def test_get_exlusion_radii(atom_1_idx, atoms, expected):
    exclusion_radii_dict = {
        6: {6: 1.0, 8: 2.0},
        8: {6: -1.0, 8: -2.0},
    }  # just simulate assymetric radii for testing purposes
    out = get_exclusion_radii(atoms, atom_1_idx, exclusion_radii_dict)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "proposed_position, atoms, expected",
    [
        (
            np.array([0.0, 0.0, 2]),
            Atoms("CC", positions=np.array([[0, 0, 0], [0, 0, 1.4]])),
            True,
        ),
        (
            np.array([0.0, 0.0, 0.7]),
            Atoms("CC", positions=np.array([[0, 0, 0], [0, 0, 1.4]])),
            False,
        ),
        (
            np.array([0.0, 0.0, 2]),
            Atoms("CCC", positions=np.array([[0, 0, 0], [0, 0, 1.4], [0, 0, 2.8]])),
            False,
        ),
        (
            np.array([0.0, 1.0, 1.4]),
            Atoms("CCC", positions=np.array([[0, 0, 0], [0, 0, 1.4], [0, 0, 2.8]])),
            True,
        ),
    ],
)
def test_check_hydrogen_position_is_valid(proposed_position, atoms, expected):
    exclusion_radii_dict = {6: {6: 0.819}}
    exlcusion_radii = get_exclusion_radii(atoms, 1, exclusion_radii_dict)
    assert (
        check_hydrogen_position_is_valid(
            proposed_position, exlcusion_radii, atoms.get_positions()
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
