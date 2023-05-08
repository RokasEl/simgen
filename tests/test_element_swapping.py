from unittest.mock import MagicMock

import numpy as np
import pytest
from mace.tools import AtomicNumberTable

from moldiff.element_swapping import (
    catch_diverged_energies,
    create_element_swapped_particles,
    replace_array_elements_at_indices_with_other_elements_in_z_table,
    swap_single_particle,
    sweep_all_elements,
)
from moldiff.utils import initialize_mol

z_table = AtomicNumberTable([1, 6, 7, 8, 9])


def get_water_with_swapped_Os():
    mol = initialize_mol("H2O")
    other_elements = [z for z in z_table.zs if z != 8]
    swapped = []
    for z in other_elements:
        new_atomic_numbers = [z, 1, 1]
        mol.set_atomic_numbers(new_atomic_numbers)
        swapped.append(mol.copy())
    return swapped


def get_water_with_swapped_Hs():
    mol = initialize_mol("H2O")
    other_elements = [z for z in z_table.zs if z != 1]
    swapped = []
    for z in other_elements:
        new_atomic_numbers = [8, z, 1]
        mol.set_atomic_numbers(new_atomic_numbers)
        swapped.append(mol.copy())
    return swapped


@pytest.mark.parametrize(
    "mol, idx, z_table, expected_swapped_mols",
    [
        (
            initialize_mol("C"),
            0,
            z_table,
            [
                initialize_mol("H"),
                initialize_mol("N"),
                initialize_mol("O"),
                initialize_mol("F"),
            ],
        ),
        (initialize_mol("H2O"), 0, z_table, get_water_with_swapped_Os()),
        (initialize_mol("H2O"), 1, z_table, get_water_with_swapped_Hs()),
    ],
)
def test_sweep_all_elements(mol, idx, z_table, expected_swapped_mols):
    original_mol = mol.copy()
    swapped_mols = sweep_all_elements(mol, idx, z_table)
    assert mol == original_mol
    assert len(swapped_mols) == len(expected_swapped_mols)
    for x, y in zip(swapped_mols, expected_swapped_mols):
        assert x == y


def test_replace_array_elements_at_indices_with_other_elements_in_z_table():
    x = np.array([1, 2, 3, 4, 5])
    idx = np.array([0, 2, 4])
    swapped = replace_array_elements_at_indices_with_other_elements_in_z_table(
        x, idx, z_table
    )
    assert x[0] == 1 and x[1] == 2 and x[2] == 3 and x[3] == 4 and x[4] == 5
    assert swapped[0] != 1 and swapped[0] in z_table.zs
    assert swapped[2] != 3 and swapped[2] in z_table.zs
    assert swapped[4] != 5 and swapped[4] in z_table.zs


def test_swap_single_particle_gives_rise_to_correct_distribution():
    mol = initialize_mol("H2O")
    ps = np.array([1, 1, 1]) / 3
    num_change = 3
    mols = [swap_single_particle(mol, ps, num_change, z_table) for _ in range(1000)]
    expected_element_distribution = (
        np.array([1 / 4, 1 / 4, 1 / 4, 0, 1 / 4]) / 3
        + np.array([0, 1 / 4, 1 / 4, 1 / 4, 1 / 4]) * 2 / 3
    )
    swapped_elements = np.asarray([m.get_atomic_numbers() for m in mols])
    element_counts = np.unique(swapped_elements.flatten(), return_counts=True)[1]
    generated_ratios = element_counts / np.sum(element_counts)
    np.testing.assert_allclose(
        generated_ratios, expected_element_distribution, atol=0.025
    )


def test_swap_single_particle_swaps_correct_index_when_probability_is_peaked():
    mol = initialize_mol("H2O")
    ps = np.array([0, 0.5, 0.5])
    num_change = 1
    mols = [swap_single_particle(mol, ps, num_change, z_table) for _ in range(10)]
    for mol in mols:
        assert mol.get_atomic_numbers()[1] != 1 or mol.get_atomic_numbers()[2] != 1
        assert mol.get_atomic_numbers()[0] == 8

    mol = initialize_mol("H2O")
    ps = np.array([1.0, 0, 0])
    mols = [swap_single_particle(mol, ps, num_change, z_table) for _ in range(10)]
    for mol in mols:
        assert mol.get_atomic_numbers()[0] != 8
        assert mol.get_atomic_numbers()[1] == 1 or mol.get_atomic_numbers()[2] == 1


@pytest.mark.parametrize(
    "energies, num_atoms, expected",
    [
        (np.array([0, -1, -1.1]), 1, np.array([0, -1, 1e3])),
        (np.array([0, -1, -1.1]), 2, np.array([0, -1, -1.1])),
    ],
)
def test_catch_diverged_energies(energies, num_atoms, expected):
    thresh = -1
    out = catch_diverged_energies(energies, num_atoms, thresh)
    np.testing.assert_allclose(out, expected)


def test_create_element_swapped_particles_doesnt_swap_masked_particles():
    atoms = initialize_mol("C6H6")
    calc = MagicMock()
    calc.get_potential_energies = MagicMock(
        return_value=np.ones_like(atoms, dtype=float)
    )
    atoms.calc = calc
    # no mask, all atoms should be swapped at least once
    ensemble = create_element_swapped_particles(atoms, 1, 100, z_table=z_table)
    swaps = [x.numbers == atoms.numbers for x in ensemble]
    swaps = np.stack(swaps, axis=0)
    all_positions_changed = (~swaps).any(axis=0).all()
    assert all_positions_changed

    mask = np.concatenate([np.ones(6), np.zeros(6)])
    ensemble = create_element_swapped_particles(
        atoms, 1, 100, z_table=z_table, mask=mask
    )
    swaps = [x.numbers == atoms.numbers for x in ensemble]
    swaps = np.stack(swaps, axis=0)
    first_six_pos_all_changed = (~swaps[:, :6]).any(axis=0).all()
    assert first_six_pos_all_changed
    masked_positions_all_unchanged = (swaps[:, 6:]).all(axis=0).all()
    assert masked_positions_all_unchanged

    # TODO: write a test for the case when energies peaked around one atom
    atoms = initialize_mol("C6H6")
    calc = MagicMock()
    energies = np.zeros(12)
    energies[-1] = 99999  # high energy of masked atoms should not matter
    calc.get_potential_energies = MagicMock(return_value=energies)
    atoms.calc = calc
    mask = np.concatenate([np.ones(6), np.zeros(6)])
    ensemble = create_element_swapped_particles(
        atoms, 1, 100, z_table=z_table, mask=mask
    )
    swaps = [x.numbers == atoms.numbers for x in ensemble]
    swaps = ~np.stack(swaps, axis=0)
    assert swaps[:, 6:].sum() == 0
    assert np.allclose(swaps[:, :6].sum(axis=0), 99 / 6, rtol=0.5)
