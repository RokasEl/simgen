import numpy as np
import pytest
from mace.tools import AtomicNumberTable

from moldiff.atoms_cleanup import (
    get_higest_energy_unswapped_idx,
    relax_elements,
    remove_isolated_atoms,
)
from moldiff.utils import initialize_mol

from .fixtures import (
    loaded_mace_similarity_calculator,
    loaded_model,
    training_molecules,
)

z_table = AtomicNumberTable([1, 6, 7, 8, 9])


@pytest.fixture()
def linear_molecule_with_increasingly_isolated_atoms():
    mol = initialize_mol("C10")
    x_positions = [2**i - 1 for i in range(10)]
    positions = np.array([x_positions, [0] * 10, [0] * 10]).T
    mol.set_positions(positions)
    return mol


@pytest.mark.parametrize(
    "atoms, cutoff, expected",
    [
        (initialize_mol("H2O"), 10.0, initialize_mol("H2O")),
        (initialize_mol("H2O"), 0.1, initialize_mol("")),
    ],
)
def test_remove_isolated_atoms(atoms, cutoff, expected):
    pruned_atoms = remove_isolated_atoms(atoms, cutoff)
    assert pruned_atoms == expected


def test_remove_isolated_atoms_with_molecule_with_increasingly_isolated_atoms(
    linear_molecule_with_increasingly_isolated_atoms,
):
    mol = linear_molecule_with_increasingly_isolated_atoms
    cutoffs = [2**i for i in range(10)]
    expected_remaining_atoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    for cutoff, expected_remaining_atom in zip(cutoffs, expected_remaining_atoms):
        pruned_atoms = remove_isolated_atoms(mol, cutoff)
        assert len(pruned_atoms) == expected_remaining_atom

    pruned_atoms = remove_isolated_atoms(mol, 0.1)
    assert len(pruned_atoms) == 0


def test_get_highest_energy_unswapped_idx():
    swapped_indices = []
    energies = np.array([1, 20, -3, 4, 5]).astype(float)
    expected_indices = (1, 4, 3, 0, 2)
    expected_swapped_indices = ([1], [1, 4], [1, 4, 3], [1, 4, 3, 0], [1, 4, 3, 0, 2])

    for expected_idx, expected_swapped_idx in zip(
        expected_indices, expected_swapped_indices
    ):
        idx = get_higest_energy_unswapped_idx(swapped_indices, energies)
        assert idx == expected_idx
        swapped_indices.append(idx)
        assert swapped_indices == expected_swapped_idx


@pytest.fixture()
def element_swapping_test_suite():
    test_suite = []
    # add a few mols that are already correct
    for mol_name in ["H2O", "CH4", "C2H6", "C2H4", "CH3COCH3"]:
        mol = initialize_mol(mol_name)
        test_suite.append((mol.copy(), mol.copy()))

    # add a few mols that are incorrect and that have one obvious element to swap
    mol = initialize_mol("H2O")
    mol.set_atomic_numbers([1, 1, 1])
    test_suite.append((mol.copy(), initialize_mol("H2O")))
    mol.set_atomic_numbers([6, 1, 1])
    test_suite.append((mol.copy(), initialize_mol("H2O")))

    mol = initialize_mol("CH4")
    mol.set_atomic_numbers([7, 1, 1, 1, 1])
    test_suite.append((mol.copy(), initialize_mol("CH4")))

    mol = initialize_mol("CO2")
    mol.set_atomic_numbers([6, 6, 8])
    test_suite.append((mol.copy(), initialize_mol("CO2")))
    return test_suite


def test_relax_elements(loaded_mace_similarity_calculator, element_swapping_test_suite):
    for mol, expected_mol in element_swapping_test_suite:
        mol.calc = loaded_mace_similarity_calculator
        relaxed_mol = relax_elements(mol, z_table=z_table)
        assert relaxed_mol == expected_mol
