import einops
import numpy as np
import pytest
import torch
from ase import Atoms
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.generation_utils import (
    ExponentialRepulsionBlock,
    batch_atoms,
    duplicate_atoms,
    remove_elements,
)
from moldiff.utils import initialize_mol


def test_exponential_repulsive_block_returns_correct_forces():
    z_table = AtomicNumberTable([1, 6, 7, 8, 9])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    water = initialize_mol("H2O")
    batched = batch_atoms(water, z_table, cutoff=10.0, device=device)

    alpha = 1.0
    repulsion_block = ExponentialRepulsionBlock(alpha=alpha).to(device)
    block_energies = repulsion_block(batched)

    distances = water.get_all_distances()
    np.fill_diagonal(distances, 1e10)
    energies = np.exp(-alpha * distances).sum(axis=1) / 2

    np.testing.assert_allclose(energies, block_energies.detach().cpu().numpy())

    vectors = (
        water.get_positions()[:, None, :] - water.get_positions()[None, :, :]
    )  # (N, N, 3)
    derivatives = -alpha * np.exp(-alpha * distances)
    one_over_distances = 1.0 / distances
    forces = einops.einsum(
        -1 * vectors,
        derivatives,
        one_over_distances,
        "from to cartesian, from to, from to -> from cartesian",
    )
    block_forces = MaceSimilarityCalculator._get_gradient(
        batched.positions, -1 * block_energies
    )
    np.testing.assert_allclose(forces, block_forces)


def test_exponential_repulsive_block_correct_for_batches_of_molecules():
    z_table = AtomicNumberTable([1, 6, 7, 8, 9])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    water = initialize_mol("H2O")
    batched = batch_atoms(
        [water.copy(), water.copy()], z_table, cutoff=10.0, device=device
    )

    alpha = 1.0
    repulsion_block = ExponentialRepulsionBlock(alpha=alpha).to(device)
    block_energies = repulsion_block(batched)

    distances = water.get_all_distances()
    np.fill_diagonal(distances, 1e10)
    energies = np.exp(-alpha * distances).sum(axis=1) / 2
    energies = np.concatenate([energies, energies])

    np.testing.assert_allclose(energies, block_energies.detach().cpu().numpy())

    vectors = (
        water.get_positions()[:, None, :] - water.get_positions()[None, :, :]
    )  # (N, N, 3)
    derivatives = -alpha * np.exp(-alpha * distances)
    one_over_distances = 1.0 / distances
    forces = einops.einsum(
        -1 * vectors,
        derivatives,
        one_over_distances,
        "from to cartesian, from to, from to -> from cartesian",
    )
    forces = np.concatenate([forces, forces], axis=0)
    block_forces = MaceSimilarityCalculator._get_gradient(
        batched.positions, -1 * block_energies
    )
    np.testing.assert_allclose(forces, block_forces)


def test_duplicate_atoms_does_not_copy_calculated_values():
    mol_1 = initialize_mol("H2O")
    mol_1.arrays["energy"] = np.random.randn(3)

    mol_1_copy = mol_1.copy()
    assert (mol_1.arrays["energy"] == mol_1_copy.arrays["energy"]).all()

    mol_1_duplicate = duplicate_atoms(mol_1)
    assert "energy" not in mol_1_duplicate.arrays


def test_remove_elements_removes_required_elements_and_leaves_other_atoms_unchanged():
    mol = initialize_mol("H2O")
    no_hs_mol = remove_elements(mol, [1])
    expected = mol.copy()[0:1]
    assert no_hs_mol == expected

    mol = initialize_mol("C2H6")
    expected = mol.copy()[:2]
    no_hs_mol = remove_elements(mol, [1])
    assert no_hs_mol == expected

    mol = ase.Atoms("CCHF", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    expected = ase.Atoms("CC", positions=[[0, 0, 0], [0, 0, 1]])
    no_hs_and_f_mol = remove_elements(mol, [1, 9])
    assert no_hs_and_f_mol == expected
