import einops
import numpy as np
import pytest
import torch
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.generation_utils import (
    ExponentialRepulsionBlock,
    batch_atoms,
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
