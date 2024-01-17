import einops
import numpy as np
import pytest
import torch
from ase import Atoms
from mace.tools import AtomicNumberTable

from simgen.calculators import MaceSimilarityCalculator
from simgen.generation_utils import (
    ExponentialRepulsionBlock,
    batch_atoms,
    calculate_path_length,
    check_atoms_outside_threshold,
    duplicate_atoms,
    get_edge_array_and_neighbour_numbers,
    get_model_dtype,
    interpolate_points,
    remove_elements,
)
from simgen.utils import get_system_torch_device_str, initialize_mol


def test_calculate_path_length():
    f = lambda x: x**2
    xs = np.array([-1, 0, 1])
    ys = f(xs)
    points = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
    length = calculate_path_length(points)
    expected_length = 2 * np.sqrt(2)
    np.testing.assert_almost_equal(length, expected_length)
    points = interpolate_points(points, num_interpolated_points=500)
    length = calculate_path_length(points)
    expected_length = 0.5 * (np.log(2 + np.sqrt(5)) + 2 * np.sqrt(5))
    np.testing.assert_almost_equal(length, expected_length, decimal=5)


def test_exponential_repulsive_block_returns_correct_forces():
    z_table = AtomicNumberTable([1, 6, 7, 8, 9])
    device = get_system_torch_device_str()
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
    block_forces = (
        MaceSimilarityCalculator._get_gradient(batched.positions, -1 * block_energies)
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(forces, block_forces)


def test_exponential_repulsive_block_correct_for_batches_of_molecules():
    z_table = AtomicNumberTable([1, 6, 7, 8, 9])
    device = get_system_torch_device_str()
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
    block_forces = (
        MaceSimilarityCalculator._get_gradient(batched.positions, -1 * block_energies)
        .detach()
        .cpu()
        .numpy()
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

    mol = Atoms("CCHF", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    expected = Atoms("CC", positions=[[0, 0, 0], [0, 0, 1]])
    no_hs_and_f_mol = remove_elements(mol, [1, 9])
    assert no_hs_and_f_mol == expected


def test_get_edge_array_and_neighbour_numbers_gets_correct_number_of_neighbours():
    mol = initialize_mol("H2O")
    _, neighbour_numbers = get_edge_array_and_neighbour_numbers(mol)
    np.testing.assert_array_equal(neighbour_numbers, np.array([2, 1, 1]))

    mol = initialize_mol("C2H4")
    _, neighbour_numbers = get_edge_array_and_neighbour_numbers(mol)
    np.testing.assert_array_equal(neighbour_numbers, np.array([3, 3, 1, 1, 1, 1]))

    mol = initialize_mol("C20")
    _, neighbour_numbers = get_edge_array_and_neighbour_numbers(mol)
    np.testing.assert_array_equal(neighbour_numbers, np.array([19] * 20))


def test_assigning_model_dtype():
    model = torch.nn.Linear(5, 7)
    model.to(torch.float64)
    assert get_model_dtype(model) == torch.float64

    model.to(torch.float32)
    assert get_model_dtype(model) == torch.float32


@pytest.mark.parametrize(
    "atoms, expected, threshold",
    [
        (initialize_mol("C6H6"), False, 10),
        (initialize_mol("C6H6"), True, 1),
        (
            Atoms("CCHF", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            False,
            1,
        ),
        (
            Atoms("CCHF", positions=[[1e9, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            True,
            1,
        ),
        (
            Atoms(
                "CCHF", positions=[[1e9, 0, 0], [1e9, 0, 1], [1e9, 1, 0], [1e9, 0, 0]]
            ),
            False,
            1,
        ),
    ],
)
def test_check_atoms_outside_threshold(atoms, expected, threshold):
    assert check_atoms_outside_threshold(atoms, threshold) == expected
