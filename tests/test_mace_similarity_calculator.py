import numpy as np
import pytest
import torch
from mace.tools.scatter import scatter_sum

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.utils import get_system_torch_device_str, initialize_mol

from .fixtures import (
    loaded_mace_similarity_calculator,
    loaded_model,
    mace_model,
    test_molecules,
    training_molecules,
    z_table,
)

torch.set_default_dtype(torch.float64)

DEVICE = get_system_torch_device_str()


def get_embedding(mace_sim_calc, atoms):
    return mace_sim_calc._get_node_embeddings(
        mace_sim_calc.batch_atoms(atoms).to_dict()
    )


def test_molecules_in_training_data_have_minimum_distance_of_zero(
    loaded_mace_similarity_calculator, training_molecules
):
    embeddings = [
        get_embedding(loaded_mace_similarity_calculator, mol)
        for mol in training_molecules
    ]
    distance_mats = [
        loaded_mace_similarity_calculator._calculate_distance_matrix(emb)
        for emb in embeddings
    ]
    distance_mats = [mat.detach().cpu().numpy() for mat in distance_mats]
    for mat in distance_mats:
        # mat shape (n_input_nodes, n_training_nodes)
        # all test nodes here are in the training set, so the minimum distance should be zero
        min_across_input_nodes = np.min(mat, axis=1)  # (n_input_nodes,)
        assert np.allclose(min_across_input_nodes, 0.0)


def test_molecules_not_in_training_set_have_non_zero_distance_to_reference_embeddings(
    loaded_mace_similarity_calculator, test_molecules
):
    embeddings = [
        get_embedding(loaded_mace_similarity_calculator, mol) for mol in test_molecules
    ]
    distance_mats = [
        loaded_mace_similarity_calculator._calculate_distance_matrix(emb)
        for emb in embeddings
    ]
    distance_mats = [mat.detach().cpu().numpy() for mat in distance_mats]
    for mat in distance_mats:
        min_across_input_nodes = np.min(mat, axis=1)  # (n_input_nodes,)
        assert np.all(min_across_input_nodes > 1e-3)


def test_log_kernel_density_of_training_data_much_higher_than_new_data(
    loaded_mace_similarity_calculator, training_molecules, test_molecules
):
    calc = loaded_mace_similarity_calculator
    for mol in test_molecules:
        mol.positions = mol.positions + 5e-1 * np.random.randn(*mol.positions.shape)
    training_embs = [get_embedding(calc, mol) for mol in training_molecules]
    test_embs = [get_embedding(calc, mol) for mol in test_molecules]
    training_log_dens = [calc._calculate_log_k(emb, 0) for emb in training_embs]
    training_log_dens = [
        dens.detach().cpu().numpy().mean() for dens in training_log_dens
    ]
    test_log_dens = [calc._calculate_log_k(emb, 0) for emb in test_embs]
    test_log_dens = [dens.detach().cpu().numpy().mean() for dens in test_log_dens]
    assert np.min(training_log_dens) - np.max(test_log_dens) > 10  # type: ignore


def test_gradient_of_log_kernel_density_smaller_for_undisturbed_data(
    loaded_mace_similarity_calculator, training_molecules
):
    calc = loaded_mace_similarity_calculator
    batch = calc.batch_atoms(training_molecules)
    emb = calc._get_node_embeddings(batch)
    log_dens = calc._calculate_log_k(emb, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    gradients = calc._get_gradient(batch.positions, log_dens)
    norms = np.sqrt((gradients**2).sum(axis=-1))

    for mol in training_molecules:
        mol.set_positions(
            mol.get_positions() + 1e-1 * np.random.randn(*mol.get_positions().shape)
        )
    batch = calc.batch_atoms(training_molecules)
    emb = calc._get_node_embeddings(batch)
    log_dens = calc._calculate_log_k(emb, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    gradients_disturbed = calc._get_gradient(batch.positions, log_dens)
    norms_disturbed = np.sqrt((gradients_disturbed**2).sum(axis=-1))
    assert all(norms <= norms_disturbed)


@pytest.mark.parametrize(
    "grad, expected_out",
    [
        (
            np.ones((5, 3)) * np.nan,
            np.zeros((5, 3)),
        ),
        (
            np.ones((5, 3)),
            np.ones((5, 3)),
        ),
        (
            np.zeros((5, 3)),
            np.zeros((5, 3)),
        ),
        (
            np.ones((5, 3)) * np.inf,
            np.zeros((5, 3)),
        ),
        (
            np.ones((5, 3)) * np.array([[np.inf], [1], [1], [1], [1]]),
            np.ones((5, 3)) * np.array([[0], [1], [1], [1], [1]]),
        ),
    ],
)
def test_gradient_magnitude_handler(
    loaded_mace_similarity_calculator, grad, expected_out
):
    grad = loaded_mace_similarity_calculator._handle_grad_nans(grad)
    np.testing.assert_allclose(grad, expected_out, atol=1e-6)


@pytest.mark.parametrize("index_pair", [(0, 1), (1, 2), (2, 1), (3, 2), (9, 0)])
def test_finite_difference(loaded_mace_similarity_calculator, index_pair):
    mol = initialize_mol("C6H6")
    mol.positions = mol.positions + 5e-2 * np.random.randn(*mol.positions.shape)
    original_positions = mol.get_positions()
    perturbation = np.zeros_like(mol.get_positions())
    h = 1e-4
    x, y = index_pair
    perturbation[x, y] = h

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions + perturbation / 2)
    atomic_data_plus_h = loaded_mace_similarity_calculator.batch_atoms(mol)
    embed_plus_h = loaded_mace_similarity_calculator._get_node_embeddings(
        atomic_data_plus_h
    )
    log_density_h_plus = loaded_mace_similarity_calculator._calculate_log_k(
        embed_plus_h, time=0
    )

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions - perturbation / 2)
    atomic_data_minus_h = loaded_mace_similarity_calculator.batch_atoms(mol)
    embed_minus_h = loaded_mace_similarity_calculator._get_node_embeddings(
        atomic_data_minus_h
    )
    log_density_h_minus = loaded_mace_similarity_calculator._calculate_log_k(
        embed_minus_h, time=0
    )

    finite_difference_gradient = (
        log_density_h_plus - log_density_h_minus
    ).detach().cpu().numpy() / h
    finite_difference_gradient = sum(finite_difference_gradient)

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions)
    calc = loaded_mace_similarity_calculator
    batch = calc.batch_atoms(mol)
    emb = calc._get_node_embeddings(batch)
    log_dens = calc._calculate_log_k(emb, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    model_gradient = calc._get_gradient(batch.positions, log_dens)

    np.testing.assert_almost_equal(
        finite_difference_gradient, model_gradient[x, y], decimal=3
    )
