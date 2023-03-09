import numpy as np
import pytest
import torch

from moldiff.sampling import MaceSimilarityScore
from moldiff.utils import initialize_mol

from .fixtures import (
    mace_model,
    test_molecules,
    training_molecules,
    z_table,
)

torch.set_default_dtype(torch.float64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mace_similarity_scorer(mace_model, z_table, training_molecules):
    return MaceSimilarityScore(mace_model, z_table, training_molecules)


def get_embedding(mace_similarity_scorer, atoms):
    return mace_similarity_scorer._get_node_embeddings(
        mace_similarity_scorer._to_atomic_data(atoms)
    )


def test_molecules_in_training_data_have_minimum_distance_of_zero(
    mace_similarity_scorer, training_molecules
):
    embeddings = [
        get_embedding(mace_similarity_scorer, mol) for mol in training_molecules
    ]
    distance_mats = [
        mace_similarity_scorer._calculate_distance_matrix(emb) for emb in embeddings
    ]
    distance_mats = [mat.detach().cpu().numpy() for mat in distance_mats]
    for mat in distance_mats:
        # mat shape (n_input_nodes, n_training_nodes)
        # all test nodes here are in the training set, so the minimum distance should be zero
        min_across_input_nodes = np.min(mat, axis=1)  # (n_input_nodes,)
        assert np.allclose(min_across_input_nodes, 0.0)


def test_molecules_not_in_training_set_have_non_zero_distance_to_reference_embeddings(
    mace_similarity_scorer, test_molecules
):
    embeddings = [get_embedding(mace_similarity_scorer, mol) for mol in test_molecules]
    distance_mats = [
        mace_similarity_scorer._calculate_distance_matrix(emb) for emb in embeddings
    ]
    distance_mats = [mat.detach().cpu().numpy() for mat in distance_mats]
    for mat in distance_mats:
        min_across_input_nodes = np.min(mat, axis=1)  # (n_input_nodes,)
        assert np.all(min_across_input_nodes > 1e-3)


def test_log_kernel_density_of_training_data_much_higher_than_new_data(
    mace_similarity_scorer, training_molecules, test_molecules
):
    for mol in test_molecules:
        mol.positions = mol.positions + 5e-1 * np.random.randn(*mol.positions.shape)
    training_embs = [
        get_embedding(mace_similarity_scorer, mol) for mol in training_molecules
    ]
    test_embs = [get_embedding(mace_similarity_scorer, mol) for mol in test_molecules]
    training_log_dens = [
        mace_similarity_scorer._get_log_kernel_density(emb, 0) for emb in training_embs
    ]
    training_log_dens = [
        dens.detach().cpu().numpy().mean() for dens in training_log_dens
    ]
    test_log_dens = [
        mace_similarity_scorer._get_log_kernel_density(emb, 0) for emb in test_embs
    ]
    test_log_dens = [dens.detach().cpu().numpy().mean() for dens in test_log_dens]
    assert np.min(training_log_dens) - np.max(test_log_dens) > 0


def test_gradient_of_log_kernel_density_is_close_to_zero_for_training_data(
    mace_similarity_scorer, training_molecules
):
    gradients = [
        mace_similarity_scorer(mol, t=0, normalise_grad=False)
        for mol in training_molecules
    ]
    for grad in gradients:
        assert np.allclose(grad, 0.0, atol=1e-6)


def test_gradient_at_high_temperature_non_zero_even_for_training_data(
    mace_similarity_scorer, training_molecules
):
    gradients = [
        mace_similarity_scorer(mol, t=1, normalise_grad=False)
        for mol in training_molecules
    ]
    for grad in gradients:
        assert np.all(np.linalg.norm(grad, axis=1) > 0)


def test_slightly_perturbed_training_molecules_have_non_zero_gradient(
    mace_similarity_scorer, training_molecules
):
    for mol in training_molecules:
        mol.set_positions(
            mol.get_positions() + 1e-2 * np.random.randn(*mol.get_positions().shape)
        )
    gradients = [mace_similarity_scorer(mol, t=1) for mol in training_molecules]
    for grad in gradients:
        assert np.all(grad != 0.0)


def test_unseen_molecules_have_finite_gradients(mace_similarity_scorer, test_molecules):
    gradients = [mace_similarity_scorer(mol, t=1) for mol in test_molecules]
    for grad in gradients:
        assert np.all(np.linalg.norm(grad, axis=1) != 0.0)


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
def test_gradient_magnitude_handler(mace_similarity_scorer, grad, expected_out):
    grad = mace_similarity_scorer._handle_grad_nans(grad)
    np.testing.assert_allclose(grad, expected_out, atol=1e-6)


@pytest.mark.parametrize("index_pair", [(0, 1), (1, 2), (2, 1), (3, 2), (9, 0)])
def test_finite_difference(mace_similarity_scorer, index_pair):
    mol = initialize_mol("C6H6")
    mol.positions = mol.positions + 5e-2 * np.random.randn(*mol.positions.shape)
    original_positions = mol.get_positions()
    perturbation = np.zeros_like(mol.get_positions())
    h = 1e-3
    x, y = index_pair
    perturbation[x, y] = h

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions + perturbation / 2)
    atomic_data_plus_h = mace_similarity_scorer._to_atomic_data(mol)
    embed_plus_h = mace_similarity_scorer._get_node_embeddings(atomic_data_plus_h)
    log_density_h_plus = mace_similarity_scorer._get_log_kernel_density(
        embed_plus_h, t=0
    )

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions - perturbation / 2)
    atomic_data_minus_h = mace_similarity_scorer._to_atomic_data(mol)
    embed_minus_h = mace_similarity_scorer._get_node_embeddings(atomic_data_minus_h)
    log_density_h_minus = mace_similarity_scorer._get_log_kernel_density(
        embed_minus_h, t=0
    )

    finite_difference_gradient = (
        log_density_h_plus - log_density_h_minus
    ).detach().cpu().numpy() / h
    finite_difference_gradient = finite_difference_gradient

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions)
    model_gradient = mace_similarity_scorer(mol, t=0, normalise_grad=False)

    np.testing.assert_almost_equal(finite_difference_gradient, model_gradient[x, y])
