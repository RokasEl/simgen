import numpy as np
import pytest
import torch
from mace.tools.scatter import scatter_sum

from simgen.calculators import MaceSimilarityCalculator
from simgen.utils import get_system_torch_device_str, initialize_mol

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


def get_distance_matrix(mace_sim_calc, atoms):
    batch = mace_sim_calc.batch_atoms(atoms)
    emb = mace_sim_calc._get_node_embeddings(batch)
    return mace_sim_calc._calculate_distance_matrix(emb, batch.node_attrs)


def test_molecules_in_training_data_have_minimum_distance_of_zero(
    loaded_mace_similarity_calculator, training_molecules
):
    calc = loaded_mace_similarity_calculator
    distance_mats = [get_distance_matrix(calc, x) for x in training_molecules]
    distance_mats = [mat.detach().cpu().numpy() for mat in distance_mats]
    for mat in distance_mats:
        # mat shape (n_input_nodes, n_training_nodes)
        # all test nodes here are in the training set, so the minimum distance should be zero
        min_across_input_nodes = np.min(mat, axis=1)  # (n_input_nodes,)
        assert np.allclose(min_across_input_nodes, 0.0)


def test_molecules_not_in_training_set_have_non_zero_distance_to_reference_embeddings(
    loaded_mace_similarity_calculator, test_molecules
):
    calc = loaded_mace_similarity_calculator
    distance_mats = [get_distance_matrix(calc, x) for x in test_molecules]
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
    node_attrs = [calc.batch_atoms(mol).node_attrs for mol in training_molecules]
    test_embs = [get_embedding(calc, mol) for mol in test_molecules]
    training_log_dens = [
        calc._calculate_log_k(emb, nd_atr, 0)
        for emb, nd_atr in zip(training_embs, node_attrs)
    ]
    training_log_dens = [
        dens.detach().cpu().numpy().mean() for dens in training_log_dens
    ]
    node_attrs = [calc.batch_atoms(mol).node_attrs for mol in test_molecules]
    test_log_dens = [
        calc._calculate_log_k(emb, nd_atr, 0)
        for emb, nd_atr in zip(test_embs, node_attrs)
    ]
    test_log_dens = [dens.detach().cpu().numpy().mean() for dens in test_log_dens]
    assert np.min(training_log_dens) - np.max(test_log_dens) > 10  # type: ignore


def test_gradient_of_log_kernel_density_smaller_for_undisturbed_data(
    loaded_mace_similarity_calculator, training_molecules
):
    calc = loaded_mace_similarity_calculator
    batch = calc.batch_atoms(training_molecules)
    nd_atr = batch.node_attrs
    emb = calc._get_node_embeddings(batch)
    log_dens = calc._calculate_log_k(emb, nd_atr, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    gradients = calc._get_gradient(batch.positions, log_dens).detach().cpu().numpy()
    norms = np.sqrt((gradients**2).sum(axis=-1))

    for mol in training_molecules:
        mol.set_positions(
            mol.get_positions() + 1e-1 * np.random.randn(*mol.get_positions().shape)
        )
    batch = calc.batch_atoms(training_molecules)
    emb = calc._get_node_embeddings(batch)
    log_dens = calc._calculate_log_k(emb, nd_atr, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    gradients_disturbed = (
        calc._get_gradient(batch.positions, log_dens).detach().cpu().numpy()
    )
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
    nd_atrs = atomic_data_plus_h.node_attrs
    embed_plus_h = loaded_mace_similarity_calculator._get_node_embeddings(
        atomic_data_plus_h
    )
    log_density_h_plus = loaded_mace_similarity_calculator._calculate_log_k(
        embed_plus_h, nd_atrs, time=0
    )

    mol = initialize_mol("C6H6")
    mol.set_positions(original_positions - perturbation / 2)
    atomic_data_minus_h = loaded_mace_similarity_calculator.batch_atoms(mol)
    embed_minus_h = loaded_mace_similarity_calculator._get_node_embeddings(
        atomic_data_minus_h
    )
    log_density_h_minus = loaded_mace_similarity_calculator._calculate_log_k(
        embed_minus_h, nd_atrs, time=0
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
    log_dens = calc._calculate_log_k(emb, nd_atrs, 0)
    log_dens = scatter_sum(log_dens, batch.batch)
    model_gradient = (
        calc._get_gradient(batch.positions, log_dens).detach().cpu().numpy()
    )

    np.testing.assert_almost_equal(
        finite_difference_gradient, model_gradient[x, y], decimal=3
    )


def test_adjust_element_sigmas(loaded_mace_similarity_calculator):
    kernel_sigmas = (
        loaded_mace_similarity_calculator.element_kernel_sigmas.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(kernel_sigmas, np.ones_like(kernel_sigmas))

    loaded_mace_similarity_calculator.adjust_element_sigmas({"H": 0.25, "C": 2})
    expected_kernel_sigmas = np.ones_like(kernel_sigmas)
    expected_kernel_sigmas[0] = 0.25
    c_index = loaded_mace_similarity_calculator.z_table.zs.index(6)
    expected_kernel_sigmas[c_index] = 2
    kernel_sigmas = (
        loaded_mace_similarity_calculator.element_kernel_sigmas.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(kernel_sigmas, expected_kernel_sigmas)


def test_scatter_element_sigmas(loaded_mace_similarity_calculator):
    mol = initialize_mol("HCCNONFH")
    loaded_mace_similarity_calculator.adjust_element_sigmas({"H": 1.0, "C": 1.0})
    batched = loaded_mace_similarity_calculator.batch_atoms(mol)
    element_sigmas = loaded_mace_similarity_calculator.element_kernel_sigmas
    scatter_element_sigmas = loaded_mace_similarity_calculator._scatter_element_sigmas(
        batched.node_attrs, element_sigmas
    )
    scatter_element_sigmas = scatter_element_sigmas.detach().cpu().numpy()
    expected_scatter = np.ones_like(scatter_element_sigmas)
    np.testing.assert_array_equal(scatter_element_sigmas, expected_scatter)

    loaded_mace_similarity_calculator.adjust_element_sigmas({"H": 0.25, "C": 2})
    scatter_element_sigmas = loaded_mace_similarity_calculator._scatter_element_sigmas(
        batched.node_attrs, element_sigmas
    )
    scatter_element_sigmas = scatter_element_sigmas.detach().cpu().numpy()
    expected_scatter = np.array([0.25, 2, 2, 1, 1, 1, 1, 0.25], dtype=np.float64)[
        :, np.newaxis
    ]
    np.testing.assert_array_equal(scatter_element_sigmas, expected_scatter)


def test_element_sigmas_adjusts_the_distance_matrix(
    loaded_mace_similarity_calculator, training_molecules
):
    # first, get original distance matrix
    loaded_mace_similarity_calculator.adjust_element_sigmas({"H": 1.0, "C": 1.0})
    embeddings = [
        get_embedding(loaded_mace_similarity_calculator, mol)
        for mol in training_molecules
    ]
    node_attrs = [
        loaded_mace_similarity_calculator.batch_atoms(mol).node_attrs
        for mol in training_molecules
    ]
    distance_mats = [
        loaded_mace_similarity_calculator._calculate_distance_matrix(emb, nd_attr)
        for emb, nd_attr in zip(embeddings, node_attrs)
    ]
    distance_mats_original = [mat.detach().cpu().numpy() for mat in distance_mats]

    # adjust and get new distance matrix
    loaded_mace_similarity_calculator.adjust_element_sigmas({"H": 0.25, "C": 2})
    distance_mats = [
        loaded_mace_similarity_calculator._calculate_distance_matrix(emb, nd_attr)
        for emb, nd_attr in zip(embeddings, node_attrs)
    ]
    distance_mats_new = [mat.detach().cpu().numpy() for mat in distance_mats]
    for i, mol in enumerate(training_molecules):
        h_indices = np.where(mol.get_atomic_numbers() == 1)[0]
        c_indices = np.where(mol.get_atomic_numbers() == 6)[0]
        the_rest_of_the_indices = np.where(
            (mol.get_atomic_numbers() != 1) & (mol.get_atomic_numbers() != 6)
        )[0]
        h_distances_original = distance_mats_original[i][h_indices]
        h_distances_new = distance_mats_new[i][h_indices]
        np.testing.assert_allclose(h_distances_new, h_distances_original / 0.25)
        c_distances_original = distance_mats_original[i][c_indices]
        c_distances_new = distance_mats_new[i][c_indices]
        np.testing.assert_allclose(c_distances_new, c_distances_original / 2)
        the_rest_distances_original = distance_mats_original[i][the_rest_of_the_indices]
        the_rest_distances_new = distance_mats_new[i][the_rest_of_the_indices]
        np.testing.assert_allclose(the_rest_distances_new, the_rest_distances_original)
