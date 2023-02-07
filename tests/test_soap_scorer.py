import numpy as np
import pytest

from moldiff.sampling import SOAPSimilarityModel
from moldiff.utils import initialize_mol

from .fixtures import test_molecules, training_molecules


@pytest.fixture
def soap_scorer(training_molecules):
    return SOAPSimilarityModel(training_molecules)


@pytest.mark.parametrize(
    "grad, index_array, expected_out",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.array([[5, 7, 9], [17, 19, 21]]),
        ),
        (
            np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [0, 0, 0], [-1, -2, -3]]
            ),
            np.array([[0, 0], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]),
            np.array([[1, 2, 3], [21, 24, 27], [-1, -2, -3]]),
        ),
    ],
)
def test_sum_over_split_gradients(grad, index_array, expected_out):
    grad_out = SOAPSimilarityModel._sum_over_split_gradients(grad, index_array)
    np.testing.assert_allclose(grad_out, expected_out, atol=1e-6)


def test_embedding_deltas_are_close_to_zero_in_multiple_places_for_symmetric_mol(
    soap_scorer,
):
    mol = initialize_mol("C6H6")  # also in training set
    embedding = soap_scorer.descriptor_calculator.calc(mol)["data"]
    delta = soap_scorer._calculate_distance_matrix(embedding)[
        0
    ]  # embedim_dim, training_atoms, 12 (benzene atom num)
    np.testing.assert_allclose(delta[:, :6, :6], 0, atol=1e-6)
    np.testing.assert_allclose(delta[:, 6:12, 6:12], 0, atol=1e-6)
    assert not np.allclose(delta[:, :6, 6:12], 0, atol=1e-6)


def test_embedding_deltas_are_zero_for_training_molecules(
    soap_scorer, training_molecules
):
    embeddings = [
        soap_scorer.descriptor_calculator.calc(mol)["data"]
        for mol in training_molecules
    ]
    deltas = [soap_scorer._calculate_distance_matrix(emb)[0] for emb in embeddings]
    for i, delta in enumerate(deltas):
        assert np.all(delta == 0, axis=0).sum() == len(training_molecules[i])


def test_embedding_deltas_are_nonzero_for_test_molecules(soap_scorer, test_molecules):
    embeddings = [
        soap_scorer.descriptor_calculator.calc(mol)["data"] for mol in test_molecules
    ]
    deltas = [soap_scorer._calculate_distance_matrix(emb)[0] for emb in embeddings]
    for delta in deltas:
        assert np.all(delta == 0, axis=0).sum() == 0


def test_grad_zero_if_training_set_and_test_set_are_the_same_mol():
    mol = initialize_mol("C6H6")
    scorer = SOAPSimilarityModel([mol])
    assert np.allclose(scorer(mol, 0, False), 0, atol=1e-23)


def test_grad_non_zero_if_training_set_and_test_set_are_different_mols():
    mol = initialize_mol("C6H6")
    scorer = SOAPSimilarityModel([mol])
    assert not np.allclose(scorer(initialize_mol("CH4"), 0, False), 0, atol=1e-18)
