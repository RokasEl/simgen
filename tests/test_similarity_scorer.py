import einops
import numpy as np
import pytest
import torch
from e3nn import o3
from mace.modules import interaction_classes
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable
from mace.tools.utils import get_atomic_number_table_from_zs

from moldiff.sampling import MaceSimilarityScore
from moldiff.utils import initialize_mol

torch.set_default_dtype(torch.float64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def training_molecules():
    mol_strs = ["C6H6", "CH4", "CH3OH", "CH3CH2OH"]
    atoms = [initialize_mol(mol_str) for mol_str in mol_strs]
    return atoms


@pytest.fixture
def test_molecules():
    mol_strs = ["CH3COOH", "C2H4", "H2O"]
    atoms = [initialize_mol(mol_str) for mol_str in mol_strs]
    return atoms


@pytest.fixture
def z_table(training_molecules) -> AtomicNumberTable:
    zs = np.concatenate([atom.get_atomic_numbers() for atom in training_molecules])
    return get_atomic_number_table_from_zs(zs)


@pytest.fixture
def mace_model(z_table):
    """Initialize a small MACE model for testing"""
    atomic_numbers = z_table.zs
    atomic_energies = np.array([-10, -1000, -2000], dtype=float)
    model = MACE(
        r_max=4.0,
        num_bessel=3,
        num_polynomial_cutoff=3,
        max_ell=3,
        interaction_cls=interaction_classes["RealAgnosticInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=1,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e"),
        MLP_irreps=o3.Irreps("128x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=atomic_numbers,
        correlation=1,
        gate=torch.nn.functional.silu,
    )
    model.to(DEVICE)
    return model


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
        # mat shape (n_training_nodes, n_input_nodes)
        # all test nodes here are in the training set, so the minimum distance should be zero
        min_across_input_nodes = np.min(mat, axis=0)  # (n_input_nodes,)
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
        min_across_input_nodes = np.min(mat, axis=0)  # (n_input_nodes,)
        assert np.any(min_across_input_nodes > 1e-6)


def test_log_kernel_density_of_training_data_much_higher_than_new_data(
    mace_similarity_scorer, training_molecules, test_molecules
):
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
    assert np.min(training_log_dens) - np.max(test_log_dens) > 100


def test_gradient_of_log_kernel_density_is_close_to_zero_for_training_data(
    mace_similarity_scorer, training_molecules
):
    gradients = [mace_similarity_scorer(mol, t=0) for mol in training_molecules]
    for grad in gradients:
        assert np.allclose(grad, 0.0, atol=1e-10)


def test_slightly_perturbed_training_molecules_have_non_zero_gradient(
    mace_similarity_scorer, training_molecules
):
    for mol in training_molecules:
        mol.set_positions(
            mol.get_positions() + 1e-2 * np.random.randn(*mol.get_positions().shape)
        )
    gradients = [mace_similarity_scorer(mol, t=1) for mol in training_molecules]
    for grad in gradients:
        assert np.any(grad != 0.0)


def test_unseen_molecules_have_finite_gradients(mace_similarity_scorer, test_molecules):
    gradients = [mace_similarity_scorer(mol, t=1) for mol in test_molecules]
    for grad in gradients:
        assert np.any(grad != 0.0)


def test_nan_gradient_when_overlapping_atoms(mace_similarity_scorer):
    mol = initialize_mol("H2")
    mol.set_positions(np.array([[0, 0, 0], [0, 0, 0.01]]))
    atomic_data = mace_similarity_scorer._to_atomic_data(mol)
    embedding = mace_similarity_scorer._get_node_embeddings(atomic_data)
    log_dens = mace_similarity_scorer._get_log_kernel_density(embedding, 1)
    grad = mace_similarity_scorer._get_gradient(atomic_data, log_dens)
    assert np.any(np.isnan(grad))


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
            np.ones((5, 3)) * np.array([[100000], [1], [1], [1], [1]]),
            np.ones((5, 3)) * np.array([[10 / np.sqrt(3)], [1], [1], [1], [1]]),
        ),
    ],
)
def test_gradient_magnitude_handler(mace_similarity_scorer, grad, expected_out):
    grad = mace_similarity_scorer._handle_grad_magnitude(grad)
    np.testing.assert_allclose(grad, expected_out, atol=1e-6)
