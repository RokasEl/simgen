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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)


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
