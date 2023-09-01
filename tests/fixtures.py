import numpy as np
import pytest
import torch
from e3nn import o3
from mace.modules import interaction_classes
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE, ScaleShiftMACE
from mace.tools import AtomicNumberTable
from mace.tools.utils import get_atomic_number_table_from_zs

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.utils import get_system_torch_device_str, initialize_mol

DEVICE = get_system_torch_device_str()
torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def training_molecules():
    mol_strs = [
        "C2H2",
        "C6H6",
        "CH4",
        "CH3OH",
    ]
    atoms = [initialize_mol(mol_str) for mol_str in mol_strs]
    atoms[1].set_positions(
        np.array(
            [
                [0.695, 1.20377531, 0.0],
                [-0.695, 1.20377531, 0.0],
                [-1.39, 0.0, 0.0],
                [-0.695, -1.20377531, 0.0],
                [0.695, -1.20377531, 0.0],
                [1.39, 0.0, 0.0],
                [1.24, 2.147743, 0.0],
                [-1.24, 2.147743, 0.0],
                [-2.48, 0.0, 0.0],
                [-1.24, -2.147743, 0.0],
                [1.24, -2.147743, 0.0],
                [2.48, 0.0, 0.0],
            ]
        )
    )
    return atoms


@pytest.fixture(scope="module")
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
        hidden_irreps=o3.Irreps("16x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=atomic_numbers,
        correlation=1,
        gate=torch.nn.functional.silu,
    )
    model.to(DEVICE)
    return model


@pytest.fixture(scope="module")
def loaded_model():
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    model = ScaleShiftMACE(
        r_max=4.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=2,
        num_elements=10,
        atomic_energies=np.zeros(10),
        avg_num_neighbors=15.653135299682617,
        correlation=3,
        interaction_cls_first=RealAgnosticInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        hidden_irreps=o3.Irreps("96x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.088502,
        atomic_inter_shift=0.0,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    device = get_system_torch_device_str()
    model.to(device)
    return model


@pytest.fixture(scope="module")
def loaded_one_layer_model():
    pretrained_mace = "./models/SPICE_1l_neut_E0_swa.model"
    pretrained_model = torch.load(pretrained_mace)
    model = ScaleShiftMACE(
        r_max=4.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=1,
        num_elements=10,
        atomic_energies=np.zeros(10),
        avg_num_neighbors=15.653135299682617,
        correlation=3,
        interaction_cls_first=RealAgnosticInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        hidden_irreps=o3.Irreps("64x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.088502,
        atomic_inter_shift=0.0,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    device = get_system_torch_device_str()
    model.to(device)
    return model


@pytest.fixture(scope="module")
def loaded_mace_similarity_calculator(loaded_model, training_molecules):
    calc = MaceSimilarityCalculator(
        loaded_model, reference_data=training_molecules, device=DEVICE
    )
    return calc
