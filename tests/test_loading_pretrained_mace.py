import numpy as np
import torch
from e3nn import o3

from moldiff.particle_filtering import ParticleFilterGenerator

from .fixtures import training_molecules

torch.set_default_dtype(torch.float64)
from ase import Atoms
from mace.data.atomic_data import AtomicData

from moldiff.utils import initialize_mol, read_qm9_xyz, setup_logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import logging

import ase.io as ase_io
import mace
import pytest
from mace.calculators import MACECalculator
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.modules.blocks import (
    RadialDistanceTransformBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE, ScaleShiftMACE
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.diffusion_tools import EDMSampler, SamplerNoiseParameters
from moldiff.sampling import MaceSimilarityScore


@pytest.fixture
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


def get_batch_data(atoms, z_table, r_max, device):
    confs = [config_from_atoms(x) for x in atoms]
    atomic_datas = [
        AtomicData.from_config(x, z_table=z_table, cutoff=r_max).to(device)
        for x in confs
    ]
    batch_data = next(
        iter(get_data_loader(atomic_datas, batch_size=len(atomic_datas), shuffle=False))
    )
    return batch_data


def test_both_loading_methods_give_same_total_energies(
    training_molecules, loaded_model
):
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model.to(device)
    z_table = z_table = AtomicNumberTable([int(z) for z in loaded_model.atomic_numbers])
    r_max = pretrained_model.r_max.item()
    batch_data = get_batch_data(training_molecules, z_table, r_max, device)
    out_2 = loaded_model(batch_data)
    batch_data = get_batch_data(training_molecules, z_table, r_max, device)
    out_1 = pretrained_model(batch_data)
    for key in out_1:
        if out_1[key] is not None:
            assert torch.allclose(out_1[key], out_2[key])
