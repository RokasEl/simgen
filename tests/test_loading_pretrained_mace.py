import numpy as np
import torch
from e3nn import o3

from simgen.generation_utils import batch_atoms
from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import get_system_torch_device_str

from .fixtures import (
    loaded_model,
    loaded_one_layer_model,
    training_molecules,
)

torch.set_default_dtype(torch.float64)
from mace.data.atomic_data import AtomicData

DEVICE = get_system_torch_device_str()

from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable


def test_both_loading_methods_give_same_total_energies(
    training_molecules, loaded_model
):
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    device = get_system_torch_device_str()
    pretrained_model.to(device)
    z_table = z_table = AtomicNumberTable([int(z) for z in loaded_model.atomic_numbers])
    r_max = pretrained_model.r_max.item()
    batch_data = batch_atoms(training_molecules, z_table, r_max, device)
    out_2 = loaded_model(batch_data)
    batch_data = batch_atoms(training_molecules, z_table, r_max, device)
    out_1 = pretrained_model(batch_data)
    for key in out_1:
        if out_1[key] is not None:
            assert torch.allclose(out_1[key], out_2[key])


def test_loading_single_layer_model_gives_same_energies(
    training_molecules, loaded_one_layer_model
):
    pretrained_mace = "./models/SPICE_1l_neut_E0_swa.model"
    pretrained_model = torch.load(pretrained_mace)
    device = get_system_torch_device_str()
    pretrained_model.to(device)
    z_table = z_table = AtomicNumberTable(
        [int(z) for z in loaded_one_layer_model.atomic_numbers]
    )
    r_max = pretrained_model.r_max.item()
    batch_data = batch_atoms(training_molecules, z_table, r_max, device)
    out_2 = loaded_one_layer_model(batch_data)
    batch_data = batch_atoms(training_molecules, z_table, r_max, device)
    out_1 = pretrained_model(batch_data)
    for key in out_1:
        if out_1[key] is not None:
            assert torch.allclose(out_1[key], out_2[key])
