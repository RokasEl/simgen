import numpy as np
import torch
from e3nn import o3

from moldiff.particle_filtering import ParticleFilterGenerator

from .fixtures import loaded_model, training_molecules

torch.set_default_dtype(torch.float64)
from mace.data.atomic_data import AtomicData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable


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
