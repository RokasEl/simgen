import numpy as np
import torch
from mace import data, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
import ase
import ase.io as aio
from ase import Atoms
from fire import Fire

from energy_model.diffusion_tools import (
    HeunSampler,
    SamplerNoiseParameters,
)
from simgen.utils import get_system_torch_device_str, initialize_mol

DEVICE = get_system_torch_device_str()


def atomic_data_to_ase(node_attrs, positions):
    elements = node_attrs.detach().cpu().numpy()
    elements = np.argmax(elements, axis=1)
    elements = [Z_TABLE.zs[z] for z in elements]
    positions = positions.detach().cpu().numpy()
    atoms = Atoms(elements, positions)
    return atoms


def batch_to_ase(batch):
    ptr = batch.ptr.detach().cpu().numpy()
    for i, j in zip(ptr[:-1], ptr[1:]):
        node_attrs = batch.node_attrs[i:j]
        positions = batch.positions[i:j]
        yield atomic_data_to_ase(node_attrs, positions)


Z_TABLE = tools.AtomicNumberTable([1, 6, 7, 8, 9])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataloader(
    min_size, max_size, num_samples_per_size, batch_size=32, cutoff=10.0
):
    dataset = []
    for size in range(min_size, max_size + 1):
        mol = initialize_mol("C" * size)
        config = data.Configuration(
            atomic_numbers=mol.get_atomic_numbers(),
            positions=mol.positions,
        )
        atomic_data = data.AtomicData.from_config(
            config, z_table=Z_TABLE, cutoff=cutoff
        )
        dataset += [atomic_data] * num_samples_per_size

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return data_loader


def check_generated_structures_for_nans(save_path, threshold=100):
    structs = aio.read(save_path, ":", format="extxyz")
    nans = 0
    for atoms in structs:
        if np.isnan(atoms.get_positions()).any():
            nans += 1
    if nans >= threshold:
        should_break = True
    else:
        should_break = False
    return should_break


def main(
    model_path="./trained_energy_mace.pt",
    save_path="./mols_energy_model.xyz",
    num_samples_per_size=10,
    sampler_params=SamplerNoiseParameters(
        sigma_max=3, sigma_min=1.5e-4, S_churn=10, S_min=2e-3, S_noise=1.014
    ),
    batch_size=128,
    track_trajectory=False,
):
    model = torch.load(model_path, map_location=DEVICE)
    cutoff = model.model.r_max.item()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    noise_params = sampler_params
    sampler = HeunSampler(model, sampler_noise_parameters=noise_params, device=DEVICE)

    batch_size = 1 if track_trajectory else batch_size
    data_loader = get_dataloader(
        15, 25, num_samples_per_size, batch_size=batch_size, cutoff=cutoff
    )
    for batch in data_loader:
        batch_data = batch.to(DEVICE)
        final, trajectory, _ = sampler.generate_samples(
            batch_data, num_steps=30, training=False, track_trajectory=track_trajectory
        )
        batch_data = None
        model.zero_grad()
        as_ase = [x for x in batch_to_ase(final)]
        if track_trajectory:
            as_ase = [y for x in trajectory for y in batch_to_ase(x)] + as_ase
        final = None
        aio.write(save_path, as_ase, format="extxyz", append=True)
        should_break = check_generated_structures_for_nans(save_path)
        if should_break:
            print("Stopping early due to too many NaNs")
            print("Try different noise parameters")
            print(f"Current noise parameters: {sampler_params}")
            return


if __name__ == "__main__":
    Fire(main)
