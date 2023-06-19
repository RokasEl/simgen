import numpy as np
import torch
from mace import data, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
import ase
import ase.io as aio
from ase import Atoms

from energy_model.diffusion_tools import (
    EDMModelWrapper,
    EDMSampler,
    EnergyMACEDiffusion,
    SamplerNoiseParameters,
)
from moldiff.utils import initialize_mol

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def atomic_data_to_ase(node_attrs, positions, energy):
    elements = node_attrs.detach().cpu().numpy()
    elements = np.argmax(elements, axis=1)
    elements = [Z_TABLE.zs[z] for z in elements]
    positions = positions.detach().cpu().numpy()
    atoms = Atoms(elements, positions)
    atoms.info["energy"] = energy.detach().cpu().numpy()
    return atoms


def batch_to_ase(batch):
    ptr = batch.ptr.detach().cpu().numpy()
    for num, (i, j) in enumerate(zip(ptr[:-1], ptr[1:])):
        energy = batch.energy[num]
        node_attrs = batch.node_attrs[i:j]
        positions = batch.positions[i:j]
        yield atomic_data_to_ase(node_attrs, positions, energy)


Z_TABLE = tools.AtomicNumberTable([1, 6, 7, 8, 9])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataloader(min_size, max_size, num_samples_per_size, batch_size=32):
    dataset = []
    for size in range(min_size, max_size + 1):
        mol = initialize_mol("C" * size)
        config = data.Configuration(
            atomic_numbers=mol.get_atomic_numbers(),
            positions=mol.positions,
            energy=-1.5,
        )
        atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=10.0)
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
    sampler_params=SamplerNoiseParameters(),
):
    save_dict = torch.load(model_path, map_location=DEVICE)
    model = EnergyMACEDiffusion(noise_embed_dim=32, **save_dict["model_params"])
    model = EDMModelWrapper(model, sigma_data=1.0).to(DEVICE)
    model.load_state_dict(save_dict["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    noise_params = sampler_params
    sampler = EDMSampler(model, sampler_noise_parameters=noise_params, device=DEVICE)

    data_loader = get_dataloader(3, 29, num_samples_per_size, batch_size=16)

    for batch in data_loader:
        batch_data = batch.to(DEVICE)
        final, _ = sampler.generate_samples(batch_data, num_steps=30, training=True)
        batch_data = None
        model.zero_grad()
        as_ase = [x for x in batch_to_ase(final)]
        final = None
        aio.write(save_path, as_ase, format="extxyz", append=True)
        should_break = check_generated_structures_for_nans(save_path)
        if should_break:
            print("Stopping early due to too many NaNs")
            print("Try different noise parameters")
            print(f"Current noise parameters: {sampler_params}")
            return


if __name__ == "__main__":
    params = SamplerNoiseParameters(
        sigma_max=20, sigma_min=1e-3, S_churn=20, S_min=0.10, S_noise=1.00
    )
    main(sampler_params=params)
