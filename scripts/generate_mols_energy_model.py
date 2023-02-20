import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
import ase
from ase import Atoms
from diffusion_tools import (
    EDMModelWrapper,
    EDMSampler,
    EnergyMACEDiffusion,
    SamplerNoiseParameters,
)

from moldiff.utils import initialize_mol

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def atomic_data_to_ase(atomic_data):
    elements = atomic_data["node_attrs"].detach().cpu().numpy()
    elements = np.argmax(elements, axis=1)
    elements = [Z_TABLE.zs[z] for z in elements]
    positions = atomic_data["positions"].detach().cpu().numpy()
    atoms = Atoms(elements, positions)
    return atoms


Z_TABLE = tools.AtomicNumberTable([1, 6])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model_config = dict(
        r_max=5,
        num_bessel=5,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("64x0e + 64x1o"),
        MLP_irreps=o3.Irreps("64x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=Z_TABLE.zs,
        correlation=3,
    )
    model = EnergyMACEDiffusion(noise_embed_dim=32, **model_config)
    model = EDMModelWrapper(model, sigma_data=1).to(DEVICE)
    model.load_state_dict(torch.load("./model.pt", map_location=DEVICE))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    mol = initialize_mol("C6H6")
    config = data.Configuration(
        atomic_numbers=mol.get_atomic_numbers(),
        positions=mol.positions,
        energy=-1.5,
    )
    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=10.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data],
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
    batch_data = next(iter(data_loader)).to(DEVICE)
    noise_params = SamplerNoiseParameters(
        sigma_max=10, sigma_min=2e-3, S_churn=40, S_min=0.01, S_noise=1.003
    )
    sampler = EDMSampler(model, sampler_noise_parameters=noise_params, device=DEVICE)
    final, trajectories = sampler.generate_samples(
        batch_data, num_steps=20, track_trajectory=True, training=True
    )
    trajectories.append(final)
    atoms = [atomic_data_to_ase(x.to_dict()) for x in trajectories]
    ase.io.write("./test.xyz", atoms, append=True)


if __name__ == "__main__":
    main()
