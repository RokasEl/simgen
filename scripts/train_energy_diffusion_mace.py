import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
from moldiff.networks import (
    EDMAtomDataPreconditioning,
    EDMLossFn,
    EnergyMACEDiffusion,
)
from moldiff.utils import initialize_mol

Z_TABLE = tools.AtomicNumberTable([1, 6])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("128x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=Z_TABLE.zs,
        correlation=3,
    )
    # model = EnergyMACEDiffusion(noise_embed_dim=16, **model_config)
    model = EDMAtomDataPreconditioning(
        sigma_max=10, noise_embed_dim=32, **model_config
    ).to(DEVICE)
    mol = initialize_mol("C6H6")
    config = data.Configuration(
        atomic_numbers=mol.get_atomic_numbers(),
        positions=mol.positions,
        energy=-1.5,
    )
    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=3.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data] * 16,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(DEVICE).to_dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(200):
        optimizer.zero_grad()
        weight, loss = EDMLossFn()(batch, model, training=True)
        model_loss = (weight * loss).mean()
        model_loss.backward()
        clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if i % 10 == 0:
            print(loss.mean().item())
    # save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
