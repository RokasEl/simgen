import logging

import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
from diffusion_tools import (
    EDMLossFn,
    EDMModelWrapper,
    EnergyMACEDiffusion,
)

from moldiff.utils import initialize_mol, setup_logger

Z_TABLE = tools.AtomicNumberTable([1, 6])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    setup_logger(
        level=logging.DEBUG, tag="train_energy_diffusion_mace", directory="./logs/"
    )
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
    try:
        model.load_state_dict(torch.load("./model.pt", map_location=DEVICE))
        logging.info("Loaded model from ./model.pt")
    except FileNotFoundError:
        pass
    mol = initialize_mol("C6H6")
    config = data.Configuration(
        atomic_numbers=mol.get_atomic_numbers(),
        positions=mol.positions,
    )
    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=10.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data] * 32,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    loss_fn = EDMLossFn(P_mean=-1.2, P_std=0.8, sigma_data=1)
    batch_data = next(iter(data_loader)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for i in range(300):
        optimizer.zero_grad()
        loss = loss_fn(batch_data, model, training=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        if i % 10 == 0:
            logging.info(f"Batch {i}: {loss.item()}")
    # save model
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
