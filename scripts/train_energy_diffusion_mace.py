import logging

import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
from functools import partial

import wandb
from diffusion_tools import (
    EDMLossFn,
    EDMModelWrapper,
    EnergyMACEDiffusion,
)
from fire import Fire
from lion_pytorch import Lion

from moldiff.utils import initialize_mol, read_qm9_xyz, setup_logger

Z_TABLE = tools.AtomicNumberTable([1, 6])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

atomic_energies = np.zeros_like(Z_TABLE.zs, dtype=np.float64)
MACE_CONFIG = dict(
    r_max=10,
    num_bessel=5,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
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

PARAMS = {
    "model_params": MACE_CONFIG,
    "lr": 1e-2,
    "batch_size": 64,
    "epochs": 3,
    "warmup_steps": 1000,
}


def main(data_path, model_path, restart=False):
    setup_logger(
        level=logging.DEBUG, tag="train_energy_diffusion_mace", directory="./logs/"
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project="energy-diffusion",
        # track hyperparameters and run metadata
        config=PARAMS,
        save_code=True,
    )

    model = EnergyMACEDiffusion(noise_embed_dim=32, **PARAMS["model_params"])
    model = EDMModelWrapper(model, sigma_data=1).to(DEVICE)
    if restart:
        save_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(save_dict["model_state_dict"])
        logging.info(f"Loaded model from {model_path}")

    rng = np.random.default_rng(0)
    training_data = [
        read_qm9_xyz(f"{data_path}/dsgdb9nsd_{i:06d}.xyz")
        for i in rng.choice(133885, 2048, replace=False)
    ]
    to_atomic_data = partial(data.AtomicData.from_config, z_table=Z_TABLE, cutoff=10.0)
    training_data = [
        data.Configuration(atomic_numbers=d.atomic_numbers, positions=d.positions)
        for d in training_data
    ]
    training_data = [to_atomic_data(conf) for conf in training_data]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=training_data,  # type: ignore
        batch_size=PARAMS["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    loss_fn = EDMLossFn(P_mean=-1.2, P_std=1.2, sigma_data=1)

    optimizer = Lion(model.parameters(), lr=PARAMS["lr"])
    for epoch in range(PARAMS["epochs"]):
        for i, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(batch_data, model, training=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1000.0)
            wandb.log({"loss": loss.item()})
            optimizer.step()
        logging.info(f"Epoch {epoch}: {loss.item()}")
    # save model
    save_dict = {
        "model_params": PARAMS["model_params"],
        "model_state_dict": model.state_dict(),
    }
    torch.save(save_dict, model_path)


if __name__ == "__main__":
    Fire(main)