import logging

import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
from functools import partial

from fire import Fire

import wandb
from energy_model.diffusion_tools import (
    EDMLossFn,
    EDMModelWrapper,
    EnergyMACEDiffusion,
    iDDPMLossFunction,
    iDDPMModelWrapper,
)
from moldiff.utils import setup_logger

Z_TABLE = tools.AtomicNumberTable([1, 6, 7, 8, 9])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

atomic_energies = np.zeros_like(Z_TABLE.zs, dtype=np.float64)
MACE_CONFIG = dict(
    r_max=10.0,
    num_bessel=4,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ],
    num_interactions=2,
    num_elements=len(Z_TABLE.zs),
    hidden_irreps=o3.Irreps("96x0e + 96x1o"),
    MLP_irreps=o3.Irreps("96x0e"),
    radial_MLP=[64] * 3,
    gate=torch.nn.functional.silu,
    atomic_energies=atomic_energies,
    avg_num_neighbors=8,
    atomic_numbers=Z_TABLE.zs,
    correlation=3,
)

PARAMS = {
    "model_params": MACE_CONFIG,
    "lr": 2e-3,
    "batch_size": 128,
    "epochs": 250,
}


def main(
    data_path="./Data/qm9_data",
    model_path="test_model.pt",
    restart=False,
):
    setup_logger(
        level=logging.DEBUG, tag="train_energy_diffusion_mace", directory="./logs/"
    )

    collections, _ = get_dataset_from_xyz(
        data_path,
        config_type_weights={},
        valid_path=None,  # type: ignore
        valid_fraction=0.1,
    )
    cutoff = MACE_CONFIG["r_max"]
    to_atomic_data = partial(
        data.AtomicData.from_config, z_table=Z_TABLE, cutoff=cutoff
    )

    training_data = [to_atomic_data(conf) for conf in collections.train]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=training_data,  # type: ignore
        batch_size=PARAMS["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    validation_data = [to_atomic_data(conf) for conf in collections.valid]
    validation_data_loader = torch_geometric.dataloader.DataLoader(
        dataset=validation_data,  # type: ignore
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    avg_num_neighbors = modules.compute_avg_num_neighbors(data_loader)
    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    PARAMS["model_params"]["avg_num_neighbors"] = avg_num_neighbors
    wandb.init(
        # set the wandb project where this run will be logged
        project="energy-diffusion",
        entity="rokasel",
        # track hyperparameters and run metadata
        config=PARAMS,
        save_code=True,
    )

    model = EnergyMACEDiffusion(noise_embed_dim=32, **PARAMS["model_params"])
    model = iDDPMModelWrapper(model).to(DEVICE)
    if restart:
        save_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(save_dict["model_state_dict"])
        logging.info(f"Loaded model from {model_path}")

    loss_fn = iDDPMLossFunction(P_mean=-1.4, P_std=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=PARAMS["lr"],
        total_steps=len(data_loader) * PARAMS["epochs"],
    )
    for epoch in range(PARAMS["epochs"]):
        model.train()
        for _, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            weight, pos_loss, elem_loss = loss_fn(batch_data, model, training=True)
            loss = (weight * (pos_loss + elem_loss)).mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), 250.0)
            optimizer.step()
            scheduler.step()
            wandb.log(
                {
                    "loss": loss.item(),
                    "unweighted_loss": (pos_loss + elem_loss).mean().item(),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
        # validation
        val_pos_loss = []
        val_elem_loss = []
        weights = []
        model.eval()
        for _, batch_data in enumerate(validation_data_loader):
            batch_data = batch_data.to(DEVICE)
            weight, pos_loss, elem_loss = loss_fn(batch_data, model, training=True)
            model.zero_grad()
            weights.append(weight.detach().cpu().numpy())
            val_pos_loss.append((pos_loss).detach().cpu().numpy())
            val_elem_loss.append((elem_loss).detach().cpu().numpy())
        val_pos_loss = np.concatenate(val_pos_loss)
        val_elem_loss = np.concatenate(val_elem_loss)
        weights = np.concatenate(weights)
        val_loss = ((val_pos_loss + val_elem_loss) * weights).mean()
        unweighted_val_loss = (val_pos_loss + val_elem_loss).mean()
        wandb.log(
            {
                "val_loss": val_loss,
                "unweighted_val_loss": unweighted_val_loss,
                "val_pos_loss": val_pos_loss.mean(),
                "val_elem_loss": val_elem_loss.mean(),
                "epoch": epoch,
            }
        )
        logging.info(f"Epoch {epoch}: {val_loss.item()}")

    # save model
    save_dict = {
        "model_params": PARAMS["model_params"],
        "model_state_dict": model.state_dict(),
    }
    torch.save(save_dict, model_path)


if __name__ == "__main__":
    Fire(main)
