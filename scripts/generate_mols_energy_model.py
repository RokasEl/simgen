import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric
from torch.nn.utils.clip_grad import clip_grad_norm_

torch.set_default_dtype(torch.float64)
import ase
from ase import Atoms

from moldiff.networks import (
    EDMAtomDataPreconditioning,
    EDMLossFn,
    EnergyMACEDiffusion,
    ModelWrapper,
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


def edm_sampler(
    net,
    batch_data,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=10,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    destination=None,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    device = DEVICE
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = batch_data.clone()
    x_next.positions = (
        torch.randn(x_next.positions.shape, device=device, dtype=torch.float64)
        * t_steps[0]
    )
    x_next.node_attrs = (
        torch.randn(x_next.node_attrs.shape, device=device, dtype=torch.float64)
        * t_steps[0]
    )
    if destination is not None:
        ase.io.write(destination, atomic_data_to_ase(x_next.to_dict()), append=True)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next.clone()

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_cur.positions = x_cur.positions + (
            t_hat**2 - t_cur**2
        ).sqrt() * S_noise * torch.randn_like(x_cur.positions)
        x_cur.node_attrs = x_cur.node_attrs + (
            t_hat**2 - t_cur**2
        ).sqrt() * S_noise * torch.randn_like(x_cur.node_attrs)
        x_hat = x_cur

        # Euler step.
        denoised = net(x_hat.to_dict(), t_hat, training=True)
        denoised["positions"] = (x_hat.positions - denoised["positions"]) / t_hat
        denoised["node_attrs"] = (x_hat.node_attrs - denoised["node_attrs"]) / t_hat

        x_next.positions = x_hat.positions + (t_next - t_hat) * denoised["positions"]
        x_next.node_attrs = x_hat.node_attrs + (t_next - t_hat) * denoised["node_attrs"]

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised_prime = net(x_next.to_dict(), t_next, training=True)
            denoised_prime["positions"] = (
                x_next.positions - denoised_prime["positions"]
            ) / t_next
            denoised_prime["node_attrs"] = (
                x_next.node_attrs - denoised_prime["node_attrs"]
            ) / t_next
            x_next.positions = x_hat.positions + (t_next - t_hat) * (
                0.5 * denoised["positions"] + 0.5 * denoised_prime["positions"]
            )
            x_next.node_attrs = x_hat.node_attrs + (t_next - t_hat) * (
                0.5 * denoised["node_attrs"] + 0.5 * denoised_prime["node_attrs"]
            )
        if destination is not None:
            ase.io.write(destination, atomic_data_to_ase(x_next.to_dict()), append=True)
    return x_next


def simple_sampler(model, batch_data, num_steps=50):

    x_T = batch_data.clone()
    x_T.positions = torch.randn(x_T.positions.shape, device=DEVICE, dtype=torch.float64)
    x_T.node_attrs = torch.randn(
        x_T.node_attrs.shape, device=DEVICE, dtype=torch.float64
    )

    def sigma_fn(t):
        return torch.sin(np.pi * t / 2) ** 2 + 1e-5

    t_space = torch.linspace(0, 1, num_steps, device=DEVICE, dtype=torch.float64)[::-1]
    sigmas = sigma_fn(t_space)

    for sigma in sigmas:
        pass


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
    model = EDMAtomDataPreconditioning(
        noise_embed_dim=32, sigma_data=1, **model_config
    ).to(DEVICE)
    model.load_state_dict(torch.load("./model.pt", map_location=DEVICE))

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
    edm_sampler(
        model,
        batch_data,
        num_steps=30,
        destination="./benzene_energy_model.xyz",
        S_churn=80,
        S_min=0.05,
        S_max=50,
        S_noise=1.003,
    )
    print("Done!")


if __name__ == "__main__":
    main()
