from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from mace.data.atomic_data import AtomicData
from mace.modules.blocks import LinearNodeEmbeddingBlock
from mace.modules.models import MACE
from mace.modules.utils import compute_forces, get_outputs
from mace.tools.scatter import scatter_sum

# Many ideas from https://github.com/NVlabs/edm/


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, max_positions=1024, endpoint=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.embed_dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.embed_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class EDMModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=50,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, model_input_dict, sigma, **model_kwargs):
        c_skip, c_out, c_in, c_noise = self.compute_prefactors(sigma)

        model_input_dict["positions"] = c_in * model_input_dict["positions"]
        model_input_dict["node_attrs"] = c_in * model_input_dict["node_attrs"]

        D_x = self.model(model_input_dict, c_noise.flatten(), **model_kwargs)
        D_x["node_forces"] = c_out * D_x["node_forces"]
        D_x["forces"] = c_out * D_x["forces"]
        D_x["positions"] = c_skip * model_input_dict["positions"] + D_x["forces"]
        D_x["node_attrs"] = c_skip * model_input_dict["node_attrs"] + D_x["node_forces"]
        return D_x

    def compute_prefactors(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise


class EDMLossFn:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, batch_data: AtomicData, model, **model_kwargs):
        molecule_sigmas = self._generate_sigmas(batch_data)
        weight = self._get_weight(molecule_sigmas)
        model_input = self._get_corrupted_input(batch_data, molecule_sigmas)
        D_yn = model(model_input.to_dict(), molecule_sigmas, **model_kwargs)
        pos_loss, elem_loss = self._calculate_loss(batch_data, D_yn)
        # reweight the element loss to penalise missing elements more when sigma is small
        weighted_loss = weight * (pos_loss + elem_loss)
        return weighted_loss.mean()

    def _generate_sigmas(self, batch_data: AtomicData):
        num_graphs = batch_data.num_graphs
        device = batch_data.positions.device  # type: ignore
        log_sigmas = self.P_mean + self.P_std * torch.randn(num_graphs).to(device)
        sigmas = torch.exp(log_sigmas)
        molecule_sigmas = sigmas[batch_data.batch.to(torch.long)]  # type: ignore
        molecule_sigmas = molecule_sigmas.to(torch.float64).reshape(-1, 1)
        return molecule_sigmas

    def _get_weight(self, molecule_sigmas: torch.Tensor):
        return (molecule_sigmas**2 + self.sigma_data**2) / (
            molecule_sigmas * self.sigma_data
        ) ** 2

    @staticmethod
    def _calculate_loss(
        original_data: AtomicData, reconstructed_data: Dict[str, torch.Tensor]
    ):
        position_loss = (original_data.positions - reconstructed_data["positions"]) ** 2
        position_loss = einops.reduce(
            position_loss, "num_nodes cartesians -> num_nodes", "sum"
        )
        element_loss = (
            original_data.node_attrs - reconstructed_data["node_attrs"]
        ) ** 2
        element_loss = einops.reduce(
            element_loss, "num_nodes elements -> num_nodes", "sum"
        )
        return position_loss, element_loss

    @staticmethod
    def _get_corrupted_input(batch_data: AtomicData, molecule_sigmas: torch.Tensor):
        model_input = batch_data.clone()
        model_input.positions += molecule_sigmas * torch.randn_like(
            model_input.positions
        )
        model_input.node_attrs += molecule_sigmas * torch.randn_like(
            model_input.node_attrs
        )
        model_input.node_attrs = model_input.node_attrs  # TODO: Discuss this line
        return model_input


@dataclass
class SamplerNoiseParameters:
    sigma_min: float = 0.1
    sigma_max: float = 30
    rho: float = 7
    S_churn: float = (
        0.0  # Churn rate of the noise level. 0 corresponds to an ODE solver.
    )
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1


@dataclass
class GradLogP:
    positions: torch.Tensor
    node_attrs: torch.Tensor


class EDMSampler:
    def __init__(
        self,
        model,
        sampler_noise_parameters=SamplerNoiseParameters(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.noise_parameters = sampler_noise_parameters

    def generate_samples(
        self,
        batch_data: AtomicData,
        num_steps=20,
        track_trajectory=False,
        **model_kwargs,
    ):
        sigmas = self._get_sigma_schedule(num_steps=num_steps)

        # Initialize the sample to gaussian noise.
        # sigmas[0] is the initial and largest noise level.
        x_next: AtomicData = self._initialize_sample(batch_data, sigmas[0])
        trajectories = []
        if track_trajectory:
            trajectories = [x_next.clone()]
        # Main sampling loop. A second order Heun integrator is used.
        # read the integrator parameters from the noise parameters.
        S_churn, S_min, S_max, S_noise = self._get_integrator_parameters()
        for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            x_cur = x_next.clone()

            # If current sigma is between S_min and S_max, then we first temporarily increase the current noise leve.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma_cur <= S_max
                else 0
            )
            # Added noise depends on the current noise level. So, it decreases over the course of the integration.
            sigma_increased = sigma_cur * (1 + gamma)
            # Add noise to the current sample.
            noise_level = (sigma_increased**2 - sigma_cur**2).sqrt() * S_noise
            x_cur.positions += torch.randn_like(x_cur.positions) * noise_level
            x_cur.node_attrs += torch.randn_like(x_cur.node_attrs) * noise_level
            x_increased = x_cur

            # Euler step.
            x_increased.positions.grad = None
            x_increased.node_attrs.grad = None
            D_x_increased = self.model(
                x_increased.to_dict(), sigma_increased, **model_kwargs
            )

            grad_log_P_noisy = GradLogP(
                positions=(x_increased.positions - D_x_increased["positions"])
                / sigma_increased,
                node_attrs=(x_increased.node_attrs - D_x_increased["node_attrs"])
                / sigma_increased,
            )
            x_next = x_cur.clone()
            x_next.positions += (
                sigma_next - sigma_increased
            ) * grad_log_P_noisy.positions
            x_next.node_attrs += (
                sigma_next - sigma_increased
            ) * grad_log_P_noisy.node_attrs

            # Apply 2nd order correction.
            if i < num_steps - 1:
                x_next.positions.grad = None
                x_next.node_attrs.grad = None
                D_x_next = self.model(x_next.to_dict(), sigma_next, **model_kwargs)
                grad_log_P_correction = GradLogP(
                    positions=(x_next.positions - D_x_next["positions"]) / sigma_next,
                    node_attrs=(x_next.node_attrs - D_x_next["node_attrs"])
                    / sigma_next,
                )
                x_next = x_increased.clone()
                x_next.positions += (
                    (sigma_next - sigma_increased)
                    * (grad_log_P_correction.positions + grad_log_P_noisy.positions)
                    / 2
                )
                x_next.node_attrs += (
                    (sigma_next - sigma_increased)
                    * (grad_log_P_correction.node_attrs + grad_log_P_noisy.node_attrs)
                    / 2
                )
                if track_trajectory:
                    trajectories.append(x_next.clone())
        if track_trajectory:
            return x_next, trajectories
        return x_next, []

    def _get_integrator_parameters(self):
        return (
            self.noise_parameters.S_churn,
            self.noise_parameters.S_min,
            self.noise_parameters.S_max,
            self.noise_parameters.S_noise,
        )

    def _get_sigma_schedule(self, num_steps: int):
        step_indices = torch.arange(num_steps).to(self.device)
        sigma_max, sigma_min, rho = (
            self.noise_parameters.sigma_max,
            self.noise_parameters.sigma_min,
            self.noise_parameters.rho,
        )
        max_noise_rhod = sigma_max ** (1 / rho)
        min_noise_rhod = sigma_min ** (1 / rho)
        noise_interpolation = (
            step_indices / (num_steps - 1) * (min_noise_rhod - max_noise_rhod)
        )
        sigmas = (max_noise_rhod + noise_interpolation) ** rho
        # Add a zero sigma at the end to get the sample.
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1]).to(self.device)])
        return sigmas

    @staticmethod
    def _initialize_sample(batch_data: AtomicData, sigma: torch.Tensor) -> AtomicData:
        x_next = batch_data
        x_next.positions = sigma * torch.randn_like(x_next.positions)
        x_next.node_attrs = sigma * torch.randn_like(x_next.node_attrs)
        return x_next


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        noise_embed_dim=16,  # Dimension of the noise embedding.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.model = EnergyMACEDiffusion(
            noise_embed_dim=noise_embed_dim, **model_kwargs
        )

    def forward(self, model_input_dict, sigma, **model_kwargs):
        sigma = sigma.to(torch.float64).reshape(-1, 1)

        c_skip, c_out, c_in, c_noise = self.compute_prefactors(sigma)

        model_input_dict["positions"] = c_in * model_input_dict[
            "positions"
        ] + sigma**2 * torch.randn_like(model_input_dict["positions"])

        model_input_dict["node_attrs"] = c_in * model_input_dict[
            "node_attrs"
        ] + sigma**2 * torch.randn_like(model_input_dict["node_attrs"])

        D_x = self.model(model_input_dict, c_noise.flatten(), **model_kwargs)
        D_x["node_forces"] = c_out * D_x["node_forces"]
        D_x["forces"] = c_out * D_x["forces"]
        D_x["positions"] = c_skip * model_input_dict["positions"] + D_x["forces"]
        D_x["node_attrs"] = c_skip * model_input_dict["node_attrs"] + D_x["node_forces"]
        return D_x

    def compute_prefactors(self, sigma):
        c_skip = 1
        c_out = sigma
        c_in = 1 - sigma
        c_noise = sigma
        return c_skip, c_out, c_in, c_noise


class TessLossFn:
    def __init__(self, P_mean=-0.8, P_std=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.pi = torch.as_tensor(np.pi)

    def __call__(self, batch_data, model, training=False):
        model_input = batch_data.clone()
        uncorrupted_dict = batch_data.to_dict()
        num_graphs = uncorrupted_dict["ptr"].numel() - 1
        log_time = self.P_mean + self.P_std**2 * torch.randn(num_graphs)
        times = torch.exp(log_time).to(uncorrupted_dict["positions"].device)
        molecule_times = times[uncorrupted_dict["batch"].to(torch.long)]
        molecule_sigmas = self._sigma_fn(molecule_times).unsqueeze(-1)
        weight = 1 / molecule_times

        corrupted_data_dict = model_input.to_dict()
        corrupted_data_dict["positions"] += molecule_sigmas * torch.randn_like(
            corrupted_data_dict["positions"]
        )
        corrupted_data_dict["node_attrs"] += molecule_sigmas * torch.randn_like(
            corrupted_data_dict["node_attrs"]
        )

        D_yn = model(corrupted_data_dict, molecule_sigmas, training=training)
        loss = (
            weight
            * (D_yn["node_attrs"] - uncorrupted_dict["node_attrs"]).pow(2).sum(dim=1)
            + (D_yn["positions"] - uncorrupted_dict["positions"]).pow(2).sum(dim=1)
            + (D_yn["node_energy"]).pow(2)
        )
        return weight, loss

    def _sigma_fn(self, t):
        return torch.sin(self.pi * t / 2) ** 2 + 1e-5


class EnergyMACEDiffusion(MACE):
    def __init__(self, noise_embed_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Change the weight initialization of the final layer to zeros.
        final_readout = [module for _, module in self.readouts[-1].named_children()][-1]
        final_readout.weight.data.zero_()
        # Add a positional embedding for the noise level.
        self.noise_embedding = PositionalEmbedding(noise_embed_dim)
        noise_in_irreps = o3.Irreps([(noise_embed_dim, (0, 1))])
        noise_out_irreps = o3.Irreps(
            [(kwargs["hidden_irreps"].count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.noise_linear = LinearNodeEmbeddingBlock(noise_in_irreps, noise_out_irreps)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        training: bool = False,
        compute_force: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        # Embeddings
        node_feats, edge_attrs, edge_feats = self._get_initial_embeddings(data)
        sigma_embedding = self.noise_embedding(sigmas)

        # Interactions
        energies = []
        node_energies_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            sigma_addition = F.silu(self.noise_linear(sigma_embedding))
            # Pad sigma_addition on the right by zeros to match node_feats.
            sigma_addition = F.pad(
                sigma_addition, (0, node_feats.shape[-1] - sigma_addition.shape[-1])
            )
            node_feats = node_feats + sigma_addition
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        # Outputs
        forces, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_stress=False,
            compute_virials=False,
        )
        node_forces = compute_forces(
            energy=total_energy, positions=data["node_attrs"], training=training
        )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "node_forces": node_forces,
        }
