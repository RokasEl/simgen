from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from e3nn import o3
from mace.modules.blocks import LinearNodeEmbeddingBlock
from mace.modules.models import MACE
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)
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


class EDMAtomDataPreconditioning(torch.nn.Module):
    def __init__(
        self,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=50,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model_type="SomeModel",  # Class name of the underlying model.
        noise_embed_dim=16,  # Dimension of the noise embedding.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        # self.model = globals()[model_type](**model_kwargs)
        self.model = EnergyMACEDiffusion(
            noise_embed_dim=noise_embed_dim, **model_kwargs
        )

    def forward(self, batch_dict, sigma, **model_kwargs):
        sigma = sigma.to(torch.float64).reshape(-1, 1)

        c_skip, c_out, c_in, c_noise = self.compute_prefactors(sigma)

        model_input_dict = batch_dict.copy()
        model_input_dict["positions"] = c_in * (
            model_input_dict["positions"]
            + sigma**2 * torch.randn_like(model_input_dict["positions"])
        )
        model_input_dict["node_attrs"] = c_in * (
            model_input_dict["node_attrs"]
            + sigma**2 * torch.randn_like(model_input_dict["node_attrs"])
        )

        D_x = self.model(model_input_dict, c_noise.flatten(), **model_kwargs)
        D_x["node_forces"] = c_out * D_x["node_forces"]
        D_x["forces"] = c_out * D_x["forces"]
        D_x["positions"] = c_skip * batch_dict["positions"] + D_x["forces"]
        D_x["node_attrs"] = c_skip * batch_dict["node_attrs"] + D_x["node_forces"]
        D_x["node_attrs"] = F.softmax(D_x["node_attrs"], dim=1)
        return D_x

    def compute_prefactors(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMLossFn:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, batch_dict, model, training=False):
        num_graphs = batch_dict["ptr"].numel() - 1
        log_sigmas = self.P_mean + self.P_std**2 * torch.randn(num_graphs)
        sigmas = torch.exp(log_sigmas).to(batch_dict["positions"].device)
        molecule_sigmas = sigmas[batch_dict["batch"].to(torch.long)]
        weight = (molecule_sigmas**2 + self.sigma_data**2) / (
            molecule_sigmas * self.sigma_data
        ) ** 2
        D_yn = model(batch_dict, molecule_sigmas, training=training)
        loss = ((D_yn["positions"] - batch_dict["positions"]) ** 2).sum(dim=1) + (
            (D_yn["node_attrs"] - batch_dict["node_attrs"]) ** 2
        ).sum(dim=1)
        return weight, loss


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
            energy=node_energy, positions=data["node_attrs"], training=training
        )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "node_forces": node_forces,
        }
