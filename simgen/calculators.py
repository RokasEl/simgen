"""ASE calculator comparing SOAP similarity"""
import logging
import warnings
from functools import partial
from typing import List

import ase
import einops
import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable
from mace.tools.scatter import scatter_sum

from simgen.generation_utils import (
    ExponentialRepulsionBlock,
    batch_atoms,
    batch_to_correct_dtype,
    convert_atoms_to_atomic_data,
    get_model_dtype,
    remove_elements,
)


class MaceSimilarityCalculator(Calculator):
    implemented_properties = ["energy", "forces", "energies"]

    def __init__(
        self,
        model: MACE,
        reference_data: List[ase.Atoms],
        device: str = "cpu",
        alpha: float = 8.0,
        element_sigma_array: np.ndarray | None = None,
        *args,
        restart=None,
        label=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        """

        Parameters
        ----------
        model
            a pretrained MACE model
        alpha
            parameter for controlling range of repulsion. High alpha -> short repulsion range
        """
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )
        self.model = model
        self.dtype = get_model_dtype(model)
        self.repulsion_block = ExponentialRepulsionBlock(alpha=alpha).to(device)
        self.device = device
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.element_kernel_sigmas = self._init_element_kernel_sigmas(
            element_sigma_array
        )
        self.cutoff = model.r_max.item()  # type: ignore

        batch_func = partial(
            batch_atoms, z_table=self.z_table, cutoff=self.cutoff, device=self.device
        )
        self.batch_atoms = lambda x: batch_to_correct_dtype(batch_func(x), self.dtype)
        self.convert_to_atomic_data = partial(
            convert_atoms_to_atomic_data,
            z_table=self.z_table,
            cutoff=self.cutoff,
            device=self.device,
        )

        self.reference_embeddings = self._calculate_reference_embeddings(reference_data)
        self.typical_length_scale = self._calculate_mean_dot_product(
            self.reference_embeddings
        )

    def __call__(
        self,
        atomic_data,
        t,
    ):
        batch_index = atomic_data.batch
        emb = self._get_node_embeddings(atomic_data)
        log_dens = self._calculate_log_k(emb, atomic_data.node_attrs, t)
        log_dens = scatter_sum(log_dens, batch_index)
        grad = self._get_gradient(atomic_data.positions, log_dens)
        repulsive_energy = self.repulsion_block(atomic_data) / 3.0
        repulsive_force = self._get_gradient(
            atomic_data.positions, repulsive_energy * -1
        )
        grad = self._clip_grad_norm(grad, max_norm=np.sqrt(3))
        return grad + repulsive_force

    def calculate(
        self,
        atoms: None | ase.Atoms = None,
        properties=None,
        system_changes=all_changes,
    ):
        self.reset()
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object must be provided")

        calculation_type = atoms.info.get("calculation_type", None)
        if calculation_type is None:
            raise ValueError(
                "`atoms.info['calculation_type']` must be either 'similarity' or 'mace'"
            )
        mask = atoms.info.get("mask", np.zeros(len(atoms), dtype=bool))
        if calculation_type == "similarity":
            (
                node_energies,
                forces,
                molecule_energies,
            ) = self._calculate_similarity_energies_and_forces(atoms)
        elif calculation_type == "mace":
            (
                node_energies,
                forces,
                molecule_energies,
            ) = self._calculate_mace_interaction_energies_and_forces(atoms)
        else:
            raise ValueError(
                f"calculation_type must be either 'similarity' or 'mace', not {calculation_type}"
            )
        forces[mask] = 0
        self.results["energies"] = node_energies
        self.results["energy"] = molecule_energies
        self.results["forces"] = forces

    def _calculate_similarity_energies_and_forces(self, atoms: ase.Atoms):
        try:
            time = atoms.info["time"]
        except KeyError:
            raise KeyError("Atoms object must have a time attribute")

        batched = self.batch_atoms(atoms)
        embedding = self._get_node_embeddings(batched)
        log_k = self._calculate_log_k(embedding, batched.node_attrs, time)
        node_energies = -1 * log_k
        molecule_energies = scatter_sum(node_energies, batched.batch, dim=0)
        node_energies = node_energies.detach().cpu().numpy()
        molecule_energies = molecule_energies.detach().cpu().numpy()
        force = self._get_gradient(batched.positions, log_k).detach().cpu().numpy()
        force = self._handle_grad_nans(force)
        return node_energies, force, molecule_energies

    def _calculate_mace_interaction_energies_and_forces(self, atoms: ase.Atoms):
        batched = self.batch_atoms(atoms)
        out = self.model(batched.to_dict())
        forces = out["forces"].detach().cpu().numpy()
        node_energies = out["node_energy"]
        node_e0s = self.model.atomic_energies_fn(batched.node_attrs)
        node_interaction_energies = node_energies - node_e0s
        molecule_energies = scatter_sum(node_interaction_energies, batched.batch, dim=0)
        molecule_energies = molecule_energies.detach().cpu().numpy()
        node_interaction_energies = node_interaction_energies.detach().cpu().numpy()
        return node_interaction_energies, forces, molecule_energies

    @staticmethod
    def _get_gradient(inp_tensor: torch.Tensor, log_dens: torch.Tensor):
        grad = torch.autograd.grad(
            outputs=log_dens,
            inputs=inp_tensor,
            grad_outputs=torch.ones_like(log_dens),
            create_graph=True,
            only_inputs=True,
            retain_graph=True,
        )[0]
        return grad

    @staticmethod
    def _clip_grad_norm(
        grad: np.ndarray | torch.Tensor, max_norm: float = 1
    ) -> np.ndarray | torch.Tensor:
        if isinstance(grad, np.ndarray):
            norm = np.linalg.norm(grad, axis=1)
        elif isinstance(grad, torch.Tensor):
            norm = torch.norm(grad, dim=1)
        norm[norm < max_norm] = max_norm
        mult = max_norm / norm
        grad = grad * mult[:, None]
        return grad

    @staticmethod
    def _calculate_mean_dot_product(x):
        dot_products = einops.reduce(x**2, "n m -> n", "sum")
        mean_dot_product = einops.reduce(dot_products, "n -> ()", "mean")
        return mean_dot_product

    def _get_node_embeddings(self, data: AtomicData):
        # Embeddings
        out = self.model(data, compute_force=False)
        node_feats = out["node_feats"]  # (n_nodes, features_per_layer*num_layers)
        irreps_out = self.model.products[0].linear.__dict__["irreps_out"]  # type: ignore
        l_max = irreps_out.lmax
        num_features = irreps_out.dim // (l_max + 1) ** 2
        node_feats = node_feats[
            :, :num_features
        ]  # extract invariant features only from the first layer
        return node_feats

    def _calculate_reference_embeddings(
        self, training_data: List[ase.Atoms]
    ) -> torch.Tensor:
        as_atomic_data = self.convert_to_atomic_data(training_data)
        dloader = get_data_loader(as_atomic_data, batch_size=128, shuffle=False)
        with torch.no_grad():
            node_embeddings = [
                self._get_node_embeddings(batch_to_correct_dtype(data, self.dtype))
                for data in dloader
            ]
        return torch.concatenate(node_embeddings, dim=0)

    def _calculate_distance_matrix(self, embedding, node_attrs):
        embedding_deltas = embedding[:, None, :] - self.reference_embeddings[None, :, :]
        # embedding_deltas (embedding_nodes, reference_nodes, embed_dim)
        embedding_deltas = embedding_deltas**2
        squared_distance_matrix = torch.sum(
            embedding_deltas, dim=-1
        )  # (embedding_nodes, reference_nodes)
        squared_distance_matrix = squared_distance_matrix / self.typical_length_scale
        element_specific_sigmas = self._scatter_element_sigmas(
            node_attrs, self.element_kernel_sigmas
        )
        squared_distance_matrix = squared_distance_matrix / element_specific_sigmas
        return squared_distance_matrix

    def _calculate_log_k(self, embedding, node_attrs, time):
        squared_distance_matrix = self._calculate_distance_matrix(embedding, node_attrs)
        additional_multiplier = 119 * (1 - (time / 10) ** 0.25) + 1 if time <= 10 else 1
        squared_distance_matrix = squared_distance_matrix * additional_multiplier
        log_k = torch.logsumexp(-squared_distance_matrix / 2, dim=1)
        return log_k

    @staticmethod
    def _handle_grad_nans(grad):
        # overlapping atoms can cause nans or very large gradients
        # convert nans to zeros
        if np.isnan(grad).any() or np.isinf(grad).any():
            warnings.warn("nan or inf in grad")
            grad = np.nan_to_num(grad, nan=0, posinf=0, neginf=0)
        return grad

    def get_property(self, *args, **kwargs):
        # Prevent caching of properties
        self.reset()
        return super().get_property(*args, **kwargs)

    def _init_element_kernel_sigmas(self, element_sigma_array: np.ndarray | None):
        if element_sigma_array is not None and len(element_sigma_array) == len(
            self.z_table
        ):
            element_kernel_sigmas = torch.tensor(
                element_sigma_array, device=self.device, dtype=self.dtype
            )
        else:
            if element_sigma_array is not None:
                warnings.warn(
                    f"element_sigma_array has length {len(element_sigma_array)}, but there are {len(self.z_table)} elements in the model. Using default values of 1.0 for all elements"
                )
            element_kernel_sigmas = torch.ones(
                len(self.z_table), device=self.device, dtype=self.dtype
            )
            element_kernel_sigmas.requires_grad_(False)
        return element_kernel_sigmas

    def adjust_element_sigmas(self, new_sigma_dict: dict[str, float]) -> None:
        """Adjust the element kernel sigmas inplace.

        Parameters
        ----------
        new_sigma_dict
            dictionary mapping element name to new sigma value
        """
        for element_name, new_sigma in new_sigma_dict.items():
            z = ase.Atom(element_name).number
            idx = self.z_table.z_to_index(z)
            self.element_kernel_sigmas[idx] = new_sigma

    @staticmethod
    def _scatter_element_sigmas(
        node_attrs: torch.Tensor, element_kernel_sigmas: torch.Tensor
    ) -> torch.Tensor:
        """
        Scatters the element kernel sigmas.
        Returns
        -------
        torch.Tensor
            The tensor with scattered element kernel sigmas.
        """
        elements = node_attrs.detach().cpu().numpy()
        element_index = np.argmax(elements, axis=1)
        return element_kernel_sigmas[element_index, None]
