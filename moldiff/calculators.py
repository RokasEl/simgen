"""ASE calculator comparing SOAP similarity"""
import logging
import warnings
from typing import List

import ase
import einops
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_mean, scatter_sum
from quippy.descriptors import Descriptor
from tqdm import tqdm


# Writen by Tamas Stenczel
class SoapSimilarityCalculator(Calculator):
    """

    Notes
    -----
    Constraints:
    - single element

    """

    implemented_properties = ["energy", "forces", "energies"]

    def __init__(
        self,
        descriptor: Descriptor,
        ref_soap_vectors: np.ndarray,
        weights=None,
        zeta=1,
        scale=1.0,
        *,
        restart=None,
        label=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        """

        Parameters
        ----------
        descriptor
            descriptor calculator object
        ref_soap_vectors
            reference SOAP vectors [n_ref, len_soap]
        weights
            of reference SOAP vectors, equal weight used if not given
        zeta
            exponent of kernel
        scale
            scaling for energy & forces, energy of calculator is
            `-1 * scale * k_soap ^ zeta` where 0 < k_soap < 1
        """
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )

        self.descriptor = descriptor
        self.zeta = zeta
        self.ref_soap_vectors = ref_soap_vectors
        self.scale = scale

        if weights is None:
            self.weights = (
                1 / len(ref_soap_vectors) * np.ones(ref_soap_vectors.shape[0])
            )
        else:
            assert len(weights) == weights.shape[0]
            self.weights = weights

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        # Descriptor calculation w/ gradients
        d_move = self.descriptor.calc(atoms, grad=True)
        print(d_move["data"].shape)
        # -1 * similarity -> 'Energy'
        # k_ab = einops.einsum(
        #     d_move["data"], self.ref_soap_vectors, "a desc, b desc -> a b"
        # ) # very slow with NumPy
        k_ab = d_move["data"] @ self.ref_soap_vectors.T  # [len(atoms), num_ref]
        print(k_ab.shape)
        local_similarity = self.scale * einops.einsum(
            self.weights, k_ab**self.zeta, "b, a b -> a"
        )  # this one is OK, speed about the same as np.dot()
        self.results["energies"] = -1 * local_similarity
        print(local_similarity)
        similarity = np.sum(local_similarity)
        self.results["energy"] = -1 * similarity

        # grad(similarity) -> forces
        # n.b. no -1 since energy is -1 * similarity
        a_cross = d_move["grad_index_0based"]  # [n_cross, 2]
        a_grad_data = d_move["grad_data"]  # [n_cross, 3, len_desc]

        forces = np.zeros(shape=(len(atoms), 3))  # type: ignore
        if self.zeta == 1:
            for i_grad, (_, ii) in enumerate(a_cross):
                # forces[ii] += einops.einsum(
                #     self.weights,
                #     a_grad_data[i_grad],
                #     self.ref_soap_vectors,
                #     "bi, cart desc, bi desc -> cart",
                # )
                forces[ii] += np.sum(
                    a_grad_data[i_grad] @ self.ref_soap_vectors.T * self.weights,
                    axis=1,
                )

        else:
            # chain rule - uses k_ab from outside, with z-1 power
            k_ab_zeta = k_ab ** (self.zeta - 1)
            for i_grad, (ci, ii) in enumerate(a_cross):
                # forces[ii] += einops.einsum(
                #     self.weights,
                #     k_ab[ci] ** (self.zeta - 1),
                #     a_grad_data[i_grad],
                #     self.ref_soap_vectors,
                #     "bi, bi, cart desc, bi desc -> cart",
                # ) # this is VERY slow, but easy to understand
                forces[ii] += np.sum(
                    a_grad_data[i_grad]
                    @ self.ref_soap_vectors.T
                    * (self.weights * k_ab_zeta[ci]),
                    axis=1,
                )
            forces *= self.zeta

        self.results["forces"] = self.scale * forces


class MaceSimilarityCalculator(Calculator):
    implemented_properties = ["energy", "forces", "energies"]

    def __init__(
        self,
        model: MACE,
        reference_data: List[ase.Atoms],
        device="cpu",
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
        ref_soap_vectors
            reference SOAP vectors [n_ref, len_soap]
        scale
            scaling for energy & forces, energy of calculator is
            `-1 * scale * k_soap ^ zeta` where 0 < k_soap < 1
        """
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )
        self.model = model
        self.device = device
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.reference_embeddings = self._calculate_reference_embeddings(reference_data)
        self.typical_length_scale = self._calculate_mean_dot_product(
            self.reference_embeddings
        )
        E0s = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
        self.E0s_dict: dict[int, float] = {z: E0 for z, E0 in zip(self.z_table.zs, E0s)}

    def __call__(self, atomic_data, t):
        batch_index = atomic_data.batch
        emb = self._get_node_embeddings(atomic_data)
        log_dens = self._calculate_log_k(emb, t)
        log_dens = scatter_sum(log_dens, batch_index)
        grad = self._get_gradient(atomic_data.positions, log_dens)
        grad = self._clip_grad_norm(grad, max_norm=np.sqrt(3))
        if t < 0.1:
            t = t.item() * 1 / 0.1
            out = self.model(atomic_data)
            forces = out["forces"].detach().cpu().numpy()
            forces = self._clip_grad_norm(forces, max_norm=10)
            grad = t * grad + (1 - t) * forces
        return grad

    def calculate(
        self,
        atoms: None | ase.Atoms = None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object must be provided")

        atomic_data = self.convert_to_atomic_data(atoms)
        batched = self._batch_atomic_data(atomic_data)
        embedding = self._get_node_embeddings(batched)
        try:
            time = atoms.info["time"]
        except KeyError:
            raise KeyError("Atoms object must have a time attribute")

        log_k = self._calculate_log_k(embedding, time)
        node_energies = -1 * log_k
        molecule_energies = scatter_sum(node_energies, batched.batch, dim=0)
        self.results["energies"] = node_energies.detach().cpu().numpy()
        self.results["energy"] = molecule_energies.detach().cpu().numpy()
        logging.debug(f"Node similarity energies: {self.results['energies']}")
        force = self._get_gradient(batched.positions, log_k)
        force = self._handle_grad_nans(force)
        self.results["forces"] = force

        if time < 0.01:
            time = time * 1 / 0.01
            out = self.model(batched.to_dict())
            node_energies = out["node_energy"].detach().cpu().numpy()
            shifted_energies = self.subtract_reference_energies(atoms, node_energies)
            shifted_energies = shifted_energies
            logging.debug(f"Node pretrained MACE energies: {shifted_energies}")
            logging.debug(f"MACE multiplier: {1-time}, similarity multiplier: {time}")
            shifted_energies = shifted_energies - shifted_energies[0]
            self.results["energies"] = (
                self.results["energies"] - self.results["energies"][0]
            )
            self.results["energies"] = (
                1 - time
            ) * shifted_energies + time * self.results["energies"]
            self.results["energy"] = scatter_sum(
                torch.tensor(self.results["energies"]).to(self.device),
                batched.batch,
                dim=0,
            )

    def subtract_reference_energies(self, atoms, energies: np.ndarray):
        zs = atoms.get_atomic_numbers()
        reference_energies = np.array([self.E0s_dict[z] for z in zs])
        return energies - reference_energies

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
        return grad.detach().cpu().numpy()

    @staticmethod
    def _clip_grad_norm(grad, max_norm=1):
        norm = np.linalg.norm(grad, axis=1)
        mask = norm > max_norm
        grad[mask] = grad[mask] / norm[mask, None] * max_norm
        return grad

    @staticmethod
    def _calculate_mean_dot_product(x):
        dot_products = einops.reduce(x**2, "n m -> n", "sum")
        mean_dot_product = einops.reduce(dot_products, "n -> ()", "mean")
        return mean_dot_product

    def _get_node_embeddings(self, data: AtomicData):
        # Embeddings
        node_feats = self.model.get_node_invariant_descriptors(
            data, track_gradient_on_positions=True
        )  # type: ignore
        node_feats = node_feats[:, :1, :]
        node_feats = einops.rearrange(
            node_feats,
            "num_nodes interactions embed_dim -> num_nodes (interactions embed_dim)",
        )
        return node_feats

    def _calculate_reference_embeddings(
        self, training_data: List[ase.Atoms]
    ) -> torch.Tensor:
        as_atomic_data = self.convert_to_atomic_data(training_data)
        dloader = get_data_loader(as_atomic_data, batch_size=128, shuffle=False)
        with torch.no_grad():
            node_embeddings = [
                self._get_node_embeddings(data) for data in tqdm(dloader)
            ]
        return torch.concatenate(node_embeddings, dim=0)

    def _calculate_distance_matrix(self, embedding):
        embedding_deltas = embedding[:, None, :] - self.reference_embeddings[None, :, :]
        # embedding_deltas (embedding_nodes, reference_nodes, embed_dim)
        embedding_deltas = embedding_deltas**2
        squared_distance_matrix = torch.sum(
            embedding_deltas, dim=-1
        )  # (embedding_nodes, reference_nodes)
        squared_distance_matrix = squared_distance_matrix / self.typical_length_scale
        return squared_distance_matrix

    def _calculate_log_k(self, embedding, time):
        squared_distance_matrix = self._calculate_distance_matrix(embedding)
        # squared_distance_matrix = squared_distance_matrix / 1.5 ** (time)
        additional_multiplier = (
            119  # * (1 - (time / 10) ** 0.25) + 1 if time <= 10 else 1
        )
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
