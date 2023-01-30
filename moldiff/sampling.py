import abc
import random
import warnings
from typing import List

import einops
import numpy as np
import torch
from ase import Atoms
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import torch_geometric
from quippy.descriptors import Descriptor

#################### Base Classes ####################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, t) -> np.ndarray:
        pass

    @staticmethod
    def _normalise_score(score):
        # score (n_nodes, 3)
        expected_norm = np.sqrt(3)
        actual_norm = np.linalg.norm(score, axis=1) + 1e-20
        return score / actual_norm[:, None] * expected_norm


class Sampler(abc.ABC):
    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass


class NoiseScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, t) -> tuple:
        pass


#################### Schedulers ####################


class ConstantScheduler(NoiseScheduler):
    def __init__(self, value=1, num_steps=100):
        self.value = value
        self.num_steps = num_steps

    def __call__(self, idx):
        return idx / self.num_steps, self.value


class ArrayScheduler(NoiseScheduler):
    def __init__(self, values, num_steps: int = 1000):
        self.values = values
        self.num_steps = num_steps

    def _get_idx(self, idx):
        return np.interp(idx, [0, self.num_steps], [0, len(self.values) - 1]).astype(
            int
        )

    def __call__(self, idx):
        interpolated_idx = self._get_idx(idx)
        scaled_time = idx / self.num_steps
        return scaled_time, self.values[interpolated_idx]

    def get_values_using_scaled_time(self, scaled_time):
        interpolated_idx = self._get_idx(scaled_time * self.num_steps)
        return self.values[interpolated_idx]


#################### Score models ####################


class ScoreModelContainer(ScoreModel):
    def __init__(self, score_models: list, scheduler: ArrayScheduler):
        self.score_models = score_models
        self.scheduler = scheduler

    def __call__(self, X, scaled_time):
        score_strengths = self.scheduler.get_values_using_scaled_time(scaled_time)
        score = 0
        for score_mode, score_strength in zip(self.score_models, score_strengths):
            if score_strength == 0:
                pass
            else:
                score += score_strength * score_mode(X, scaled_time)
        return score


class GaussianScoreModel(ScoreModel):
    def __init__(self, spring_constant: float = 1.0):
        self.spring_constant = spring_constant

    def __call__(self, X, t):
        return -X.positions * self.spring_constant


class ASECalculatorScoreModel(ScoreModel):
    def __init__(self, calculator):
        self.calculator = calculator

    def __call__(self, X, t):
        X.set_calculator(self.calculator)
        return X.get_forces()


class SOAPSimilarityModel(ScoreModel):
    def __init__(
        self,
        training_data: List[Atoms],
        soap_features="soap l_max=6 n_max=12 cutoff=5.0 atom_sigma=0.5",
    ):
        self.descriptor_calculator = Descriptor(soap_features)
        self.training_data = training_data
        self.reference_embeddings = self._get_reference_embeddings(training_data)
        to_corrupt = random.sample(training_data, 3)
        for atoms in to_corrupt:
            atoms.set_positions(1 * np.random.randn(*atoms.positions.shape))
        self.corrupted_ref_embeddings = self._get_reference_embeddings(to_corrupt)
        self.temperature_scale = self._calibrate_temperature_scale()

    def __call__(self, X, t, normalise_grad=True):
        descriptor_data = self.descriptor_calculator.calc(X, grad=True)
        gradient = self._calculate_gradients(descriptor_data, t)
        if normalise_grad:
            gradient = self._normalise_score(gradient)
        return gradient

    def _calculate_gradients(self, descriptor_data, t):
        # deltas shape (embed_dim, num_old_data, num_new_data)
        # squared_distance_matrix shape (num_old_data, num_new_data)
        embedding = descriptor_data["data"]
        deltas, squared_distance_matrix = self._calculate_distance_matrix(embedding)
        temp = self.temperature_scale(t)
        inverse_t = 1 / temp
        exponents = np.exp(
            -squared_distance_matrix * inverse_t
        )  # shape (num_old_data, num_new_data)
        embedding_gradients = descriptor_data["grad_data"]
        # need to use the gradient index to sum over the flattned first dimension
        gradient_index = descriptor_data["grad_index_0based"]
        embedding_gradients = self._sum_over_split_gradients(
            embedding_gradients, gradient_index
        )
        numerator = einops.einsum(
            deltas,
            embedding_gradients,
            "embed_dim num_old_data num_new_data, num_new_data space embed_dim -> num_old_data num_new_data space",
        )
        numerator = einops.einsum(
            exponents,
            numerator,
            "num_old_data num_new_data, num_old_data num_new_data space -> num_new_data space",
        )
        denum = exponents.sum(axis=0)  # (num_new_data, )
        grad = numerator / denum[:, None]  # (num_new_data, 3)
        return grad

    @staticmethod
    def _sum_over_split_gradients(gradient_array, index_array):
        num_atoms = index_array.max() + 1
        summed_array = np.zeros((num_atoms, *gradient_array.shape[1:]))
        np.add.at(summed_array, index_array[:, 0], gradient_array)
        return summed_array

    def _get_reference_embeddings(self, training_data):
        reference_embeddings = []
        for atoms in training_data:
            reference_embeddings.append(self.descriptor_calculator.calc(atoms)["data"])
        return np.concatenate(reference_embeddings)

    def _calculate_distance_matrix(self, embedding):
        embedding_deltas = einops.rearrange(
            embedding, "num_new_data embed_dim  -> embed_dim () num_new_data"
        ) - einops.rearrange(
            self.reference_embeddings,
            "num_old_data embed_dim  -> embed_dim num_old_data () ",
        )  # (embed_dim, num_old_data, num_new_data)

        squared_distance_matrix = np.sum(embedding_deltas**2, axis=0)
        return embedding_deltas, squared_distance_matrix

    def _calibrate_temperature_scale(self):
        rand_idx = np.random.randint(self.reference_embeddings.shape[0], size=10)
        sample_reference_data = self.reference_embeddings[rand_idx]
        distances_in_reference_data = self._calculate_distance_matrix(
            sample_reference_data
        )[1].flatten()
        min_quantile = 1e-3
        lower_bound = np.quantile(distances_in_reference_data, min_quantile) + 1e-6
        distances_in_reference_data = None
        distances_to_corrupted_data = self._calculate_distance_matrix(
            self.corrupted_ref_embeddings
        )[1].flatten()
        upper_bound = np.quantile(distances_to_corrupted_data, 1 - min_quantile)
        # Set the temperature scale so that at t=1 environments away by `upper_bound` are rescaled to be distance 1 away
        # And at t=0 environments away by `lower_bound` are rescaled to be distance 1 away
        lower_temp, upper_temp = np.log(lower_bound), np.log(upper_bound)
        sigmoid = lambda t: 1 / (1 + np.exp(-1 * (t - 0.5)))
        return lambda t: np.exp(lower_temp + (upper_temp - lower_temp) * sigmoid(t))


class MaceSimilarityScore(ScoreModel):
    def __init__(
        self,
        mace_model,
        z_table,
        training_data: List,
        device: str = "cuda",
    ):
        self.model = mace_model
        self.z_table = z_table
        self.device = device
        self.training_data = training_data

    def __call__(self, atoms, t, normalise_grad=True):
        atomic_data = self._to_atomic_data(atoms)
        emb = self._get_node_embeddings(atomic_data)
        reference_atoms = random.sample(self.training_data, 64)
        corrupted_atoms = self._corrupt_atoms(reference_atoms, t)
        self.reference_embeddings = self._get_reference_embeddings(corrupted_atoms)
        log_dens = self._get_log_kernel_density(emb, t)
        grad = self._get_gradient(atomic_data, log_dens)
        grad = self._handle_grad_nans(grad)
        if normalise_grad:
            grad = self._normalise_score(grad)
        return grad

    def _corrupt_atoms(self, atoms_list, t):
        corrupted = [x.copy() for x in atoms_list]
        for mol in corrupted:
            mol.set_positions(
                mol.get_positions() * np.sqrt(1 - t)
                + np.random.normal(0, t, mol.get_positions().shape)
            )
        return corrupted

    def _to_atomic_data(self, atoms):
        conf = config_from_atoms(atoms)
        atomic_data = AtomicData.from_config(
            conf, z_table=self.z_table, cutoff=self.model.r_max
        ).to(self.device)
        atomic_data.positions.requires_grad = True
        return atomic_data

    @staticmethod
    def _get_gradient(inp_atoms: AtomicData, log_dens: torch.Tensor):
        grad = torch.autograd.grad(
            outputs=log_dens,
            inputs=inp_atoms.positions,
            grad_outputs=torch.ones_like(log_dens),
            create_graph=True,
            only_inputs=True,
        )[0]
        return grad.detach().cpu().numpy()

    @staticmethod
    def _handle_grad_nans(grad):
        # overlapping atoms can cause nans or very large gradients
        # convert nans to zeros
        if np.isnan(grad).any() or np.isinf(grad).any():
            warnings.warn("nan or inf in grad")
            grad = np.nan_to_num(grad, nan=0, posinf=0, neginf=0)
        return grad

    def _get_node_embeddings(self, data: AtomicData):
        # Embeddings
        node_feats = self.model.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats = self.model.radial_embedding(lengths)
        node_feats_all = []
        for interaction, product, _ in zip(
            self.model.interactions, self.model.products, self.model.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data.node_attrs
            )
            node_feats_all.append(node_feats)
        if len(node_feats_all) == 1:
            return node_feats_all[0]
        else:
            return torch.concatenate(node_feats_all, dim=-1)

    def _get_reference_embeddings(self, training_data):
        configs = [config_from_atoms(atoms) for atoms in training_data]
        # get some corrupted configs to calibrate the variance
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.model.r_max
                )
                for config in configs
            ],  # type:ignore
            batch_size=128,
            shuffle=False,
            drop_last=False,
        )
        ref_embeddings = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                ref_embeddings.append(self._get_node_embeddings(data))
        ref_embeddings = torch.concatenate(ref_embeddings, dim=0)
        return ref_embeddings

    def _calculate_distance_matrix(self, embedding):
        embedding_deltas = einops.rearrange(
            embedding, "num_new_data embed_dim  -> embed_dim () num_new_data"
        ) - einops.rearrange(
            self.reference_embeddings,
            "num_old_data embed_dim  -> embed_dim num_old_data () ",
        )  # (embed_dim, num_old_data, num_new_data)

        squared_distance_matrix = torch.sum(embedding_deltas**2, dim=0)
        squared_distance_matrix[squared_distance_matrix <= 1e-12] = 0
        return squared_distance_matrix

    def _get_log_kernel_density(self, embedding, t):

        squared_distances = self._calculate_distance_matrix(
            embedding
        )  # (num_old_data, num_new_data)
        density = torch.exp(-squared_distances / (2)).mean(dim=0)  # (num_new_data)
        log_density = torch.log(density)
        return log_density


#################### Samplers ####################


class VarriancePreservingBackwardEulerSampler(Sampler):
    def __init__(self, score_model: ScoreModel):
        self.score_model = score_model

    def step(self, X, t, beta: float):
        score = self.score_model(X, t)
        X.positions = (2 - np.sqrt(1 - beta)) * X.positions + beta * score
        noise = np.random.normal(size=X.positions.shape)
        X.positions += np.sqrt(beta) * noise
        return X


class LangevinSampler(Sampler):
    def __init__(
        self,
        score_model: ScoreModel,
        drag_coefficient: float = 0,
        signal_to_noise_ratio=0.1,
        temperature: float = 1,
        adjust_step_size=True,
        eta=1e-3,
    ):
        self.score_model = score_model

        self.drag_coefficient = drag_coefficient
        if drag_coefficient == 0:
            self.drag = self._drag_zero
        else:
            self.drag = self._drag_strength
        self.adjust_step_size = adjust_step_size
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.eta = eta
        self.temperature = temperature

    def step(self, X, t, step_size, X_prev=None):
        score = self.score_model(X, t)
        X.arrays["forces"] = score
        noise = np.random.normal(size=X.positions.shape)
        step_size_modifier = (
            1
            if not self.adjust_step_size
            else self.get_dynamic_step_size_modifier(score, noise, eta=self.eta)
        )
        # step_size_modifier = np.clip(step_size_modifier, 1e-6, 1e6)
        step_size *= step_size_modifier
        force_term = step_size * score
        drag_term = step_size * self.drag_coefficient * self.drag(X, X_prev)
        brownian_term = np.sqrt(2 * step_size * self.temperature) * noise
        X.positions += force_term + drag_term + brownian_term
        return X

    @staticmethod
    def _drag_zero(*args):
        return 0

    @staticmethod
    def _drag_strength(x, x_prev):
        return x_prev.positions - x.positions

    def get_dynamic_step_size_modifier(
        self, score: np.ndarray, noise: np.ndarray, eta: float = 1e-3
    ) -> float:
        """ "
        Following the predictor-corrector algorithm suggested in https://github.com/yang-song/score_sde
        """
        score_norm = np.linalg.norm(score, axis=1).mean()
        noise_norm = np.linalg.norm(noise, axis=1).mean()
        step_size_modifier = (
            self.signal_to_noise_ratio * noise_norm / (score_norm + eta)
        ) ** 2
        return step_size_modifier
