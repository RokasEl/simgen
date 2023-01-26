import abc
from typing import List

import einops
import numpy as np
import torch
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import torch_geometric

#################### Base Classes ####################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, t) -> np.ndarray:
        pass


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
    def __init__(self, spring_constant=1):
        self.spring_constant = spring_constant

    def __call__(self, X, t):
        return -X.positions * self.spring_constant


class ASECalculatorScoreModel(ScoreModel):
    def __init__(self, calculator):
        self.calculator = calculator

    def __call__(self, X, t):
        X.set_calculator(self.calculator)
        return X.get_forces()


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
        self.reference_embeddings = self._get_reference_embeddings(training_data)

    def __call__(self, atoms, t):
        atomic_data = self._to_atomic_data(atoms)
        emb = self._get_node_embeddings(atomic_data)
        log_dens = self._get_log_kernel_density(emb, t)
        grad = self._get_gradient(atomic_data, log_dens)
        if np.isnan(grad).any():
            print("nan in grad")
            grad = np.nan_to_num(grad, nan=0)
            return grad
        # limit the norm of the gradient to 1e4
        grad_norm = np.linalg.norm(grad, axis=1)
        limit_to_grad_ratio = np.clip(1e4 / grad_norm, 0, 1)[:, None]
        grad = grad * limit_to_grad_ratio
        return grad

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
        return torch.concatenate(ref_embeddings, dim=0)

    def _calculate_distance_matrix(self, embedding):
        embedding_deltas = einops.rearrange(
            embedding, "num_new_data embed_dim  -> embed_dim () num_new_data"
        ) - einops.rearrange(
            self.reference_embeddings,
            "num_old_data embed_dim  -> embed_dim num_old_data () ",
        )  # (embed_dim, num_old_data, num_new_data)

        squared_distance_matrix = torch.sum(embedding_deltas**2, dim=0)

        return squared_distance_matrix

    def _get_log_kernel_density(self, embedding, t):

        squared_distances = self._calculate_distance_matrix(
            embedding
        )  # (num_old_data, num_new_data)
        # dynamically set the length scale based on closest 10 neighbors
        # sorted_distance_kernel, _ = torch.sort(squared_distances, dim=0)
        # pick 5 closest neighbors
        # sorted_distance_kernel = sorted_distance_kernel[:5, :]  # (5, num_new_data)
        # variance = sorted_distance_kernel[:10, :].mean(dim=0) # (num_new_data)
        variance = 1e-5
        density = torch.exp(-squared_distances / (2 * variance)).mean(
            dim=0
        )  # (num_new_data)
        log_density = torch.log(density)
        log_density = torch.nan_to_num(log_density, posinf=1000, neginf=-1000)
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
        noise = np.random.normal(size=X.positions.shape)
        step_size_modifier = (
            1
            if not self.adjust_step_size
            else self.get_dynamic_step_size_modifier(score, noise, eta=self.eta)
        )
        # step_size_modifier = np.clip(step_size_modifier, 1e-6, 1e6)
        # step_size *= step_size_modifier
        force_term = step_size * score * step_size_modifier
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
        )
        return step_size_modifier
