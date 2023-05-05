from abc import ABC, abstractmethod

import ase
import numpy as np
import numpy.typing as npt
import torch
from scipy.special import softmax


class PriorManifold(ABC):
    @abstractmethod
    def initialise_positions(self, molecule: ase.Atoms) -> ase.Atoms:
        raise NotImplementedError

    @abstractmethod
    def calculate_resorative_forces(
        self, positions: npt.NDArray | torch.Tensor
    ) -> npt.NDArray | torch.Tensor:
        raise NotImplementedError


class StandardGaussianPrior(PriorManifold):
    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        mol = molecule.copy()
        mol.set_positions(np.random.randn(*mol.positions.shape) * scale)
        return mol

    @staticmethod
    def calculate_resorative_forces(
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        return -1 * positions


class MultivariateGaussianPrior(PriorManifold):
    def __init__(self, covariance_matrix: npt.NDArray[np.float64]):
        assert (
            covariance_matrix.ndim == 2
            and covariance_matrix.shape[0] == covariance_matrix.shape[1]
        )
        self.covariance = self._ensure_covariance_determinant_is_one(covariance_matrix)
        self.precision_matrix = np.linalg.inv(self.covariance)
        self.mean = np.zeros(self.covariance.shape[0])

    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        mol = molecule.copy()
        random_positions = (
            np.random.multivariate_normal(self.mean, self.covariance, len(mol)) * scale
        )
        mol.set_positions(random_positions)
        return mol

    def calculate_resorative_forces(
        self,
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        precision_matrix = self.precision_matrix
        if isinstance(positions, torch.Tensor):
            precision_matrix = torch.from_numpy(precision_matrix).to(positions.device)
        return -1 * positions @ precision_matrix

    @staticmethod
    def _ensure_covariance_determinant_is_one(
        covariance_matrix: npt.NDArray[np.float64],
    ):
        determinant = np.linalg.det(covariance_matrix)
        if determinant <= 0:
            raise ValueError(
                f"Covariance matrix has non-positive determinant: {determinant}"
            )
        covariance_matrix /= determinant ** (1 / covariance_matrix.shape[0])
        return covariance_matrix


class PointCloudPrior(PriorManifold):
    def __init__(self, points: npt.NDArray[np.float64], beta: float = 1.0):
        """
        points: (N, 3) array of points
        beta: 1/sigma of each point in the manifold, for calculating restorative force
        """
        self.points = points
        self.beta = beta

    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        """
        Treat the point cload as a mixture of gaussians, and sample from it
        """
        mol = molecule.copy()
        atom_centres = np.random.randint(0, len(self.points), len(mol))
        positions = (
            self.points[atom_centres] + np.random.randn(*mol.positions.shape) * scale
        )
        mol.set_positions(positions)
        return mol

    def calculate_resorative_forces(
        self,
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        """
        Calculate the forces to move to the point cloud weighted by the softmin of the distance
        """
        if isinstance(positions, torch.Tensor):
            pos = positions.detach().cpu().numpy()
        else:
            pos = positions
        distances = (
            np.linalg.norm(pos[:, None, :] - self.points[None, :, :], axis=-1)
            * self.beta
        )
        weights = softmax(-distances, axis=-1)
        forces = np.sum(
            weights[:, :, None] * (pos[:, None, :] - self.points[None, :, :]),
            axis=1,
        )
        if isinstance(positions, torch.Tensor):
            forces = torch.from_numpy(forces).to(positions.device)
        return -forces


class HeartPointCloudPrior(PointCloudPrior):
    def __init__(self, num_points=20, rescaling=5.0, beta: float = 1):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        z = np.zeros(num_points)
        points = np.stack([x, y, z], axis=1) / rescaling
        super().__init__(points, beta)
