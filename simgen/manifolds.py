import warnings
from abc import ABC, abstractmethod

import ase
import numpy as np
import numpy.typing as npt
import torch
from einops import einsum, reduce
from scipy.special import softmax  # type: ignore


class PriorManifold(ABC):
    @abstractmethod
    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        raise NotImplementedError

    @abstractmethod
    def calculate_resorative_forces(
        self, positions: npt.NDArray | torch.Tensor
    ) -> npt.NDArray | torch.Tensor:
        raise NotImplementedError


class PointShape(ABC):
    precision_matrix = np.eye(3)

    @abstractmethod
    def get_n_positions(self, n: int) -> npt.NDArray[np.floating]:
        raise NotImplementedError


class StandardGaussianPrior(PriorManifold, PointShape):
    precision_matrix = np.eye(3)

    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        mol = molecule.copy()
        mol.set_positions(np.random.randn(*mol.positions.shape) * scale)
        return mol

    @staticmethod
    def calculate_resorative_forces(
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        return -1 * positions

    def get_n_positions(self, n: int) -> npt.NDArray[np.floating]:
        return np.random.randn(n, 3)


class MultivariateGaussianPrior(PriorManifold, PointShape):
    def __init__(
        self, covariance_matrix: npt.NDArray[np.floating], normalise_covariance=True
    ):
        assert (
            covariance_matrix.ndim == 2
            and covariance_matrix.shape[0] == covariance_matrix.shape[1]
        )
        self.covariance_matrix = self._check_covariance_determinant(
            covariance_matrix,
            normalise_covariance=normalise_covariance,
        )
        self.precision_matrix = np.linalg.inv(self.covariance_matrix)
        self.mean = np.zeros(self.covariance_matrix.shape[0])

    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        mol = molecule.copy()
        random_positions = (
            np.random.multivariate_normal(self.mean, self.covariance_matrix, len(mol))
            * scale
        )
        mol.set_positions(random_positions)
        return mol

    def calculate_resorative_forces(
        self,
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        precision_matrix = self.precision_matrix
        if isinstance(positions, torch.Tensor):
            dtype, device = positions.dtype, positions.device
            precision_matrix = torch.tensor(
                precision_matrix, dtype=dtype, device=device
            )
        return -1 * positions @ precision_matrix

    @staticmethod
    def _check_covariance_determinant(
        covariance_matrix: npt.NDArray[np.floating],
        normalise_covariance: bool = True,
    ):
        determinant = np.linalg.det(covariance_matrix)
        if determinant <= 0:
            raise ValueError(
                f"Covariance matrix has non-positive determinant: {determinant}"
            )
        if normalise_covariance:
            covariance_matrix /= determinant ** (1 / covariance_matrix.shape[0])
        return covariance_matrix

    def get_n_positions(self, n: int) -> npt.NDArray[np.floating]:
        return np.random.multivariate_normal(self.mean, self.covariance_matrix, (n,))


class PointCloudPrior(PriorManifold):
    def __init__(
        self,
        points: npt.NDArray[np.floating],
        beta: float = 1.0,
        point_shape: PointShape = StandardGaussianPrior(),
    ):
        """
        points: (N, 3) array of points
        beta: 1/sigma of each point in the manifold, for calculating restorative force
        """
        self.points = points
        self.beta = beta
        self.point_shape = point_shape

    def initialise_positions(self, molecule: ase.Atoms, scale: float) -> ase.Atoms:
        """
        Initialise the atom positions randomly around the point cloud
        """
        mol = molecule.copy()
        replace = len(mol) > len(self.points)
        atom_centres = np.random.choice(len(self.points), len(mol), replace=replace)
        offsets = self.point_shape.get_n_positions(len(mol))
        positions = self.points[atom_centres] + offsets * scale
        mol.set_positions(positions)
        return mol

    def calculate_resorative_forces(
        self,
        positions: np.ndarray | torch.Tensor,
    ) -> npt.ArrayLike:
        """
        Calculate the forces to move to the point cloud weighted by the softmin of the distance
        """
        points, precision_matrix = self.points, self.point_shape.precision_matrix
        if isinstance(positions, torch.Tensor):
            dtype, device = positions.dtype, positions.device
            points = torch.tensor(points, dtype=dtype, device=device)
            precision_matrix = torch.tensor(
                precision_matrix, dtype=dtype, device=device
            )

        differences = (
            positions[:, None, :] - points[None, :, :]
        )  # (n_atoms, n_points, 3)
        distances = (
            reduce(differences**2, "i j k -> i j", "sum") ** (0.5) * self.beta
        )  # (n_atoms, n_points)

        weights = self.get_weights(distances)  # (n_atoms, n_points)
        forces = einsum(
            differences, precision_matrix, "i j k, k l -> i j l"
        )  # (n_atoms, n_points, 3)
        forces = reduce(
            weights[:, :, None] * forces, "i j k -> i k", "sum"
        )  # (n_atoms, 3)
        return -forces

    @staticmethod
    def get_weights(distance_matrix):
        if isinstance(distance_matrix, torch.Tensor):
            return torch.softmax(-distance_matrix, dim=-1)
        else:
            return softmax(-distance_matrix, axis=-1)


class HeartPointCloudPrior(PointCloudPrior):
    def __init__(self, num_points=20, rescaling=5.0, beta: float = 1):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        z = np.zeros(num_points)
        points = np.stack([x, y, z], axis=1) / rescaling
        super().__init__(points, beta)


class CirclePrior(PointCloudPrior):
    def __init__(
        self,
        radius,
        num_points=20,
        beta: float = 1,
        point_shape: PointShape = StandardGaussianPrior(),
    ):
        self.radius = radius
        t = np.linspace(0, 2 * np.pi, num_points)
        x = np.cos(t) * radius
        y = np.sin(t) * radius
        z = np.zeros(num_points)
        points = np.stack([x, y, z], axis=1)
        super().__init__(points, beta, point_shape)

    @property
    def curve_length(self):
        return 2 * np.pi * self.radius
