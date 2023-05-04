from abc import ABC, abstractmethod

import ase
import numpy as np
import numpy.typing as npt
import torch


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
    def initialise_positions(self, molecule: ase.Atoms) -> ase.Atoms:
        mol = molecule.copy()
        mol.set_positions(np.random.randn(*mol.positions.shape))
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

    def initialise_positions(self, molecule: ase.Atoms) -> ase.Atoms:
        mol = molecule.copy()
        random_positions = np.random.multivariate_normal(
            self.mean, self.covariance, len(mol)
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
