import numpy as np
import pytest
import torch

from moldiff.manifolds import (
    MultivariateGaussianPrior,
    StandardGaussianPrior,
)


def test_standard_gaussian_prior_calculates_correct_restorative_forces():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    forces = StandardGaussianPrior.calculate_resorative_forces(positions)
    assert np.allclose(forces, -1 * positions)


@pytest.mark.parametrize(
    "covariance_matrix, positions, expected_forces",
    [
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
            np.array([[-1.0, 1.0], [1.0, -1.0]]),
        ),
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor(np.array([[1.0, -1.0], [-1.0, 1.0]])),
            torch.tensor([[-1.0, 1.0], [1.0, -1.0]]),
        ),
        (
            np.array([[0.5, 0.0], [0.0, 2.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
            np.array([[-2.0, 0.5], [2.0, -0.5]]),
        ),
        (
            np.array([[1.0, 0.0], [0.0, 4.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
            np.array([[-2.0, 0.5], [2.0, -0.5]]),
        ),
    ],
)
def test_multivariate_gaussian_prior_calculates_correct_restorative_forces(
    covariance_matrix, positions, expected_forces
):
    prior = MultivariateGaussianPrior(covariance_matrix)
    forces = prior.calculate_resorative_forces(positions)
    assert np.allclose(forces, expected_forces)
