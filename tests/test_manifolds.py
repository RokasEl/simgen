import numpy as np
import pytest
import torch

from simgen.manifolds import (
    MultivariateGaussianPrior,
    PointCloudPrior,
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


@pytest.mark.parametrize(
    "point_cloud,beta, positions, expected_forces",
    [
        (
            np.array([[0.0, 0.0, 0.0]]),
            1.0,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
        ),
        (
            np.array([[1.0, 0.0, 0.0]]),
            1.0,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            1.0,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            1.0,
            np.array([[0.5, 0.0, 0.0]]),
            np.array([[-0.037883, 0.0, 0.0]]),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            10.0,
            np.array([[0.5, 0.0, 0.0]]),
            np.array([[0.49991, 0.0, 0.0]]),
        ),
    ],
)
def test_point_cloud_prior_calculates_correct_restorative_forces(
    point_cloud, beta, positions, expected_forces
):
    prior = PointCloudPrior(point_cloud, beta=beta)
    forces = prior.calculate_resorative_forces(positions)
    assert np.allclose(forces, expected_forces)
