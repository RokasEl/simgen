import numpy as np
import pytest
import torch

from moldiff.sampling import ScoreModel


@pytest.mark.parametrize(
    "training_embeddings, expected_scale",
    [
        (
            np.array([[1, 1], [1, 1]], dtype=np.float64),
            np.array([1, 1], dtype=np.float64) * np.sqrt(2),
        ),
        (
            np.array([[1, 1], [0, -1]], dtype=np.float64),
            np.array([0.5, 1], dtype=np.float64) * np.sqrt(2),
        ),
        (
            torch.tensor([[1, 1, 5], [0, -1, 10]], dtype=torch.float64),
            torch.tensor([0.5, 1, 62.5], dtype=torch.float64) * np.sqrt(3),
        ),
    ],
)
def test_calculate_squared_embedding_scale(training_embeddings, expected_scale):
    scales = ScoreModel._calculate_reference_dot_prod(training_embeddings)
    np.testing.assert_array_equal(scales, expected_scale)


def test_calculate_squared_embedding_scale_raises_warning_for_zero_scale():
    with pytest.warns(UserWarning):
        scale = ScoreModel._calculate_reference_dot_prod(
            np.array([[0, 0], [0, 0]], dtype=np.float64)
        )
        np.testing.assert_array_equal(scale, np.array([1e-16, 1e-16]))
