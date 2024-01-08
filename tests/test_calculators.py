import numpy as np
import pytest
import torch

from simgen.calculators import MaceSimilarityCalculator


@pytest.mark.parametrize(
    "inp, max_norm, expected_out",
    [
        (
            np.array([[1, 2, 3], [1, 0, 0]]),
            2,
            np.array([[0.53452248, 1.06904497, 1.60356745], [1, 0, 0]]),
        ),
        (np.array([[0, 0]]), 1, np.array([[0, 0]])),
        (
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]]),
            2,
            torch.tensor([[0.53452248, 1.06904497, 1.60356745], [1.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], device="cuda"),
            2,
            torch.tensor([[0.53452248, 1.06904497, 1.60356745], [0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_clip_grad_norm(inp, max_norm, expected_out):
    clipped = MaceSimilarityCalculator._clip_grad_norm(inp, max_norm)
    if isinstance(clipped, torch.Tensor):
        clipped = clipped.detach().cpu().numpy()
    np.testing.assert_array_almost_equal(clipped, expected_out)
