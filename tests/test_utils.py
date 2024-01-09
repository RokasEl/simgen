import numpy as np
import pytest
import torch
from e3nn import o3
from mace.modules import interaction_classes

from simgen.utils import get_mace_config

from .fixtures import mace_model, training_molecules, z_table


def test_get_mace_config(mace_model, z_table):
    atomic_numbers = z_table.zs
    config = get_mace_config(mace_model)
    expected_config = dict(
        r_max=4.0,
        num_bessel=3,
        num_polynomial_cutoff=3,
        max_ell=3,
        interaction_cls=interaction_classes["RealAgnosticInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=1,
        num_elements=3,
        hidden_irreps=o3.Irreps("16x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.array([-10, -1000, -2000], dtype=float),
        avg_num_neighbors=8,
        atomic_numbers=atomic_numbers,
        correlation=1,
        gate=None,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    for key in config:
        comparison = config[key] == expected_config[key]
        if isinstance(comparison, np.ndarray):
            np.testing.assert_allclose(config[key], expected_config[key])
        else:
            assert comparison, f"{key} {config[key]} != {expected_config[key]}"


def test_get_mace_config_of_remote_model():
    import zntrack

    model_loader = zntrack.from_rev(
        "ani500k_small", remote="/home/rokas/Programming/MACE-Models", rev="main"
    )
    pretrained_model = model_loader.get_model()

    config = get_mace_config(pretrained_model)

    expected_config = dict(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=2,
        num_elements=4,
        atomic_energies=np.array(
            [-13.62222754, -1029.41308397, -1484.87103581, -2041.83962771], dtype=float
        ),
        avg_num_neighbors=12.673066139221191,
        correlation=3,
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        hidden_irreps=o3.Irreps("64x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8],
        gate=torch.nn.functional.silu,
        atomic_inter_scale=2.2886035442352295,
        atomic_inter_shift=0.0,
    )

    for key in config:
        comparison = config[key] == expected_config[key]
        if isinstance(comparison, np.ndarray):
            np.testing.assert_allclose(config[key], expected_config[key])
        else:
            assert comparison, f"{key} {config[key]} != {expected_config[key]}"
