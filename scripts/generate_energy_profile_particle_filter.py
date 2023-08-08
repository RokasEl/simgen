import numpy as np
import torch
from e3nn import o3

torch.set_default_dtype(torch.float64)
from ase import Atoms
from mace.data.atomic_data import AtomicData

from moldiff.generation_utils import RadialDistanceTransformBlock
from moldiff.utils import (
    get_system_torch_device_str,
    initialize_mol,
    read_qm9_xyz,
    setup_logger,
)

DEVICE = get_system_torch_device_str()

import logging

import ase.io as ase_io
import matplotlib.pyplot as plt
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator


def main():
    setup_logger(level=logging.DEBUG, tag="particle_filter", directory="./logs")
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    model = MACE(
        r_max=4.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=0,
        num_elements=10,
        atomic_energies=np.zeros(10),
        avg_num_neighbors=12,
        correlation=3,
        interaction_cls_first=RealAgnosticInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        hidden_irreps=o3.Irreps("96x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        gate=torch.nn.functional.silu,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.radial_embedding = RadialDistanceTransformBlock(
        r_min=0.5, **dict(r_max=4.5, num_bessel=8, num_polynomial_cutoff=5)
    )
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    rng = np.random.default_rng(0)
    data_path = "./Data/qm9_data/"
    training_data = [
        read_qm9_xyz(f"{data_path}/dsgdb9nsd_{i:06d}.xyz")
        for i in rng.choice(133885, 128, replace=False)
    ]

    score_model = MaceSimilarityCalculator(
        model, reference_data=training_data, device=DEVICE
    )
    all_atoms = ase_io.read(
        "./particle_filter_generation_CHONF_2.xyz", format="extxyz", index=":"
    )
    # select the first, the last, and every 10th in between
    atoms = [all_atoms[0]] + all_atoms[1:-1:100] + [all_atoms[-1]]
    sigmas = torch.concatenate(
        [
            torch.linspace(10, 2.5, 20),
            torch.linspace(2.5, 1, 80),
            torch.linspace(0.95, 0.05, 400),
            torch.logspace(-1.31, -3, 20),
        ]
    )
    # select the first, the last, and every 10th in between
    sigmas = torch.concatenate([sigmas[0:1], sigmas[1:-1:10], sigmas[-1:]])
    print(len(atoms))
    print(len(sigmas))
    assert len(atoms) == len(sigmas)
    energies = np.empty(len(atoms))
    for i, mol in enumerate(atoms):
        mol.calc = score_model
        mol.info["time"] = sigmas[i].item()
        print(sigmas[i].item())
        energies[i] = mol.get_potential_energy()
    print(energies)
    plt.plot(range(len(energies)), energies)
    plt.xticks(list(reversed(sigmas.numpy())))
    plt.ylim(100, 300)
    plt.xlim(0, 3)
    plt.show()


if __name__ == "__main__":
    main()
