import numpy as np
import torch
from e3nn import o3
from moldiff.particle_filtering import ParticleFilterGenerator

torch.set_default_dtype(torch.float64)
from ase import Atoms
from mace.data.atomic_data import AtomicData

from moldiff.utils import initialize_mol, read_qm9_xyz, setup_logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import logging

import ase.io as ase_io
import mace
from mace.modules.blocks import (
    RadialDistanceTransformBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE, ScaleShiftMACE
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.diffusion_tools import EDMSampler, SamplerNoiseParameters
from moldiff.sampling import MaceSimilarityScore


def main():
    setup_logger(level=logging.DEBUG, tag="particle_filter", directory="./logs")
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    print(pretrained_model)
    model = ScaleShiftMACE(
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
        atomic_inter_scale=1.088502,
        atomic_inter_shift=0.0,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.radial_embedding = RadialDistanceTransformBlock(
        r_min=0.7, **dict(r_max=4.5, num_bessel=8, num_polynomial_cutoff=5)
    )
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    rng = np.random.default_rng(0)
    data_path = "../data/qm9/"
    training_data = [
        read_qm9_xyz(f"{data_path}/dsgdb9nsd_{i:06d}.xyz")
        for i in rng.choice(133885, 256, replace=False)
    ]

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    score_model = MaceSimilarityCalculator(
        model, reference_data=training_data, device=DEVICE
    )
    energies = np.empty(len(training_data))
    for i, mol in enumerate(training_data):
        mol.calc = score_model
        mol.info["time"] = 1e-2
        energies[i] = mol.get_potential_energy()
    logging.debug(f"Energies of training data: {energies}")
    noise_params = SamplerNoiseParameters(
        sigma_max=10, sigma_min=2e-3, S_churn=80, S_min=2e-3, S_noise=1
    )
    particle_filter = ParticleFilterGenerator(
        score_model, num_steps=150, noise_params=noise_params
    )
    mol = initialize_mol("C30")
    trajectories = particle_filter.generate(mol, num_particles=10)
    print(len(trajectories))
    ase_io.write(
        "particle_filter_generation_CHONF_6.xyz",
        trajectories,
        format="extxyz",
        append=True,
    )


if __name__ == "__main__":
    main()
