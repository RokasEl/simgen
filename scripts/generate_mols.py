import argparse
import os

import ase
import mace.tools
import numpy as np
import torch

from moldiff.sampling import (
    ArrayScheduler,
    GaussianScoreModel,
    LangevinSampler,
    MaceSimilarityScore,
    ScoreModelContainer,
    SOAPSimilarityModel,
    VarriancePreservingBackwardEulerSampler,
)
from moldiff.utils import initialize_mol, read_qm9_xyz

mace.tools.set_default_dtype("float64")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./trained_models/MACE_3bpa_run-123.model"
    )
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="./Data/qm9_data/")
    parser.add_argument("--save_path", type=str, default="./trajectories/test.xyz")
    return parser.parse_args()


def main(model_path, num_steps, data_path, save_path):
    # Initialize score models
    model = torch.load(model_path)
    model.to("cuda")
    model.eval()

    rng = np.random.default_rng(0)
    # select 1000 random molecules from the qm9 dataset
    training_data = [
        read_qm9_xyz(f"{data_path}/dsgdb9nsd_{i:06d}.xyz")
        for i in rng.choice(133885, 512, replace=False)
    ]
    training_data = list(
        filter(lambda atoms: "F" not in atoms.get_chemical_formula(), training_data)
    )
    z_table = mace.tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    score_model = MaceSimilarityScore(
        model, z_table, training_data=training_data, device=DEVICE
    )
    # score_model = SOAPSimilarityModel(training_data=training_data)
    num_steps = num_steps

    # kernel_strength = np.sin(np.pi * np.linspace(0, 1, 1000)) ** 2
    kernel_strength = np.ones(1000)
    restorative_strength = np.linspace(0, 1, 1000)
    restorative_scorer = GaussianScoreModel(spring_constant=0.5)
    model_strengths = np.stack([kernel_strength, restorative_strength], axis=1)
    score_model_scheduler = ArrayScheduler(model_strengths, num_steps=num_steps)
    scorer = ScoreModelContainer(
        [score_model, restorative_scorer], score_model_scheduler
    )

    # Initialize samplers
    predictor = VarriancePreservingBackwardEulerSampler(score_model=scorer)
    corrector = LangevinSampler(
        score_model=scorer,
        signal_to_noise_ratio=0.1,
        temperature=0.01,
        adjust_step_size=False,
    )
    noise_scheduler = ArrayScheduler(np.linspace(1e-4, 1e-2, 1000), num_steps=num_steps)

    # Generation
    mol = initialize_mol("C6H6")
    destination = save_path
    if os.path.exists(destination):
        os.remove(destination)
    mol.set_positions(1 * np.random.randn(*mol.positions.shape))
    ase.io.write(destination, mol, append=True)
    for step_num in reversed(range(num_steps - 1)):
        scaled_time, beta = noise_scheduler(step_num + 1)
        mol = predictor.step(mol, scaled_time, beta)
        ase.io.write(destination, mol, append=True)
        scaled_time, beta = noise_scheduler(step_num)
        previous_state = mol.copy()
        for _ in range(5):  # 10 steps of MD
            mol = corrector.step(mol, scaled_time, beta, X_prev=previous_state)
            ase.io.write(destination, mol, append=True)


if __name__ == "__main__":
    args = init_args()
    main(args.model_path, args.num_steps, args.data_path, args.save_path)
