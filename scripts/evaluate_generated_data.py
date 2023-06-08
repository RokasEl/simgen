import pathlib
from io import StringIO

import ase
import ase.io as aio
import numpy as np
import pandas as pd
import torch

from moldiff.analysis import (
    analyse_base,
    analyse_calculator,
    analyse_rdkit,
)
from moldiff.utils import get_mace_similarity_calculator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atoms_directory", help="Path to directory containing atoms files"
    )
    args = parser.parse_args()
    # use pathlib to get all atoms files
    rng = np.random.default_rng(0)
    calc = get_mace_similarity_calculator(
        model_path="/home/rokas/Programming/Generative_model_energy/models/SPICE_sm_inv_neut_E0_swa.model",
        reference_data_path="/home/rokas/Programming/data/spice_subset.xyz",
        remove_halogenides=False,
        num_reference_mols=32,
        num_to_sample_uniformly_per_size=0,
        device=DEVICE,
        rng=rng,
    )
    index = []
    base_reports = []
    rdkit_reports = []
    calc_reports = []
    for atoms_file in pathlib.Path(args.atoms_directory).glob("*.xyz"):
        print(atoms_file)
        atoms = aio.read(atoms_file, index=":", format="xyz")
        index.append(atoms_file.stem)
        base_reports.append(analyse_base(atoms[-5]))
        rdkit_reports.append(analyse_rdkit(atoms[-1]))
        calc_reports.append(analyse_calculator(atoms[-1], calc))

    base_df = pd.DataFrame(base_reports, index=index)
    rdkit_df = pd.DataFrame(rdkit_reports, index=index)
    calc_df = pd.DataFrame(calc_reports, index=index)
    base_df.to_csv("base.csv")
    rdkit_df.to_csv("rdkit.csv")
    calc_df.to_csv("calc_with_h_reference.csv")
