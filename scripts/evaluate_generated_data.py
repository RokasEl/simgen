import pathlib
from typing import Iterable

import ase
import ase.io as aio
import fire
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


def main(
    atoms_path,
    experiment_name: str,
    save_path: str = "./results/",
    trajectory_index: int = -1,
    do_calculator_analysis=False,
    model_path: str | None = None,
    reference_data_path: str | None = None,
    remove_halogenides: bool = False,
    num_reference_mols: int = 32,
    num_to_sample_uniformly_per_size: int = 0,
):
    atoms_iterator = get_atoms_iterator(atoms_path, trajectory_index=trajectory_index)

    if do_calculator_analysis:
        if model_path is None or reference_data_path is None:
            raise ValueError(
                "model_path and reference_data_path must be specified for calculator analysis"
            )
        rng = np.random.default_rng(0)
        calc = get_mace_similarity_calculator(
            model_path=model_path,
            reference_data_path=reference_data_path,
            remove_halogenides=remove_halogenides,
            num_reference_mols=num_reference_mols,
            num_to_sample_uniformly_per_size=num_to_sample_uniformly_per_size,
            device=DEVICE,
            rng=rng,
        )
    else:
        calc = None
    index = []
    reports = {"base": [], "rdkit": [], "calc": []}
    for name, atoms in atoms_iterator:
        print(name, atoms)
        index.append(name)
        reports["base"].append(analyse_base(atoms))
        reports["rdkit"].append(analyse_rdkit(atoms))
        if do_calculator_analysis and calc is not None:
            reports["calc"].append(analyse_calculator(atoms, calc))
    save_path = pathlib.Path(save_path)
    print(f"Saving to {save_path}")
    save_path.mkdir(exist_ok=True, parents=True)
    for key, value in reports.items():
        df = pd.DataFrame(value, index=index)
        df.to_json(f"{save_path}/{experiment_name}_{key}.json")


def get_atoms_iterator(atoms_path: str, trajectory_index: int = -1) -> Iterable:
    if pathlib.Path(atoms_path).is_dir():
        atoms_iterator = read_atoms_from_directory(atoms_path, trajectory_index)
    else:
        atoms_iterator = enumerate(aio.read(atoms_path, index=":", format="xyz"))
    return atoms_iterator


def read_atoms_from_directory(atoms_path: str, trajectory_index: int = -1):
    names, read_atoms = [], []
    for atoms_file in pathlib.Path(atoms_path).glob("*.xyz"):
        all_atoms = aio.read(atoms_file, index=":", format="xyz")
        atoms = all_atoms[trajectory_index]
        file_name = atoms_file.stem
        names.append(file_name)
        read_atoms.append(atoms)
    return zip(names, read_atoms)


if __name__ == "__main__":
    fire.Fire(main)
