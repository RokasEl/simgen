import pathlib
from dataclasses import asdict
from io import StringIO
from typing import Iterable

import ase
import ase.io as aio
import fire
import h5py
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
    save_path: str = "./results.hdf5",
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

    with h5py.File(save_path, "a") as f:
        group = f.require_group(experiment_name)
        write_dataset(group, "index", index)
        for key, value in reports.items():
            flattened_reports = [flatten_report(report) for report in value]
            df = pd.DataFrame(flattened_reports, index=index)
            report_group = group.require_group(key)
            # purge old data
            for key in report_group.keys():
                del report_group[key]
            for column in df.columns:
                if df[column].dtype == object:
                    adict = df[column].to_dict()
                    for k, v in adict.items():
                        if v is None:
                            v = np.nan
                        report_group.create_dataset(f"{column}/{k}", data=v)
                else:
                    data = np.array(df[column])
                    report_group.create_dataset(column, data=data)


def write_dataset(group, name, data):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def flatten_report(report):
    flattened_report = {}
    report_as_dict = asdict(report)
    for key, value in report_as_dict.items():
        if isinstance(value, dict):
            flattened_report.update(value)
        else:
            flattened_report[key] = value
    return flattened_report


def get_atoms_iterator(atoms_path: str, trajectory_index: int = -1) -> Iterable:
    if pathlib.Path(atoms_path).is_dir():
        atoms_iterator = read_atoms_from_directory(atoms_path, trajectory_index)
    else:
        atoms_iterator = enumerate(aio.read(atoms_path, index=":", format="xyz"))
    return atoms_iterator


def read_atoms_from_directory(atoms_path: str, trajectory_index: int = -1):
    for atoms_file in pathlib.Path(atoms_path).glob("*.xyz"):
        all_atoms = aio.read(atoms_file, index=":", format="xyz")
        atoms = all_atoms[trajectory_index]
        file_name = atoms_file.stem
        yield file_name, atoms


if __name__ == "__main__":
    fire.Fire(main)
