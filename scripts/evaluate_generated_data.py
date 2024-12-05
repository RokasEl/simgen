import pathlib
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass

import ase
import ase.io as aio
import fire
import numpy as np
import pandas as pd
import zntrack
from rdkit import Chem

from simgen.analysis import (
    analyse_base,
    analyse_calculator,
    analyse_rdkit,
)
from simgen.utils import (
    get_mace_similarity_calculator,
    get_system_torch_device_str,
)

DEVICE = get_system_torch_device_str()


@dataclass
class Summary:
    name: str
    atom_validity: float
    atom_validity_no_h: float
    molecule_validity: float
    molecule_validity_std: float
    uniqueness: float
    uniqueness_std: float
    synthesizability: tuple[float, float, float]
    synthesizability_std: tuple[float, float, float]
    novelty: tuple[
        float, float, float
    ]  # Counting from small, medium or large reference set
    novelty_std: tuple[float, float, float]
    connected: float
    connected_std: float


def get_summary(
    name: str,
    reports: dict[str, dict[str, list]],
    ref_smile_sets: tuple[set[str]],
    resampling_seed: int = 0,
    num_resamples: int = 10,
    resample_frac: float = 0.9,
) -> Summary:
    base_reports = reports["base"]
    atom_validity = np.mean(
        [report.num_atoms_stable / report.total_num_atoms for report in base_reports]
    )
    atom_validity_no_h = np.mean(
        [
            report.num_atoms_stable_no_h / report.num_heavy_atoms
            for report in base_reports
        ]
    )
    print(f"Atom validity: {atom_validity}, Atom validity no H: {atom_validity_no_h}")
    rdkit_reports = reports["rdkit"]
    rng = np.random.default_rng(resampling_seed)

    mol_validities = np.zeros(num_resamples)
    uniquenesses = np.zeros(num_resamples)
    connected_fracs = np.zeros(num_resamples)
    novelty_fracs = np.zeros((num_resamples, len(ref_smile_sets)))
    sa_distribs = np.zeros((num_resamples, 3))

    num_to_subsample = int(len(rdkit_reports) * resample_frac)
    for i in range(num_resamples):
        indices = rng.choice(len(rdkit_reports), num_to_subsample, replace=False)
        resampled = [rdkit_reports[i] for i in indices]
        mol_validities[i] = np.mean([report.molecule_stable for report in resampled])
        num_successful_conversions = sum(
            report.smiles is not None for report in resampled
        )
        unique_smiles = {
            report.smiles for report in resampled if report.smiles is not None
        }
        uniquenesses[i] = len(unique_smiles) / num_successful_conversions
        sa_scores = [
            report.sa_score for report in resampled if report.smiles is not None
        ]
        sa_distribs[i, 0] = sum(score < 3 for score in sa_scores) / len(sa_scores)
        sa_distribs[i, 1] = sum(3 <= score < 6 for score in sa_scores) / len(sa_scores)
        sa_distribs[i, 2] = sum(6 <= score for score in sa_scores) / len(sa_scores)

        connected_fracs[i] = np.mean(
            [
                report.num_fragments == 1
                for report in resampled
                if report.smiles is not None
            ]
        )
        novelty = np.array(
            [
                1 - len(unique_smiles.intersection(ref_set)) / len(unique_smiles)
                for ref_set in ref_smile_sets
            ]
        )
        novelty_fracs[i] = novelty

    mol_validity = np.mean(mol_validities)
    mol_std = np.std(mol_validities)
    uniqueness = np.mean(uniquenesses)
    uniqueness_std = np.std(uniquenesses)
    synthesizability = tuple(np.mean(sa_distribs, axis=0))
    synthesizability_std = tuple(np.std(sa_distribs, axis=0))
    novelty = tuple(np.mean(novelty_fracs, axis=0))
    novelty_std = tuple(np.std(novelty_fracs, axis=0))
    connected_fracs = np.mean(connected_fracs)
    connected_fracs_std = np.std(connected_fracs)
    summary = Summary(
        name=name,
        atom_validity=atom_validity,
        atom_validity_no_h=atom_validity_no_h,
        molecule_validity=mol_validity,
        molecule_validity_std=mol_std,
        uniqueness=uniqueness,
        uniqueness_std=uniqueness_std,
        synthesizability=synthesizability,
        synthesizability_std=synthesizability_std,
        novelty=novelty,
        novelty_std=novelty_std,
        connected=connected_fracs,
        connected_std=connected_fracs_std,
    )
    return summary


def _convert_xyz_file_to_smi(xyz_path: pathlib.Path):
    out = subprocess.run(
        ["obabel", "-ixyz", xyz_path, "-osmi", xyz_path.with_suffix(".smi"), "-xn"],
        capture_output=True,
    )
    if out.returncode != 0:
        print(out.stderr)
        return None
    else:
        smiles = out.stdout.decode().strip().split("\n")
        return smiles


def convert_to_smiles(
    xyz_path: pathlib.Path | None = None, atoms_list: list[ase.Atoms] | None = None
):
    if xyz_path is None and atoms_list is None:
        raise ValueError("Either xyz_path or atoms_list must be provided")
    if atoms_list is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            xyz_path = pathlib.Path(tmp_dir) / "tmp.xyz"
            aio.write(xyz_path, atoms_list, format="extxyz")
            smiles = _convert_xyz_file_to_smi(xyz_path)
        return smiles
    else:
        assert xyz_path is not None, "xyz_path must be provided if `atoms_list` is None"
        smiles = _convert_xyz_file_to_smi(xyz_path)
        return smiles


def get_atoms_iterator(atoms_dir: str) -> dict[str, Iterable]:
    path = pathlib.Path(atoms_dir)
    out_dict = {}
    if not path.is_dir():
        atoms = aio.read(atoms_dir, index=":", format="extxyz")
        smiles_list = convert_to_smiles(atoms_list=atoms)
        if smiles_list is None:
            smiles_list = [None for _ in range(len(atoms))]
        assert len(atoms) == len(smiles_list)
        out_dict[path.stem] = zip(atoms, smiles_list, strict=False)
        return out_dict
    for xyz_file in path.glob("*.xyz"):
        atoms = aio.read(xyz_file, index=":", format="extxyz")
        print(f"Converting {xyz_file}")
        smiles_list = convert_to_smiles(xyz_file)
        if smiles_list is None:
            smiles_list = [None for _ in range(len(atoms))]
        assert len(atoms) == len(smiles_list)
        out_dict[xyz_file.stem] = zip(atoms, smiles_list, strict=False)
    return out_dict


def save_reports(reports: dict, save_dir: str | pathlib.Path, file_name: str):
    for key, value in reports.items():
        df = pd.DataFrame(value)
        df.to_json(f"{save_dir}/{file_name}_{key}.json")


def main(
    atoms_dir: str,
    save_path: str = "./results/",
    do_calculator_analysis=False,
    model_repo_path: str = "https://github.com/RokasEl/MACE-Models",
    remove_halogenides: bool = False,
    num_reference_mols: int = -1,
    num_to_sample_uniformly_per_size: int = 0,
):
    ref_data_names = ("small", "medium", "large")
    ref_data = {
        name: zntrack.from_rev(
            f"simgen_reference_data_{name}", remote=model_repo_path
        ).get_atoms()
        for name in ref_data_names
    }
    ref_data_smiles = {
        name: convert_to_smiles(atoms_list=atoms_list)
        for name, atoms_list in ref_data.items()
    }
    ref_data_smiles = {
        name: [Chem.CanonSmiles(smi) for smi in smiles]
        for name, smiles in ref_data_smiles.items()
    }
    ref_smile_sets = tuple(set(ref_data_smiles[name]) for name in ref_data_names)

    file_dict = get_atoms_iterator(atoms_dir)

    if do_calculator_analysis:
        rng = np.random.default_rng(0)
        calc = get_mace_similarity_calculator(
            model_repo_path=model_repo_path,
            remove_halogenides=remove_halogenides,
            num_reference_mols=num_reference_mols,
            num_to_sample_uniformly_per_size=num_to_sample_uniformly_per_size,
            device=DEVICE,
            rng=rng,
        )
    else:
        calc = None

    all_reports = {}
    summary_reports = []
    for file_name, objs in file_dict.items():
        all_reports[file_name] = {"base": [], "rdkit": [], "calc": []}
        current_reports = all_reports[file_name]
        print(file_name)
        for atoms, smiles in objs:
            current_reports["base"].append(analyse_base(atoms))
            if smiles is not None:
                current_reports["rdkit"].append(analyse_rdkit(smiles))
            if do_calculator_analysis and calc is not None:
                current_reports["calc"].append(analyse_calculator(atoms, calc))
        result_summary = get_summary(file_name, current_reports, ref_smile_sets)
        summary_reports.append(result_summary)

    save_dir = pathlib.Path(save_path)
    print(f"Saving to {save_path}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for file_name, reports in all_reports.items():
        save_reports(reports, save_dir, file_name)

    df = pd.DataFrame(summary_reports)
    df.to_json(f"{save_dir}/summary.json")


if __name__ == "__main__":
    fire.Fire(main)
