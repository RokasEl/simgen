import pathlib
from dataclasses import asdict

import ase.io as aio
import pandas as pd
from rdkit import Chem

from simgen.analysis import RDKitReport, analyse_base, get_rdkit_mol


def main(size_sweep_path="./results/energy_model/sweep_sampler_params/"):
    reports = []
    for atoms_file in pathlib.Path(size_sweep_path).glob("*.xyz"):
        print(atoms_file.stem)
        all_atoms = aio.read(atoms_file, index=":", format="xyz")
        for atoms in all_atoms:
            base_report = analyse_base(atoms)
            if base_report.bond_lengths is None:
                error_text = "No bonds found in molecule"
                report = RDKitReport(error=error_text)
                report_dict = asdict(report)
                report_dict["group"] = atoms_file.stem
                report_dict["smiles_exists"] = report_dict["smiles"] is not None
                reports.append(report_dict)
                continue
            mol, error = get_rdkit_mol(atoms)
            if mol is None:
                error_text = f"Failed to convert atoms to rdkit mol: {error}"
                report = RDKitReport(error=error_text)
            else:
                report = RDKitReport(smiles=Chem.MolToSmiles(mol))
            report_dict = asdict(report)
            report_dict["group"] = atoms_file.stem
            report_dict["smiles_exists"] = report_dict["smiles"] is not None
            reports.append(report_dict)

    df = pd.DataFrame(reports)
    df.to_json(f"{size_sweep_path}/mols_parsed_as_smiles.json")
    groups = df.groupby("group")
    for name, group in groups:
        print(f"Group {name}, Succesful: {group['smiles_exists'].sum()}")
    max_avg_group = df.groupby("group")["smiles_exists"].sum().idxmax()
    print(max_avg_group)
    print(groups.get_group(max_avg_group))


if __name__ == "__main__":
    main()
