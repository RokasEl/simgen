import pathlib
from io import StringIO

import ase
import ase.io as aio
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from xyz2mol import read_xyz_file, xyz2mol

from moldiff.hydrogenation import NATURAL_VALENCES
from moldiff.hydrogenation_deterministic import build_xae_molecule


def evaluate_atoms_with_no_hs_stability(atoms: ase.Atoms):
    positions, atomic_symbols = atoms.get_positions(), atoms.get_chemical_symbols()
    _, _, edge_array = build_xae_molecule(
        positions,
        atomic_symbols,
        use_margins=True,
    )
    max_bonds = np.array(
        [
            NATURAL_VALENCES[atomic_number]
            for atomic_number in atoms.get_atomic_numbers()
        ]
    )
    num_bonds = edge_array.sum(axis=1)
    num_atoms_stable = (num_bonds <= max_bonds).sum()
    molecule_stable = num_atoms_stable == len(atoms)
    return num_atoms_stable, molecule_stable


def try_build_molecule_using_rdkit(atoms: ase.Atoms):
    holder = StringIO()
    aio.write(holder, atoms, format="xyz")
    try:
        raw_mol = Chem.MolFromXYZBlock(holder.getvalue())
        conn_mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(conn_mol, charge=0)
    except Exception as e:
        print(e)
        return False
    if conn_mol is None:
        return False
    else:
        print(Chem.MolToSmiles(conn_mol))
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atoms_directory", help="Path to directory containing atoms files"
    )
    args = parser.parse_args()
    # use pathlib to get all atoms files
    total_atoms, total_stable_atoms, total_molecules, total_stable_molecules = (
        0,
        0,
        0,
        0,
    )
    rdkit_stable_molecules = 0
    for atoms_file in pathlib.Path(args.atoms_directory).glob("*.xyz"):
        print(atoms_file)
        try:
            trajectory = aio.read(atoms_file, index=":")
        except:
            continue
        no_h_atoms = trajectory[-6]
        with_h_atoms = trajectory[-1]
        num_atoms_stable, molecule_stable = evaluate_atoms_with_no_hs_stability(
            no_h_atoms
        )
        total_atoms += len(no_h_atoms)
        total_stable_atoms += num_atoms_stable
        total_stable_molecules += molecule_stable
        rdkit_stable_molecules += try_build_molecule_using_rdkit(with_h_atoms)
        total_molecules += 1

    print(
        f"Total atoms: {total_atoms}, total stable atoms: {total_stable_atoms}, i.e. {total_stable_atoms/total_atoms*100:.2f}%"
    )
    print(
        f"Total molecules: {total_molecules}, total stable molecules: {total_stable_molecules}, i.e. {total_stable_molecules/total_molecules*100:.2f}%"
    )
    print(
        f"Total molecules that rdkit could build: {rdkit_stable_molecules}, i.e. {rdkit_stable_molecules/total_molecules*100:.2f}%"
    )
