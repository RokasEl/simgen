from collections import Counter
from dataclasses import dataclass
from io import StringIO

import ase
import ase.io as aio
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds

from .calculators import MaceSimilarityCalculator
from .generation_utils import get_edge_array_and_neighbour_numbers
from .hydrogenation import NATURAL_VALENCES
from .hydrogenation_deterministic import build_xae_molecule


@dataclass
class BaseReport:
    composition: None | str = None
    num_heavy_atoms: None | int = None
    atom_valences_possible: None | np.ndarray = None
    num_atoms_stable: None | int = None
    molecule_stable: None | bool = None
    bond_lengths: None | np.ndarray = None


@dataclass
class RDKitReport:
    smiles: None | str = None
    error: None | str = None
    descriptors: None | dict = None
    num_fragments: None | int = None
    num_rings: None | int = None
    ring_sizes: None | list = None


@dataclass
class CalculatorReport:
    similarity_energies: None | np.ndarray = None
    similarity_energy: None | float = None
    mace_energies: None | np.ndarray = None
    mace_energy: None | float = None


def get_number_of_nearest_neighbours(atoms: ase.Atoms):
    positions, atomic_symbols = atoms.get_positions(), atoms.get_chemical_symbols()
    _, _, edge_array = build_xae_molecule(
        positions,
        atomic_symbols,
        use_margins=True,
    )
    return edge_array, edge_array.sum(axis=1)


def check_valences(atoms: ase.Atoms, nearest_neigh_array):
    max_bonds_allowed = np.array(
        [
            NATURAL_VALENCES[atomic_number]
            for atomic_number in atoms.get_atomic_numbers()
        ]
    )
    valences_possible = nearest_neigh_array <= max_bonds_allowed
    num_atoms_stable = valences_possible.sum()
    molecule_stable = num_atoms_stable == len(atoms)
    return valences_possible, num_atoms_stable, molecule_stable


def get_bond_lengths(atoms: ase.Atoms, edge_array):
    atoms.pbc = False
    all_distances = atoms.get_all_distances()
    mask = np.triu(edge_array, k=1).astype(bool)
    bond_lengths = all_distances[mask].flatten()
    if len(bond_lengths) == 0:
        return None
    bond_lengths = np.round(bond_lengths, 3)
    return bond_lengths


def atoms_to_rdkit_mol(atoms: ase.Atoms):
    if np.isnan(atoms.get_positions()).any():
        raise ValueError("Atoms has NaN positions")
    holder = StringIO()
    aio.write(holder, atoms, format="xyz")
    raw_mol = Chem.MolFromXYZBlock(holder.getvalue())
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(conn_mol, charge=0)
    return conn_mol


def get_rdkit_mol(atoms: ase.Atoms):
    try:
        mol = atoms_to_rdkit_mol(atoms)
        return mol, None
    except Exception as e:
        return None, e


def analyse_rings(mol):
    ssr = Chem.GetSymmSSSR(mol)
    num_rings = len(ssr)
    if num_rings == 0:
        return num_rings, []
    else:
        ring_sizes = [len(ring) for ring in ssr]
        return num_rings, ring_sizes


def analyse_base(atoms: ase.Atoms):
    edge_array, neigbour_numbers = get_number_of_nearest_neighbours(atoms)
    valences_possible, num_atoms_stable, molecule_stable = check_valences(
        atoms, neigbour_numbers
    )
    composition = str(atoms.symbols)
    # bond lengths and num of heavy atoms
    bond_lengths = get_bond_lengths(atoms, edge_array)
    num_heavy_atoms = len([x for x in atoms.symbols if x != "H"])
    report = BaseReport(
        composition=composition,
        num_heavy_atoms=num_heavy_atoms,
        atom_valences_possible=valences_possible,
        num_atoms_stable=num_atoms_stable,
        molecule_stable=molecule_stable,
        bond_lengths=bond_lengths,
    )
    return report


def analyse_rdkit(atoms: ase.Atoms):
    mol, error = get_rdkit_mol(atoms)

    if mol is None:
        error_text = f"Failed to convert atoms to rdkit mol: {error}"
        report = RDKitReport(error=error_text)
        return report
    else:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(largest_mol))
        num_fragments = len(mol_frags)
        num_rings, ring_sizes = analyse_rings(largest_mol)
        descriptors = Descriptors.CalcMolDescriptors(largest_mol)
        report = RDKitReport(
            smiles=smiles,
            descriptors=descriptors,
            num_fragments=num_fragments,
            num_rings=num_rings,
            ring_sizes=ring_sizes,
        )
        return report


def analyse_calculator(atoms: ase.Atoms, calculator: MaceSimilarityCalculator):
    # get the log similarity of the molecule and the energy from the MACE model
    mask = np.zeros(len(atoms), dtype=bool)
    atoms.info["mask"] = mask
    atoms.info["calculation_type"] = "similarity"
    atoms.info["time"] = 0
    calculator.calculate(atoms)
    sim_energies = calculator.results["energies"]
    sim_total_energy = calculator.results["energy"]

    atoms.info["calculation_type"] = "mace"
    calculator.calculate(atoms)
    mace_energies = calculator.results["energies"]
    mace_total_energy = calculator.results["energy"]

    report = CalculatorReport(
        similarity_energies=sim_energies,
        similarity_energy=sim_total_energy,
        mace_energies=mace_energies,
        mace_energy=mace_total_energy,
    )
    return report
