from dataclasses import dataclass

import ase
import numpy as np
from rdkit import Chem

from .calculators import MaceSimilarityCalculator
from .hydrogenation import NATURAL_VALENCES
from .hydrogenation_deterministic import build_xae_molecule
from .sascrorer import calculateScore


@dataclass
class BaseReport:
    composition: None | str = None
    total_num_atoms: None | int = None
    num_heavy_atoms: None | int = None
    atom_valences_possible: None | np.ndarray = None
    num_atoms_stable: None | int = None
    num_atoms_stable_no_h: None | int = None
    molecule_stable: None | bool = None
    bond_lengths: None | np.ndarray = None


@dataclass
class RDKitReport:
    smiles: None | str = None
    error: None | str = None
    atom_is_radical: None | list[bool] = None
    num_atoms_radical: None | int = None
    num_heavy_atoms_stable: None | int = None
    molecule_stable: None | bool = None
    sa_score: None | float = None
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


def get_number_of_nearest_neighbours(atoms: ase.Atoms, strict=False):
    positions, atomic_symbols = atoms.get_positions(), atoms.get_chemical_symbols()
    _, _, edge_array = build_xae_molecule(
        positions,
        atomic_symbols,
        use_margins=True,
    )
    if strict:
        num_neighs = edge_array.sum(axis=1)
    else:
        adj_mat = edge_array != 0
        num_neighs = adj_mat.sum(axis=1)
    return edge_array, num_neighs


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
    atoms_no_h = atoms.copy()
    numbers = atoms.numbers
    atoms_no_h = atoms_no_h[numbers != 1]
    edge_array_no_h, neigbour_numbers_no_h = get_number_of_nearest_neighbours(
        atoms_no_h
    )
    valences_possible_no_h, num_atoms_stable_no_h, molecule_stable_no_h = (
        check_valences(atoms_no_h, neigbour_numbers_no_h)
    )
    composition = str(atoms.symbols)
    # bond lengths and num of heavy atoms
    bond_lengths = get_bond_lengths(atoms, edge_array)
    num_heavy_atoms = len([x for x in atoms.symbols if x != "H"])
    report = BaseReport(
        composition=composition,
        num_heavy_atoms=num_heavy_atoms,
        total_num_atoms=len(atoms),
        atom_valences_possible=valences_possible,
        num_atoms_stable=num_atoms_stable,
        num_atoms_stable_no_h=num_atoms_stable_no_h,
        molecule_stable=molecule_stable,
        bond_lengths=bond_lengths,
    )
    return report


def check_for_unrecognized_bonds(atom):
    connected_atoms = atom.GetNeighbors()
    neighbor_radicals = [
        neighbor.GetNumRadicalElectrons() for neighbor in connected_atoms
    ]
    if sum(neighbor_radicals) == 1:
        return True
    else:
        return False


def check_mol_is_radical(mol) -> tuple[list[bool], int]:
    is_radical = []
    num_heavy_atoms_stable = 0
    for atom in mol.GetAtoms():
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical == 0:
            is_radical.append(False)
            num_heavy_atoms_stable += 1 if atom.GetAtomicNum() != 1 else 0
        elif num_radical == 1:
            # Check if there's a connected atom with a radical, i.e., an unrecognized bond
            _stable = check_for_unrecognized_bonds(atom)
            is_radical.append(not _stable)
            num_heavy_atoms_stable += 1 if (atom.GetAtomicNum() != 1 and _stable) else 0
        else:
            is_radical.append(True)
    return is_radical, num_heavy_atoms_stable


def analyse_rdkit(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        error_text = f"Failed to build mol from smiles: {smiles}"
        report = RDKitReport(error=error_text, molecule_stable=False)
        return report
    else:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        sa_score = calculateScore(largest_mol)
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(largest_mol))
        num_fragments = len(mol_frags)
        num_rings, ring_sizes = analyse_rings(largest_mol)
        # descriptors = Descriptors.CalcMolDescriptors(largest_mol)
        atom_is_radical, num_heavy_atoms_stable = check_mol_is_radical(largest_mol)
        num_atoms_radical = sum(atom_is_radical)
        molecule_stable = num_atoms_radical == 0
        report = RDKitReport(
            smiles=smiles,
            # descriptors=descriptors,
            atom_is_radical=atom_is_radical,
            num_atoms_radical=num_atoms_radical,
            sa_score=sa_score,
            num_heavy_atoms_stable=num_heavy_atoms_stable,
            molecule_stable=molecule_stable,
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
