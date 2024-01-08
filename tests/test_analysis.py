from collections import Counter

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from simgen.analysis import (
    BaseReport,
    RDKitReport,
    analyse_base,
    analyse_rdkit,
)
from simgen.utils import initialize_mol


def test_analyse_base():
    atoms = initialize_mol("C")
    report = analyse_base(atoms)
    expected_report = BaseReport(
        composition="C",
        num_heavy_atoms=1,
        atom_valences_possible=np.array([True]),
        num_atoms_stable=1,
        molecule_stable=True,
        bond_lengths=None,
    )
    assert report == expected_report

    atoms = initialize_mol("C6H6")
    report = analyse_base(atoms)
    expected_report = BaseReport(
        composition="C6H6",
        num_heavy_atoms=6,
        atom_valences_possible=None,
        num_atoms_stable=12,
        molecule_stable=True,
        bond_lengths=None,
    )
    expected_bond_lengths = np.array([1.395, 1.087] * 6)
    expected_atom_valences_possible = np.array([True] * 12)
    actual_bond_lengths = np.array(report.bond_lengths)
    actual_atom_valences_possible = np.array(report.atom_valences_possible)
    report.bond_lengths = None
    report.atom_valences_possible = None
    assert Counter(expected_bond_lengths) == Counter(actual_bond_lengths)
    assert (expected_atom_valences_possible == actual_atom_valences_possible).all()
    assert report == expected_report


def test_analyse_rdkit():
    atoms = initialize_mol("CH4")
    report = analyse_rdkit(atoms)
    expected_report = RDKitReport(
        smiles="C",
        error=None,
        descriptors=Descriptors.CalcMolDescriptors(Chem.AddHs(Chem.MolFromSmiles("C"))),
        num_fragments=1,
        num_rings=0,
        ring_sizes=[],
    )
    assert report == expected_report

    atoms = initialize_mol("C6H6")
    small_frag = initialize_mol("H2O")
    small_frag.positions += 10
    atoms = atoms + small_frag
    report = analyse_rdkit(atoms)
    expected_report = RDKitReport(
        smiles="c1ccccc1",
        error=None,
        descriptors=Descriptors.CalcMolDescriptors(
            Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        ),
        num_fragments=2,
        num_rings=1,
        ring_sizes=[6],
    )
    assert report == expected_report

    atoms = initialize_mol("C19")
    report = analyse_rdkit(atoms)
    expected_report = RDKitReport(
        error="Failed to convert atoms to rdkit mol: Valence of atom 0 is 18, which is larger than the allowed maximum, 4",
    )
    assert report == expected_report
