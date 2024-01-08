import ase
import numpy as np
import pytest
import torch
from mace.tools import AtomicNumberTable

from simgen.atoms_cleanup import (
    get_higest_energy_unswapped_idx,
    get_swapping_candidates,
    relax_elements,
    relax_hydrogens,
    remove_isolated_atoms_fixed_cutoff,
    remove_isolated_atoms_using_covalent_radii,
)
from simgen.generation_utils import batch_atoms
from simgen.utils import get_system_torch_device_str, initialize_mol

from .fixtures import (
    loaded_mace_similarity_calculator,
    loaded_model,
    test_molecules,
    training_molecules,
)

z_table = AtomicNumberTable([1, 6, 7, 8, 9])


@pytest.fixture()
def linear_molecule_with_increasingly_isolated_atoms():
    mol = initialize_mol("C10")
    x_positions = [2**i - 1 for i in range(10)]
    positions = np.array([x_positions, [0] * 10, [0] * 10]).T
    mol.set_positions(positions)
    return mol


@pytest.mark.parametrize(
    "atoms, cutoff, expected",
    [
        (initialize_mol("H2O"), 10.0, initialize_mol("H2O")),
        (initialize_mol("H2O"), 0.1, initialize_mol("")),
    ],
)
def test_remove_isolated_atoms(atoms, cutoff, expected):
    pruned_atoms = remove_isolated_atoms_fixed_cutoff(atoms, cutoff)
    assert pruned_atoms == expected


def test_remove_isolated_atoms_with_molecule_with_increasingly_isolated_atoms(
    linear_molecule_with_increasingly_isolated_atoms,
):
    mol = linear_molecule_with_increasingly_isolated_atoms
    cutoffs = [2**i for i in range(10)]
    expected_remaining_atoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    for cutoff, expected_remaining_atom in zip(cutoffs, expected_remaining_atoms):
        pruned_atoms = remove_isolated_atoms_fixed_cutoff(mol, cutoff)
        assert len(pruned_atoms) == expected_remaining_atom

    pruned_atoms = remove_isolated_atoms_fixed_cutoff(mol, 0.1)
    assert len(pruned_atoms) == 0


stretched_CH = ase.Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.1]])
stretched_CC = ase.Atoms("CC", positions=[[0, 0, 0], [0, 0, 1.1]])


@pytest.mark.parametrize(
    "atoms, expected",
    [
        (initialize_mol("H2O"), initialize_mol("H2O")),
        (stretched_CH, initialize_mol("")),  # too far apart
        (stretched_CC, stretched_CC),  # not too far apart for a CC bond
    ],
)
def test_remove_isolated_atoms_covalent_cutoff(atoms, expected):
    atoms_copy = atoms.copy()
    pruned_atoms = remove_isolated_atoms_using_covalent_radii(atoms, multiplier=1.0)
    assert atoms_copy == atoms  # check that the original atoms are not modified
    assert pruned_atoms == expected


def test_get_highest_energy_unswapped_idx():
    swapped_indices = []
    energies = np.array([1, 20, -3, 4, 5]).astype(float)
    expected_indices = (1, 4, 3, 0, 2)
    expected_swapped_indices = ([1], [1, 4], [1, 4, 3], [1, 4, 3, 0], [1, 4, 3, 0, 2])

    for expected_idx, expected_swapped_idx in zip(
        expected_indices, expected_swapped_indices
    ):
        idx = get_higest_energy_unswapped_idx(swapped_indices, energies)
        assert idx == expected_idx
        swapped_indices.append(idx)
        assert swapped_indices == expected_swapped_idx

    # raise a value error if all indices are swapped
    swapped_indices = [1, 4, 3, 0, 2]
    with pytest.raises(ValueError):
        get_higest_energy_unswapped_idx(swapped_indices, energies)


def test_relax_hydrogens_keeps_positions_of_heavy_elements_unchanged(
    loaded_mace_similarity_calculator,
):
    mols = [initialize_mol("H2O"), initialize_mol("CH4"), initialize_mol("C2H6")]
    mol_with_bad_geometry = ase.Atoms(
        "HCN", positions=np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    )
    mols.append(mol_with_bad_geometry)
    for mol in mols:
        mol.info["calculation_type"] = "mace"
    original_mols = [mol.copy() for mol in mols]
    relaxed_mols = relax_hydrogens(mols, loaded_mace_similarity_calculator)
    for mol, relaxed_mol in zip(original_mols, relaxed_mols):
        non_h_indices = np.where(mol.get_atomic_numbers() != 1)[0]
        np.testing.assert_allclose(
            mol.get_positions()[non_h_indices],
            relaxed_mol.get_positions()[non_h_indices],
        )


def test_get_swapping_candidates():
    mol = initialize_mol("C2H6")
    idx = 0
    neighbours = np.array([1, 2, 3, 4])
    already_switched = [2, 3, 4]
    z_table = AtomicNumberTable([6, 7, 8])
    ensemble, swapped_indices = get_swapping_candidates(
        mol, idx, neighbours, already_switched, z_table
    )
    expected_ensemble = []
    for elem in [7, 8]:
        m = mol.copy()
        m.set_atomic_numbers([elem, 6, 1, 1, 1, 1, 1, 1])
        expected_ensemble.append(m)
    for elem in [7, 8]:
        m = mol.copy()
        m.set_atomic_numbers([6, elem, 1, 1, 1, 1, 1, 1])
        expected_ensemble.append(m)
    expected_swapped_indices = [0, 0, 1, 1]
    assert ensemble == expected_ensemble
    assert swapped_indices == expected_swapped_indices


@pytest.fixture()
def element_swapping_test_suite():
    test_suite = []
    # add a few mols that are already correct
    for mol_name in ["H2O", "CH4", "C2H6", "C2H4", "CH3COCH3"]:
        mol = initialize_mol(mol_name)
        test_suite.append((mol.copy(), mol.copy()))

    # add a few mols that are incorrect and that have one obvious element to swap
    mol = initialize_mol("C2H6")
    mol.set_atomic_numbers([6, 8, 1, 1, 1, 1, 1, 1])
    test_suite.append((mol.copy(), initialize_mol("C2H6")))

    mol = initialize_mol("CH4")
    mol.set_atomic_numbers([7, 1, 1, 1, 1])
    test_suite.append((mol.copy(), initialize_mol("CH4")))

    mol = initialize_mol("CO2")
    mol.set_atomic_numbers([6, 6, 8])
    test_suite.append((mol.copy(), initialize_mol("CO2")))
    return test_suite


def test_relax_elements(loaded_mace_similarity_calculator, element_swapping_test_suite):
    for mol, expected_mol in element_swapping_test_suite:
        mol.calc = loaded_mace_similarity_calculator
        relaxed_mol = relax_elements(mol, z_table=z_table)
        assert relaxed_mol == expected_mol


def test_calculate_mace_interaction_energies_and_forces_gives_same_node_energies_and_forces(
    loaded_mace_similarity_calculator, test_molecules
):
    pretrained_mace = "./models/SPICE_sm_inv_neut_E0.model"
    pretrained_model = torch.load(pretrained_mace)
    device = get_system_torch_device_str()
    pretrained_model.to(device)
    z_table = z_table = AtomicNumberTable(
        [int(z) for z in pretrained_model.atomic_numbers]
    )
    r_max = pretrained_model.r_max.item()

    batched_mols = batch_atoms(test_molecules, z_table, r_max, device)
    out = pretrained_model(batched_mols.to_dict())
    node_energies = out["node_energy"].detach().cpu().numpy()
    node_e0s = (
        pretrained_model.atomic_energies_fn(batched_mols.node_attrs)
        .detach()
        .cpu()
        .numpy()
    )
    mace_interaction_energies = node_energies - node_e0s
    mace_forces = out["forces"].detach().cpu().numpy()
    mace_mol_energies = out["interaction_energy"].detach().cpu().numpy()
    calculator_interaction_energies = []
    calculator_forces = []
    mol_energies = []
    for mol in test_molecules:
        (
            energies,
            forces,
            mol_energy,
        ) = loaded_mace_similarity_calculator._calculate_mace_interaction_energies_and_forces(
            mol
        )
        calculator_interaction_energies.append(energies)
        calculator_forces.append(forces)
        mol_energies.append(mol_energy)
    calculator_interaction_energies = np.concatenate(
        calculator_interaction_energies
    ).flatten()
    calculator_forces = np.concatenate(calculator_forces, axis=0)
    calculator_mol_energies = np.concatenate(mol_energies).flatten()
    np.testing.assert_allclose(
        mace_interaction_energies, calculator_interaction_energies, atol=1e-5
    )
    np.testing.assert_allclose(mace_forces, calculator_forces, atol=1e-5)
    np.testing.assert_allclose(mace_mol_energies, calculator_mol_energies, atol=1e-5)
