import numpy as np
import pytest
from ase import Atoms

from simgen.particle_filtering import ParticleFilterGenerator
from simgen.utils import initialize_mol


def test_merge_scaffold_and_create_mask():
    generated_atoms = initialize_mol("C3")
    scaffold = initialize_mol("C6H6")
    merged, mask, _ = ParticleFilterGenerator._merge_scaffold_and_create_mask(
        generated_atoms, scaffold, 1
    )
    expected_merged_positions = np.concatenate(
        [generated_atoms.positions, scaffold.positions], axis=0
    )
    expected_atomic_numbers = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(merged.positions, expected_merged_positions)
    np.testing.assert_array_almost_equal(merged.numbers, expected_atomic_numbers)
    expected_mask = np.zeros(len(merged))
    expected_mask[:3] = 1
    np.testing.assert_array_equal(expected_mask, mask)
