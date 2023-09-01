import numpy as np
from ase.build import molecule
from ase.calculators.mixing import LinearCombinationCalculator

from airss_baseline.calculators import (
    MopacCalculator,
    RestorativeCalculator,
)


def test_mopac_calculator():
    atoms = molecule("H2O")
    calc = MopacCalculator()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert energy == -54.09724 * 0.0433641153087705
    expected_forces = -1 * np.array(
        [[0.0, 0.0, 12.482147], [0.0, 2.931553, -6.228044], [0.0, -2.931553, -6.254103]]
    )
    np.testing.assert_allclose(forces, expected_forces * 0.0433641153087705)

    atoms = molecule("CH4")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy == -12.25217 * 0.0433641153087705
    expected_forces = -1 * np.array(
        [
            [0.000004, 0.000004, 0.000004],
            [2.010724, 2.010724, 2.010724],
            [-2.027094, -2.027094, 2.041648],
            [2.042554, -2.027094, -2.026188],
            [-2.026188, 2.043460, -2.026188],
        ]
    )
    np.testing.assert_allclose(forces, expected_forces * 0.0433641153087705)


def test_combining_mopac_and_restorative_calc():
    mopac_calc = MopacCalculator()
    restorative_calc = RestorativeCalculator(prior_manifold=None)
    atoms = molecule("H2O")
    atoms.calc = mopac_calc
    forces_mopac = atoms.get_forces()
    atoms.calc = restorative_calc
    forces_restorative = atoms.get_forces()
    expected_forces = forces_mopac + forces_restorative

    combined_calc = LinearCombinationCalculator([mopac_calc, restorative_calc], [1, 1])
    atoms.calc = combined_calc
    actual_forces = atoms.get_forces()
    np.testing.assert_allclose(actual_forces, expected_forces)

    # Test changing the weights
    combined_calc = LinearCombinationCalculator(
        [mopac_calc, restorative_calc], [1, 0.1]
    )
    atoms.calc = combined_calc
    actual_forces = atoms.get_forces()
    expected_forces = forces_mopac + 0.1 * forces_restorative
    np.testing.assert_allclose(actual_forces, expected_forces)


def test_restorative_calculator_zero_energy_sphere():
    calc = RestorativeCalculator(zero_energy_radius=0.0)
    atoms = molecule("H2O")
    atoms.positions += 100  # should be tranlationally invariant
    positions_without_mean = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    expected_forces = -1 * positions_without_mean
    atoms.calc = calc
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, expected_forces)

    calc = RestorativeCalculator(zero_energy_radius=2.0)
    atoms = molecule("H2O")
    expected_forces = np.zeros_like(atoms.get_positions())
    atoms.calc = calc
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, expected_forces)

    atoms = molecule("H2")
    atoms.set_positions(np.array([[0, 0, -2], [0, 0, 2]]))
    atoms.calc = calc
    expected_forces = np.array([[0, 0, 0], [0, 0, 0]])
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, expected_forces)
    atoms.set_positions(np.array([[0.0, 0.0, -2.1], [0.0, 0.0, 2.1]]))
    expected_forces = np.array([[0.0, 0.0, +0.1], [0.0, 0.0, -0.1]])
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, expected_forces)
