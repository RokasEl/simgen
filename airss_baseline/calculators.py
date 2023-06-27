import re
import subprocess
from typing import Tuple

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class RestorativeCalculator(Calculator):
    implemented_properties = ["forces", "energy", "energies"]
    """
    'zero_energy_radius' if > 0, then atoms within this radius of the origin will experience no force.
    If the atoms are outside this radius, then the force will be proportional to the distance from the origin minus the zero_energy_radius.
    """

    def __init__(
        self,
        prior_manifold=None,
        zero_energy_radius: float = 0.0,
        restart=None,
        label=None,
        atoms=None,
        directory=".",
        **kwargs,
    ):
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )
        self.prior_manifold = prior_manifold
        self.zero_energy_radius = zero_energy_radius

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms  # type: ignore
        if atoms is None:
            raise ValueError("atoms is None")
        atoms.pbc = False
        positions = atoms.get_positions()
        positions -= positions.mean(axis=0)
        positions = self.get_positions_with_respect_zero_energy_sphere(positions)
        energies = 0.5 * (positions**2).sum(axis=1)
        energy = energies.sum()
        if self.prior_manifold is not None:
            forces = self.prior_manifold.calculate_resorative_forces(positions)
        else:
            forces = -1 * positions
        self.results["energies"] = energies
        self.results["energy"] = energy
        self.results["forces"] = forces

    def get_positions_with_respect_zero_energy_sphere(self, positions):
        # if a point is inside the sphere, then the force is zero
        # so set its position to origin, otherwise subtract the radius
        if self.zero_energy_radius > 0:
            distance_from_origin = np.linalg.norm(positions, axis=1)
            position_unit_vectors = positions / distance_from_origin[:, None]
            scaled_distances = distance_from_origin - self.zero_energy_radius
            scaled_distances[scaled_distances < 0] = 0
            scaled_positions = position_unit_vectors * scaled_distances[:, None]
            return scaled_positions
        else:
            return positions


class MopacCalculator(Calculator):
    name = "MopacCalculator"
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        method="PM6",
        input_file_name="mopac",
        f_max: float = 10.0,
        **kwargs,
    ):
        self.method = method
        self.input_file_name = input_file_name
        self.f_max = f_max
        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                "No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation."
            )
        super().calculate(atoms, properties, system_changes)
        self.write_input(atoms)
        subprocess.run(
            ["mopac " + self.input_file_name + ".in"],
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        # kcal/mol and kcal/angstrom
        energy, forces, _ = self.read_results()
        forces = self._clip_grad_norm(forces, self.f_max)
        self.results["energy"] = energy
        self.results["forces"] = forces

    def do_full_relaxation(self, atoms):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                "No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation."
            )
        self.write_input(atoms, full_relaxation=True)
        subprocess.run(
            ["mopac " + self.input_file_name + ".in"],
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        energy, forces, final_atoms = self.read_results()
        final_atoms.info["energy"] = energy
        final_atoms.arrays["forces"] = forces
        return final_atoms

    def write_input(self, atoms, full_relaxation=False):
        # Build string to hold .mop input file:
        if not full_relaxation:
            command = f"{self.method} 1SCF XYZ GRADIENTS GEO-OK"
        else:
            command = f"{self.method} XYZ GRADIENTS GEO-OK"
        command += "\nTitle: ASE calculation\n\n"

        # Write coordinates:
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            command += " {0:2} {1} {2} {3}\n".format(symbol, *xyz)

        with open(f"{self.input_file_name}.in", "w") as fd:
            fd.write(command)

    def read_results(self) -> Tuple[float, np.ndarray, ase.Atoms]:
        with open(f"{self.input_file_name}.in.out") as fd:
            lines = fd.readlines()
        energy = self.find_energy(lines) * 0.0433641153087705
        forces = self.find_forces(lines) * 0.0433641153087705
        atoms = self.get_final_atoms(lines)
        return energy, forces, atoms

    def find_energy(self, lines):
        for line in lines:
            if line.startswith("          FINAL HEAT OF FORMATION"):
                match = re.search(r"([-+]?\d*\.\d+|\d+) KCAL\/MOL", line)
                if match:
                    number = match.group(1)
                    return float(number)
        raise ValueError("Could not find energy in MOPAC output file.")

    def find_forces(self, lines):
        gradient_start_idx = 0
        gradient_end_idx = 0

        for i, line in enumerate(lines):
            if "FINAL  POINT  AND  DERIVATIVES" in line:
                gradient_start_idx = i + 3
                break
        for i, line in enumerate(lines[gradient_start_idx:]):
            if line == "\n":
                gradient_end_idx = gradient_start_idx + i
                break
        assert (
            gradient_end_idx > gradient_start_idx
            and (gradient_end_idx - gradient_start_idx) % 3 == 0
        )

        gradients = []
        for line in lines[gradient_start_idx:gradient_end_idx]:
            split_line = line.split()
            if len(split_line) == 8:
                gradients.append(split_line[-2])
            elif len(split_line) == 7:
                # mopac freaked out and didn't print the gradient
                # let's hope next step it will be fine
                gradients.append("0.0")
            else:  # something went horribly wrong
                raise ValueError("Could not parse gradient line in MOPAC output file.")
        forces = np.array(gradients, dtype=float).reshape(-1, 3) * -1
        return forces

    def get_final_atoms(self, lines) -> ase.Atoms:
        final_coordinates_start_idx = 0
        final_coordinates_end_idx = 0
        for i in range(len(lines))[::-1]:
            if "CARTESIAN COORDINATES" in lines[i]:
                final_coordinates_start_idx = i + 2
                break
        for i, line in enumerate(lines[final_coordinates_start_idx:]):
            if line == "\n":
                final_coordinates_end_idx = final_coordinates_start_idx + i
                break
        symbols, positions = [], []
        for line in lines[final_coordinates_start_idx:final_coordinates_end_idx]:
            symbols.append(line.split()[1])
            positions.append(line.split()[2:])
        positions = np.array(positions, dtype=float)
        atoms = ase.Atoms(symbols=symbols, positions=positions)
        return atoms

    @staticmethod
    def _clip_grad_norm(grad, max_norm: float = 1.0):
        norm = np.linalg.norm(grad, axis=1)
        mask = norm > max_norm
        grad[mask] = grad[mask] / norm[mask, None] * max_norm
        return grad
