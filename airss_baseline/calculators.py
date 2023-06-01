import os

import ase
import numpy as np
from ase.calculators.calculator import (
    Calculator,
    FileIOCalculator,
    Parameters,
    ReadError,
    all_changes,
)
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.mopac import MOPAC
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, LBFGS
from ase.visualize import view

from moldiff.manifolds import MultivariateGaussianPrior


class RestorativeCalculator(Calculator):
    implemented_properties = ["forces", "energy", "energies"]

    def __init__(
        self,
        prior_manifold=None,
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

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        atoms: ase.Atoms = atoms if atoms is not None else self.atoms  # type: ignore
        atoms.pbc = False
        positions = atoms.get_positions()
        energies = 0.5 * (positions**2).sum(axis=1)
        energy = energies.sum()
        if self.prior_manifold is not None:
            forces = self.prior_manifold.calculate_resorative_forces(positions)
        else:
            forces = -1 * positions
        self.results["energies"] = energies
        self.results["energy"] = energy
        self.results["forces"] = forces


class MOPACLight(MOPAC):
    def __init__(
        self,
        parameters=Parameters(task="DISP", method="PM6", charge=0, relscf=0.0001),
        restart=None,
        ignore_bad_restart_file=FileIOCalculator._deprecated,
        label="mopac",
        atoms=None,
        **kwargs,
    ):
        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        self.parameters = parameters

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters

        # Build string to hold .mop input file:
        s = p.method + " " + p.task + " "

        if p.relscf:
            s += "RELSCF={0} ".format(p.relscf)

        # Write charge:
        if p.charge:
            charge = p.charge
        else:
            charge = atoms.get_initial_charges().sum()

        if charge != 0:
            s += "CHARGE={0} ".format(int(round(charge)))

        s += "\nTitle: ASE calculation\n\n"

        # Write coordinates:
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            s += " {0:2} {1} 1 {2} 1 {3} 1\n".format(symbol, *xyz)

        with open(self.label + ".mop", "w") as fd:
            fd.write(s)

    def read_results(self):
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + ".out"):
            raise ReadError

        with open(self.label + ".out") as fd:
            lines = fd.readlines()

        self.atoms = self.read_atoms_from_file(lines)

    def read_atoms_from_file(self, lines):
        i = self.get_index(
            lines, "GEOMETRY OPTIMISED USING EIGENVECTOR FOLLOWING (EF)."
        )
        lines1 = lines[i:]
        i = self.get_index(lines1, "CARTESIAN COORDINATES")
        j = i + 2
        symbols = []
        positions = []
        while not lines1[j].isspace():  # continue until we hit a blank line
            l = lines1[j].split()
            symbols.append(l[1])
            positions.append([float(c) for c in l[2 : 2 + 3]])
            j += 1

        return ase.Atoms(symbols=symbols, positions=positions)
