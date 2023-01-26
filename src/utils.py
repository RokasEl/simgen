import numpy as np
from ase import Atoms

QM9_PROPERTIES = (
    "rotational_constants",
    "dipole_moment",
    "isotropic_polarizability",
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "electronic_spatial_extent",
    "zero_point_vibrational_energy",
    "internal_energy_0K",
    "internal_energy_298K",
    "enthalpy_298K",
    "free_energy_298K",
    "heat_capacity_298K",
)

FIELD_IN_HARTREE = (
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "zero_point_vibrational_energy",
    "internal_energy_0K",
    "internal_energy_298K",
    "enthalpy_298K",
    "free_energy_298K",
)


def _parse_to_float(float_str):
    try:
        num = float(float_str)
    except ValueError:
        if "*^" in float_str:
            whole, exp = float_str.split("*^")
            num = float(whole) * 10 ** float(exp)
        else:
            num = 0.0
    return num


def _process_line(line):
    """Processes a line from the xyz file"""
    element, *coord, charge = line.split()
    coord = np.asarray([_parse_to_float(c) for c in coord])
    charge = _parse_to_float(charge)
    return element, coord, charge


def _get_qm9_props(line):
    "Processes the second line in the XYZ, which contains the properties"
    properties = line.split()
    assert len(properties) == 17
    assert properties[0] == "gdb"
    clean_floats = [_parse_to_float(p) for p in properties[2:]]
    rotational_constants = clean_floats[:3]
    remaining_properties = clean_floats[3:]
    parsed_props = dict(zip(QM9_PROPERTIES, remaining_properties))
    parsed_props["rotational_constants"] = rotational_constants  # type: ignore
    for key in FIELD_IN_HARTREE:
        parsed_props[key] *= 27.211396641308
    parsed_props["energy"] = parsed_props["internal_energy_0K"]  # type: ignore
    return parsed_props


def read_qm9_xyz(filename):
    """Reads xyz file with QM9 dataset"""
    with open(filename) as f:
        lines = f.readlines()
    natoms = int(lines[0])
    parsed_props = _get_qm9_props(lines[1])
    elements, coords, charges = [], [], []
    for l in lines[2:-3]:
        element, *coord, charge = _process_line(l)
        elements.append(element)
        coords.append(coord)
        charges.append(charge)
    coords = np.concatenate(coords)
    assert len(elements) == natoms
    info = parsed_props
    atoms = Atoms(elements, coords, charges=charges, info=info)
    # since we are reading optimised geometries, set the forces to 0
    atoms.arrays["forces"] = np.zeros((natoms, 3))
    return atoms
