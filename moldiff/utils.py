import logging
import os
import sys
from typing import List, Optional, Tuple, Union

import ase.io as aio
import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3
from mace.modules.blocks import (
    RadialDistanceTransformBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import ScaleShiftMACE
from torch import nn

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.generation_utils import remove_elements

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


def initialize_mol(molecule_str="C6H6"):
    try:
        mol = molecule(molecule_str)
    except:
        mol = Atoms(molecule_str)
    return mol


# Taken from MACE
def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


def get_mace_similarity_calculator(
    model_path: str,
    reference_data_path: str,
    num_reference_mols: int = 256,
    num_to_sample_uniformly_per_size: int = 2,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
) -> MaceSimilarityCalculator:
    mace_model = get_loaded_mace_model(model_path, device)
    if rng is None:
        rng = np.random.default_rng(0)
    reference_data = get_reference_data(
        reference_data_path, rng, num_reference_mols, num_to_sample_uniformly_per_size
    )
    mace_similarity_calculator = MaceSimilarityCalculator(
        mace_model, reference_data=reference_data, device=device
    )
    return mace_similarity_calculator


def get_loaded_mace_model(model_path: str, device: str = "cuda") -> nn.Module:
    pretrained_model = torch.load(model_path)
    model = ScaleShiftMACE(
        r_max=4.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        radial_MLP=[64, 64, 64],
        max_ell=3,
        num_interactions=2,
        num_elements=10,
        atomic_energies=np.zeros(10),
        avg_num_neighbors=15.653135299682617,
        correlation=3,
        interaction_cls_first=RealAgnosticInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        hidden_irreps=o3.Irreps("96x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_numbers=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.088502,
        atomic_inter_shift=0.0,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.radial_embedding = RadialDistanceTransformBlock(
        r_min=0.75, **dict(r_max=4.5, num_bessel=8, num_polynomial_cutoff=5)
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def get_reference_data(
    reference_data_path: str,
    rng: np.random.Generator,
    num_reference_mols: int = 256,
    num_to_sample_uniformly_per_size: int = 2,
) -> List[Atoms]:
    """
    Reference data is assumed to be a single xyz file containing all reference molecules.
    """
    all_data = aio.read(reference_data_path, index=":", format="extxyz")
    all_data = [remove_elements(mol, [1, 9]) for mol in all_data]  # type: ignore
    if num_reference_mols == -1:
        return all_data

    if num_to_sample_uniformly_per_size > 0:
        training_data, already_sampled_indices = sample_uniformly_across_heavy_atom_number(all_data, num_to_sample_uniformly_per_size, rng)  # type: ignore
        already_sampled = len(training_data)
        all_data = [
            mol
            for idx, mol in enumerate(all_data)
            if idx not in already_sampled_indices
        ]
    else:
        training_data = []
        already_sampled = 0

    # now add further random molecules
    too_add = num_reference_mols - already_sampled
    all_data = np.asarray(all_data, dtype=object)
    rand_mols = [x for x in rng.choice(all_data, size=too_add, replace=False)]
    training_data.extend(rand_mols)

    return training_data


def sample_uniformly_across_heavy_atom_number(
    data: List[Atoms], num_mols_per_size: int, rng: np.random.Generator
) -> Tuple[List[Atoms], List[int]]:
    mol_sizes = {len(mol) for mol in data}
    selected_indices = []
    for size in mol_sizes:
        indices = np.where(np.asarray([len(mol) for mol in data]) == size)[0]
        num_to_pick = min(num_mols_per_size, len(indices))
        selected_indices.extend(rng.choice(indices, size=num_to_pick, replace=False))
    sampled_mols = [data[idx].copy() for idx in selected_indices]
    return sampled_mols, selected_indices
