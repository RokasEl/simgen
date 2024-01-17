import logging
import os
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import zntrack
from ase import Atoms
from ase.build import molecule
from e3nn import o3
from hydromace.interface import HydroMaceCalculator
from mace.calculators.foundations_models import mace_off
from mace.modules import interaction_classes
from mace.modules.models import ScaleShiftMACE
from torch import nn

from simgen.calculators import MaceSimilarityCalculator
from simgen.generation_utils import (
    RadialDistanceTransformBlock,
    remove_elements,
)

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
    name: str | None = None,
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger(name)
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
    model_repo_path: str,
    model_name: str = "small",
    data_name: str = "simgen_reference_data_small",
    remove_halogenides: bool = True,
    num_reference_mols: int = 256,
    num_to_sample_uniformly_per_size: int = 2,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
) -> MaceSimilarityCalculator:
    """
    model_name: name of the model to load from the model_repo_path. Either small_spice, medium_spice
    data_name: name of the reference data to load from the model_repo_path. Either simgen_reference_data_small or simgen_reference_data_medium
    remove_halogenides: whether to remove Hydrogen and Fluorine from the reference data, this is only for evaluation purposes. Should be True for all other purposes.
    """
    mace_model = get_loaded_mace_model(model_name, device)
    if rng is None:
        rng = np.random.default_rng(0)
    reference_data = get_reference_data(
        model_repo_path,
        data_name,
        rng,
        num_reference_mols,
        num_to_sample_uniformly_per_size,
        remove_halogenides,
    )
    mace_similarity_calculator = MaceSimilarityCalculator(
        mace_model, reference_data=reference_data, device=device
    )
    return mace_similarity_calculator


def get_hydromace_calculator(model_repo_path, device):
    try:
        model_loader = zntrack.from_rev("hydromace", remote=model_repo_path)
        model = model_loader.get_model(device=device)
        hydrogenation_model = HydroMaceCalculator(model, device=device)
        return hydrogenation_model
    except Exception as e:
        print(f"Could not load hydrogenation model due to exception: {e}")
        return None


def get_loaded_mace_model(
    model_name="medium",
    device: str = "cuda",
) -> nn.Module:
    assert model_name in (
        "small",
        "medium",
    ), "Only small and medium models are supported"
    pretrained_model: torch.nn.Module = mace_off(
        model=model_name, device=device, return_raw_model=True, default_dtype="float32"
    )
    model_config = get_mace_config(pretrained_model)
    model = ScaleShiftMACE(
        **model_config,
    )
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    model.radial_embedding = RadialDistanceTransformBlock(
        r_min=0.75,
        **dict(
            r_max=model_config["r_max"],
            num_bessel=model_config["num_bessel"],
            num_polynomial_cutoff=model_config["num_polynomial_cutoff"],
        ),
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def get_reference_data(
    model_repo_path: str,
    data_name: str = "simgen_reference_data_small",
    rng: np.random.Generator | None = None,
    num_reference_mols: int = 256,
    num_to_sample_uniformly_per_size: int = 2,
    remove_halogenides: bool = True,
) -> List[Atoms]:
    """
    Reference data is assumed to be a single xyz file containing all reference molecules.
    """
    data_loader = zntrack.from_rev(data_name, remote=model_repo_path)
    all_data = data_loader.get_atoms()
    if remove_halogenides:
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
    if rng is None:
        rng = np.random.default_rng(0)
    too_add = num_reference_mols - already_sampled
    if too_add <= 0:
        logging.info("After sampling uniformly, got more molecules than requested.")
        return training_data
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


def get_mace_config(model) -> dict:
    r_max = model.radial_embedding.bessel_fn.r_max.item()
    num_bessel = len(model.radial_embedding.bessel_fn.bessel_weights)
    num_polynomial_cutoff = int(model.radial_embedding.cutoff_fn.p.item())
    max_ell = model.spherical_harmonics._lmax
    num_interactions = model.num_interactions.item()
    num_species = model.node_embedding.linear.irreps_in.count(o3.Irrep(0, 1))
    hidden_irreps = str(model.interactions[0].hidden_irreps)
    readout_mlp_irreps = (
        "16x0e" if num_interactions == 1 else model.readouts[-1].hidden_irreps
    )  # not used when num_interactions == 1
    avg_num_neighbors = model.interactions[0].avg_num_neighbors
    correlation = model.products[0].symmetric_contractions.contractions[0].correlation
    atomic_energies = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    atomic_numbers = model.atomic_numbers.detach().cpu().numpy()
    interaction_class_first = interaction_classes[
        model.interactions[0].__class__.__name__
    ]
    interaction_class = interaction_classes[model.interactions[-1].__class__.__name__]
    mean = 0.0
    std = 1.0
    if hasattr(model, "scale_shift"):
        mean = model.scale_shift.shift.detach().cpu().numpy()
        std = model.scale_shift.scale.detach().cpu().numpy()
    if num_interactions == 1:
        activation = None
    else:
        acts = model.readouts[-1].non_linearity.acts
        activation = None
        for act in acts:
            if act is not None:
                gate = act.f
                activation = getattr(torch.nn.functional, gate.__name__)
                break
    config = dict(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        interaction_cls=interaction_class,
        interaction_cls_first=interaction_class_first,
        num_interactions=num_interactions,
        num_elements=num_species,
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps(readout_mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=correlation,
        gate=activation,
        atomic_inter_scale=std,
        atomic_inter_shift=mean,
    )
    return config


def get_system_torch_device_str() -> str:
    # the hierarchy is: MPS > CUDA > CPU
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.backends.cuda.is_built():
        return "cuda"
    else:
        print("No GPU acceleration available, using CPU")
        return "cpu"


@contextmanager
def time_function(name: str):
    start = perf_counter()
    yield
    end = perf_counter()
    logging.info(f"{name} took {end-start:.2f} seconds")
