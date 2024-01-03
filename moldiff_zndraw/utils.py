import logging
import os
import pathlib
import sys
from typing import Optional, Union

import ase
import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.interpolate import splev, splprep
from zndraw.settings import GlobalConfig

from moldiff.hydrogenation import get_edge_array_from_atoms

"""
These functions are the same as in the main repo.
Repeating here to enable a light-weight installation compatible with ZnDraw.
"""


def get_edge_array(atoms: ase.Atoms) -> np.ndarray:
    try:
        connectivity_graph = atoms.info["graph_representation"]
        edge_array = nx.adjacency_matrix(connectivity_graph).todense()  # type: ignore
        return edge_array
    except KeyError:
        logging.info(
            "No graph representation found, try resetting the scene or clicking `Save` in the `Bonds` tab"
        )
        logging.info("Will build an edge array for hydrogenation using covalent radii")
        edge_array = get_edge_array_from_atoms(atoms)
        return edge_array
    except Exception:
        raise Exception


def make_mace_config_jsonifiable(mace_config: dict) -> dict:
    jsonifiable_mace = mace_config.copy()
    for key, value in jsonifiable_mace.items():
        if isinstance(value, np.ndarray):
            jsonifiable_mace[key] = value.tolist()
        elif isinstance(value, (int, float)):
            pass
        else:
            jsonifiable_mace[key] = str(value)
    return jsonifiable_mace


def calculate_path_length(points: np.ndarray):
    path_length = 0.0
    for p1, p2 in zip(points[:-1], points[1:]):
        path_length += np.linalg.norm(p1 - p2)
    return path_length


def interpolate_points(points, num_interpolated_points=100):
    k = min(3, len(points) - 1)
    tck, u = splprep(points.T, s=0, k=k)
    u = np.linspace(0, 1, num_interpolated_points)
    new_points = np.array(splev(u, tck)).T
    return new_points


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


def remove_isolated_atoms_using_covalent_radii(
    atoms: ase.Atoms, multiplier: float = 1.2
) -> ase.Atoms:
    """
    Remove unconnected atoms from the final atoms object.
    """
    cutoffs = natural_cutoffs(atoms, mult=multiplier)  # type: ignore
    indices_of_connected_atoms = neighbor_list("i", atoms, cutoffs)
    unique_indices = np.unique(indices_of_connected_atoms)
    stripped_atoms = atoms.copy()
    stripped_atoms = stripped_atoms[unique_indices]
    return stripped_atoms  # type: ignore


def get_default_mace_models_path() -> str:
    config_path = "~/.zincware/zndraw/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing configuration at {config_path}")
        config = GlobalConfig.from_file(config_path)  # type: ignore
        mace_models_path = config.function_schema[
            "moldiff_zndraw.main.DiffusionModelling"
        ]["path"]
        return mace_models_path
    else:
        raise ValueError(
            "Could not find a config file at ~/.zincware/zndraw/config.json, specify the path to the MACE-models repo with --path"
        )
