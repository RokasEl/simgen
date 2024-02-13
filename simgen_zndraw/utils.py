import logging
import pathlib

import ase
import networkx as nx
import numpy as np
from znframe.frame import get_radius

from simgen.hydrogenation import get_edge_array_from_atoms
from simgen_zndraw import DefaultGenerationParams


def get_anchor_point_positions(
    atoms: ase.Atoms, selection: list[int], camera_dict: dict[str, list[float]]
) -> np.ndarray:
    if len(selection) < 2:
        raise ValueError("Need at least two atoms to define a connection")
    positions = atoms.positions[selection]
    numbers = atoms.numbers[selection]
    camera_position = np.array(camera_dict["position"])[None, :]  # 1x3

    radii: np.ndarray = get_radius(numbers)[0][:, None]  # Nx1
    direction = camera_position - positions  # Nx3
    direction /= np.linalg.norm(direction, axis=1)[:, None]  # Nx3
    anchor_point_positions = positions + direction * radii
    return anchor_point_positions


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


def get_default_mace_models_path() -> str:
    config_path = "~/.simgen/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing configuration at {config_path}")
        config = DefaultGenerationParams.from_file(config_path)  # type: ignore
        path = config.default_model_path
        return path
    else:
        return DefaultGenerationParams.default_model_path
