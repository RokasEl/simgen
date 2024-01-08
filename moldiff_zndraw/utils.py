import logging
import pathlib

import ase
import numpy as np

from moldiff.hydrogenation import get_edge_array_from_atoms
from moldiff_zndraw import DefaultGenerationParams


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
        if path is None:
            print("No default model path found, will use remote models")
            return "https://github.com/RokasEl/MACE-Models"
        else:
            return path
    else:
        print("No config file found, will use remote models")
        return "https://github.com/RokasEl/MACE-Models"
