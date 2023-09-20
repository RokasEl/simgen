import logging

import ase
import networkx as nx
import numpy as np

from moldiff.hydrogenation import get_edge_array_from_atoms


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
