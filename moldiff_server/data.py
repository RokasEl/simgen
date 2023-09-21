import logging
from dataclasses import dataclass
from typing import Any, Dict

import ase
import numpy as np
from pydantic import BaseModel
from zndraw.data import atoms_from_json, atoms_to_json


@dataclass
class RequestAtoms:
    run_type: str
    # all request have these
    atom_ids: list[int]
    atoms: ase.Atoms
    points: np.ndarray
    segments: np.ndarray
    # generate specific params
    num_atoms_to_add: int = 5
    restorative_force_multiplier: float = 1
    # hydrogenate and relax params
    max_steps: int = 50
    # experimental
    url: str | None = None


class RequestStructure(BaseModel):
    run_type: str
    run_specific_params: Dict[str, Any]
    common_data: Dict[str, Any]


def format_common_data(request: RequestStructure):
    data = request.common_data
    specific_params = request.run_specific_params
    atom_ids = data["atom_ids"]
    atoms = atoms_from_json(data["atoms"])
    points = data["points"]
    if points is None:
        points = [[0.0, 0.0, 0.0]]
    points = np.array(points)
    segments = np.array(data["segments"])
    formated_data = RequestAtoms(
        run_type=request.run_type,
        atom_ids=atom_ids,
        atoms=atoms,
        points=points,
        segments=segments,
        url=data["url"],
        **specific_params,
    )
    return formated_data


def jsonify_atoms(*args: ase.Atoms) -> dict:
    atoms = [atoms_to_json(x) for x in args]
    return {"atoms": atoms}


def parse_request(data: dict) -> RequestAtoms:
    try:
        parsed_data = RequestStructure.parse_obj(data)
        formatted_request = format_common_data(parsed_data)
        return formatted_request
    except Exception as e:
        logging.error("Couldn't parse the request to the inference serve")
        raise e
