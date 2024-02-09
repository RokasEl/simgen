import json
from dataclasses import dataclass

import ase
import numpy as np
from znframe import Frame


def atoms_from_json(atoms_json: dict) -> ase.Atoms:
    try:
        return Frame.from_dict(atoms_json).to_atoms()
    except:
        return ase.Atoms()


def atoms_to_json(atoms: ase.Atoms) -> dict:
    try:
        return Frame.from_atoms(atoms).to_dict()
    except:
        return {}


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
    timeout: float = 10.0


def format_run_settings(vis, **kwargs) -> RequestAtoms:
    if kwargs.get("points", None) is None:
        kwargs["points"] = vis.points
    settings = RequestAtoms(
        atom_ids=vis.selection,
        atoms=vis.atoms,
        segments=vis.segments,
        **kwargs,
    )
    return settings


def settings_to_json(settings: RequestAtoms) -> str:
    serializable_settings = {
        "run_type": settings.run_type,
        "atom_ids": settings.atom_ids,
        "atoms": atoms_to_json(settings.atoms),
        "points": settings.points.tolist(),
        "segments": settings.segments.tolist(),
        "num_atoms_to_add": settings.num_atoms_to_add,
        "restorative_force_multiplier": settings.restorative_force_multiplier,
        "max_steps": settings.max_steps,
        "timeout": settings.timeout,
    }
    return json.dumps(serializable_settings)


def settings_from_json(settings_json: str | bytes) -> RequestAtoms:
    serializable_settings = json.loads(settings_json)
    print(serializable_settings)
    settings = RequestAtoms(
        run_type=serializable_settings["run_type"],
        atom_ids=serializable_settings["atom_ids"],
        atoms=atoms_from_json(serializable_settings["atoms"]),
        points=np.array(serializable_settings["points"]),
        segments=np.array(serializable_settings["segments"]),
        num_atoms_to_add=serializable_settings["num_atoms_to_add"],
        restorative_force_multiplier=serializable_settings[
            "restorative_force_multiplier"
        ],
        max_steps=serializable_settings["max_steps"],
        timeout=serializable_settings["timeout"],
    )
    return settings


def jsonify_atoms(*args: ase.Atoms) -> dict:
    atoms = [atoms_to_json(x) for x in args]
    return {"atoms": atoms}
