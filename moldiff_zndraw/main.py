import abc
import json
import logging
import typing as t

import ase
import numpy as np
import requests
from pydantic import BaseModel, Field
from zndraw.data import atoms_from_json

from .utils import (
    calculate_path_length,
    remove_isolated_atoms_using_covalent_radii,
    setup_logger,
)

setup_logger()


def _format_data_from_zndraw(atom_ids, **kwargs) -> dict:
    points = kwargs["points"]
    formatted_points = points if points is None else points.tolist()
    data = {
        "atom_ids": atom_ids,
        "atoms": kwargs["json_data"],
        "points": formatted_points,
        "segments": kwargs["segments"].tolist(),
    }
    return data


def _post_request(address: str, data: dict, name: str):
    logging.info(f"Posted {name} request")
    try:
        response = requests.post(str(address), data=json.dumps(data))
        logging.info(f"Received {name} response with code {response}")
        return response
    except Exception as e:
        logging.error(f"Failed to get response with error: {e}")
        raise e


class UpdateScene(BaseModel, abc.ABC):
    @abc.abstractmethod
    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        pass


class Generate(BaseModel):
    method: t.Literal["Generate"] = Field("Generate")
    num_steps: int = Field(
        50, le=100, ge=20, description="Number of steps in the generation."
    )
    num_atoms_to_add: int = Field(
        5, ge=1, le=30, description="Number of atoms to generate"
    )
    atoms_per_angstrom: float = Field(
        -1.0,
        ge=-1.0,
        le=3.0,
        description="Will supersede num_atoms_to_add if positive",
    )
    guiding_force_multiplier: float = Field(
        1.0,
        ge=1.0,
        le=10.0,
        description="Multiplier for guiding force. Default value should be enough for simple geometries.",
    )

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        points = self._handle_points(kwargs["points"], kwargs["segments"])
        kwargs["points"] = points
        num_atoms_to_add = self._get_num_atoms_to_add(
            points, self.atoms_per_angstrom, self.num_atoms_to_add
        )
        request = {
            "run_type": "generate",
            "run_specific_params": {
                "num_atoms_to_add": int(num_atoms_to_add),
                "restorative_force_multiplier": float(self.guiding_force_multiplier),
                "max_steps": int(self.num_steps),
            },
            "common_data": _format_data_from_zndraw(atom_ids, **kwargs),
        }
        response = _post_request(
            kwargs["client_address"], data=request, name="generation"
        )
        return [atoms_from_json(x) for x in response.json()["atoms"]]

    @staticmethod
    def _handle_points(points, segments) -> np.ndarray:
        if points.size == 0:
            logging.info("No location provided, will generate at origin")
            return np.array([[0.0, 0.0, 0.0]])
        elif points.shape[0] == 1:
            return points
        else:
            return segments

    @staticmethod
    def _get_num_atoms_to_add(
        points: np.ndarray, atoms_per_angstrom: float, num_static: int
    ) -> int:
        if atoms_per_angstrom > 0 and points is not None and points.shape[0] > 0:
            logging.info(
                "Calculating number of atoms to add based on curve length and density"
            )
            curve_length = calculate_path_length(points)
            num_atoms_to_add = np.ceil(curve_length * atoms_per_angstrom).astype(int)
            logging.info(
                f"Path length defined by points: {curve_length:.1f} A; atoms to add: {num_atoms_to_add}"
            )
            return num_atoms_to_add
        else:
            return num_static


class Relax(BaseModel):
    method: t.Literal["Relax"] = Field("Relax")
    max_steps: int = Field(100, ge=1)

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        request = {
            "run_type": "relax",
            "run_specific_params": {
                "max_steps": int(self.max_steps),
            },
            "common_data": _format_data_from_zndraw(atom_ids, **kwargs),
        }
        response = _post_request(
            kwargs["client_address"], data=request, name="relaxation"
        )
        return [atoms_from_json(x) for x in response.json()["atoms"]]


class Hydrogenate(BaseModel):
    method: t.Literal["Hydrogenate"] = Field("Hydrogenate")
    max_steps: int = Field(100, ge=1)

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        request = {
            "run_type": "hydrogenate",
            "run_specific_params": {
                "max_steps": int(self.max_steps),
            },
            "common_data": _format_data_from_zndraw(atom_ids, **kwargs),
        }
        response = _post_request(
            kwargs["client_address"], data=request, name="hydrogenation"
        )
        return [atoms_from_json(x) for x in response.json()["atoms"]]


run_types = t.Union[Generate, Relax, Hydrogenate]


class DiffusionModelling(UpdateScene):
    method: t.Literal["DiffusionModelling"] = "DiffusionModelling"
    run_type: run_types = Field(discriminator="method")
    path: str = Field(
        "/home/rokas/Programming/MACE-Models",
        description="Path to the repo holding the required models",
    )
    client_address: str = Field("http://127.0.0.1:8000")

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, t.Any]:
        schema = super().model_json_schema(*args, **kwargs)
        for prop in [x.__name__ for x in t.get_args(run_types)]:
            schema["$defs"][prop]["properties"]["method"]["options"] = {"hidden": True}
            schema["$defs"][prop]["properties"]["method"]["type"] = "string"
        return schema

    def run(self, atom_ids: list[int], atoms: ase.Atoms, **kwargs) -> list[ase.Atoms]:
        logging.info("Sending request to inference server.")
        modified_atoms = self.run_type.run(
            atom_ids=atom_ids,
            atoms=atoms,
            client_address=self.client_address,
            **kwargs,
        )
        modified_atoms[-1] = remove_isolated_atoms_using_covalent_radii(
            modified_atoms[-1]
        )
        logging.info(f"Received back {len(modified_atoms)} atoms.")
        return modified_atoms
