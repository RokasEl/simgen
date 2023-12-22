import abc
import json
import logging
import typing as t

import ase
import numpy as np
import requests
from pydantic import BaseModel, Field
from zndraw import ZnDraw
from zndraw.frame import Frame
from zndraw.modify import UpdateScene

from .utils import (
    calculate_path_length,
    interpolate_points,
    remove_isolated_atoms_using_covalent_radii,
    setup_logger,
)

setup_logger()

def atoms_from_json(atoms_json: dict) -> ase.Atoms:
    return Frame.from_dict(atoms_json).to_atoms()

def atoms_to_json(atoms: ase.Atoms) -> dict:
    return Frame.from_atoms(atoms).to_dict()


def _format_data_from_zndraw(vis: ZnDraw, **kwargs) -> dict:
    points = kwargs.get("points", vis.points)
    formatted_points = points if points is None else points.tolist()
    data = {
        "atom_ids": vis.selection,
        "atoms": atoms_to_json(vis.atoms),
        "points": formatted_points,
        "segments": vis.segments.tolist(),
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


class PerAngstrom(BaseModel):
    discriminator: t.Literal["PerAngstrom"] = "PerAngstrom"
    atoms_per_angstrom: float = Field(
        1.2,
        ge=0,
        le=3.0,
        description="Num atoms added = atoms_per_angstrom * curve_length",
    )

    @property
    def parameters(self):
        return self.discriminator, self.atoms_per_angstrom


class FixedNumber(BaseModel):
    discriminator: t.Literal["FixedNumber"] = "FixedNumber"
    number_of_atoms: int = Field(
        5, ge=1, le=30, description="Number of atoms to generate"
    )

    @property
    def parameters(self):
        return self.discriminator, self.number_of_atoms


class Generate(UpdateScene):
    discriminator: t.Literal["Generate"] = Field("Generate")
    num_steps: int = Field(
        50, le=100, ge=20, description="Number of steps in the generation."
    )
    atom_number: t.Union[FixedNumber, PerAngstrom]
    guiding_force_multiplier: float = Field(
        1.0,
        ge=1.0,
        le=10.0,
        description="Multiplier for guiding force. Default value should be enough for simple geometries.",
    )

    def run(self, vis: ZnDraw, client_address) -> list[ase.Atoms]:
        vis.log("Running Generation")
        points = self._handle_points(vis.points, vis.segments)
        if len(vis.atoms):
            points = self._remove_collisions_between_prior_and_atoms(
                points, vis.atoms.get_positions()
            )
        atom_number_type, atom_number = self.atom_number.parameters
        num_atoms_to_add = self._get_num_atoms_to_add(
            points, atom_number_type, atom_number
        )
        request = {
            "run_type": "generate",
            "run_specific_params": {
                "num_atoms_to_add": int(num_atoms_to_add),
                "restorative_force_multiplier": float(self.guiding_force_multiplier),
                "max_steps": int(self.num_steps),
            },
            "common_data": _format_data_from_zndraw(vis, points=points),
        }
        response = _post_request(client_address, data=request, name="generation")
        modified_atoms = [
            atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
        ]
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.extend(modified_atoms)
        vis.play()

    @staticmethod
    def _handle_points(points, segments) -> np.ndarray:
        if points.size == 0:
            logging.info("No location provided, will generate at origin")
            return np.array([[0.0, 0.0, 0.0]])
        elif points.shape[0] == 1:
            return points
        else:
            """ZnDraw interpolates between each user placed point.
            However, this means that the points are not spread evenly along the curve.
            We will then do an interpolation between each point to get a more even spread.
            """
            segments = interpolate_points(segments, 100)
            return segments

    @staticmethod
    def _remove_collisions_between_prior_and_atoms(
        points: np.ndarray, atoms_positions: np.ndarray, cutoff=0.8
    ):
        if points is None or points.shape[0] == 0:
            return points
        else:
            distances = (
                points[:, None, :] - atoms_positions[None, :, :]
            )  # (points, atoms, 3)
            distances = np.linalg.norm(distances, axis=-1)  # (points, atoms)
            stripped_points = points[np.min(distances, axis=1) > cutoff]
            assert (
                stripped_points.shape[0] > 0
            ), "No points left after removing collisions"
            return stripped_points

    @staticmethod
    def _get_num_atoms_to_add(
        points: np.ndarray,
        atom_number_determination_type: str,
        atom_parameter_value: int | float,
    ) -> int:
        if atom_number_determination_type == "FixedNumber":
            return atom_parameter_value
        elif atom_number_determination_type == "PerAngstrom":
            logging.info(
                "Calculating number of atoms to add based on curve length and density"
            )
            assert (
                points.shape[0] > 1
            ), "`PerAngstrom` requires at least 2 points drawn in the interface"
            curve_length = calculate_path_length(points)
            num_atoms_to_add = np.ceil(curve_length * atom_parameter_value).astype(int)
            logging.info(
                f"Path length defined by points: {curve_length:.1f} A; atoms to add: {num_atoms_to_add}"
            )
            return num_atoms_to_add
        else:
            raise ValueError(
                f"Unknown atom number determination type: {atom_number_determination_type}"
            )


class Relax(UpdateScene):
    discriminator: t.Literal["Relax"] = Field("Relax")
    max_steps: int = Field(50, ge=1)

    def run(self, vis: ZnDraw, client_address) -> list[ase.Atoms]:
        vis.log("Running Relax")
        request = {
            "run_type": "relax",
            "run_specific_params": {
                "max_steps": int(self.max_steps),
            },
            "common_data": _format_data_from_zndraw(vis),
        }
        response = _post_request(client_address, data=request, name="relaxation")
        modified_atoms = [
            atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
        ]
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.extend(modified_atoms)
        vis.play()


class Hydrogenate(UpdateScene):
    discriminator: t.Literal["Hydrogenate"] = Field("Hydrogenate")
    max_steps: int = Field(30, ge=1)

    def run(self, vis: ZnDraw, client_address) -> list[ase.Atoms]:
        vis.log("Running Hydrogenate")
        request = {
            "run_type": "hydrogenate",
            "run_specific_params": {
                "max_steps": int(self.max_steps),
            },
            "common_data": _format_data_from_zndraw(vis),
        }
        response = _post_request(client_address, data=request, name="hydrogenation")

        modified_atoms = [
            atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
        ]
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.extend(modified_atoms)
        vis.play()


run_types = t.Union[Generate, Relax, Hydrogenate]


class DiffusionModelling(UpdateScene):
    discriminator: t.Literal["DiffusionModelling"] = "DiffusionModelling"
    run_type: run_types = Field(discriminator="discriminator")
    path: str = Field(
        "/home/rokas/Programming/MACE-Models",
        description="Path to the repo holding the required models",
    )
    client_address: str = Field("http://127.0.0.1:5000/run")

    def run(self, vis: ZnDraw) -> list[ase.Atoms]:
        vis.log("Sending request to inference server.")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]

        self.run_type.run(
            vis=vis,
            client_address=self.client_address,
        )
        vis.append(remove_isolated_atoms_using_covalent_radii(vis[-1]))

    @staticmethod
    def get_documentation_url() -> str:
        return "https://rokasel.github.io/EnergyMolecularDiffusion"
