import logging
import typing as t

import ase
import numpy as np
import requests
from pydantic import BaseModel, Field
from zndraw import ZnDraw
from zndraw.frame import Frame
from zndraw.modify import UpdateScene

from simgen.atoms_cleanup import (
    remove_isolated_atoms_using_covalent_radii,
)
from simgen.generation_utils import (
    calculate_path_length,
    interpolate_points,
)
from simgen.utils import setup_logger

from .data import atoms_from_json, format_run_settings, settings_to_json
from .endpoints import generate, hydrogenate, relax

setup_logger()


def _post_request(address: str, json_data_str: str, name: str):
    logging.info(f"Posted {name} request")
    try:
        response = requests.post(str(address), data=json_data_str)
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
        description="Multiplier for guiding force. Increase if molecules falls apart.",
    )

    def run(self, vis: ZnDraw, client_address, calculators: dict) -> None:
        vis.log("Running Generation")
        run_specific_settings = self._get_run_specific_settings(vis)
        run_settings = format_run_settings(vis, **run_specific_settings)
        generation_calc = calculators.get("generation", None)
        if generation_calc is None:
            vis.log("No loaded generation model, will try posting remote request")
            json_request = settings_to_json(run_settings)
            response = _post_request(
                client_address, json_data_str=json_request, name="generation"
            )
            modified_atoms = [
                atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
            ]
        else:
            modified_atoms = generate(run_settings, generation_calc)
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.extend(modified_atoms)
        vis.play()

    def _get_run_specific_settings(self, vis: ZnDraw) -> dict:
        points = self._handle_points(vis.points, vis.segments)
        if len(vis.atoms):
            points = self._remove_collisions_between_prior_and_atoms(
                points, vis.atoms.get_positions()
            )
        atom_number_type, atom_number = self.atom_number.parameters
        num_atoms_to_add = self._get_num_atoms_to_add(
            points, atom_number_type, atom_number
        )
        return {
            "run_type": "generate",
            "num_atoms_to_add": int(num_atoms_to_add),
            "restorative_force_multiplier": float(self.guiding_force_multiplier),
            "max_steps": int(self.num_steps),
            "points": points,
        }

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
            return int(atom_parameter_value)
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

    def run(self, vis: ZnDraw, client_address, calculators) -> None:
        vis.log("Running Relax")
        run_settings = format_run_settings(
            vis, run_type="relax", max_steps=self.max_steps
        )
        generation_calc = calculators.get("generation", None)
        if generation_calc is None:
            vis.log("No loaded generation model, will try posting remote request")
            json_request = settings_to_json(run_settings)
            response = _post_request(
                client_address, json_data_str=json_request, name="relaxation"
            )
            modified_atoms = [
                atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
            ]
        else:
            modified_atoms = relax(run_settings, generation_calc)
        vis.extend(modified_atoms)
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.play()


class Hydrogenate(UpdateScene):
    discriminator: t.Literal["Hydrogenate"] = Field("Hydrogenate")
    max_steps: int = Field(30, ge=1)

    def run(self, vis: ZnDraw, client_address, calculators) -> None:
        vis.log("Running Hydrogenate")
        run_settings = format_run_settings(
            vis, run_type="hydrogenate", max_steps=self.max_steps
        )
        generation_calc = calculators.get("generation", None)
        hydrogenation_calc = calculators.get("hydrogenation", None)

        if generation_calc is None or hydrogenation_calc is None:
            vis.log("No loaded generation model, will try posting remote request")
            json_request = settings_to_json(run_settings)
            response = _post_request(
                client_address, json_data_str=json_request, name="hydrogenate"
            )
            modified_atoms = [
                atoms_from_json(atoms_json) for atoms_json in response.json()["atoms"]
            ]
        else:
            modified_atoms = hydrogenate(
                run_settings, generation_calc, hydrogenation_calc
            )
        vis.extend(modified_atoms)
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        vis.play()


run_types = t.Union[Generate, Relax, Hydrogenate]


class DiffusionModelling(UpdateScene):
    """
    Click on `run type` to select the type of run to perform.\n
    The usual workflow is to first generate a structure, then hydrogenate it, and finally relax it.
    """

    discriminator: t.Literal["DiffusionModelling"] = "DiffusionModelling"
    run_type: run_types = Field(discriminator="discriminator")
    client_address: str = Field("http://127.0.0.1:5000/run")

    def run(self, vis: ZnDraw, calculators: dict | None = None) -> None:
        vis.log("Sending request to inference server.")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        if calculators is None:
            calculators = dict()
        vis.bookmarks = vis.bookmarks | {
            vis.step: f"Running {self.run_type.discriminator}"
        }
        self.run_type.run(
            vis=vis,
            client_address=self.client_address,
            calculators=calculators,
        )
        vis.append(remove_isolated_atoms_using_covalent_radii(vis[-1]))

    @staticmethod
    def get_documentation_url() -> str:
        return "https://rokasel.github.io/EnergyMolecularDiffusion"


class DiffusionModellingNoPort(UpdateScene):
    """
    Click on `run type` to select the type of run to perform.\n
    The usual workflow is to first generate a structure, then hydrogenate it, and finally relax it.
    """

    discriminator: t.Literal["DiffusionModellingNoPort"] = "DiffusionModellingNoPort"
    run_type: run_types = Field(discriminator="discriminator")

    def run(self, vis: ZnDraw, calculators: dict | None = None) -> None:
        vis.log("Sending request to inference server.")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        if calculators is None:
            raise ValueError("No calculators provided")
        vis.bookmarks = vis.bookmarks | {
            vis.step: f"Running {self.run_type.discriminator}"
        }
        self.run_type.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
        )
        vis.append(remove_isolated_atoms_using_covalent_radii(vis[-1]))

    @staticmethod
    def get_documentation_url() -> str:
        return "https://rokasel.github.io/EnergyMolecularDiffusion"
