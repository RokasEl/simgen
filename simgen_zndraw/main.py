import logging
import typing as t
from typing import Any

import ase
import numpy as np
import requests
from pydantic import BaseModel, ConfigDict, Field
from zndraw.modify import UpdateScene
from zndraw.zndraw_frozen import ZnDrawFrozen as ZnDraw

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
from .utils import get_anchor_point_positions

setup_logger(directory="./logs", tag="simgen_zndraw", level=logging.INFO)


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
        ge=0.5,
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

    def run(self, vis: ZnDraw, client_address, calculators: dict, timeout) -> ase.Atoms:
        vis.log("Running Generation")
        logging.debug("Reached Generate run method")
        run_specific_settings = self._get_run_specific_settings(vis)
        run_settings = format_run_settings(
            vis, **run_specific_settings, timeout=timeout
        )
        logging.debug("Formated run settings; vis.atoms was accessed")
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
            e = "Exceptions currently not sent via http"
        else:
            logging.debug("Calling generate function")
            modified_atoms, e = generate(run_settings, generation_calc)
        if len(modified_atoms) == 0:
            logging.error(f"Generation did not return any atoms. Error: {e}")
            vis.log(
                f"Generation did not return any atoms. Error: {e}. Please try again."
            )
        else:
            logging.debug("Generate function returned, adding atoms to vis")
            vis.log(f"Received back {len(modified_atoms)} atoms.")
            modified_atoms.append(
                remove_isolated_atoms_using_covalent_radii(modified_atoms[-1])
            )
            vis.extend(modified_atoms)
            return modified_atoms[-1]

    def _get_run_specific_settings(self, vis: ZnDraw) -> dict:
        points = self._handle_points(vis.points, vis.segments)
        if points is None:
            if len(vis.selection) <= 1:
                logging.info("No location provided, will generate at origin")
                points = np.array([[0.0, 0.0, 0.0]])
            else:
                points = get_anchor_point_positions(
                    vis.atoms, vis.selection, vis.camera
                )
                vis.points = points
                vis.selection = []
                points = interpolate_points(points, 100)

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
    def _handle_points(points, segments) -> np.ndarray | None:
        if points.size == 0:
            return None
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
            if stripped_points.shape[0] == 0:
                logging.info(
                    "All guiding points are close to existing atoms. Consider drawing a different path."
                )
                return points
            else:
                return stripped_points

    @staticmethod
    def _get_num_atoms_to_add(
        points: np.ndarray,
        atom_number_determination_type: str,
        atom_parameter_value: int | float,
    ) -> int:
        logging.debug(
            f"Getting how many atoms to add {atom_number_determination_type}, {atom_parameter_value}"
        )
        if atom_number_determination_type == "FixedNumber":
            return int(atom_parameter_value)
        elif atom_number_determination_type == "PerAngstrom":
            logging.info(
                "Calculating number of atoms to add based on curve length and density"
            )
            if points.shape[0] <= 1:
                raise ValueError(
                    "`PerAngstrom` requires at least 2 points drawn in the interface"
                )
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

    def run(self, vis: ZnDraw, client_address, calculators, timeout) -> ase.Atoms:
        vis.log("Running Relax")
        logging.debug("Reached Relax run method")
        run_settings = format_run_settings(
            vis, run_type="relax", max_steps=self.max_steps, timeout=timeout
        )
        if run_settings.atoms is None or len(run_settings.atoms) == 0:
            vis.log("No atoms to relax")
            return
        logging.debug("Formated run settings; vis.atoms was accessed")
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
            logging.debug("Calling relax function")
            modified_atoms, _ = relax(run_settings, generation_calc)
        logging.debug("Relax function returned, adding atoms to vis")
        modified_atoms.append(
            remove_isolated_atoms_using_covalent_radii(modified_atoms[-1])
        )
        vis.extend(modified_atoms)
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        return modified_atoms[-1]


class Hydrogenate(UpdateScene):
    discriminator: t.Literal["Hydrogenate"] = Field("Hydrogenate")
    max_steps: int = Field(30, ge=1)

    def run(self, vis: ZnDraw, client_address, calculators, timeout) -> ase.Atoms:
        logging.debug("Reached Hydrogenate run method")
        vis.log("Running Hydrogenate")
        run_settings = format_run_settings(
            vis, run_type="hydrogenate", max_steps=self.max_steps, timeout=timeout
        )
        if run_settings.atoms is None or len(run_settings.atoms) == 0:
            vis.log("No atoms to hydrogenate")
            return
        run_settings.atoms = self._check_and_remove_existing_hydrogen_atoms(
            vis, run_settings.atoms
        )
        logging.debug("Formated run settings; vis.atoms was accessed")
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
            logging.debug("Calling hydrogenate function")
            modified_atoms, _ = hydrogenate(
                run_settings, generation_calc, hydrogenation_calc
            )
        logging.debug("Hydrogenate function returned, adding atoms to vis")
        modified_atoms.append(
            remove_isolated_atoms_using_covalent_radii(modified_atoms[-1])
        )
        vis.extend(modified_atoms)
        vis.log(f"Received back {len(modified_atoms)} atoms.")
        return modified_atoms[-1]

    @staticmethod
    def _check_and_remove_existing_hydrogen_atoms(
        vis: ZnDraw, atoms: ase.Atoms
    ) -> ase.Atoms:
        numbers = atoms.get_atomic_numbers()
        atoms_are_hydrogen = numbers == 1
        if any(atoms_are_hydrogen):
            vis.log(
                "Removing existing hydrogen atoms. Currently partial hydrogenation is not supported."
            )
            del atoms[atoms_are_hydrogen]
        return atoms


run_types = t.Union[Generate, Hydrogenate, Relax]


class DiffusionModelling(UpdateScene):
    """
    Click on `run type` to select the type of run to perform.\n
    The usual workflow is to first generate a structure, then hydrogenate it, and finally relax it.
    """

    discriminator: t.Literal["DiffusionModelling"] = "DiffusionModelling"
    run_type: run_types = Field(discriminator="discriminator")
    client_address: str = Field("http://127.0.0.1:5000/run")

    def run(
        self, vis: ZnDraw, calculators: dict | None = None, timeout: float = 60
    ) -> None:
        logging.debug("-" * 72)
        vis.log("Sending request to inference server.")
        logging.debug(f"Vis token: {vis.token}")
        logging.debug("Accessing vis and vis.step for the first time")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        if calculators is None:
            raise ValueError("No calculators provided")
        logging.debug("Accessing vis.bookmarks")
        vis.bookmarks = vis.bookmarks | {
            vis.step: f"Running {self.run_type.discriminator}"
        }
        self.run_type.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
            timeout=timeout,
        )
        logging.debug("Accessing vis.append when removing isolated atoms")
        logging.debug("-" * 72)


class SiMGen(UpdateScene):
    """
    Click on `run type` to select the type of run to perform.\n
    The usual workflow is to first generate a structure, then hydrogenate it, and finally relax it.
    """

    discriminator: t.Literal["SiMGen"] = "SiMGen"
    run_type: run_types = Field(discriminator="discriminator")

    def run(self, vis: ZnDraw, calculators: dict | None = None, **kwargs) -> None:
        logging.debug("-" * 72)
        vis.log("Sending request to inference server.")
        logging.debug(f"Vis token: {vis.token}")
        logging.debug("Accessing vis and vis.step for the first time")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        if calculators is None:
            raise ValueError("No calculators provided")
        logging.debug("Accessing vis.bookmarks")
        vis.bookmarks = vis.bookmarks | {
            vis.step: f"Running {self.run_type.discriminator}"
        }
        timeout = kwargs.get("timeout", 60)
        self.run_type.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
            timeout=timeout,
        )
        logging.debug("-" * 72)


def _format_fields(schema, cls):
    return cls._update_schema(schema)


class SiMGenDemo(UpdateScene):
    """
    Demo of SiMGen. Generates a structure, hydrogenates it, and relaxes it.
    See the tutorial for more information.
    """

    discriminator: t.Literal["SiMGenDemo"] = "SiMGenDemo"
    atoms_per_angstrom: float = Field(
        1.2,
        ge=0.8,
        le=2.0,
        description="Num atoms added = atoms_per_angstrom * curve_length",
    )
    guiding_force_multiplier: float = Field(
        1.5,
        ge=1.0,
        le=10.0,
        description="Multiplier for guiding force. Increase if molecules falls apart.",
    )

    # model_config = ConfigDict(json_schema_extra=_format_fields) # Not working on ZnDraw side yet

    def run(self, vis: ZnDraw, calculators: dict | None = None, **kwargs) -> None:
        vis.log("Sending request to inference server.")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        if calculators is None:
            raise ValueError("No calculators provided")
        bookmarks = vis.bookmarks.copy()
        bookmarks = bookmarks | {int(vis.step): f"SiMGen: Generating a structure."}
        vis.bookmarks = bookmarks
        timeout = kwargs.get("timeout", 60)
        gen_class = Generate(
            discriminator="Generate",
            num_steps=50,
            atom_number=PerAngstrom(atoms_per_angstrom=self.atoms_per_angstrom),
            guiding_force_multiplier=self.guiding_force_multiplier,
        )
        atoms = gen_class.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
            timeout=timeout,
        )

        vis._cached_data["atoms"] = atoms
        bookmarks = bookmarks | {len(vis): f"SiMGen: Hydrogenating the structure."}
        vis.bookmarks = bookmarks
        hydrogenate_class = Hydrogenate(
            discriminator="Hydrogenate",
            max_steps=30,
        )
        atoms = hydrogenate_class.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
            timeout=timeout,
        )

        vis._cached_data["atoms"] = atoms
        vis._cached_data["selection"] = []
        bookmarks = bookmarks | {len(vis): f"SiMGen: Relaxing the structure."}
        vis.bookmarks = bookmarks
        relax_class = Relax(
            discriminator="Relax",
            max_steps=50,
        )
        relax_class.run(
            vis=vis,
            client_address=None,
            calculators=calculators,
            timeout=timeout,
        )
        vis.log("Finished running SiMGen demo.")

    @classmethod
    def _update_schema(cls, schema: dict) -> dict:
        schema["properties"]["atoms_per_angstrom"]["format"] = "range"
        schema["properties"]["guiding_force_multiplier"]["format"] = "range"
        return schema
