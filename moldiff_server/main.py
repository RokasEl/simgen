import json
import logging

from flask import Flask, Response, jsonify, request

from moldiff.utils import (
    get_hydromace_calculator,
    get_mace_config,
    get_mace_similarity_calculator,
    setup_logger,
)

from . import endpoints
from .data import parse_request
from .utils import make_mace_config_jsonifiable

app = Flask(__name__)
setup_logger()

models = {}


def _moldiff_factory():
    return get_mace_similarity_calculator(
        app.config["mace_models_path"],
        num_reference_mols=-1,
        device=app.config["device"],
    )


def _hydromace_factory():
    return get_hydromace_calculator(
        app.config["mace_models_path"], device=app.config["device"]
    )


def get_models():
    try:
        moldiff_calc = models.get("moldiff_calc", _moldiff_factory())
        hydromace_calc = models.get("hydromace_calc", _hydromace_factory())
        models["moldiff_calc"] = moldiff_calc
        models["hydromace_calc"] = hydromace_calc
        return moldiff_calc, hydromace_calc
    except KeyError as e:
        logging.error("Could not get the model due to missing app config")
        raise e
    except Exception as e:
        logging.error("Error trying to get model")
        raise e


@app.route("/run", methods=["POST"])
def run():
    raw_data = json.loads(request.data)
    formatted_request = parse_request(raw_data)
    endpoint_name = formatted_request.run_type
    endpoint = getattr(endpoints, endpoint_name)
    moldiff_calc, hydromace_calc = get_models()
    logging.info(f"Received request: {formatted_request.run_type}.\nRunning...")
    results = endpoint(formatted_request, moldiff_calc, hydromace_calc)
    logging.info("Completed. Sending back the response.")
    return results


@app.route("/config", methods=["GET"])
def get_config() -> Response:
    moldiff_calc, _ = get_models()
    mace_config = get_mace_config(moldiff_calc.model)
    return jsonify(make_mace_config_jsonifiable(mace_config))