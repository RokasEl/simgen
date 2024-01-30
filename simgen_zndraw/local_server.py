import logging

from flask import Flask, Response, jsonify, request

from simgen.utils import (
    get_hydromace_calculator,
    get_mace_config,
    get_mace_similarity_calculator,
    setup_logger,
)

from . import endpoints
from .data import jsonify_atoms, settings_from_json
from .utils import make_mace_config_jsonifiable

app = Flask(__name__)
setup_logger()

models = {}


def _simgen_factory():
    return get_mace_similarity_calculator(
        app.config["mace_models_path"],
        app.config["mace_model_name"],
        app.config["reference_data_name"],
        num_reference_mols=-1,
        device=app.config["device"],
    )


def _hydromace_factory():
    return get_hydromace_calculator(
        app.config["mace_models_path"], device=app.config["device"]
    )


def get_models():
    try:
        simgen_calc = models.get("simgen_calc", _simgen_factory())
        hydromace_calc = models.get("hydromace_calc", _hydromace_factory())
        models["simgen_calc"] = simgen_calc
        models["hydromace_calc"] = hydromace_calc
        return simgen_calc, hydromace_calc
    except KeyError as e:
        logging.error("Could not get the model due to missing app config")
        raise e
    except Exception as e:
        logging.error("Error trying to get model")
        raise e


@app.route("/run", methods=["POST"])
def run():
    raw_data = request.data
    formatted_request = settings_from_json(raw_data)
    endpoint_name = formatted_request.run_type
    endpoint = getattr(endpoints, endpoint_name)
    simgen_calc, hydromace_calc = get_models()
    logging.info(f"Received request: {formatted_request.run_type}.\nRunning...")
    results, _ = endpoint(formatted_request, simgen_calc, hydromace_calc)
    results = jsonify_atoms(*results)
    logging.info("Completed. Sending back the response.")
    return results


@app.route("/config", methods=["GET"])
def get_config() -> Response:
    simgen_calc, _ = get_models()
    mace_config = get_mace_config(simgen_calc.model)
    return jsonify(make_mace_config_jsonifiable(mace_config))
