import time

import torch
from zndraw import ZnDraw

from moldiff.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
)
from moldiff_zndraw.main import DiffusionModelling
from moldiff_zndraw.utils import get_default_mace_models_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
models = {}


def _moldiff_factory():
    return get_mace_similarity_calculator(
        get_default_mace_models_path(),
        num_reference_mols=-1,
        device=DEVICE,
    )


def _hydromace_factory():
    return get_hydromace_calculator(get_default_mace_models_path(), device=DEVICE)


def get_models():
    try:
        moldiff_calc = models.get("moldiff_calc", _moldiff_factory())
        hydromace_calc = models.get("hydromace_calc", _hydromace_factory())
        models["generation"] = moldiff_calc
        models["hydrogenation"] = hydromace_calc
        return moldiff_calc, hydromace_calc
    except KeyError as e:
        print("Could not get the model due to missing app config")
        raise e
    except Exception as e:
        print("Error trying to get model")
        raise e


get_models()
print("Starting server...")
while True:
    vis = ZnDraw(url="http://127.0.0.1:1234/")
    vis.register_modifier(
        DiffusionModelling, run_kwargs={"calculators": models}, default=True
    )
    while vis.socket.connected:
        time.sleep(5)
    print("Connection lost, stopping...")
