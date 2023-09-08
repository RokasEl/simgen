import logging

from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from moldiff.utils import (
    get_hydromace_calculator,
    get_mace_config,
    get_mace_similarity_calculator,
    setup_logger,
)

from . import endpoints
from .data import parse_request

setup_logger(level=logging.INFO)

app = FastAPI()


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 16},
    autoscaling_config={"min_replicas": 0, "max_replicas": 1},
)
@serve.ingress(app)
class GenerationServer:
    def __init__(
        self,
        mace_models_path,
        device="cpu",
    ) -> None:
        self.moldiff_calc = get_mace_similarity_calculator(
            mace_models_path, num_reference_mols=-1, device=device
        )
        self.hydromace_calc = get_hydromace_calculator(mace_models_path, device=device)
        self.device = device

    @app.post("/")
    async def run(self, request: Request) -> dict:
        raw_data = await request.json()
        formatted_request = parse_request(raw_data)
        endpoint_name = formatted_request.run_type
        endpoint = getattr(endpoints, endpoint_name)
        return endpoint(formatted_request, self.moldiff_calc, self.hydromace_calc)

    @app.get("/config")
    def get_model_config(
        self,
    ) -> dict:
        return get_mace_config(self.moldiff_calc.model)
