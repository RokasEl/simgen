from enum import Enum

import ray
import typer
from ray import serve

from .ray_app import GenerationServer

app = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


@app.command()
def launch_server(
    path: str = typer.Argument(..., help="Path to clone of MACE-models repo"),
    device: Device = typer.Option(Device.cpu),
):
    ray.init(
        address="auto",
        namespace="serve-example",
        ignore_reinit_error=True,
    )
    serve.start(detached=True)
    GenerationServer.deploy(path, device)
