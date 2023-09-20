import logging
from enum import Enum

import typer
from zndraw.utils import get_port

from moldiff_server.main import app

cli = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


@cli.command()
def launch(
    path: str = typer.Argument(..., help="Path to clone of MACE-models repo"),
    device: Device = typer.Option(Device.cpu),
    port: int = 5000,
):
    print(app)
    app.config["device"] = device.value
    app.config["mace_models_path"] = path
    url = f"http://127.0.0.1:{port}"
    logging.info(f"Starting generation server at {url}")
    app.run(port=port, host="0.0.0.0")
