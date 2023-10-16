import logging
import pathlib
from enum import Enum
from typing import Optional

import typer
from zndraw.settings import GlobalConfig

from moldiff_server.main import app

cli = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


def get_default_mace_models_path() -> str:
    config_path = "~/.zincware/zndraw/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing configuration at {config_path}")
        config = GlobalConfig.from_file(config_path)  # type: ignore
        mace_models_path = config.function_schema[
            "moldiff_zndraw.main.DiffusionModelling"
        ]["path"]
        return mace_models_path
    else:
        raise ValueError(
            "Could not find a config file at ~/.zincware/zndraw/config.json, specify the path to the MACE-models repo with --path"
        )


@cli.command()
def launch(
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to clone of MACE-models repo"
    ),
    device: Device = typer.Option(Device.cpu),
    port: int = 5000,
):
    if path is None:
        path = get_default_mace_models_path()
    app.config["device"] = device.value
    app.config["mace_models_path"] = path
    url = f"http://127.0.0.1:{port}"
    logging.info(f"Starting generation server at {url}")
    app.run(port=port, host="0.0.0.0")
