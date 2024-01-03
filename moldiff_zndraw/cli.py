import logging
import pathlib
from enum import Enum
from typing import Optional

import typer
from zndraw.settings import GlobalConfig

from .local_server import app
from .utils import get_default_mace_models_path

cli = typer.Typer()


@cli.command()
def init(path: str = typer.Argument(..., help="Path to clone of MACE-models repo")):
    print(f"Initializing moldiff ZnDraw integration with the model path at {path}")

    config_path = "~/.zincware/zndraw/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing configuration at {config_path}")
        config = GlobalConfig.from_file(config_path)  # type: ignore
    else:
        config = GlobalConfig()
    pkg = "moldiff_zndraw.main.DiffusionModelling"
    config.modify_functions = list(
        filter(
            lambda x: not "DiffusionModelling".lower() in x.lower(),
            config.modify_functions,
        )
    )
    config.modify_functions.append(pkg)
    path = pathlib.Path(path).expanduser().absolute()
    moldiff_settings = {
        "path": path.as_posix(),
    }
    config.function_schema["moldiff_zndraw.main.DiffusionModelling"] = moldiff_settings
    config.save()
    print(f"Saved configuration to {path}")


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


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


if __name__ == "__main__":
    cli()
