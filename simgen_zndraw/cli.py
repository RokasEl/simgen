import logging
import pathlib
import time
from enum import Enum
from typing import Optional

import typer
import zntrack
from zndraw import ZnDraw
from zndraw.settings import GlobalConfig

from simgen.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
)
from simgen_zndraw import DefaultGenerationParams
from simgen_zndraw.main import DiffusionModellingNoPort

from .local_server import app
from .utils import get_default_mace_models_path

cli_app = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


@cli_app.command(help="Set the default path to the MACE-models repo")
def init(
    path: str = typer.Argument(..., help="Path to clone of MACE-models repo"),
    force: bool = typer.Option(False, help="Overwrite existing config file"),
    add_to_zndraw: bool = typer.Option(
        True, help="Add the DiffusionModelling class to the list of ZnDraw modifiers"
    ),
):
    print(f"Initializing SiMGen ZnDraw integration with the model path at {path}")

    config_path = "~/.simgen/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing SiMGen configuration at {config_path}")
        config = DefaultGenerationParams.from_file(config_path)  # type: ignore
    else:
        config = DefaultGenerationParams()

    current_default_path = config.default_model_path
    if current_default_path is not None and not force:
        print(f"Default model path is already set to {current_default_path}")
        print(f"If you want to change it, run the command with the --force flag")
    else:
        config.default_model_path = (
            pathlib.Path(path).expanduser().absolute().as_posix()
        )
        config.save()
        print(f"Saved configuration to {config_path}")

    if add_to_zndraw:
        print("Adding the DiffusionModelling class to the list of ZnDraw modifiers")
        config_path = "~/.zincware/zndraw/config.json"
        config_path = pathlib.Path(config_path).expanduser()
        if config_path.exists():
            print(f"Found an existing ZnDraw configuration at {config_path}")
            config = GlobalConfig.from_file(config_path)  # type: ignore
        else:
            config = GlobalConfig()

        pkg = "simgen_zndraw.main.DiffusionModelling"
        config.modify_functions = list(
            filter(
                lambda x: not "DiffusionModelling".lower() in x.lower(),
                config.modify_functions,
            )
        )
        config.modify_functions = [pkg] + config.modify_functions
        config.save()


@cli_app.command(help="Launch a local server for molecule generation")
def launch(
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to clone of MACE-models repo"
    ),
    mace_model_name: str = typer.Option(
        "small_spice", help="Name of MACE model to use"
    ),
    reference_data_name: str = typer.Option(
        "similarity_reference_data_small", help="Name of reference data to use"
    ),
    device: Device = typer.Option(Device.cpu),
    port: int = 5000,
):
    if path is None:
        path = get_default_mace_models_path()
    app.config["device"] = device.value
    app.config["mace_models_path"] = path
    app.config["mace_model_name"] = mace_model_name
    app.config["reference_data_name"] = reference_data_name
    url = f"http://127.0.0.1:{port}"
    logging.info(f"Starting generation server at {url}")
    app.run(port=port, host="0.0.0.0")


@cli_app.command(help="connect to a running ZnDraw instance")
def connect(
    url: str = typer.Option(
        "http://127.0.0.1:1234", help="URL of the ZnDraw instance to connect to"
    ),
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to clone of MACE-models repo"
    ),
    mace_model_name: str = typer.Option(
        "small_spice", help="Name of MACE model to use"
    ),
    reference_data_name: str = typer.Option(
        "simgen_reference_data_small", help="Name of reference data to use"
    ),
    add_linkers: bool = typer.Option(False, help="Add example linkers to the scene"),
    device: Device = typer.Option(Device.cpu),
):
    print("Loading models...")
    if path is None:
        path = get_default_mace_models_path()

    models = {
        "generation": get_mace_similarity_calculator(
            path,
            mace_model_name,
            reference_data_name,
            num_reference_mols=-1,
            device=device.value,
        ),
        "hydrogenation": get_hydromace_calculator(path, device=device.value),
    }
    print("Connecting to ZnDraw...")
    linkers_added = False
    vis = ZnDraw(url=url)
    if add_linkers and not linkers_added:
        linkers = zntrack.from_rev("linker_examples", path).get_atoms()
        vis.extend(linkers)
        linkers_added = True
    vis.register_modifier(
        DiffusionModellingNoPort, run_kwargs={"calculators": models}, default=True  # type: ignore
    )
    while vis.socket.connected:
        vis.socket.sleep(5)
    print("Connection lost, stopping...")


if __name__ == "__main__":
    cli_app()