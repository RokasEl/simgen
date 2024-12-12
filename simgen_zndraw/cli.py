import logging
import pathlib
from enum import Enum

import typer
import zntrack
from zndraw import ZnDraw

from simgen.utils import (
    get_hydromace_calculator,
    get_mace_similarity_calculator,
)
from simgen_zndraw import DefaultGenerationParams
from simgen_zndraw.main import SiMGen, SiMGenDemo

from .local_server import app
from .utils import get_default_mace_models_path

cli_app = typer.Typer()


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class SupportedModels(str, Enum):
    small_spice = "small_spice"
    medium_spice = "medium_spice"


@cli_app.command(help="Set the default path to the MACE-models repo")
def init(
    path: str = typer.Argument(..., help="Path to clone of MACE-models repo"),
    force: bool = typer.Option(False, help="Overwrite existing config file"),
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
        print("If you want to change it, run the command with the --force flag")
    else:
        config.default_model_path = (
            pathlib.Path(path).expanduser().absolute().as_posix()
        )
        config.save()
        print(f"Saved configuration to {config_path}")


@cli_app.command(help="Launch a local server for molecule generation")
def launch(
    path: str | None = typer.Option(
        None, "--path", help="Path to clone of MACE-models repo"
    ),
    mace_model_name: SupportedModels = typer.Option(
        SupportedModels.medium_spice, help="Name of MACE model to use"
    ),
    reference_data_name: str = typer.Option(
        "simgen_reference_data_small", help="Name of reference data to use"
    ),
    device: Device = typer.Option(Device.cpu),
    port: int = 5000,
):
    if path is None:
        path = get_default_mace_models_path()
    model_name = mace_model_name.value.split("_")[0]
    app.config["device"] = device.value
    app.config["mace_models_path"] = path
    app.config["mace_model_name"] = model_name
    app.config["reference_data_name"] = reference_data_name
    url = f"http://127.0.0.1:{port}"
    logging.info(f"Starting generation server at {url}")
    app.run(port=port, host="0.0.0.0")


@cli_app.command(help="connect to a running ZnDraw instance")
def connect(
    url: str = typer.Option(
        "http://127.0.0.1:1234", help="URL of the ZnDraw instance to connect to", envvar="SIMGEN_URL"
    ),
    path: str | None = typer.Option(
        None, "--path", help="Path to clone of MACE-models repo"
    ),
    mace_model_name: SupportedModels = typer.Option(
        SupportedModels.medium_spice, help="Name of MACE model to use"
    ),
    reference_data_name: str = typer.Option(
        "simgen_reference_data_small", help="Name of reference data to use"
    ),
    add_linkers: bool = typer.Option(False, help="Add example linkers to the scene"),
    auth_token: str | None = typer.Option(None, help="Authentication token", envvar="SIMGEN_AUTH_TOKEN"),
    device: Device = typer.Option(Device.cpu),
):
    logging.info("Loading models...")
    if path is None:
        path = get_default_mace_models_path()
    model_name = mace_model_name.value.split("_")[0]
    models = {
        "generation": get_mace_similarity_calculator(
            path,
            model_name,
            reference_data_name,
            num_reference_mols=-1,
            device=device.value,
        ),
        "hydrogenation": get_hydromace_calculator(path, device=device.value),
    }
    logging.info("Connecting to ZnDraw...")
    if add_linkers:
        linkers = zntrack.from_rev("linker_examples", path).frames
    else:
        linkers = []
    vis = ZnDraw(
        url=url,
        token="SIMGenModifier",
        auth_token=auth_token,
    )
    vis.timeout["modifier"] = 1.0
    vis.timeout["emit_retries"] = 5
    vis.timeout["call_retries"] = 5

    if add_linkers:
        vis.extend(linkers)
    vis.register_modifier(
        SiMGenDemo,
        run_kwargs={"calculators": models},
        public=True,  # type: ignore
    )
    vis.register_modifier(SiMGen, run_kwargs={"calculators": models}, public=True)
    logging.info("All modifiers registered. Waiting for requests...")
    vis.socket.wait()


if __name__ == "__main__":
    cli_app()
