import pathlib

import typer
from zndraw.settings import GlobalConfig

app = typer.Typer()


@app.command()
def init(path: str = typer.Argument(..., help="Path to clone of MACE-models repo")):
    print(f"Initializing moldiff_zndraw with the model path at {path}")

    config_path = "~/.zincware/zndraw/config.json"
    config_path = pathlib.Path(config_path).expanduser()
    if config_path.exists():
        print(f"Found an existing configuration at {config_path}")
        config = GlobalConfig.from_file(config_path)  # type: ignore
    else:
        config = GlobalConfig()
    pkg = "moldiff_zndraw.main.MoldiffGeneration"
    config.modify_functions = list(
        filter(
            lambda x: not "MoldiffGeneration".lower() in x.lower(),
            config.modify_functions,
        )
    )
    config.modify_functions.append(pkg)
    path = pathlib.Path(path).expanduser().absolute()
    moldiff_settings = {
        "model_repo_path": path.as_posix(),
    }
    config.function_schema["moldiff_zndraw.main.MoldiffGeneration"] = moldiff_settings
    config.save()
    print(f"Saved configuration to {path}")


@app.command()
def readme():
    print("This is a CLI for moldiff_zndraw")


if __name__ == "__main__":
    app()
