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


@app.command()
def test_repulsive_block():
    import torch

    from moldiff.element_swapping import SwappingAtomicNumberTable
    from moldiff.generation_utils import (
        ExponentialRepulsionBlock,
        batch_atoms,
    )
    from moldiff.utils import get_system_torch_device_str

    device = get_system_torch_device_str()
    if device == "mps":
        torch.set_default_dtype(torch.float32)

    repulsion_block = ExponentialRepulsionBlock(alpha=8.0).to(device)
    from moldiff.utils import initialize_mol

    mol = initialize_mol("C6H6")
    z_table = SwappingAtomicNumberTable([1, 6, 7, 8], [1, 1, 1, 1])
    batched = batch_atoms([mol.copy(), mol.copy()], z_table, cutoff=5, device=device)
    energies = repulsion_block(batched)
    print(energies, energies.shape, energies.dtype)
    print("success!")


if __name__ == "__main__":
    app()
