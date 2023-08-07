import argparse
import pathlib

from zndraw.settings import GlobalConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_model_path", type=str, required=True)
    args = parser.parse_args()

    config_path = "~/.zincware/zndraw/config.json"
    path = pathlib.Path(config_path).expanduser()
    if path.exists():
        print(f"Found an existing configuration at {path}")
        config = GlobalConfig.from_file(path)  # type: ignore
    else:
        config = GlobalConfig()
    pkg = "moldiff_zndraw.main.MoldiffGeneration"
    if pkg not in config.modify_functions:
        config.modify_functions.append(pkg)
    moldiff_settings = {
        "model_repo_path": args.default_model_path,
    }
    config.function_schema["moldiff_zndraw.main.MoldiffGeneration"] = moldiff_settings
    config.save()
    print(f"Saved configuration to {path}")
