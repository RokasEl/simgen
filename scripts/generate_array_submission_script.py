from itertools import product

import fire


def main(param_save_file, save_path, molecule_str: str, data_path="../data/qm9/", num_steps=1000):
    molecules = molecule_str.split("-")
    langevin_temp = [0.005, 0.01, 0.1]
    rng_seeds = [0, 1, 2]
    with open(param_save_file, "w") as f:
        for molecule, temperature, rng_seed in product(
            molecules, langevin_temp, rng_seeds
        ):
            traj_path = f"{save_path}/{molecule}_{temperature}_{rng_seed}.xyz"
            f.write(f"{molecule};{temperature};{rng_seed};{traj_path};{data_path};{num_steps}\n")


if __name__ == "__main__":
    fire.Fire(main)
