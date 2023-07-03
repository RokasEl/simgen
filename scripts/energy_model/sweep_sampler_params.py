import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from generate_mols_energy_model import (
    main as generate_mols_energy_model,
)

from energy_model.diffusion_tools import SamplerNoiseParameters


def main():
    rng = np.random.default_rng(0)
    sigma_max_gen = lambda: rng.uniform(low=2.5, high=10.0)
    sigma_min_gen = lambda: 10 ** rng.uniform(low=-4, high=-3)
    s_churn_gen = lambda: rng.uniform(low=1, high=80)
    s_min_gen = lambda x: np.maximum(x, 10 ** rng.uniform(low=-3, high=0))
    s_noise_gen = lambda: rng.uniform(low=0.1, high=2.0)

    save_path = Path("./results/energy_model/sweep_sampler_params/")
    save_path.mkdir(parents=True, exist_ok=True)
    params = {}
    for i in range(100):
        sigma_min = sigma_min_gen()
        sampler_params = SamplerNoiseParameters(
            sigma_max=sigma_max_gen(),
            sigma_min=sigma_min,
            S_churn=s_churn_gen(),
            S_min=s_min_gen(sigma_min),
            S_noise=s_noise_gen(),
        )
        params[i] = asdict(sampler_params)
        this_save_path = str(save_path / f"sampler_params_{i}.xyz")
        generate_mols_energy_model(
            model_path="./trained_energy_mace_iddpm_deeper_readouts.pt",
            sampler_params=sampler_params,
            num_samples_per_size=100,
            save_path=this_save_path,
            batch_size=256,
        )
        with open(save_path / "params.json", "w") as f:
            json.dump(params, f, indent=4)


if __name__ == "__main__":
    main()
