import json
import pathlib
import typing as t
from dataclasses import dataclass

from simgen.element_swapping import SwappingAtomicNumberTable
from simgen.integrators import IntegrationParameters


@dataclass()
class DefaultGenerationParams:
    prior_beta: float = 5.0
    swapping_table_zs: t.Tuple[int, ...] = (6, 7, 8)
    swapping_table_freqs: t.Tuple[int, ...] = (1, 1, 1)
    num_particles: int = 10
    particle_swap_frequency: int = 4
    default_model_path: str = "https://github.com/RokasEl/MACE-Models"

    @property
    def swapping_table(self):
        return SwappingAtomicNumberTable(
            self.swapping_table_zs, self.swapping_table_freqs
        )

    @classmethod
    def from_file(cls, path="~/.simgen/config.json"):
        load_path = pathlib.Path(path).expanduser()
        with open(load_path, "r") as f:
            return cls(**json.load(f))

    @classmethod
    def load(cls):
        if pathlib.Path("~/.simgen/config.json").expanduser().exists():
            return cls.from_file()
        else:
            return cls()

    def save(self, path="~/.simgen/config.json"):
        save_path = pathlib.Path(path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(json.dumps(self.__dict__, indent=4))


DEFAULT_INTEGRATION_PARAMS = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
DEFAULT_GENERATION_PARAMS = DefaultGenerationParams.load()
