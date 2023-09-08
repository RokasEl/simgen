from dataclasses import dataclass

from moldiff.element_swapping import SwappingAtomicNumberTable
from moldiff.integrators import IntegrationParameters


@dataclass()
class DefaultGenerationParams:
    prior_beta: float = 5.0
    swapping_table = SwappingAtomicNumberTable([6, 7, 8], [1, 1, 1])
    num_particles: int = 10
    particle_swap_frequency: int = 4


DEFAULT_INTEGRATION_PARAMS = IntegrationParameters(S_churn=1.3, S_min=2e-3, S_noise=0.5)
DEFAULT_GENERATION_PARAMS = DefaultGenerationParams()
