import logging
from dataclasses import dataclass

import torch
from mace.data import AtomicData

from simgen.manifolds import PriorManifold
from simgen.utils import get_system_torch_device_str

# if get_system_torch_device_str() == "mps":
#     torch.set_default_dtype(torch.float32)
# else:
#     torch.set_default_dtype(torch.float64)


@dataclass
class IntegrationParameters:
    S_churn: float = (
        0.0  # Churn rate of the noise level. 0 corresponds to an ODE solver.
    )
    S_min: float = 0.0  # S_min and S_max define the noise level range, in which additional noise is added to the positions.
    S_max: float = float("inf")
    S_noise: float = 1  # The calculator sees the current noise level as `sigma_cur * S_churn`, whereas the noise added to the positions is `sigma_cur * S_churn * S_noise`.
    min_noise: float = 1e-2  # Minimum noise level. If not zero, then `min_noise` is added to the positions even after `sigma_cur` has dropped below `S_min`.


class HeunIntegrator:
    def __init__(
        self,
        similarity_calculator,
        guiding_manifold: PriorManifold,
        integration_parameters=IntegrationParameters(),
        restorative_force_strength: float = 1.5,
        device=get_system_torch_device_str(),
    ):
        self.prior_manifold = guiding_manifold
        self.similarity_calculator = similarity_calculator
        self.integration_parameters = integration_parameters
        self.restorative_force_strength = restorative_force_strength
        self.device = device

    def __call__(self, x: AtomicData, step: int, sigma_cur, sigma_next, mask=None):
        """
        This does NOT update the neighbout list. DANGER!
        """
        S_churn, S_min, S_max, S_noise, min_noise = self._get_integrator_parameters()
        mol_cur = x
        if mask is None:
            mask = torch.ones(
                mol_cur.positions.shape[0],
                device=self.device,
                dtype=mol_cur.positions.dtype,
            )
        # If current sigma is between S_min and S_max, then we first temporarily increase the current noise leve.
        gamma = S_churn if S_min <= sigma_cur <= S_max else 1
        # Added noise depends on the current noise level. So, it decreases over the course of the integration.
        sigma_increased = sigma_cur * gamma
        # Add noise to the current sample.
        noise_level = torch.sqrt(sigma_increased**2 - sigma_cur**2) * S_noise
        noise_level = torch.max(noise_level, min_noise)
        logging.debug(f"Current step: {step}")
        logging.debug(f"Noise added to positions: {noise_level:.2e}")
        with torch.no_grad():
            mol_cur.positions += (
                torch.randn_like(mol_cur.positions) * noise_level * mask[:, None]
            )
        noised_positions = mol_cur.positions.clone().detach()
        mol_increased = mol_cur
        # Euler step.
        mol_increased.positions.grad = None
        forces = self.similarity_calculator(mol_increased, sigma_increased)
        restorative_forces = self.prior_manifold.calculate_resorative_forces(
            mol_increased.positions
        )
        forces += (
            self.restorative_force_strength
            * restorative_forces
            * torch.tanh(20 * sigma_cur**2)
        )
        forces *= mask[:, None]
        mol_next = mol_cur
        logging.debug(f"Step size = {abs(sigma_next - sigma_increased):.2e}")
        with torch.no_grad():
            mol_next.positions += -1 * (sigma_next - sigma_increased) * forces
        # Apply 2nd order correction.
        if sigma_next != 0:
            mol_next.positions.grad = None

            forces_next = self.similarity_calculator(mol_next, sigma_next)
            restorative_forces = self.prior_manifold.calculate_resorative_forces(
                mol_next.positions
            )
            forces_next += (
                self.restorative_force_strength
                * restorative_forces
                * torch.tanh(20 * sigma_next**2)
            )
            forces_next *= mask[:, None]
            with torch.no_grad():
                mol_next.positions = noised_positions - 1 * (
                    (sigma_next - sigma_increased) * (forces + forces_next) / 2
                )
        return mol_next

    def _get_integrator_parameters(self):
        return (
            self.integration_parameters.S_churn,
            self.integration_parameters.S_min,
            self.integration_parameters.S_max,
            self.integration_parameters.S_noise,
            torch.tensor(self.integration_parameters.min_noise)
            .to(self.device)
            .to(self.similarity_calculator.dtype),
        )
