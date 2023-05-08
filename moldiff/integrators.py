import logging

import torch
from mace.data import AtomicData

from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.manifolds import PriorManifold


class HeunIntegrator:
    def __init__(
        self,
        similarity_calculator,
        guiding_manifold: PriorManifold,
        sampler_noise_parameters=SamplerNoiseParameters(),
        restorative_force_strength: float = 1.5,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        self.prior_manifold = guiding_manifold
        self.similarity_calculator = similarity_calculator
        self.noise_parameters = sampler_noise_parameters
        self.restorative_force_strength = restorative_force_strength
        self.device = device

    def __call__(self, x: AtomicData, step: int, sigma_cur, sigma_next, mask=None):
        """
        This does NOT update the neighbout list. DANGER!
        """
        S_churn, S_min, S_max, S_noise = self._get_integrator_parameters()
        mol_cur = x.clone()
        if mask is None:
            mask = torch.ones(len(x)).to(self.device)
        # mol_cur.positions.requires_grad = True

        # If current sigma is between S_min and S_max, then we first temporarily increase the current noise leve.
        gamma = S_churn if S_min <= sigma_cur <= S_max else 1
        # Added noise depends on the current noise level. So, it decreases over the course of the integration.
        sigma_increased = sigma_cur * gamma
        # Add noise to the current sample.
        noise_level = torch.sqrt(sigma_increased**2 - sigma_cur**2) * S_noise
        noise_level = torch.max(noise_level, torch.tensor(1e-2).to(self.device))
        logging.debug(f"Current step: {step}")
        logging.debug(f"Noise added to positions: {noise_level:.2e}")
        with torch.no_grad():
            mol_cur.positions += (
                torch.randn_like(mol_cur.positions) * noise_level * mask
            )

        mol_increased = mol_cur
        device = mol_increased.positions.device
        # Euler step.
        # mol_increased.positions.requires_grad = True
        mol_increased.positions.grad = None
        forces = self.similarity_calculator(mol_increased, sigma_increased)
        forces = torch.tensor(forces, device=device)
        restorative_forces = self.prior_manifold.calculate_resorative_forces(
            mol_increased.positions
        )

        forces += (
            self.restorative_force_strength
            * restorative_forces
            * torch.tanh(20 * sigma_cur**2)
        )
        forces *= mask
        mol_next = mol_cur.clone()
        logging.debug(f"Step size = {abs(sigma_next - sigma_increased):.2e}")
        with torch.no_grad():
            mol_next.positions += -1 * (sigma_next - sigma_increased) * forces

        # Apply 2nd order correction.
        if sigma_next != 0:
            mol_next.positions.grad = None

            forces_next = self.similarity_calculator(mol_next, sigma_next)
            forces_next = torch.tensor(forces_next, device=device) * mask

            mol_next = mol_increased.clone()
            with torch.no_grad():
                mol_next.positions += -1 * (
                    (sigma_next - sigma_increased) * (forces + forces_next) / 2
                )
        return mol_next

    def _get_integrator_parameters(self):
        return (
            self.noise_parameters.S_churn,
            self.noise_parameters.S_min,
            self.noise_parameters.S_max,
            self.noise_parameters.S_noise,
        )
