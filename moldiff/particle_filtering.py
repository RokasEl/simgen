import logging
import pprint
from functools import partial
from typing import List

import ase
import numpy as np
import torch
from mace.data import AtomicData
from mace.tools import AtomicNumberTable

from moldiff.atoms_cleanup import cleanup_atoms
from moldiff.calculators import MaceSimilarityCalculator
from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import (
    collect_particles,
    create_element_swapped_particles,
)
from moldiff.generation_utils import (
    batch_atoms,
    duplicate_atoms,
    get_atoms_from_batch,
)
from moldiff.temperature_annealing import ExponentialThermostat

logger = logging.getLogger(__name__)


class ParticleFilterGenerator:
    def __init__(
        self,
        similarity_calculator: MaceSimilarityCalculator,
        num_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        restorative_force_strength: float = 1.5,
        noise_params=SamplerNoiseParameters(),
    ):
        self.similarity_calculator = similarity_calculator

        self.z_table = AtomicNumberTable(
            [int(z) for z in similarity_calculator.model.atomic_numbers]
        )
        self.device = device
        self.noise_parameters = noise_params
        self.sigmas = self._get_sigma_schedule(num_steps=num_steps)
        self.sigmas = torch.concatenate(
            [
                torch.linspace(1, 0.05, 50),
                torch.logspace(-1.31, -3, 50),
            ]
        ).to(device)
        num_steps = len(self.sigmas)
        self.integrator = HeunIntegrator(
            similarity_calculator=similarity_calculator,
            num_steps=num_steps,
            sampler_noise_parameters=noise_params,
            restorative_force_strength=restorative_force_strength,
        )
        self.thermostat = ExponentialThermostat(
            initial_T_log_10=6, final_T_log_10=-1, sigma_max=self.sigmas[0]
        )
        self.num_steps = num_steps
        self.batch_atoms = partial(
            batch_atoms,
            z_table=self.z_table,
            device=self.device,
            cutoff=self.similarity_calculator.cutoff,
        )

    def generate(
        self,
        molecule: ase.Atoms,
        swapping_z_table: AtomicNumberTable,
        num_particles: int = 10,
        particle_swap_frequency: int = 1,
        do_final_cleanup: bool = True,
    ):
        # initialise mol
        molecule.positions = (
            np.random.randn(*molecule.positions.shape) * self.sigmas[0].item()
        )
        trajectories = [molecule]

        atoms = [duplicate_atoms(molecule)]
        batched = self.batch_atoms(atoms)
        self.swapped = False
        for step in range(self.num_steps - 1):
            sigma_cur, sigma_next = self.sigmas[step], self.sigmas[step + 1]
            if step % particle_swap_frequency == 0 and num_particles > 1:
                atoms = self._prepare_atoms_for_swap(atoms, sigma_next)
                atoms = self._collect_and_swap(
                    atoms_list=atoms,
                    beta=self.thermostat(sigma_next),
                    num_particles=num_particles,
                    z_table=swapping_z_table,
                )
                batched = self.batch_atoms(atoms)

            batched = self.integrator(batched, step, sigma_cur, sigma_next)
            atoms = get_atoms_from_batch(batched, self.z_table)
            trajectories.extend(atoms)

        if self.swapped:
            atoms = self._prepare_atoms_for_swap(atoms, self.sigmas[-1])
            atoms = collect_particles(atoms, self.thermostat(self.sigmas[-1]))
            trajectories.append(atoms)

        if do_final_cleanup:
            atoms = duplicate_atoms(trajectories[-1])
            atoms.calc = self.similarity_calculator
            cleaned = cleanup_atoms(
                atoms,
                swapping_z_table,
                num_element_sweeps=10,
            )
            trajectories.extend(cleaned)
        return trajectories

    def _collect_and_swap(
        self, atoms_list: List[ase.Atoms], beta, num_particles, z_table
    ):
        if self.swapped:
            collected_mol = collect_particles(atoms_list, beta)  # type: ignore
            collected_mol.calc = self.similarity_calculator
            atoms_list = [collected_mol]
            self.swapped = False
        assert len(atoms_list) == 1
        atoms = atoms_list[0]
        atom_ensemble = create_element_swapped_particles(
            atoms=atoms,
            beta=beta,
            num_particles=num_particles,
            z_table=z_table,
        )
        self.swapped = True
        return atom_ensemble

    def _prepare_atoms_for_swap(self, atoms_list: List[ase.Atoms], sigma_next):
        if sigma_next is isinstance(sigma_next, torch.Tensor):
            time = sigma_next.item()
        else:
            time = sigma_next

        for mol in atoms_list:
            mol.info["time"] = time
            mol.info["calculation_type"] = "similarity"
            mol.calc = self.similarity_calculator
        return atoms_list

    def _get_sigma_schedule(self, num_steps: int):
        step_indices = torch.arange(num_steps).to(self.device)
        sigma_max, sigma_min, rho = (
            self.noise_parameters.sigma_max,
            self.noise_parameters.sigma_min,
            self.noise_parameters.rho,
        )
        max_noise_rhod = sigma_max ** (1 / rho)
        min_noise_rhod = sigma_min ** (1 / rho)
        noise_interpolation = (
            step_indices / (num_steps - 1) * (min_noise_rhod - max_noise_rhod)
        )
        sigmas = (max_noise_rhod + noise_interpolation) ** rho
        # Add a zero sigma at the end to get the sample.
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1]).to(self.device)])
        return sigmas


class HeunIntegrator:
    def __init__(
        self,
        similarity_calculator,
        sampler_noise_parameters=SamplerNoiseParameters(),
        num_steps=100,
        restorative_force_strength: float = 1.5,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        self.similarity_calculator = similarity_calculator
        self.noise_parameters = sampler_noise_parameters
        self.restorative_force_strength = restorative_force_strength
        self.num_steps = num_steps
        self.device = device
        # TODO: remove num_steps dependency; can replace the final if statement with a condition
        # on sigma being zero.

    def __call__(self, x: AtomicData, step: int, sigma_cur, sigma_next):
        """
        This does NOT update the neighbout list. DANGER!
        """
        S_churn, S_min, S_max, S_noise = self._get_integrator_parameters()
        mol_cur = x.clone()
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
            mol_cur.positions += torch.randn_like(mol_cur.positions) * noise_level

        mol_increased = mol_cur
        device = mol_increased.positions.device
        # Euler step.
        # mol_increased.positions.requires_grad = True
        mol_increased.positions.grad = None
        forces = self.similarity_calculator(mol_increased, sigma_increased)
        forces = torch.tensor(forces, device=device)
        forces += (
            -1
            * self.restorative_force_strength
            * mol_increased.positions
            * torch.tanh(20 * sigma_cur**2)
        )

        mol_next = mol_cur.clone()
        logging.debug(f"Step size = {abs(sigma_next - sigma_increased):.2e}")
        with torch.no_grad():
            mol_next.positions += -1 * (sigma_next - sigma_increased) * forces

        # Apply 2nd order correction.
        if step < self.num_steps - 1:
            mol_next.positions.grad = None

            forces_next = self.similarity_calculator(mol_next, sigma_next)
            forces_next = torch.tensor(forces_next, device=device)

            mol_next = mol_increased.clone()
            with torch.no_grad():
                mol_next.positions += -1 * (
                    (sigma_next - sigma_increased) * (forces + forces_next) / 2
                )
                # mol_next.arrays["forces"] = (forces + forces_next) / 2
        return mol_next

    def _get_integrator_parameters(self):
        return (
            self.noise_parameters.S_churn,
            self.noise_parameters.S_min,
            self.noise_parameters.S_max,
            self.noise_parameters.S_noise,
        )
