import logging
import pprint
from functools import partial
from typing import List

import ase
import numpy as np
import torch
from mace.data import AtomicData
from mace.tools import AtomicNumberTable

from moldiff.calculators import MaceSimilarityCalculator
from moldiff.diffusion_tools import SamplerNoiseParameters
from moldiff.element_swapping import (
    collect_particles,
    create_element_swapped_particles,
)
from moldiff.generation_utils import batch_atoms, get_atoms_from_batch
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
        self.cutoff = similarity_calculator.model.cutoff.item()  # type: ignore
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
            initial_T_log_10=5, sigma_max=self.sigmas[0]
        )
        self.num_steps = num_steps
        self.batch_atoms = partial(
            batch_atoms, z_table=self.z_table, device=self.device, cutoff=self.cutoff
        )

    def generate(
        self,
        molecule: ase.Atoms,
        num_particles: int = 10,
        particle_swap_frequency: int = 1,
        do_final_cleanup: bool = True,
    ):
        # initialise mol
        molecule.positions = (
            np.random.randn(*molecule.positions.shape) * self.sigmas[0].item()
        )
        trajectories = [molecule]

        atoms = [molecule.copy()]
        batched = self.batch_atoms(atoms)
        self.swapped = False

        for step in range(self.num_steps - 1):
            sigma_cur, sigma_next = self.sigmas[step], self.sigmas[step + 1]
            if step % 4 == particle_swap_frequency and num_particles > 1:
                atoms = self._collect_and_swap(
                    atoms, sigma_next, self.thermostat(sigma_next), num_particles
                )
                batched = self.batch_atoms(atoms)

            batched = self.integrator(batched, step, sigma_cur, sigma_next)
            atoms = get_atoms_from_batch(batched, self.z_table)
            trajectories.extend(atoms)

        if self.swapped:
            atoms = collect_particles(atoms, self.thermostat(self.sigmas[-1]))
            trajectories.append(atoms)

        if do_final_cleanup:
            cleaned = self._clean_final_atoms(batched)
            trajectories.append(cleaned)
        return trajectories

    def _collect_and_swap(
        self, atoms_list: List[ase.Atoms], sigma_next, beta, num_particles
    ):
        if self.swapped:
            atoms_list = collect_particles(atoms_list, beta)
            self.swapped = False
        assert len(atoms_list) == 1
        # rewrite into a separate "prepare" function
        atoms = atoms_list[0]
        atoms.info["time"] = sigma_next
        atom_ensemble = create_element_swapped_particles(
            atoms=atoms,
            beta=beta,
            num_particles=num_particles,
            z_table=self.z_table,
        )
        self.swapped = True
        return atom_ensemble

    def _clean_final_atoms(self, atomic_data):
        """
        Remove unconnected atoms from the final atoms object.
        Then run a final element swap using the pretrained model energies.
        """
        # TODO: test this function and add a try except for failure
        atoms = atoms_from_batch(atomic_data, self.z_table)
        assert len(atoms) == 1
        atoms = atoms[0]
        # remove unconnected atoms
        distances = atoms.get_all_distances()
        distances = distances + np.eye(len(atoms)) * 100
        min_distances_per_atom = np.min(distances, axis=1)
        logging.debug(f"Min distances per atom: {min_distances_per_atom}")
        to_keep = np.where(min_distances_per_atom <= 1.7)[0]
        logging.debug(f"to_keep: {to_keep}")
        atoms.arrays["numbers"] = atoms.arrays["numbers"][to_keep]
        atoms.arrays["positions"] = atoms.arrays["positions"][to_keep]
        # run a loop of element swaps
        atomic_data = self.similarity_calculator.convert_to_atomic_data(atoms)
        new_atomic_data = self.similarity_calculator._batch_atomic_data(atomic_data)
        already_switched = []
        for _ in range(len(atoms)):
            atoms = atoms_from_batch(new_atomic_data, self.z_table)
            atoms = atoms[0]

            out = self.similarity_calculator.model(new_atomic_data.to_dict())
            node_energies = out["node_energy"].detach().cpu().numpy()
            shifted_energies = self.similarity_calculator.subtract_reference_energies(
                atoms, node_energies
            )
            logging.debug(f"Shifted energies: {shifted_energies}")
            sorted_energies = np.argsort(shifted_energies)
            sorted_energies = sorted_energies[
                ~np.isin(sorted_energies, already_switched)
            ]
            to_switch = sorted_energies[-1]
            already_switched.append(to_switch)
            logging.debug(
                f"To switch: {to_switch}, already switched: {already_switched}"
            )
            numbers = atoms.get_atomic_numbers()
            ensemble = []
            for z in [1, 6, 7, 8, 9]:
                mol = atoms.copy()
                numbers[to_switch] = z
                mol.set_atomic_numbers(numbers)
                ensemble.append(mol)

            atomic_data = self.similarity_calculator.convert_to_atomic_data(ensemble)
            new_atomic_data = self.similarity_calculator._batch_atomic_data(atomic_data)
            # get lowest energy
            new_atomic_data = self.collect(
                new_atomic_data, sigma_cur=torch.tensor(1e-3)
            )

        return new_atomic_data

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
        mol_cur = clone(x)
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

        mol_next = clone(mol_cur)
        logging.debug(f"Step size = {abs(sigma_next - sigma_increased):.2e}")
        with torch.no_grad():
            mol_next.positions += -1 * (sigma_next - sigma_increased) * forces

        # Apply 2nd order correction.
        if step < self.num_steps - 1:
            mol_next.positions.grad = None

            forces_next = self.similarity_calculator(mol_next, sigma_next)
            forces_next = torch.tensor(forces_next, device=device)

            mol_next = clone(mol_increased)
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
