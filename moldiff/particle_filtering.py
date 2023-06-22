import logging
from functools import partial
from typing import List, Literal

import ase
import numpy as np
import torch
from hydromace.interface import HydroMaceCalculator
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
from moldiff.integrators import HeunIntegrator
from moldiff.manifolds import MultivariateGaussianPrior, PriorManifold
from moldiff.temperature_annealing import ExponentialThermostat


class ParticleFilterGenerator:
    def __init__(
        self,
        similarity_calculator: MaceSimilarityCalculator,
        guiding_manifold: PriorManifold = MultivariateGaussianPrior(
            covariance_matrix=np.diag([1.0, 1.0, 4.0])
        ),
        num_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        restorative_force_strength: float = 1.5,
        noise_params=SamplerNoiseParameters(),
    ):
        self.similarity_calculator = similarity_calculator
        self.guiding_manifold = guiding_manifold
        self.z_table = AtomicNumberTable(
            [int(z) for z in similarity_calculator.model.atomic_numbers]  # type: ignore
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
            guiding_manifold=guiding_manifold,
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
        scaffold: ase.Atoms | None = None,
        hydrogenation_type: Literal["valence"] | Literal["hydromace"] = "valence",
        hydrogenation_calc: HydroMaceCalculator | None = None,
    ):
        # initialise mol
        molecule = self.guiding_manifold.initialise_positions(molecule, scale=0.5)
        molecule, mask, torch_mask = self._merge_scaffold_and_create_mask(
            molecule, scaffold, num_particles, self.device
        )
        trajectories = [molecule]

        # main generation loop
        intermediate_configs = self._maximise_log_similarity(
            molecule.copy(),
            particle_swap_frequency,
            num_particles,
            swapping_z_table,
            mask,
            torch_mask,
        )
        trajectories.extend(intermediate_configs)

        if do_final_cleanup:
            atoms = duplicate_atoms(trajectories[-1])
            atoms.calc = self.similarity_calculator
            cleaned = cleanup_atoms(
                atoms,
                hydrogenation_type,
                hydrogenation_calc,
                swapping_z_table,
                num_element_sweeps=10,
            )
            trajectories.extend(cleaned)
        return trajectories

    def _maximise_log_similarity(
        self,
        initial_atoms: ase.Atoms,
        particle_swap_frequency: int,
        num_particles: int,
        swapping_z_table: AtomicNumberTable,
        mask: np.ndarray,
        torch_mask: torch.Tensor,
    ):
        atoms = [duplicate_atoms(initial_atoms)]
        batched = self.batch_atoms(atoms)
        self.swapped = False
        intermediate_configs = []
        for step in range(self.num_steps - 1):
            sigma_cur, sigma_next = self.sigmas[step], self.sigmas[step + 1]
            if step % particle_swap_frequency == 0 and num_particles > 1:
                atoms = self._prepare_atoms_for_swap(atoms, sigma_next)
                atoms = self._collect_and_swap(
                    atoms_list=atoms,
                    beta=self.thermostat(sigma_next),
                    num_particles=num_particles,
                    z_table=swapping_z_table,
                    mask=mask,
                )
                batched = self.batch_atoms(atoms)

            batched = self.integrator(batched, step, sigma_cur, sigma_next, torch_mask)
            atoms = get_atoms_from_batch(batched, self.z_table)
            intermediate_configs.extend(atoms)

        if self.swapped:
            atoms = self._prepare_atoms_for_swap(atoms, self.sigmas[-1])
            atoms, _ = collect_particles(atoms, self.thermostat(self.sigmas[-1]))
            intermediate_configs.append(atoms)
        return intermediate_configs

    def _collect_and_swap(
        self, atoms_list: List[ase.Atoms], beta, num_particles, z_table, mask
    ):
        if self.swapped:
            collected_mol, _ = collect_particles(atoms_list, beta)  # type: ignore
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
            mask=mask,
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

    @staticmethod
    def _merge_scaffold_and_create_mask(
        molecule: ase.Atoms,
        scaffold: ase.Atoms | None,
        num_particles: int,
        device: str = "cpu",
    ):
        if scaffold is None:
            return (
                molecule,
                np.ones(len(molecule)),
                torch.ones(len(molecule)).repeat(num_particles).to(device),
            )
        merged = molecule.copy() + scaffold.copy()
        mask = np.concatenate([np.ones(len(molecule)), np.zeros(len(scaffold))], axis=0)
        torch_mask = torch.tensor(mask).repeat(num_particles).to(device)
        return merged, mask, torch_mask

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
