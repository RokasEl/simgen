import logging
from functools import partial
from time import monotonic
from typing import List, Literal, Tuple

import ase
import numpy as np
import torch
from hydromace.interface import HydroMaceCalculator
from mace.tools import AtomicNumberTable

from simgen.atoms_cleanup import cleanup_atoms
from simgen.calculators import MaceSimilarityCalculator
from simgen.element_swapping import (
    collect_particles,
    create_element_swapped_particles,
)
from simgen.generation_utils import (
    batch_atoms,
    batch_to_correct_dtype,
    check_atoms_outside_threshold,
    duplicate_atoms,
    get_atoms_from_batch,
)
from simgen.integrators import HeunIntegrator, IntegrationParameters
from simgen.manifolds import MultivariateGaussianPrior, PriorManifold
from simgen.temperature_annealing import ExponentialThermostat
from simgen.utils import get_system_torch_device_str

# if get_system_torch_device_str() == "mps":
#     torch.set_default_dtype(torch.float32)
# else:
#     torch.set_default_dtype(torch.float64)


class ParticleFilterGenerator:
    def __init__(
        self,
        similarity_calculator: MaceSimilarityCalculator,
        guiding_manifold: PriorManifold = MultivariateGaussianPrior(
            covariance_matrix=np.diag([1.0, 1.0, 4.0])
        ),
        device=get_system_torch_device_str(),
        restorative_force_strength: float = 1.5,
        integration_parameters=IntegrationParameters(),
        num_steps: int = 100,
    ):
        self.similarity_calculator = similarity_calculator
        self.dtype = self.similarity_calculator.dtype
        self.guiding_manifold = guiding_manifold
        self.z_table = AtomicNumberTable(
            [int(z) for z in similarity_calculator.model.atomic_numbers]  # type: ignore
        )
        self.device = device
        self.sigmas = (
            torch.concatenate(
                [
                    torch.linspace(1, 0.05, num_steps // 2),
                    torch.logspace(-1.31, -3, num_steps // 2),
                ]
            )
            .to(device)
            .to(self.dtype)
        )
        self.integrator = HeunIntegrator(
            similarity_calculator=similarity_calculator,
            guiding_manifold=guiding_manifold,
            integration_parameters=integration_parameters,
            restorative_force_strength=restorative_force_strength,
        )
        self.thermostat = ExponentialThermostat(
            initial_T_log_10=6, final_T_log_10=-1, sigma_max=self.sigmas[0]
        )
        batch_func = partial(
            batch_atoms,
            z_table=self.z_table,
            device=self.device,
            cutoff=self.similarity_calculator.cutoff,
        )
        self.batch_atoms = lambda x: batch_to_correct_dtype(batch_func(x), self.dtype)

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
        timeout: float = np.inf,  # only needed for the web interface
    ):
        # initialise mol
        molecule = self.guiding_manifold.initialise_positions(molecule, scale=0.5)
        molecule, mask, torch_mask = self._merge_scaffold_and_create_mask(
            molecule,
            scaffold,
            num_particles,
            self.device,
            self.similarity_calculator.dtype,
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
            timeout=timeout,
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
                num_element_sweeps="all",
                mask=mask,
            )
            trajectories.extend(cleaned)
        self._values_to_numpy(trajectories)
        return trajectories

    def _maximise_log_similarity(
        self,
        initial_atoms: ase.Atoms,
        particle_swap_frequency: int,
        num_particles: int,
        swapping_z_table: AtomicNumberTable,
        mask: np.ndarray,
        torch_mask: torch.Tensor,
        timeout: float = np.inf,
    ):
        atoms = [duplicate_atoms(initial_atoms)]
        batched = self.batch_atoms(atoms)
        self.swapped = False
        intermediate_configs = []
        start_time = monotonic()
        for step, (sigma_cur, sigma_next) in enumerate(
            zip(self.sigmas[:-1], self.sigmas[1:]),
        ):
            run_time = monotonic() - start_time
            if run_time > timeout:
                raise RuntimeError(f"Generation timed out after {run_time:.2f}s")
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
        if check_atoms_outside_threshold(atoms, 100):
            logging.critical(
                "Atoms exploded during the main loop of generation. Adjust the restorative force strength."
            )
            raise ValueError(
                "Atoms exploded during the main loop of generation. Adjust the restorative force strength."
            )
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
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[ase.Atoms, np.ndarray, torch.Tensor]:
        if scaffold is None:
            return (
                molecule,
                np.ones(len(molecule)),
                torch.ones(len(molecule), device=device, dtype=dtype).repeat(
                    num_particles
                ),
            )
        merged = molecule.copy() + scaffold.copy()
        mask = np.concatenate([np.ones(len(molecule)), np.zeros(len(scaffold))], axis=0)
        if device == "mps":
            mask = mask.astype(np.float32)
        torch_mask = torch.tensor(mask, device=device, dtype=dtype).repeat(
            num_particles
        )
        return merged, mask, torch_mask

    @staticmethod
    def _values_to_numpy(atoms: List[ase.Atoms]):
        for _atoms in atoms:
            for k, v in _atoms.info.items():
                if isinstance(v, torch.Tensor):
                    _atoms.info[k] = v.detach().cpu().numpy()
