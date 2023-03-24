import logging
import pprint
from typing import List

import ase
import numpy as np
import torch
from mace.data import AtomicData
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_sum
from mace.tools.torch_geometric import Batch
from scipy.special import softmax

from .calculators import MaceSimilarityCalculator
from .diffusion_tools import SamplerNoiseParameters


def clone(atomic_data):
    try:
        return atomic_data.clone()
    except:
        x = atomic_data.to_dict()
        x["dipole"] = None
        return AtomicData(**x)


def indices_to_atomic_numbers(
    indices: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_atomic_numbers_fn = np.vectorize(z_table.index_to_z)
    return to_atomic_numbers_fn(indices)


def atoms_from_batch(batch, z_table) -> List[ase.Atoms]:
    """Convert batch to ase.Atoms"""
    atoms_list = []
    for i in range(len(batch.ptr) - 1):
        indices = np.argmax(
            batch.node_attrs[batch.ptr[i] : batch.ptr[i + 1], :].detach().cpu().numpy(),
            axis=-1,
        )
        numbers = indices_to_atomic_numbers(indices=indices, z_table=z_table)
        atoms = ase.Atoms(
            numbers=numbers,
            positions=batch.positions[batch.ptr[i] : batch.ptr[i + 1], :]
            .detach()
            .cpu()
            .numpy(),
            cell=None,
        )
        atoms_list.append(atoms)
    return atoms_list


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
                torch.linspace(10, 2.5, 20),
                torch.linspace(2.5, 1, 40),
                torch.linspace(0.95, 0.05, 200),
                torch.logspace(-1.31, -3, 20),
            ]
        ).to(device)
        num_steps = len(self.sigmas)
        self.integrator = HeunIntegrator(
            similarity_calculator=similarity_calculator,
            num_steps=num_steps,
            sampler_noise_parameters=noise_params,
            restorative_force_strength=restorative_force_strength,
        )
        self.num_steps = num_steps

    def generate(self, molecule: ase.Atoms, num_particles: int):
        # initialise mol
        molecule.positions = (
            np.random.randn(*molecule.positions.shape) * self.sigmas[0].item()
        )

        batched = self.similarity_calculator.convert_to_atomic_data(molecule)
        batched = self.similarity_calculator._batch_atomic_data(batched)

        swapped = False
        trajectories = [molecule]

        for step in range(self.num_steps - 1):
            sigma_cur, sigma_next = self.sigmas[step], self.sigmas[step + 1]
            if step % 10 == 0 and num_particles > 1:
                if swapped:
                    batched = self.collect(batched, sigma_next)
                    swapped = False
                batched = self.swap_elements(batched, step, num_particles)
                swapped = True

            batched = self.integrator(batched, step, sigma_cur, sigma_next)

            atoms = atoms_from_batch(batched, self.z_table)
            trajectories.extend(atoms)

        if swapped:
            batched = self.collect(
                batched, torch.tensor(self.noise_parameters.sigma_min)
            )
            trajectories.append(self.atomic_data_to_ase(batched, self.z_table))

        cleaned = self._clean_final_atoms(atomic_data=batched)
        trajectories.append(self.atomic_data_to_ase(cleaned, self.z_table))

        if num_particles == 1:
            trajectories.append(self.atomic_data_to_ase(batched, self.z_table))
        return trajectories

    @staticmethod
    def atomic_data_to_ase(atomic_data, z_table):
        elements = atomic_data["node_attrs"].detach().cpu().numpy()
        elements = np.argmax(elements, axis=1)
        elements = [z_table.zs[z] for z in elements]
        positions = atomic_data["positions"].detach().cpu().numpy()
        atoms = ase.Atoms(elements, positions)
        return atoms

    def swap_elements(self, atomic_data: AtomicData, step: int, num_particles: int):
        atoms = atoms_from_batch(atomic_data, self.z_table)
        assert len(atoms) == 1
        atoms = atoms[0]

        atoms.info["time"] = self.sigmas[step].item()
        atoms.calc = self.similarity_calculator
        # beta should be 1e3 until sigma < 0.1, then loglinearly decrease to 1
        sigma_cut = 0.5
        sigma = self.sigmas[step].item()
        log_beta = -2 + np.max([0, 3 * (sigma_cut - sigma) / sigma_cut])
        beta = 10**log_beta

        energies = atoms.get_potential_energies()
        logging.debug(f"beta = {beta}, energies = {energies}")
        energies = energies * beta
        probabilities = softmax(energies)
        num_change = np.ceil(0.1 * len(probabilities)).astype(int)

        logging.debug(f"Element swap probabilities: {probabilities}")
        ensemble = [atoms.copy()]
        for _ in range(num_particles - 1):
            x = atoms.copy()
            try:
                to_change = np.random.choice(
                    len(probabilities),
                    size=int(num_change),
                    replace=False,
                    p=probabilities,
                )
            except ValueError:
                # fewer than num_change elements have non-zero probability
                to_change = np.where(probabilities > 0.5)[0]
            mask = np.zeros(len(probabilities)).astype(bool)
            mask[to_change] = True

            logging.debug(f"Element swap mask: {mask}")
            numbers = x.get_atomic_numbers()
            zs = np.array([1, 6, 7, 8, 9])
            numbers[mask] = np.random.choice(zs, size=len(x))[mask]
            x.set_atomic_numbers(numbers)
            ensemble.append(x)

        confs = [config_from_atoms(x) for x in ensemble]
        atomic_datas = [
            AtomicData.from_config(
                conf,
                z_table=self.z_table,
                cutoff=self.similarity_calculator.model.r_max.item(),
            ).to(self.device)
            for conf in confs
        ]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=atomic_datas,
            batch_size=num_particles,
            shuffle=False,
            drop_last=False,
        )
        return next(iter(data_loader))

    def collect(self, atomic_data: Batch, sigma_cur):
        atoms = atoms_from_batch(atomic_data, self.z_table)
        # energies = np.zeros(len(atoms))
        # for i, mol in enumerate(atoms):
        #     mol.info["time"] = sigma_cur.item()
        #     mol.calc = self.similarity_calculator
        #     energies[i] = mol.get_potential_energy()

        embeds = self.similarity_calculator._get_node_embeddings(atomic_data)
        log_k = self.similarity_calculator._calculate_log_k(embeds, sigma_cur)
        energies_v2 = scatter_sum(-1 * log_k, atomic_data["batch"], dim=0)
        beta = 1 / (sigma_cur * 3000 + 1)
        probabilities = torch.softmax(-1 * beta * energies_v2, dim=0)
        debug_str = f"""
        Collecting at time {sigma_cur.item()}
        Particle energies: {energies_v2}
        Probabilities: {probabilities}
        """
        debug_str = pprint.pformat(debug_str)
        logging.debug(debug_str)
        collect_idx = np.random.choice(
            len(atoms), p=probabilities.detach().cpu().numpy()
        )
        logging.debug(f"Collecting particle {collect_idx}")
        lowest_energy_atoms = atoms[collect_idx]
        new_atomic_data = self.similarity_calculator.convert_to_atomic_data(
            lowest_energy_atoms
        )
        new_atomic_data = self.similarity_calculator._batch_atomic_data(new_atomic_data)
        return new_atomic_data

    def _clean_final_atoms(self, atomic_data):
        """
        Remove unconnected atoms from the final atoms object.
        Then run a final element swap using the pretrained model energies.
        """
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
        gamma = (
            min(S_churn / self.num_steps, np.sqrt(2) - 1)
            if S_min <= sigma_cur <= S_max
            else 0
        )
        # Added noise depends on the current noise level. So, it decreases over the course of the integration.
        sigma_increased = sigma_cur * (1 + gamma)
        # Add noise to the current sample.
        noise_level = torch.sqrt((sigma_increased**2 - sigma_cur**2)) * S_noise
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
            * torch.tanh(25 * sigma_cur**2)
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
