from copy import deepcopy
from typing import Dict, List

import ase
import numpy as np
import torch
from ase.neighborlist import natural_cutoffs, neighbor_list
from e3nn.util.jit import compile_mode
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.modules.blocks import RadialEmbeddingBlock
from mace.tools import AtomicNumberTable
from scipy.interpolate import splev, splprep  # type: ignore
from torch import nn


def duplicate_atoms(atoms: ase.Atoms, copy_info=True) -> ase.Atoms:
    """
    Create a deep copy of the atoms object
    """
    atoms_copy = ase.Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.cell,
        pbc=atoms.pbc,
        info=deepcopy(atoms.info) if copy_info else None,
    )
    return atoms_copy


def calculate_restorative_force_strength(num_atoms: int | float) -> float:
    sqrt_prefactor = 1.5664519  # prefactor fit to qm9
    bounding_sphere_diameter = sqrt_prefactor * np.sqrt(num_atoms)
    force_strength = 1 / (0.2 + 0.1 * bounding_sphere_diameter) ** 2
    return force_strength


def interpolate_points(points, num_interpolated_points=100):
    k = min(3, len(points) - 1)
    tck, u = splprep(points.T, s=0, k=k)
    u = np.linspace(0, 1, num_interpolated_points)
    new_points = np.array(splev(u, tck)).T
    return new_points


def calculate_path_length(points):
    path_length = 0
    for p1, p2 in zip(points[:-1], points[1:]):
        path_length += np.linalg.norm(p1 - p2)
    return path_length


def change_indices_to_atomic_numbers(
    indices: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_atomic_numbers_fn = np.vectorize(z_table.index_to_z)
    return to_atomic_numbers_fn(indices)


def get_atoms_from_batch(batch, z_table: AtomicNumberTable) -> List[ase.Atoms]:
    """Convert batch to ase.Atoms"""
    atoms_list = []
    for i in range(len(batch.ptr) - 1):
        indices = np.argmax(
            batch.node_attrs[batch.ptr[i] : batch.ptr[i + 1], :].detach().cpu().numpy(),
            axis=-1,
        )
        numbers = change_indices_to_atomic_numbers(indices=indices, z_table=z_table)
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


def convert_atoms_to_atomic_data(
    atoms: ase.Atoms | List[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
):
    if isinstance(atoms, ase.Atoms):
        atoms = [atoms]
    confs = [config_from_atoms(x) for x in atoms]
    atomic_datas = [
        AtomicData.from_config(conf, z_table, cutoff).to(device) for conf in confs
    ]
    return atomic_datas


def batch_atoms(
    atoms: ase.Atoms | list[ase.Atoms],
    z_table: AtomicNumberTable,
    cutoff: float,
    device: str,
) -> AtomicData:
    atomic_datas = convert_atoms_to_atomic_data(atoms, z_table, cutoff, device)
    return next(
        iter(get_data_loader(atomic_datas, batch_size=len(atomic_datas), shuffle=False))
    )


def remove_elements(atoms: ase.Atoms, atomic_numbers_to_remove: List[int]) -> ase.Atoms:
    """
    Remove all hydrogens from the atoms object
    """
    atoms_copy = atoms.copy()
    for atomic_number in atomic_numbers_to_remove:
        to_remove = atoms_copy.get_atomic_numbers() == atomic_number
        del atoms_copy[to_remove]
    return atoms_copy


def get_edge_array_and_neighbour_numbers(atoms: ase.Atoms, mult: float = 1.2):
    """
    Get the edge array and the number of neighbours for each atom
    """
    cutoffs = natural_cutoffs(atoms, mult=mult)  # type: ignore
    edge_array = neighbor_list("ij", atoms, cutoffs)
    edge_array = np.stack(edge_array, axis=1)
    neighbour_numbers = np.bincount(edge_array[:, 0])
    return edge_array, neighbour_numbers


from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools.scatter import scatter_sum


@compile_mode("script")
class ExponentialRepulsionBlock(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        data["positions"].requires_grad_(True)
        print("getting edge vectors")
        print(f"edge_index shape {data['edge_index'].shape}")
        _, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        receiver = data["edge_index"][1]
        print("running torch exp")
        all_energies = torch.exp(-self.alpha * lengths)  # [n_edges, 1]
        all_energies = all_energies.squeeze()
        print(
            f"got energies with shape {all_energies.shape} and device {all_energies.device}, and dtype {all_energies.dtype}"
        )
        index = data["edge_index"][0]
        print(f"sender shape {index.shape}, device {index.device}, dtype {index.dtype}")
        print(
            f"receiver shape {receiver.shape}, device {receiver.device}, dtype {receiver.dtype}"
        )
        print("running scatter sum")
        num_nodes = data["positions"].shape[0]
        print(f"num nodes {num_nodes}")
        energies = 0.5 * scatter_sum(
            src=all_energies, index=receiver, dim=-1, dim_size=num_nodes
        )  # [n_nodes]
        print(f"got energies with shape {energies.shape}")
        return energies


class RadialDistanceTransformBlock(RadialEmbeddingBlock):
    def __init__(self, r_min: float = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "r_min", torch.tensor(r_min, dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        transformed_edges = (
            torch.nn.functional.relu(edge_lengths - self.r_min) + self.r_min
        )
        return super().forward(transformed_edges)
