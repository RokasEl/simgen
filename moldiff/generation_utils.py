from typing import List

import ase
import numpy as np
from mace.data.atomic_data import AtomicData, get_data_loader
from mace.data.utils import config_from_atoms
from mace.tools import AtomicNumberTable


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
    atoms: ase.Atoms | list[ase.Atoms],
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