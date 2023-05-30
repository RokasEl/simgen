from typing import List

import numpy as np
from frozendict import frozendict
from numpy import typing as npt

"""
Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9
"""


BONDS1 = frozendict(
    {
        "H": {
            "H": 74,
            "C": 109,
            "N": 101,
            "O": 96,
            "F": 92,
            "B": 119,
            "Si": 148,
            "P": 144,
            "As": 152,
            "S": 134,
            "Cl": 127,
            "Br": 141,
            "I": 161,
        },
        "C": {
            "H": 109,
            "C": 154,
            "N": 147,
            "O": 143,
            "F": 135,
            "Si": 185,
            "P": 184,
            "S": 182,
            "Cl": 177,
            "Br": 194,
            "I": 214,
        },
        "N": {
            "H": 101,
            "C": 147,
            "N": 145,
            "O": 140,
            "F": 136,
            "Cl": 175,
            "Br": 214,
            "S": 168,
            "I": 222,
            "P": 177,
        },
        "O": {
            "H": 96,
            "C": 143,
            "N": 140,
            "O": 148,
            "F": 142,
            "Br": 172,
            "S": 151,
            "P": 163,
            "Si": 163,
            "Cl": 164,
            "I": 194,
        },
        "F": {
            "H": 92,
            "C": 135,
            "N": 136,
            "O": 142,
            "F": 142,
            "S": 158,
            "Si": 160,
            "Cl": 166,
            "Br": 178,
            "P": 156,
            "I": 187,
        },
        "B": {"H": 119, "Cl": 175},
        "Si": {
            "Si": 233,
            "H": 148,
            "C": 185,
            "O": 163,
            "S": 200,
            "F": 160,
            "Cl": 202,
            "Br": 215,
            "I": 243,
        },
        "Cl": {
            "Cl": 199,
            "H": 127,
            "C": 177,
            "N": 175,
            "O": 164,
            "P": 203,
            "S": 207,
            "B": 175,
            "Si": 202,
            "F": 166,
            "Br": 214,
        },
        "S": {
            "H": 134,
            "C": 182,
            "N": 168,
            "O": 151,
            "S": 204,
            "F": 158,
            "Cl": 207,
            "Br": 225,
            "Si": 200,
            "P": 210,
            "I": 234,
        },
        "Br": {
            "Br": 228,
            "H": 141,
            "C": 194,
            "O": 172,
            "N": 214,
            "Si": 215,
            "S": 225,
            "F": 178,
            "Cl": 214,
            "P": 222,
        },
        "P": {
            "P": 221,
            "H": 144,
            "C": 184,
            "O": 163,
            "Cl": 203,
            "S": 210,
            "F": 156,
            "N": 177,
            "Br": 222,
        },
        "I": {
            "H": 161,
            "C": 214,
            "Si": 243,
            "N": 222,
            "O": 194,
            "S": 234,
            "F": 187,
            "I": 266,
        },
        "As": {"H": 152},
    }
)

BONDS2 = frozendict(
    {
        "C": {"C": 134, "N": 129, "O": 120, "S": 160},
        "N": {"C": 129, "N": 125, "O": 121},
        "O": {"C": 120, "N": 121, "O": 121, "P": 150},
        "P": {"O": 150, "S": 186},
        "S": {"P": 186},
    }
)


BONDS3 = frozendict(
    {
        "C": {"C": 120, "N": 116, "O": 113},
        "N": {"C": 116, "N": 110},
        "O": {"C": 113},
    }
)


NATURAL_VALENCES = frozendict(
    {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
        "B": 3,
        "Al": 3,
        "Si": 4,
        "P": [3, 5],
        "S": 4,
        "Cl": 1,
        "As": 3,
        "Br": 1,
        "I": 1,
        "Hg": [1, 2],
        "Bi": [3, 5],
    }
)

MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3


def get_bond_order(
    atom1: str,
    atom2: str,
    distance: float,
    check_exists: bool = False,
    single_bond_stretch_factor: float = 1.0,
    multi_bond_stretch_factor: float = 1.0,
    use_margins: bool = False,
):
    distance = 100 * distance

    def adjust_threshold(use_margins, threshold, margin, multiplier):
        if use_margins:
            return threshold + margin
        else:
            return threshold * multiplier

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in BONDS1:
            return 0
        if atom2 not in BONDS1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    bond1_thresh = BONDS1[atom1][atom2]
    bond1_thresh = adjust_threshold(
        use_margins, bond1_thresh, MARGIN1, single_bond_stretch_factor
    )
    if distance < bond1_thresh:
        # Check if atoms in bonds2 dictionary.
        if atom1 in BONDS2 and atom2 in BONDS2[atom1]:
            bond2_thresh = BONDS2[atom1][atom2]
            bond2_thresh = adjust_threshold(
                use_margins, bond2_thresh, MARGIN2, multi_bond_stretch_factor
            )
            if distance < bond2_thresh:
                if atom1 in BONDS3 and atom2 in BONDS3[atom1]:
                    bond3_thresh = BONDS3[atom1][atom2]
                    bond3_thresh = adjust_threshold(
                        use_margins, bond3_thresh, MARGIN3, multi_bond_stretch_factor
                    )
                    if distance < bond3_thresh:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def single_bond_only(threshold: float, length: float, stretch_factor: float = 1.0):
    if length < threshold * stretch_factor:
        return 1
    return 0


def build_xae_molecule(
    positions: npt.NDArray[np.float64],
    atom_types: List[str],
    single_bond_stretch_factor: float = 1.0,
    multi_bond_stretch_factor: float = 1.0,
    use_margins: bool = False,
):
    """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
    args:
    positions: N x 3  (already masked to keep final number nodes)
    atom_types: N
    returns:
    X: N         (int)
    A: N x N     (bool)                  (binary adjacency matrix)
    E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = np.zeros((n, n), dtype=bool)
    E = np.zeros((n, n), dtype=int)

    pos = positions[None, ...]
    dists = np.linalg.norm(pos - pos.transpose(1, 0, 2), axis=-1)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(
                pair[0],
                pair[1],
                dists[i, j],
                single_bond_stretch_factor=single_bond_stretch_factor,
                multi_bond_stretch_factor=multi_bond_stretch_factor,
                use_margins=use_margins,
            )
            if order > 0:
                A[i, j] = 1
                E[i, j], E[j, i] = order, order
    return X, A, E
