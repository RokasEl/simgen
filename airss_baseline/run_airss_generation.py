import ase.io as aio
import numpy as np

from airss_baseline.utils import build_mol, do_mopac_relaxation
from moldiff.manifolds import MultivariateGaussianPrior


def main():
    most_common_qm9_composition = "C7H10O2"
    prior = MultivariateGaussianPrior(np.diag([1, 1, 2]).astype(np.float32))
    for _ in range(50):
        atoms = build_mol(most_common_qm9_composition, prior=prior)
        relaxed_atoms = do_mopac_relaxation(atoms)
        aio.write(
            f"most_common_qm9_composition.xyz",
            relaxed_atoms,
            append=True,
            format="extxyz",
        )


if __name__ == "__main__":
    main()
