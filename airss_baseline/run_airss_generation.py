from collections import Counter

import ase.io as aio
import numpy as np
from utils import (
    build_mol,
    do_mopac_relaxation,
    get_composition_counter,
)

from simgen.manifolds import MultivariateGaussianPrior

most_common_qm9_composition = "C7H10O2"


def get_composition_generator(
    composition_counter: Counter | None, rng: np.random.Generator
):
    if composition_counter is None:
        composition_counter = Counter([most_common_qm9_composition])
    compositions, counts = zip(*composition_counter.items())
    compositions = np.array(compositions, dtype=object)
    counts = np.asarray(counts)
    probs = counts / np.sum(counts)
    while True:
        yield rng.choice(compositions, p=probs)


def main(save_path: str, qm9_path: str | None = None):
    rng = np.random.default_rng(0)
    if qm9_path is not None:
        composition_counter = get_composition_counter(qm9_path)
        composition_generator = get_composition_generator(composition_counter, rng)
    else:
        composition_generator = get_composition_generator(None, rng)
    prior = MultivariateGaussianPrior(np.diag([1, 1, 0.5]).astype(np.float32))
    for _ in range(1000):
        composition = next(composition_generator)
        atoms = build_mol(composition, prior=prior)
        relaxed_atoms = do_mopac_relaxation(atoms)
        aio.write(
            save_path,
            relaxed_atoms,
            append=True,
            format="extxyz",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./airss_generations.xyz")
    parser.add_argument("--qm9_path", type=str, default=None)
    args = parser.parse_args()
    main(args.save_path, args.qm9_path)
