import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import jaccard
from sklearn.cluster import DBSCAN
from typer import Typer

app = Typer(pretty_exceptions_show_locals=False)


def get_dbscan_results(
    fingerprints, min_dist: float = 0.3, min_samples: int = 2, n_procs: int = -1
):
    db = DBSCAN(
        eps=min_dist, min_samples=min_samples, metric=jaccard, p=1, n_jobs=n_procs
    )
    db.fit(fingerprints)
    return db


def calculate_diversity(db: DBSCAN) -> float:
    uniques, counts = np.unique(db.labels_, return_counts=True)
    # len(uniques) -1 == number of clusters with more than 1 similar molecule
    # counts[0] == number of singletons
    return (len(uniques) - 1 + counts[0]) / counts.sum()


@app.command()
def main(qm9_summary_path: str, n_procs: int = -1):
    qm9_df = pd.read_json(qm9_summary_path)
    qm9_smiles = qm9_df["smiles"].tolist()
    qm9_mols = [Chem.MolFromSmiles(smi) for smi in qm9_smiles]
    qm9_fingerprints = [Chem.RDKFingerprint(mol) for mol in qm9_mols]
    db = get_dbscan_results(qm9_fingerprints, n_procs=n_procs)
    diversity = calculate_diversity(db)
    np.save("./results/qm9_diversity.npy", db.labels_)
    print(f"Diversity: {diversity}")
