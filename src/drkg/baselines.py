"""Non-learned baselines for cold-start drug -> disease ranking.

These set the bar a GNN has to clear. For a drug with no known indications,
"which diseases are common?" and "which diseases do similar drugs treat?" are
surprisingly hard to beat.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .kg import KG
from .split import Split


def adjacency(kg: KG) -> sp.csr_matrix:
    """Symmetric binary node x node adjacency of the message graph (relations collapsed)."""
    a = sp.coo_matrix((np.ones(len(kg.src)), (kg.src, kg.dst)), shape=(kg.n_nodes, kg.n_nodes))
    return ((a + a.T) > 0).astype(np.float32).tocsr()


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    norm[norm == 0] = 1.0
    return sp.diags(1.0 / norm) @ m


def popularity(split: Split, drugs: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Score every drug with how many *training* drugs each disease is indicated for."""
    col = {d: j for j, d in enumerate(candidates)}
    counts = np.zeros(len(candidates))
    for _, disease in split.train:
        counts[col[disease]] += 1
    return np.tile(counts, (len(drugs), 1))


def drug_knn(kg: KG, adj: sp.csr_matrix, split: Split, drugs: np.ndarray, candidates: np.ndarray, k: int = 25) -> np.ndarray:
    """Predict the diseases of the most similar training drugs.

    Similarity is cosine over each drug's neighbours (targets and side effects).
    """
    feats = _row_normalize(adj[:, np.concatenate([kg.nodes_of("gene/protein"), kg.nodes_of("effect/phenotype")])])
    train_drugs = np.unique(split.train[:, 0])
    col = {d: j for j, d in enumerate(candidates)}
    row = {d: i for i, d in enumerate(train_drugs)}
    y = np.zeros((len(train_drugs), len(candidates)), dtype=np.float32)
    for drug, disease in split.train:
        y[row[drug], col[disease]] = 1.0

    sim = (feats[drugs] @ feats[train_drugs].T).toarray()
    scores = np.zeros((len(drugs), len(candidates)), dtype=np.float32)
    for i in range(len(drugs)):
        top = np.argpartition(-sim[i], min(k, sim.shape[1] - 1))[:k]
        scores[i] = sim[i, top] @ y[top]
    return scores


def target_overlap(kg: KG, adj: sp.csr_matrix, drugs: np.ndarray, candidates: np.ndarray, hops: int = 0) -> np.ndarray:
    """Guilt by association: do a drug's targets overlap a disease's associated genes?

    With ``hops=1`` each drug's targets are first expanded to their direct
    protein-protein neighbours, a cheap stand-in for network proximity.
    """
    genes = kg.nodes_of("gene/protein")
    targets = adj[drugs][:, genes]
    if hops:
        ppi = adj[genes][:, genes]
        targets = ((targets + targets @ ppi) > 0).astype(np.float32)
    disease_genes = adj[candidates][:, genes]
    return (_row_normalize(targets) @ _row_normalize(disease_genes).T).toarray()
