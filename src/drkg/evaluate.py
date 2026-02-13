"""Filtered ranking metrics for drug -> disease retrieval."""

from __future__ import annotations

import numpy as np


def rank_metrics(scores: np.ndarray, drugs: np.ndarray, pairs: np.ndarray, candidates: np.ndarray, ks=(1, 10, 50)) -> dict:
    """Rank each held-out true disease among all candidate diseases.

    ``scores[i, j]`` is the model's score for ``drugs[i]`` against ``candidates[j]``.
    Ranking is *filtered*: the drug's other known indications are removed from the
    competition, so a drug is not penalised for ranking one true disease above
    another. Ties get their average rank, which matters for coarse scores like
    popularity where many diseases share a value.

    Returns pair-level (micro) metrics and per-drug (macro) averages.
    """
    col = {d: j for j, d in enumerate(candidates)}
    row = {d: i for i, d in enumerate(drugs)}
    ranks_by_drug: dict[int, list[float]] = {}
    for drug in drugs:
        true_cols = [col[x] for x in pairs[pairs[:, 0] == drug, 1] if x in col]
        if not true_cols:
            continue
        s = scores[row[drug]]
        others = np.ones(len(s), dtype=bool)
        others[true_cols] = False
        competing = s[others]
        ranks = []
        for j in true_cols:
            greater = (competing > s[j]).sum()
            ties = (competing == s[j]).sum()
            ranks.append(1.0 + greater + 0.5 * ties)
        ranks_by_drug[int(drug)] = ranks

    def summarise(rank_lists: list[list[float]], pooled: bool) -> dict:
        groups = [np.concatenate(rank_lists)] if pooled else [np.array(r) for r in rank_lists]
        out = {"mrr": np.mean([np.mean(1.0 / g) for g in groups])}
        for k in ks:
            out[f"hits@{k}"] = np.mean([np.mean(g <= k) for g in groups])
        out["mean_rank"] = np.mean([np.mean(g) for g in groups])
        return {k: float(v) for k, v in out.items()}

    lists = list(ranks_by_drug.values())
    micro = summarise(lists, pooled=True)
    macro = summarise(lists, pooled=False)
    return {**micro, **{f"macro_{k}": v for k, v in macro.items()},
            "n_pairs": int(sum(len(r) for r in lists)), "n_drugs": len(lists)}
