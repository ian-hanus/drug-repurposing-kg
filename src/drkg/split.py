"""Cold-start split: whole drugs are held out, never individual drug-disease pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray  # [n, 2] (drug, disease) pairs
    val: np.ndarray
    test: np.ndarray

    @property
    def drugs(self) -> dict[str, np.ndarray]:
        return {k: np.unique(getattr(self, k)[:, 0]) for k in ("train", "val", "test")}


def cold_start_split(indications: np.ndarray, seed: int, frac=(0.8, 0.1, 0.1)) -> Split:
    """Assign each *drug* (with all its indications) to train, val or test.

    A random split over pairs would let the model see other indications of a
    test drug during training, which is not the situation a new drug is in.
    """
    drugs = np.unique(indications[:, 0])
    rng = np.random.RandomState(seed)
    drugs = drugs[rng.permutation(len(drugs))]
    n_train, n_val = int(frac[0] * len(drugs)), int(frac[1] * len(drugs))
    groups = {"train": drugs[:n_train], "val": drugs[n_train:n_train + n_val], "test": drugs[n_train + n_val:]}
    return Split(**{k: indications[np.isin(indications[:, 0], v)] for k, v in groups.items()})
