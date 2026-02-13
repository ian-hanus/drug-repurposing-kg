import numpy as np

from drkg.evaluate import rank_metrics
from drkg.split import cold_start_split

PAIRS = np.array([(d, 100 + (d * 7 + k) % 30) for d in range(50) for k in range(3)])


def test_no_drug_appears_in_two_splits():
    s = cold_start_split(PAIRS, seed=0)
    d = s.drugs
    assert not (set(d["train"]) & set(d["val"])) and not (set(d["train"]) & set(d["test"]))
    assert not (set(d["val"]) & set(d["test"]))
    assert len(s.train) + len(s.val) + len(s.test) == len(PAIRS)


def test_all_of_a_drugs_pairs_stay_together():
    s = cold_start_split(PAIRS, seed=3)
    for d in s.drugs["test"]:
        assert (PAIRS[:, 0] == d).sum() == (s.test[:, 0] == d).sum()


def test_seed_changes_the_split():
    assert not np.array_equal(cold_start_split(PAIRS, 0).drugs["test"], cold_start_split(PAIRS, 1).drugs["test"])


def test_perfect_ranking():
    cands = np.array([10, 11, 12, 13])
    scores = np.array([[0.1, 0.9, 0.2, 0.0]])
    m = rank_metrics(scores, np.array([0]), np.array([[0, 11]]), cands)
    assert m["mrr"] == 1.0 and m["hits@1"] == 1.0 and m["mean_rank"] == 1.0


def test_filtering_ignores_other_true_diseases():
    cands = np.array([10, 11, 12])
    scores = np.array([[0.9, 0.8, 0.1]])  # both true diseases outrank the negative
    m = rank_metrics(scores, np.array([0]), np.array([[0, 10], [0, 11]]), cands)
    assert m["hits@1"] == 1.0  # 11 is not pushed down by the other true disease 10


def test_ties_get_average_rank():
    cands = np.array([10, 11, 12, 13])
    scores = np.zeros((1, 4))  # nothing distinguishes the true disease from 3 negatives
    m = rank_metrics(scores, np.array([0]), np.array([[0, 10]]), cands)
    assert np.isclose(m["mean_rank"], 2.5)


def test_macro_weights_each_drug_equally():
    cands = np.arange(5)
    scores = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=float)
    pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])  # drug 0 has 3 true, drug 1 has 1 (ranked last)
    m = rank_metrics(scores, np.array([0, 1]), pairs, cands)
    assert m["macro_hits@1"] < 1.0 and m["n_pairs"] == 4 and m["n_drugs"] == 2
