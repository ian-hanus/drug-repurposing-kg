"""Aggregate results/results.csv into results/summary.md and a figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["random", "popularity", "target_overlap", "drug_knn", "distmult", "rgcn"]
METRICS = [("mrr", "MRR"), ("hits@10", "Hits@10"), ("hits@50", "Hits@50"), ("mean_rank", "mean rank"),
           ("macro_mrr", "per-drug MRR")]


def fmt(s: pd.Series, digits: int = 3) -> str:
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}" if len(s) > 1 else f"{s.mean():.{digits}f}"


def main() -> None:
    df = pd.read_csv(RES / "results.csv")
    models = [m for m in ORDER if m in set(df.model)]
    n_seeds = df.seed.nunique()
    n_cand = 1363

    out = ["# Results", "",
           f"Cold-start drug split (whole drugs held out), mean ± std over {n_seeds} random splits. "
           f"Each held-out indication is ranked against {n_cand} candidate diseases (filtered). "
           "Random scoring has a mean rank of about 680.", "",
           "| model | " + " | ".join(label for _, label in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for m in models:
        sub = df[df.model == m]
        out.append(f"| {m} | " + " | ".join(fmt(sub[k], 1 if k == "mean_rank" else 3) for k, _ in METRICS) + " |")

    out += ["", "## Per-seed comparison against the strongest baseline", "",
            "MRR difference per split (model minus drug_knn); positive means the model won that split.", "",
            "| model | " + " | ".join(f"seed {s}" for s in sorted(df.seed.unique())) + " | wins |",
            "|---|" + "---|" * (n_seeds + 1)]
    base = df[df.model == "drug_knn"].set_index("seed").mrr
    for m in [x for x in models if x not in ("drug_knn", "random")]:
        mrr = df[df.model == m].set_index("seed").mrr
        diff = (mrr - base).dropna()
        out.append(f"| {m} | " + " | ".join(f"{diff.get(s, float('nan')):+.3f}" for s in sorted(df.seed.unique()))
                   + f" | {(diff > 0).sum()}/{len(diff)} |")
    (RES / "summary.md").write_text("\n".join(out) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, label) in zip(axes, [("mrr", "MRR"), ("hits@10", "Hits@10")]):
        g = [df[df.model == m][key] for m in models]
        ax.bar(range(len(models)), [x.mean() for x in g], yerr=[x.std(ddof=1) if len(x) > 1 else 0 for x in g],
               color=["#9aa5b1"] * 3 + ["#4c78a8"] + ["#d1495b"] * 2, capsize=3)
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
        ax.set_ylabel(label)
    fig.suptitle("Cold-start drug repurposing on PrimeKG")
    fig.tight_layout()
    fig.savefig(RES / "fig_models.png", dpi=150)
    print("\n".join(out))


if __name__ == "__main__":
    main()
