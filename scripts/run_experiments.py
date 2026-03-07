"""Run every model on cold-start splits. Resumable: finished (seed, model) pairs are skipped."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drkg import baselines as B  # noqa: E402
from drkg.evaluate import rank_metrics  # noqa: E402
from drkg.kg import load_primekg  # noqa: E402
from drkg.split import cold_start_split  # noqa: E402
from drkg.train import TrainConfig, score_test, train_distmult, train_rgcn  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--models", nargs="+",
                    default=["random", "popularity", "drug_knn", "target_overlap", "distmult", "rgcn"])
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--out", default="results/results.csv")
args = parser.parse_args()

torch.set_num_threads(args.threads)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    prev = pd.read_csv(out)
    done = set(zip(prev.seed, prev.model))

kg = load_primekg()
adj = B.adjacency(kg)
candidates = np.unique(kg.indications[:, 1])
cfg = TrainConfig(steps=args.steps, lr=args.lr)

for seed in range(args.seeds):
    split = cold_start_split(kg.indications, seed)
    drugs = split.drugs["test"]
    rng = np.random.RandomState(seed)
    for model in args.models:
        if (seed, model) in done:
            continue
        val_mrr, history = np.nan, ""
        if model == "random":
            scores = rng.rand(len(drugs), len(candidates))
        elif model == "popularity":
            scores = B.popularity(split, drugs, candidates)
        elif model == "drug_knn":
            scores = B.drug_knn(kg, adj, split, drugs, candidates)
        elif model == "target_overlap":
            scores = B.target_overlap(kg, adj, drugs, candidates, hops=0)
        else:
            trainer = train_distmult if model == "distmult" else train_rgcn
            result = trainer(kg, split, candidates, cfg, seed)
            drugs, scores = score_test(result, split, candidates)
            val_mrr, history = result["val_mrr"], json.dumps(result["history"])
        m = rank_metrics(scores, drugs, split.test, candidates)
        row = {"seed": seed, "model": model, "val_mrr": val_mrr, **m, "history": history}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"seed={seed} {model:15} MRR={m['mrr']:.3f} H@10={m['hits@10']:.3f} H@50={m['hits@50']:.3f}", flush=True)
