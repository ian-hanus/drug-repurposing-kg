"""Training for the shallow DistMult baseline and the R-GCN, with val-MRR early stopping."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import rank_metrics
from .kg import KG
from .models import RGCN, DistMult, build_edge_sets
from .split import Split


@dataclass
class TrainConfig:
    dim: int = 64
    lr: float = 5e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    n_neg: int = 16
    # shallow DistMult
    epochs: int = 40
    batch_size: int = 16384
    indication_repeat: int = 10   # indication triples are rare next to 748k graph edges
    # R-GCN
    steps: int = 150
    pos_batch: int = 2048
    eval_every: int = 10
    patience: int = 5             # evaluations without improvement
    inductive_drugs: bool = False  # R-GCN: drugs share one input embedding


class TypeSampler:
    """Sample random nodes of a given type: negatives for a corrupted triple keep their type."""

    def __init__(self, node_type: np.ndarray, n_types: int):
        order = np.argsort(node_type, kind="stable")
        self.nodes = torch.tensor(order)
        counts = np.bincount(node_type, minlength=n_types)
        self.count = torch.tensor(counts)
        self.start = torch.tensor(np.concatenate([[0], np.cumsum(counts)[:-1]]))
        self.node_type = torch.tensor(node_type)

    def like(self, nodes: torch.Tensor) -> torch.Tensor:
        t = self.node_type[nodes]
        offset = (torch.rand(nodes.shape) * self.count[t]).long()
        return self.nodes[self.start[t] + offset]


def _scores(rep: torch.Tensor, w: torch.Tensor, drugs: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return ((rep[torch.tensor(drugs)] * w) @ rep[torch.tensor(candidates)].T).numpy()


def _val_mrr(rep, w, split: Split, candidates) -> float:
    drugs = split.drugs["val"]
    return rank_metrics(_scores(rep, w, drugs, candidates), drugs, split.val, candidates)["mrr"]


def train_distmult(kg: KG, split: Split, candidates: np.ndarray, cfg: TrainConfig, seed: int):
    torch.manual_seed(seed)
    R = len(kg.relations)
    ind_rel = R  # extra relation id for "indication"
    h = np.concatenate([kg.src, np.repeat(split.train[:, 0], cfg.indication_repeat)])
    r = np.concatenate([kg.rel, np.full(len(split.train) * cfg.indication_repeat, ind_rel)])
    t = np.concatenate([kg.dst, np.repeat(split.train[:, 1], cfg.indication_repeat)])
    h, r, t = map(torch.tensor, (h, r, t))

    sampler = TypeSampler(kg.node_type, len(kg.type_names))
    model = DistMult(kg.n_nodes, R + 1, cfg.dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best, best_rep, stale, history = -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(h))
        for i in range(0, len(h), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            bh, br, bt = h[b], r[b], t[b]
            pos = model.score(bh, br, bt)
            nh = bh.repeat_interleave(cfg.n_neg)
            nt = bt.repeat_interleave(cfg.n_neg)
            corrupt_tail = torch.rand(len(nh)) < 0.5
            nh = torch.where(corrupt_tail, nh, sampler.like(nh))
            nt = torch.where(corrupt_tail, sampler.like(nt), nt)
            neg = model.score(nh, br.repeat_interleave(cfg.n_neg), nt)
            loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 2 == 0:
            rep, w = model.emb.weight.detach(), model.rel.weight[ind_rel].detach()
            mrr = _val_mrr(rep, w, split, candidates)
            history.append((epoch + 1, float(loss), mrr))
            if mrr > best:
                best, best_rep, best_w, stale = mrr, rep.clone(), w.clone(), 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    return {"rep": best_rep, "w": best_w, "val_mrr": best, "history": history}


def train_rgcn(kg: KG, split: Split, candidates: np.ndarray, cfg: TrainConfig, seed: int, log=None):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mask = (kg.node_type == kg.type_names.index("drug")) if cfg.inductive_drugs else None
    model = RGCN(kg.n_nodes, build_edge_sets(kg), cfg.dim, depth=2, dropout=cfg.dropout, inductive_mask=mask)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cand = torch.tensor(candidates)

    best, best_rep, best_w, stale, history = -1.0, None, None, 0, []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        sel = rng.choice(len(split.train), size=min(cfg.pos_batch, len(split.train)), replace=False)
        d = torch.tensor(split.train[sel, 0])
        c = torch.tensor(split.train[sel, 1])
        rep = model.node_repr()
        pos = model.score(d, c, rep)
        neg_c = cand[torch.randint(len(cand), (len(d), cfg.n_neg))]
        neg = model.score(d[:, None].expand_as(neg_c), neg_c, rep)
        loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                rep = model.node_repr()
            mrr = _val_mrr(rep, model.rel.detach(), split, candidates)
            history.append((step, float(loss), mrr))
            if log:
                log(f"step {step} loss={float(loss):.3f} val_mrr={mrr:.3f} ({time.time() - t0:.0f}s)")
            if mrr > best:
                best, best_rep, best_w, stale = mrr, rep.clone(), model.rel.detach().clone(), 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    return {"rep": best_rep, "w": best_w, "val_mrr": best, "history": history}


def score_test(result: dict, split: Split, candidates: np.ndarray):
    drugs = split.drugs["test"]
    scores = _scores(result["rep"], result["w"], drugs, candidates)
    return drugs, scores
