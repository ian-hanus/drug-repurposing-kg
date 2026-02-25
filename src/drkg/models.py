"""Knowledge-graph embedding models: shallow DistMult and an R-GCN encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .kg import KG


class DistMult(nn.Module):
    """Shallow KG embedding: score(h, r, t) = sum(e_h * w_r * e_t)."""

    def __init__(self, n_nodes: int, n_relations: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.rel = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def node_repr(self) -> torch.Tensor:
        return self.emb.weight

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, node_repr: torch.Tensor | None = None) -> torch.Tensor:
        e = self.node_repr() if node_repr is None else node_repr
        return (e[h] * self.rel(r) * e[t]).sum(-1)


@dataclass
class EdgeSet:
    """Edges of one (relation, direction), with mean-aggregation weights 1/in-degree."""

    src: torch.Tensor
    dst: torch.Tensor
    norm: torch.Tensor


def build_edge_sets(kg: KG) -> list[EdgeSet]:
    """One EdgeSet per relation and direction: index r is forward, r + R is the inverse."""
    sets = []
    for direction in (0, 1):
        for r in range(len(kg.relations)):
            m = kg.rel == r
            src, dst = (kg.src[m], kg.dst[m]) if direction == 0 else (kg.dst[m], kg.src[m])
            deg = np.bincount(dst, minlength=kg.n_nodes).astype(np.float32)
            sets.append(EdgeSet(torch.tensor(src), torch.tensor(dst), torch.tensor(1.0 / deg[dst])))
    return sets


class RGCNLayer(nn.Module):
    def __init__(self, dim: int, n_edge_sets: int):
        super().__init__()
        self.w_rel = nn.Parameter(torch.empty(n_edge_sets, dim, dim))
        self.w_self = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.w_rel.view(-1, dim))

    def forward(self, h: torch.Tensor, edge_sets: list[EdgeSet]) -> torch.Tensor:
        out = self.w_self(h)
        for r, es in enumerate(edge_sets):
            if es.src.numel() == 0:
                continue
            msg = (h @ self.w_rel[r])[es.src] * es.norm[:, None]
            out = out.index_add(0, es.dst, msg)
        return out


class RGCN(nn.Module):
    """R-GCN node encoder over learnable input embeddings + DistMult decoder."""

    def __init__(self, n_nodes: int, edge_sets: list[EdgeSet], dim: int = 64, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.edge_sets = edge_sets
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.layers = nn.ModuleList(RGCNLayer(dim, len(edge_sets)) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.rel = nn.Parameter(torch.ones(dim))  # the single "indication" relation

    def node_repr(self) -> torch.Tensor:
        h = self.emb.weight
        for i, layer in enumerate(self.layers):
            h = layer(self.dropout(h), self.edge_sets)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def score(self, h: torch.Tensor, t: torch.Tensor, node_repr: torch.Tensor) -> torch.Tensor:
        return (node_repr[h] * self.rel * node_repr[t]).sum(-1)
