"""Load PrimeKG and turn it into a message-passing graph plus indication targets.

Design choices that matter for honest evaluation:
  * Every drug-disease edge (indication, contraindication, off-label use) is
    removed from the message-passing graph for *all* drugs. Indications are the
    prediction target, and the other two are so tightly related to them that
    keeping them for some drugs would leak labels or create a train/test
    mismatch. A drug is therefore represented only through its targets, side
    effects and other non-drug-disease relations.
  * drug-drug interactions and the huge ontology/anatomy subgraphs are dropped
    to keep the graph small enough for CPU training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
KEEP_TYPES = ["drug", "disease", "gene/protein", "effect/phenotype", "pathway", "exposure"]
DRUG_DISEASE_RELATIONS = {"indication", "contraindication", "off-label use"}
DROP_RELATIONS = {"drug_drug"}


@dataclass
class KG:
    node_type: np.ndarray       # [N] index into type_names
    type_names: list[str]
    node_names: np.ndarray      # [N]
    relations: list[str]        # message-passing relations (no drug-disease ones)
    src: np.ndarray             # [E] one orientation per undirected edge
    dst: np.ndarray
    rel: np.ndarray             # [E] index into relations
    indications: np.ndarray     # [M, 2] unique (drug, disease) pairs, contiguous node ids

    @property
    def n_nodes(self) -> int:
        return len(self.node_type)

    def nodes_of(self, type_name: str) -> np.ndarray:
        return np.where(self.node_type == self.type_names.index(type_name))[0]


def build_kg(nodes: pd.DataFrame, edges: pd.DataFrame) -> KG:
    nodes = nodes[nodes.node_type.isin(KEEP_TYPES)].reset_index(drop=True)
    new_id = pd.Series(np.arange(len(nodes)), index=nodes.node_index)
    type_names = list(KEEP_TYPES)
    node_type = nodes.node_type.map({t: i for i, t in enumerate(type_names)}).to_numpy()

    e = edges[edges.x_index.isin(new_id.index) & edges.y_index.isin(new_id.index)]
    e = e[~e.relation.isin(DROP_RELATIONS)].copy()
    e["x"], e["y"] = new_id[e.x_index].to_numpy(), new_id[e.y_index].to_numpy()

    drug = type_names.index("drug")
    ind = e[(e.relation == "indication") & (node_type[e.x.to_numpy()] == drug)]
    indications = np.unique(ind[["x", "y"]].to_numpy(), axis=0)

    msg = e[~e.relation.isin(DRUG_DISEASE_RELATIONS)].copy()
    lo, hi = np.minimum(msg.x, msg.y), np.maximum(msg.x, msg.y)  # PrimeKG lists both directions
    msg = pd.DataFrame({"src": lo.to_numpy(), "dst": hi.to_numpy(), "relation": msg.relation.to_numpy()})
    msg = msg.drop_duplicates().reset_index(drop=True)
    relations = sorted(msg.relation.unique())
    rel = msg.relation.map({r: i for i, r in enumerate(relations)}).to_numpy()

    return KG(node_type, type_names, nodes.node_name.to_numpy(), relations,
              msg.src.to_numpy(), msg.dst.to_numpy(), rel, indications)


def load_primekg(raw: Path = RAW) -> KG:
    nodes = pd.read_csv(raw / "nodes.tab", sep="\t")
    edges = pd.read_csv(raw / "edges.csv", usecols=["relation", "x_index", "y_index"])
    return build_kg(nodes, edges)
