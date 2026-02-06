import pandas as pd
import pytest

from drkg import kg as kgmod
from drkg.kg import build_kg

NODES = pd.DataFrame({
    "node_index": [10, 11, 12, 13, 14, 15],
    "node_type": ["drug", "drug", "disease", "disease", "gene/protein", "anatomy"],
    "node_name": ["d1", "d2", "x1", "x2", "g1", "a1"],
})
EDGES = pd.DataFrame([
    # PrimeKG lists every undirected edge in both directions
    ("indication", 10, 12), ("indication", 12, 10),
    ("indication", 11, 13), ("indication", 13, 11),
    ("contraindication", 10, 13), ("contraindication", 13, 10),
    ("drug_protein", 10, 14), ("drug_protein", 14, 10),
    ("disease_protein", 12, 14), ("disease_protein", 14, 12),
    ("drug_drug", 10, 11), ("drug_drug", 11, 10),
    ("anatomy_protein_present", 15, 14), ("anatomy_protein_present", 14, 15),
], columns=["relation", "x_index", "y_index"])


def test_drug_disease_edges_never_enter_message_graph():
    kg = build_kg(NODES, EDGES)
    assert not set(kg.relations) & kgmod.DRUG_DISEASE_RELATIONS
    assert set(kg.relations) == {"drug_protein", "disease_protein"}


def test_indications_are_unique_drug_disease_pairs():
    kg = build_kg(NODES, EDGES)
    assert len(kg.indications) == 2
    drugs = set(kg.nodes_of("drug"))
    diseases = set(kg.nodes_of("disease"))
    assert all(d in drugs and x in diseases for d, x in kg.indications)


def test_undirected_edges_are_deduplicated():
    kg = build_kg(NODES, EDGES)
    assert len(kg.src) == 2  # one drug_protein, one disease_protein
    assert (kg.src < kg.dst).all()


def test_dropped_types_and_relations_are_gone():
    kg = build_kg(NODES, EDGES)
    assert kg.n_nodes == 5  # anatomy node removed
    assert "drug_drug" not in kg.relations and "anatomy_protein_present" not in kg.relations


needs_data = pytest.mark.skipif(not (kgmod.RAW / "edges.csv").exists(), reason="PrimeKG not downloaded")


@needs_data
def test_real_primekg_shapes():
    kg = kgmod.load_primekg()
    assert len(kg.indications) == 9388
    assert len(set(kg.indications[:, 0])) == 1801
    assert not set(kg.relations) & kgmod.DRUG_DISEASE_RELATIONS
