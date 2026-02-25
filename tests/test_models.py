import numpy as np
import torch

from drkg.kg import KG
from drkg.models import RGCN, DistMult, build_edge_sets


def _tiny_kg() -> KG:
    # 0,1 drugs; 2,3 diseases; 4,5 genes. relation 0 = drug_protein, relation 1 = disease_protein
    src = np.array([0, 1, 2, 3])
    dst = np.array([4, 5, 4, 5])
    rel = np.array([0, 0, 1, 1])
    return KG(
        node_type=np.array([0, 0, 1, 1, 2, 2]), type_names=["drug", "disease", "gene/protein"],
        node_names=np.array(list("abcdef")), relations=["drug_protein", "disease_protein"],
        src=src, dst=dst, rel=rel, indications=np.array([[0, 2], [1, 3]]),
    )


def test_edge_sets_cover_both_directions():
    sets = build_edge_sets(_tiny_kg())
    assert len(sets) == 4  # 2 relations x 2 directions
    assert sets[0].src.tolist() == [0, 1] and sets[2].src.tolist() == [4, 5]


def test_mean_aggregation_weights_sum_to_one_per_destination():
    kg = _tiny_kg()
    es = build_edge_sets(kg)[1]  # disease_protein forward: 2->4, 3->5
    assert torch.allclose(es.norm, torch.ones(2))


def test_rgcn_representation_shape_and_gradient():
    kg = _tiny_kg()
    model = RGCN(kg.n_nodes, build_edge_sets(kg), dim=8)
    rep = model.node_repr()
    assert rep.shape == (6, 8)
    model.score(torch.tensor([0, 1]), torch.tensor([2, 3]), rep).sum().backward()
    assert model.emb.weight.grad is not None and model.layers[0].w_rel.grad is not None


def test_rgcn_drug_representation_depends_on_its_targets_not_unrelated_nodes():
    torch.manual_seed(0)
    kg = _tiny_kg()
    model = RGCN(kg.n_nodes, build_edge_sets(kg), dim=8, depth=1, dropout=0.0).eval()
    base = model.node_repr()[0].detach().clone()
    with torch.no_grad():
        model.emb.weight[4] += 1.0   # drug 0's target gene
    assert not torch.allclose(model.node_repr()[0], base)
    with torch.no_grad():
        model.emb.weight[4] -= 1.0
        model.emb.weight[5] += 1.0   # gene unrelated to drug 0 within one hop
    assert torch.allclose(model.node_repr()[0], base, atol=1e-6)


def test_distmult_is_symmetric_in_head_and_tail():
    m = DistMult(6, 2, dim=8)
    a = m.score(torch.tensor([0]), torch.tensor([1]), torch.tensor([3]))
    b = m.score(torch.tensor([3]), torch.tensor([1]), torch.tensor([0]))
    assert torch.allclose(a, b)
