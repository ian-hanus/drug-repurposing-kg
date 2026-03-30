# Cold-start drug repurposing on PrimeKG: can a relational GNN rank diseases for drugs with no known indications?

## Abstract

We test whether a relational graph neural network on the PrimeKG biomedical knowledge graph can rank the diseases a drug treats when the drug has no known indications (cold-start split by drug, 3 random splits). An R-GCN with a free embedding per node memorised the training drugs (test MRR 0.068 ± 0.009). Giving all drugs one shared input embedding, so that a drug is represented only through its neighbourhood, raised test MRR to 0.156 ± 0.024 and gave the best mean rank (205 ± 21). A simple baseline that scores diseases by those of the 25 most similar training drugs still did better on MRR (0.227 ± 0.030) and Hits@10 (0.331 ± 0.028). Results are retrospective rankings against known labels, not clinical evidence.

![models](results/fig_models.png)

## 1. Question

Can a GNN on PrimeKG predict which diseases a drug treats for drugs it has never seen an indication for, and does it beat simple similarity baselines?

## 2. Methods

**Graph.** [PrimeKG](https://github.com/mims-harvard/PrimeKG) restricted to drugs, diseases, genes/proteins, effects/phenotypes, pathways and exposures: 71,353 nodes, 748,275 undirected edges, 14 relations. Drug-drug interactions and the anatomy and gene-ontology subgraphs are dropped to keep CPU training practical.

**Target.** The 9,388 drug-disease `indication` pairs (1,801 drugs, 1,363 diseases).

**Leakage control.** Every drug-disease edge (indication, contraindication, off-label use) is removed from the message-passing graph for all drugs. A drug is represented only through its targets, side effects and other non-drug-disease relations. Keeping contraindications for some drugs would leak labels or create a train/test mismatch.

**Split.** Whole drugs are held out (80/10/10 by drug, 3 random seeds), so a test drug has no known indications.

**Evaluation.** Each held-out indication is ranked against all 1,363 candidate diseases. Ranks are filtered (a drug's other true diseases do not compete) and ties receive their average rank. Random scoring has a mean rank of about 680.

**Models.**

| model | description |
|---|---|
| `random`, `popularity` | sanity floor; popularity ranks diseases by how many training drugs they treat |
| `drug_knn` | diseases of the 25 most similar training drugs (cosine over targets and side effects) |
| `target_overlap` | overlap between a drug's targets and a disease's associated genes |
| `distmult` | shallow KG embeddings trained on all graph triples |
| `rgcn` | 2-layer R-GCN with a free embedding per node, DistMult decoder |
| `rgcn_inductive` | same R-GCN, but all drugs share one input embedding, so a drug's representation comes only from its neighbourhood |

Dimension 64, learning rate 5e-3 and 2 layers were fixed; there was no hyperparameter search.

## 3. Results

Mean ± std over 3 cold-start splits:

| model | MRR | Hits@10 | Hits@50 | mean rank |
|---|---|---|---|---|
| random | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.033 ± 0.004 | 690 ± 10 |
| popularity | 0.054 ± 0.009 | 0.108 ± 0.019 | 0.277 ± 0.019 | 305 ± 20 |
| target_overlap | 0.029 ± 0.006 | 0.052 ± 0.002 | 0.148 ± 0.007 | 573 ± 4 |
| **drug_knn** | **0.227 ± 0.030** | **0.331 ± 0.028** | **0.520 ± 0.041** | 286 ± 17 |
| distmult | 0.019 ± 0.009 | 0.032 ± 0.018 | 0.099 ± 0.048 | 578 ± 107 |
| rgcn | 0.068 ± 0.009 | 0.131 ± 0.025 | 0.324 ± 0.031 | 299 ± 21 |
| rgcn_inductive | 0.156 ± 0.024 | 0.270 ± 0.038 | 0.489 ± 0.060 | **205 ± 21** |

The inductive R-GCN loses to `drug_knn` on MRR in all 3 splits (by 0.02 to 0.12) but has the best mean rank. See [`results/summary.md`](results/summary.md).

**Deviations from the planned protocol.**
- *Premature early stopping, seed 1.* The first inductive run on seed 1 stopped at step 70 (patience of 5 evaluations during a slow start) and scored MRR 0.046. Seeds 0 and 2 trained through all 400 steps. Seed 1 was re-run with patience 15, trained all 400 steps and scored 0.128. Seeds 0 and 2 never triggered early stopping, so their results are unchanged. The original row is kept in `results/rgcn_inductive_seed1_early_stopped.csv`. The re-run happened after seeing the bad number; it is a post hoc fix, and the other models kept patience 5.
- *Unequal budgets.* The plain `rgcn` and `distmult` ran 150 steps with patience 5, while `rgcn_inductive` ran 400 steps. Re-running the plain R-GCN on seed 0 with 400 steps and patience 15 (`results/ablation_rgcn_equal_budget_seed0.csv`) moved test MRR from 0.077 to 0.085, against 0.168 for the inductive version on the same split. This check covers one seed, and `distmult` was not re-run.

## 4. Discussion

1. **Free per-node embeddings are the wrong tool for cold-start.** The plain R-GCN's training loss fell from 1.15 to 0.25 while validation MRR stalled near 0.08, consistent with memorising the training drugs. `distmult` is worse, and its validation MRR fell after the second epoch.
2. **Making drugs inductive fixes most of that.** With one shared drug input embedding, validation MRR roughly doubled (0.083 to 0.167 on the development seed) and test MRR is 0.156 versus 0.068. This change was chosen from validation curves before test numbers were examined.
3. **A similarity baseline is hard to beat.** `drug_knn` has the best MRR and Hits@10. Part of that may come from near-duplicate drugs (analogues sharing targets), which a random drug split cannot rule out.
4. **Target-gene overlap alone is a weak signal**, below popularity on MRR.

## 5. Limitations

- **No temporal split.** PrimeKG is an undated snapshot, so indications approved after the graph was built could not be used as a test set.
- **Similar drugs are not separated.** Random drug holdout leaves close analogues of test drugs in training, which favours similarity-based methods such as kNN. A split holding out clusters of similar drugs would be harder and fairer; it was not run.
- **Only 3 splits**, with MRR std of about 0.02 to 0.03, so small gaps are not meaningful.
- **No hyperparameter search.** Only steps and patience were adjusted, based on validation curves.
- **No heterogeneous graph transformer.** One was planned and not completed.
- **Candidate set** is the 1,363 diseases with at least one indication in the data, easier than ranking all 17,080 diseases.
- **Not clinical evidence.** These are rankings against known labels. Nothing here supports a medical claim about any drug, and no per-drug predictions are included.

## 6. Reproduce

```bash
conda env create -f environment.yml && conda activate bioml
# download PrimeKG nodes.tab and edges.csv (Harvard Dataverse, doi:10.7910/DVN/IXA7BM) into data/raw/
python scripts/run_experiments.py --seeds 3                                    # baselines, distmult, rgcn
python scripts/run_experiments.py --seeds 3 --models rgcn_inductive --steps 400 --patience 15
python scripts/make_report.py
pytest
```
