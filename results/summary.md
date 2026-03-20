# Results

Cold-start drug split (whole drugs held out), mean ± std over 3 random splits. Each held-out indication is ranked against 1363 candidate diseases (filtered). Random scoring has a mean rank of about 680.

| model | MRR | Hits@10 | Hits@50 | mean rank | per-drug MRR |
|---|---|---|---|---|---|
| random | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.033 ± 0.004 | 690.0 ± 9.7 | 0.006 ± 0.002 |
| popularity | 0.054 ± 0.009 | 0.108 ± 0.019 | 0.277 ± 0.019 | 304.7 ± 19.9 | 0.072 ± 0.014 |
| target_overlap | 0.029 ± 0.006 | 0.052 ± 0.002 | 0.148 ± 0.007 | 573.2 ± 3.8 | 0.032 ± 0.005 |
| drug_knn | 0.227 ± 0.030 | 0.331 ± 0.028 | 0.520 ± 0.041 | 285.9 ± 17.4 | 0.218 ± 0.024 |
| distmult | 0.019 ± 0.009 | 0.032 ± 0.018 | 0.099 ± 0.048 | 577.7 ± 106.6 | 0.025 ± 0.010 |
| rgcn | 0.068 ± 0.009 | 0.131 ± 0.025 | 0.324 ± 0.031 | 299.1 ± 21.3 | 0.066 ± 0.006 |
| rgcn_inductive | 0.156 ± 0.024 | 0.270 ± 0.038 | 0.489 ± 0.060 | 204.8 ± 21.3 | 0.162 ± 0.010 |

## Per-seed comparison against the strongest baseline

MRR difference per split (model minus drug_knn); positive means the model won that split.

| model | seed 0 | seed 1 | seed 2 | wins |
|---|---|---|---|---|
| popularity | -0.193 | -0.198 | -0.129 | 0/3 |
| target_overlap | -0.212 | -0.213 | -0.170 | 0/3 |
| distmult | -0.225 | -0.236 | -0.165 | 0/3 |
| rgcn | -0.164 | -0.190 | -0.124 | 0/3 |
| rgcn_inductive | -0.073 | -0.120 | -0.022 | 0/3 |
