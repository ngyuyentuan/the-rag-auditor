# Product Proof Constrained Matrix

## scifact

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact_baseline_uncertain_only_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 1361.82 | 3554.17 | 25 | 25 |
| scifact_baseline_always_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 1.0000 | 0.0400 | 0.0000 | 0.9600 | 10650.06 | 17639.67 | 50 | 50 |
| scifact_constrained_uncertain_only_50.jsonl | 0.1600 | 0.0000 | 0.8400 | 0.8400 | 0.0200 | 0.0000 | 0.9800 | 1361.82 | 3554.17 | 25 | 25 |
| scifact_constrained_always_50.jsonl | 0.1905 | 0.0000 | 0.8095 | 1.0000 | 0.0238 | 0.0000 | 0.9762 | 11829.09 | 17719.31 | 42 | 42 |

## fever

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fever_baseline_uncertain_only_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 0.9400 | 0.0000 | 0.0200 | 0.9800 | 405.21 | 661.22 | 47 | 47 |
| fever_baseline_always_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 1.0000 | 0.0000 | 0.0200 | 0.9800 | 4533.58 | 5998.20 | 50 | 50 |
| fever_constrained_uncertain_only_50.jsonl | 0.0400 | 0.1200 | 0.8400 | 0.8400 | 0.0200 | 0.0200 | 0.9600 | 368.44 | 661.22 | 42 | 42 |
| fever_constrained_always_50.jsonl | 0.0400 | 0.1200 | 0.8400 | 1.0000 | 0.0200 | 0.0200 | 0.9600 | 4533.58 | 5998.20 | 50 | 50 |
