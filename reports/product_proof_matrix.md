# Product Proof Matrix

## scifact

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact_baseline_uncertain_only_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 4274.13 | 11136.50 | 25 | 25 |
| scifact_baseline_always_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 1.0000 | 0.0400 | 0.0000 | 0.9600 | 13829.64 | 22005.67 | 50 | 50 |
| scifact_prod_uncertain_only_50.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 4274.13 | 11136.50 | 25 | 25 |
| scifact_prod_always_50.jsonl | 0.4400 | 0.0600 | 0.5000 | 1.0000 | 0.0400 | 0.0000 | 0.9600 | 13829.64 | 22005.67 | 50 | 50 |
| scifact_baseline_uncertain_only_50_dry.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.0000 | 0.0400 | 0.0000 | 0.9600 | 0.03 | 0.04 | 0 | 0 |

## fever

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fever_baseline_uncertain_only_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 0.9400 | 0.0000 | 0.0200 | 0.9800 | 4076.18 | 6079.34 | 47 | 47 |
| fever_baseline_always_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 1.0000 | 0.0000 | 0.0200 | 0.9800 | 3821.39 | 5633.21 | 50 | 50 |
| fever_prod_uncertain_only_50.jsonl | 0.0000 | 0.0600 | 0.9400 | 0.9400 | 0.0000 | 0.0200 | 0.9800 | 4076.18 | 6079.34 | 47 | 47 |
| fever_prod_always_50.jsonl | 0.0000 | 0.0600 | 0.9400 | 1.0000 | 0.0000 | 0.0200 | 0.9800 | 3821.39 | 5633.21 | 50 | 50 |
| fever_baseline_uncertain_only_50_dry.jsonl | 0.0000 | 0.0600 | 0.9400 | 0.0000 | 0.0000 | 0.0200 | 0.9800 | 0.13 | 0.06 | 0 | 0 |
