# Product Proof Cost Matrix

## scifact

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact_baseline_uncertain_only_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 2211.53 | 5279.83 | 25 | 25 |
| scifact_baseline_always_50_real.jsonl | 0.4400 | 0.0600 | 0.5000 | 1.0000 | 0.0400 | 0.0000 | 0.9600 | 4445.03 | 5584.49 | 50 | 50 |
| scifact_cost_uncertain_only_50.jsonl | 0.1200 | 0.0000 | 0.8800 | 0.5000 | 0.0200 | 0.0000 | 0.9800 | 2211.53 | 5279.83 | 25 | 25 |
| scifact_cost_always_50.jsonl | 0.1200 | 0.0000 | 0.8800 | 1.0000 | 0.0200 | 0.0000 | 0.9800 | 4445.03 | 5584.49 | 50 | 50 |

## fever

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fever_baseline_uncertain_only_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 0.9400 | 0.0000 | 0.0200 | 0.9800 | 986.58 | 1589.76 | 47 | 47 |
| fever_baseline_always_50_real.jsonl | 0.0000 | 0.0600 | 0.9400 | 1.0000 | 0.0000 | 0.0200 | 0.9800 | 1009.21 | 1298.14 | 50 | 50 |
| fever_cost_uncertain_only_50.jsonl | 0.0000 | 0.8800 | 0.1200 | 0.9400 | 0.0000 | 0.1600 | 0.8400 | 986.58 | 1589.76 | 47 | 47 |
| fever_cost_always_50.jsonl | 0.0000 | 0.8800 | 0.1200 | 1.0000 | 0.0000 | 0.1600 | 0.8400 | 1009.21 | 1298.14 | 50 | 50 |
