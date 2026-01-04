# Matrix Stats (scifact)

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact_calibrated_always_50.jsonl | 0.4400 | 0.0600 | 0.5000 | 1.0000 | 0.0400 | 0.0000 | 0.9600 | 3545.13 | 6425.57 | 50 | 50 |
| scifact_calibrated_uncertain_only_50.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 2224.00 | 6267.21 | 25 | 25 |
| scifact_calibrated_uncertain_only_50_tuned.jsonl | 0.4400 | 0.0600 | 0.5000 | 0.5000 | 0.0400 | 0.0000 | 0.9600 | 2224.00 | 6267.21 | 25 | 25 |
| scifact_sample_50.jsonl | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0 | 0 |

Gold evaluation

- scifact_calibrated_always_50.jsonl: gold unavailable
- scifact_calibrated_uncertain_only_50.jsonl: gold unavailable
- scifact_calibrated_uncertain_only_50_tuned.jsonl: gold unavailable
- scifact_sample_50.jsonl: gold unavailable
