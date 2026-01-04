# Day12 Matrix Stats (fever)

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fever_calibrated_always_200.jsonl | 0.0000 | 0.0950 | 0.9050 | 1.0000 | 0.0000 | 0.0050 | 0.9950 | 491.36 | 1168.54 | 200 | 200 |
| fever_calibrated_uncertain_only_200.jsonl | 0.0000 | 0.0950 | 0.9050 | 0.9050 | 0.0000 | 0.0050 | 0.9950 | 798.66 | 1243.26 | 181 | 181 |
| fever_sample_200.jsonl | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0 | 0 |

Gold evaluation

- fever_calibrated_always_200.jsonl: gold unavailable
- fever_calibrated_uncertain_only_200.jsonl: gold unavailable
- fever_sample_200.jsonl: gold unavailable
