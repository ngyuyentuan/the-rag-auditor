# Product Proof Router v2 Matrix

## scifact

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | stage2_ran_rate | stage2_route_rate | capped_count | capped_rate | budget_k | stage2_should_run | missing_stage2_outputs | missing_stage2_reason_count | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | final_accept_rate | final_reject_rate | final_uncertain_rate | final_ok_rate | abstain_rate | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 | n | uncertain_count | stage2_ran_count | stage2_budget | missing_stage2_outputs_negative_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact_baseline_uncertain_only_200_real.jsonl | 0.4800 | 0.0850 | 0.4350 | 0.4350 | 0.4350 | 0.4350 | 0 | 0.0000 | na | na | na | na | 0.0400 | 0.0450 | 0.9150 | 0.6050 | 0.2250 | 0.1700 | 0.7600 | 0.1700 | 839.13 | 2333.59 | 87 | 87 | 200 | 87 | 87 | na | na |
| scifact_baseline_always_200_real.jsonl | 0.4800 | 0.0850 | 0.4350 | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0000 | na | na | na | na | 0.0400 | 0.0450 | 0.9150 | 0.4600 | 0.3700 | 0.1700 | 0.6150 | 0.1700 | 1869.30 | 2432.30 | 200 | 200 | 200 | 87 | 200 | na | na |
| scifact_router_v2_uncertain_only_200.jsonl | 0.2600 | 0.0000 | 0.7400 | 0.2300 | 0.2300 | 0.7400 | 58 | 0.2900 | 90 | 90 | 44 | 44 | 0.0100 | 0.0000 | 0.9900 | 0.3150 | 0.0800 | 0.6050 | 0.8950 | 0.6050 | 448.29 | 2053.71 | 46 | 46 | 200 | 148 | 46 | 0.4500 | 0 |
| scifact_router_v2_always_200.jsonl | 0.2600 | 0.0000 | 0.7400 | 0.7100 | 0.7100 | 0.7400 | 58 | 0.2900 | na | na | na | na | 0.0100 | 0.0000 | 0.9900 | 0.2650 | 0.2350 | 0.5000 | 0.7250 | 0.5000 | 1349.22 | 2417.39 | 142 | 142 | 200 | 148 | 142 | 0.4500 | na |

## fever

| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | stage2_ran_rate | stage2_route_rate | capped_count | capped_rate | budget_k | stage2_should_run | missing_stage2_outputs | missing_stage2_reason_count | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | final_accept_rate | final_reject_rate | final_uncertain_rate | final_ok_rate | abstain_rate | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 | n | uncertain_count | stage2_ran_count | stage2_budget | missing_stage2_outputs_negative_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fever_baseline_uncertain_only_200_real.jsonl | 0.0000 | 0.0950 | 0.9050 | 0.9050 | 0.9050 | 0.9050 | 0 | 0.0000 | na | na | na | na | 0.0000 | 0.0050 | 0.9950 | 0.2650 | 0.4750 | 0.2600 | 0.7500 | 0.2600 | 346.39 | 650.37 | 181 | 181 | 200 | 181 | 181 | na | na |
| fever_baseline_always_200_real.jsonl | 0.0000 | 0.0950 | 0.9050 | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0000 | na | na | na | na | 0.0000 | 0.0050 | 0.9950 | 0.2800 | 0.4600 | 0.2600 | 0.7350 | 0.2600 | 351.86 | 585.42 | 200 | 200 | 200 | 181 | 200 | na | na |
| fever_router_v2_uncertain_only_200.jsonl | 0.0000 | 0.1700 | 0.8300 | 0.7700 | 0.7700 | 0.8300 | 0 | 0.0000 | na | na | na | na | 0.0000 | 0.0000 | 1.0000 | 0.2650 | 0.4750 | 0.2600 | 0.7550 | 0.2600 | 294.90 | 644.95 | 154 | 154 | 200 | 166 | 154 | na | na |
| fever_router_v2_always_200.jsonl | 0.0000 | 0.1700 | 0.8300 | 1.0000 | 1.0000 | 0.8300 | 0 | 0.0000 | na | na | na | na | 0.0000 | 0.0000 | 1.0000 | 0.2800 | 0.5100 | 0.2100 | 0.7350 | 0.2100 | 351.86 | 585.42 | 200 | 200 | 200 | 166 | 200 | na | na |

Ship recommendation

- scifact: scifact_router_v2_uncertain_only_200.jsonl cost=0.5500 fp=0.0100 fn=0.0000 stage2_effective_rate=0.4500 stage2_ran_rate=0.2300 feasible=True
- fever: fever_baseline_uncertain_only_200_real.jsonl cost=0.9550 fp=0.0000 fn=0.0050 stage2_effective_rate=0.9050 stage2_ran_rate=0.9050 feasible=True