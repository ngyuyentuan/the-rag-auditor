# SciFact HGB Param Sweep Summary

Constraints: fp<=0.01, fn<=0.01, stage2_ran_rate<=0.25, capped_rate<=0.20

| tag | max_depth | learning_rate | min_samples_leaf | l2 | best_budget | ok_rate_stage1 | fp | fn | stage2_ran_rate | capped_rate | p95_ms | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d2_lr0.05_leaf20_l20.0 | 2 | 0.05 | 20 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d2_lr0.05_leaf50_l20.0 | 2 | 0.05 | 50 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d2_lr0.1_leaf20_l20.0 | 2 | 0.1 | 20 | 0.0 | 0.28 | 0.9850 | 0.0100 | 0.0050 | 0.1700 | 0.1850 | 2076.93 | True |
| d2_lr0.1_leaf50_l20.0 | 2 | 0.1 | 50 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d3_lr0.05_leaf20_l21.0 | 3 | 0.05 | 20 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d3_lr0.05_leaf50_l21.0 | 3 | 0.05 | 50 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d3_lr0.1_leaf20_l21.0 | 3 | 0.1 | 20 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| d3_lr0.1_leaf50_l21.0 | 3 | 0.1 | 50 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |

Ship recommendation

- d2_lr0.1_leaf20_l20.0 budget=0.28 ok_rate_stage1=0.9850 fp=0.0100 fn=0.0050 stage2_ran_rate=0.1700 capped_rate=0.1850

Repro commands

- scifact_in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- baseline_jsonl: `runs/product_proof_router_v2/scifact_baseline_uncertain_only_200_real.jsonl`
- budgets: `0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.35`