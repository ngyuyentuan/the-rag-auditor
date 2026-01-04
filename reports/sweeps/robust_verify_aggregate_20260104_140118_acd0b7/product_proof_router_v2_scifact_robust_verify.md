# SciFact Robust Verify Aggregate (Filtered v2)

Discovery summary

- in_dirs: ['runs/sweeps/robust_verify_20260104_200820_b81003', 'runs/sweeps/20260103_123758_43c1ad']
- glob: `*.jsonl`
- recursive: True
- files_matched_total: 57
- files_kept_scifact: 57
- files_kept_after_tag_filter: 7
- files_with_budget: 7
- files_missing_budget: 0
- include_tag: `d2_lr0.1_leaf20_l20.0`
- dropped_tag_mismatch: 50
- unknown_tag_dropped: 1

Top 20 kept file paths

- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.30.jsonl

Budget summary

| tag | budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst | n_files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d2_lr0.1_leaf20_l20.0 | 0.220 | 0.9850 | 0.0100 | 0.0050 | 0.1500 | 0.2450 | 2050.9293 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.240 | 0.9850 | 0.0100 | 0.0050 | 0.1550 | 0.2250 | 2050.9293 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.260 | 0.9850 | 0.0100 | 0.0050 | 0.1600 | 0.2050 | 2066.2319 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.280 | 0.9850 | 0.0100 | 0.0050 | 0.1700 | 0.1850 | 2076.9164 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.300 | 0.9850 | 0.0100 | 0.0050 | 0.1700 | 0.1650 | 2076.9164 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.320 | 0.9850 | 0.0100 | 0.0050 | 0.1850 | 0.1450 | 2077.1122 | 1 |
| d2_lr0.1_leaf20_l20.0 | 0.350 | 0.9850 | 0.0100 | 0.0050 | 0.1850 | 0.1150 | 2077.1122 | 1 |

Ship recommendation

- group=(tag=d2_lr0.1_leaf20_l20.0, budget=0.280) fp_worst=0.0100 fn_worst=0.0050 ran_worst=0.1700 cap_worst=0.1850 ok_mean=0.9850 p95_worst=2076.9164
- worst_files:
  - fp: ['runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl']
  - fn: ['runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl']
  - ran: ['runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl']
  - cap: ['runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl']
  - p95: ['runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl']

Unusable files (missing budget)


Dropped by tag mismatch

- runs/sweeps/robust_verify_20260104_200820_b81003/baseline_seed_5_500.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf20_l21.0/scifact_router_v2_d3_lr0.05_leaf20_l21.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf50_l20.0/scifact_router_v2_d2_lr0.1_leaf50_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf20_l21.0/scifact_router_v2_d3_lr0.1_leaf20_l21.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.05_leaf50_l21.0/scifact_router_v2_d3_lr0.05_leaf50_l21.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf20_l20.0/scifact_router_v2_d2_lr0.05_leaf20_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.30.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d3_lr0.1_leaf50_l21.0/scifact_router_v2_d3_lr0.1_leaf50_l21.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.05_leaf50_l20.0/scifact_router_v2_d2_lr0.05_leaf50_l20.0_budget_0.30.jsonl
