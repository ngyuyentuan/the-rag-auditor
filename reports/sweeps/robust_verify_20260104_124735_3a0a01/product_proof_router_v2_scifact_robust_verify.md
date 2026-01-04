# Product Proof Router v2 SciFact Robust Verify

- models: `models/sweeps/20260103_123758_43c1ad/stage1_router_v2_scifact_d2_lr0.1_leaf20_l20.0.joblib,models/sweeps/20260103_123758_43c1ad/stage1_router_v2_scifact_d2_lr0.05_leaf20_l20.0.joblib,models/sweeps/20260103_123758_43c1ad/stage1_router_v2_scifact_d3_lr0.1_leaf20_l21.0.joblib`
- baseline_jsonl: `runs/product_proof_router_v2/scifact_baseline_uncertain_only_200_real.jsonl`
- scifact_in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- fitted_t_accept: 0.8583
- fitted_t_reject: 0.6031
- fit_n: 20000
- fit_seed: 14
- max_scan_rows: 800000
runtime mitigation: thresholds were fit on fit_n subsample; feature_map was built by bounded scan with early-stop once all qids were found; robust verify metrics are computed from n=200 runs per seed/budget

## Per-seed metrics

| seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.260 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4200 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 5 | 0.280 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4000 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 5 | 0.300 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1800 | 0.3800 | 0.0050 | 0.0050 | 0.9900 | 344.83 | 2066.72 |
| 12 | 0.260 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4200 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 12 | 0.280 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4000 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 12 | 0.300 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1800 | 0.3800 | 0.0050 | 0.0050 | 0.9900 | 344.83 | 2066.72 |
| 21 | 0.260 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4200 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 21 | 0.280 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4000 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 21 | 0.300 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1800 | 0.3800 | 0.0050 | 0.0050 | 0.9900 | 344.83 | 2066.72 |
| 33 | 0.260 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4200 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 33 | 0.280 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4000 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 33 | 0.300 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1800 | 0.3800 | 0.0050 | 0.0050 | 0.9900 | 344.83 | 2066.72 |
| 42 | 0.260 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4200 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 42 | 0.280 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1700 | 0.4000 | 0.0050 | 0.0050 | 0.9900 | 325.26 | 2051.69 |
| 42 | 0.300 | 0.2850 | 0.0350 | 0.6800 | 0.6800 | 0.1800 | 0.3800 | 0.0050 | 0.0050 | 0.9900 | 344.83 | 2066.72 |

## Budget summary

| budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst |
|---:|---:|---:|---:|---:|---:|---:|
| 0.260 | 0.9900 | 0.0050 | 0.0050 | 0.1700 | 0.4200 | 2051.69 |
| 0.280 | 0.9900 | 0.0050 | 0.0050 | 0.1700 | 0.4000 | 2051.69 |
| 0.300 | 0.9900 | 0.0050 | 0.0050 | 0.1800 | 0.3800 | 2066.72 |

## Ship recommendation

- no budget met fp_worst<=0.01 fn_worst<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case
- closest: budget=0.300 fp_worst=0.0050 fn_worst=0.0050 ran_worst=0.1800 cap_worst=0.3800 ok_mean=0.9900