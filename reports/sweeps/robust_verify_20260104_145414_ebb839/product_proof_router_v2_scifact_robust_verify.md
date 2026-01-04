# Product Proof Router v2 SciFact Robust Verify

- models: `models/sweeps/20260103_123758_43c1ad/stage1_router_v2_scifact_d2_lr0.1_leaf20_l20.0.joblib`
- baseline_jsonl: `runs/product_proof_router_v2/scifact_baseline_uncertain_only_200_real.jsonl`
- scifact_in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- fitted_t_accept: 0.8592
- fitted_t_reject: 0.5714
- fit_n: 2000
- fit_seed: 14
runtime mitigation: thresholds were fit on fit_n subsample; feature_map built with parquet filter on qid_set; robust verify metrics are computed from n=200 runs per seed/budget

## Per-seed metrics

| seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.280 | 0.2800 | 0.0050 | 0.7150 | 0.7150 | 0.1850 | 0.4350 | 0.0050 | 0.0000 | 0.9950 | 346.79 | 2051.69 |

## Budget summary

| budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst |
|---:|---:|---:|---:|---:|---:|---:|
| 0.280 | 0.9950 | 0.0050 | 0.0000 | 0.1850 | 0.4350 | 2051.69 |

## Ship recommendation

- no budget met fp_worst<=0.01 fn_worst<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case
- closest: budget=0.280 fp_worst=0.0050 fn_worst=0.0000 ran_worst=0.1850 cap_worst=0.4350 ok_mean=0.9950