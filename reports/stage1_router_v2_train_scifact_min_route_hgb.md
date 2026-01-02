# Stage1 Router v2 Train (scifact)

Generated: `2026-01-02T16:10:15.851398+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- feature_cols: `raw_max_top3, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_gap, topk_ratio, topk_entropy, score_span, topk_mean, topk_std`
- model_type: `hgb`
- objective: `min_route`
- out_model: `models/stage1_router_v2_scifact_min_route_hgb.joblib`
- threshold_selection: `full_data`

Validation

- auc: `0.6617`
- ece: `0.0179`
- mean_prob_y1: `0.8279`
- mean_prob_y0: `0.7854`

Thresholds

- alpha_list: `0.01, 0.02`
- beta_list: `0.01, 0.02`
- selected_gate: `fp<=0.01 fn<=0.01`
- gate_feasible: `True`
- gate_penalty: `0.0000`
- c_fp: `10.00`
- c_fn: `10.00`
- c_stage2_list: `1.0, 2.0, 5.0, 10.0`
- ship_c_stage2: `1.00`
- prob_flip: `False`
- t_reject_max: `0.6772521304801965`
- stage2_budget: `0.45`
- t_accept: `0.8400`
- t_reject: `0.6750`
- fp_accept_rate: `0.0090`
- fn_reject_rate: `0.0077`
- uncertain_rate: `0.5064`
- stage2_route_rate: `0.5064`
- expected_cost: `0.6740`
- ok_rate: `0.9832`

Per c_stage2

| c_stage2 | t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible | penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.8400 | 0.6750 | 0.0090 | 0.0077 | 0.5064 | 0.5064 | 0.6740 | True | 0.0000 |
| 2.00 | 0.8400 | 0.6750 | 0.0090 | 0.0077 | 0.5064 | 0.5064 | 1.1804 | True | 0.0000 |
| 5.00 | 0.8400 | 0.6750 | 0.0090 | 0.0077 | 0.5064 | 0.5064 | 2.6997 | True | 0.0000 |
| 10.00 | 0.8400 | 0.6750 | 0.0090 | 0.0077 | 0.5064 | 0.5064 | 5.2320 | True | 0.0000 |

Top 10 by route

| t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8400 | 0.6750 | 0.0090 | 0.0077 | 0.5064 | 0.5064 | 0.6740 | True |
| 0.8400 | 0.6700 | 0.0090 | 0.0077 | 0.5103 | 0.5103 | 0.6778 | True |
| 0.8400 | 0.6650 | 0.0090 | 0.0052 | 0.5206 | 0.5206 | 0.6624 | True |
| 0.8400 | 0.6600 | 0.0090 | 0.0039 | 0.5271 | 0.5271 | 0.6559 | True |
| 0.8400 | 0.6550 | 0.0090 | 0.0039 | 0.5296 | 0.5296 | 0.6585 | True |
| 0.8450 | 0.6750 | 0.0064 | 0.0077 | 0.5335 | 0.5335 | 0.6753 | True |
| 0.8400 | 0.6500 | 0.0090 | 0.0039 | 0.5374 | 0.5374 | 0.6662 | True |
| 0.8450 | 0.6700 | 0.0064 | 0.0077 | 0.5374 | 0.5374 | 0.6791 | True |
| 0.8400 | 0.6450 | 0.0090 | 0.0026 | 0.5412 | 0.5412 | 0.6572 | True |
| 0.8400 | 0.6400 | 0.0090 | 0.0013 | 0.5438 | 0.5438 | 0.6469 | True |

Calibration bins

| bin | count | avg_prob | frac_pos |
|---|---:|---:|---:|
| [0.00,0.10) | 0 | 0.0000 | 0.0000 |
| [0.10,0.20) | 0 | 0.0000 | 0.0000 |
| [0.20,0.30) | 0 | 0.0000 | 0.0000 |
| [0.30,0.40) | 0 | 0.0000 | 0.0000 |
| [0.40,0.50) | 0 | 0.0000 | 0.0000 |
| [0.50,0.60) | 0 | 0.0000 | 0.0000 |
| [0.60,0.70) | 13 | 0.6631 | 0.6154 |
| [0.70,0.80) | 44 | 0.7540 | 0.7727 |
| [0.80,0.90) | 73 | 0.8525 | 0.8356 |
| [0.90,1.00) | 26 | 0.9187 | 0.9231 |