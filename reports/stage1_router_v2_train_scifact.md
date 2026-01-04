# Stage1 Router v2 Train (scifact)

Generated: `2026-01-02T15:03:08.088597+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- feature_cols: `raw_max_top3, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_mean, topk_std`
- out_model: `models/stage1_router_v2_scifact.joblib`
- threshold_selection: `full_data`

Validation

- auc: `0.7198`
- ece: `0.0797`
- mean_prob_y1: `0.8319`
- mean_prob_y0: `0.7734`

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
- t_reject_max: `0.6168227119679016`
- stage2_budget: `0.45`
- t_accept: `0.8700`
- t_reject: `0.0000`
- fp_accept_rate: `0.0077`
- fn_reject_rate: `0.0000`
- uncertain_rate: `0.7616`
- stage2_route_rate: `0.7616`
- expected_cost: `0.8389`
- ok_rate: `0.9923`

Per c_stage2

| c_stage2 | t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible | penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.8700 | 0.0000 | 0.0077 | 0.0000 | 0.7616 | 0.7616 | 0.8389 | True | 0.0000 |
| 2.00 | 0.8700 | 0.0000 | 0.0077 | 0.0000 | 0.7616 | 0.7616 | 1.6005 | True | 0.0000 |
| 5.00 | 0.8700 | 0.6150 | 0.0077 | 0.0077 | 0.7448 | 0.7448 | 3.8789 | True | 0.0000 |
| 10.00 | 0.8700 | 0.6150 | 0.0077 | 0.0077 | 0.7448 | 0.7448 | 7.6031 | True | 0.0000 |

Calibration bins

| bin | count | avg_prob | frac_pos |
|---|---:|---:|---:|
| [0.00,0.10) | 0 | 0.0000 | 0.0000 |
| [0.10,0.20) | 0 | 0.0000 | 0.0000 |
| [0.20,0.30) | 0 | 0.0000 | 0.0000 |
| [0.30,0.40) | 0 | 0.0000 | 0.0000 |
| [0.40,0.50) | 0 | 0.0000 | 0.0000 |
| [0.50,0.60) | 0 | 0.0000 | 0.0000 |
| [0.60,0.70) | 9 | 0.6566 | 0.4444 |
| [0.70,0.80) | 39 | 0.7652 | 0.6410 |
| [0.80,0.90) | 98 | 0.8490 | 0.8980 |
| [0.90,1.00) | 10 | 0.9116 | 1.0000 |