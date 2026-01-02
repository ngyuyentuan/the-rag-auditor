# Stage1 Router v2 Train (fever)

Generated: `2026-01-02T15:03:17.516892+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `5715`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- feature_cols: `logit_platt, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_mean, topk_std`
- out_model: `models/stage1_router_v2_fever.joblib`
- threshold_selection: `full_data`

Validation

- auc: `0.8655`
- ece: `0.2242`
- mean_prob_y1: `0.7025`
- mean_prob_y0: `0.2860`

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
- t_reject_max: `0.06915042028965547`
- stage2_budget: `None`
- t_accept: `0.9800`
- t_reject: `0.0650`
- fp_accept_rate: `0.0000`
- fn_reject_rate: `0.0007`
- uncertain_rate: `0.8213`
- stage2_route_rate: `0.8213`
- expected_cost: `0.8283`
- ok_rate: `0.9993`

Per c_stage2

| c_stage2 | t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible | penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.9800 | 0.0650 | 0.0000 | 0.0007 | 0.8213 | 0.8213 | 0.8283 | True | 0.0000 |
| 2.00 | 0.9750 | 0.0650 | 0.0002 | 0.0007 | 0.8201 | 0.8201 | 1.6490 | True | 0.0000 |
| 5.00 | 0.9400 | 0.0650 | 0.0075 | 0.0007 | 0.7981 | 0.7981 | 4.0726 | True | 0.0000 |
| 10.00 | 0.9350 | 0.0650 | 0.0098 | 0.0007 | 0.7937 | 0.7937 | 8.0420 | True | 0.0000 |

Calibration bins

| bin | count | avg_prob | frac_pos |
|---|---:|---:|---:|
| [0.00,0.10) | 299 | 0.0500 | 0.0134 |
| [0.10,0.20) | 217 | 0.1429 | 0.0184 |
| [0.20,0.30) | 138 | 0.2484 | 0.0290 |
| [0.30,0.40) | 109 | 0.3496 | 0.0642 |
| [0.40,0.50) | 74 | 0.4547 | 0.0811 |
| [0.50,0.60) | 78 | 0.5547 | 0.1410 |
| [0.60,0.70) | 54 | 0.6472 | 0.2037 |
| [0.70,0.80) | 57 | 0.7503 | 0.2807 |
| [0.80,0.90) | 67 | 0.8546 | 0.4179 |
| [0.90,1.00) | 50 | 0.9402 | 0.6000 |