# Stage1 Router v2 Train (scifact)

Generated: `2026-01-02T16:06:58.864482+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- feature_cols: `raw_max_top3, abs_logit, logit_sq, logit_sigmoid, logit_cube, cs_ret, top1_sim, top2_sim, top3_sim, delta12, count_sim_ge_t, margin, topk_gap, topk_ratio, topk_entropy, score_span, topk_mean, topk_std`
- model_type: `logreg`
- objective: `min_route`
- out_model: `models/stage1_router_v2_scifact_min_route_logreg.joblib`
- threshold_selection: `full_data`

Validation

- auc: `0.7206`
- ece: `0.0769`
- mean_prob_y1: `0.8319`
- mean_prob_y0: `0.7732`

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
- t_reject_max: `0.6162738888083599`
- stage2_budget: `0.45`
- t_accept: `0.8700`
- t_reject: `0.6100`
- fp_accept_rate: `0.0077`
- fn_reject_rate: `0.0077`
- uncertain_rate: `0.7436`
- stage2_route_rate: `0.7436`
- expected_cost: `0.8982`
- ok_rate: `0.9845`

Per c_stage2

| c_stage2 | t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible | penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.8700 | 0.6100 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 0.8982 | True | 0.0000 |
| 2.00 | 0.8700 | 0.6100 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 1.6418 | True | 0.0000 |
| 5.00 | 0.8700 | 0.6100 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 3.8724 | True | 0.0000 |
| 10.00 | 0.8700 | 0.6100 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 7.5902 | True | 0.0000 |

Top 10 by route

| t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8700 | 0.6100 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 0.8982 | True |
| 0.8700 | 0.6150 | 0.0077 | 0.0077 | 0.7436 | 0.7436 | 0.8982 | True |
| 0.8700 | 0.6050 | 0.0077 | 0.0077 | 0.7461 | 0.7461 | 0.9008 | True |
| 0.8700 | 0.6000 | 0.0077 | 0.0077 | 0.7474 | 0.7474 | 0.9021 | True |
| 0.8700 | 0.5950 | 0.0077 | 0.0077 | 0.7500 | 0.7500 | 0.9046 | True |
| 0.8700 | 0.5800 | 0.0077 | 0.0077 | 0.7513 | 0.7513 | 0.9059 | True |
| 0.8700 | 0.5850 | 0.0077 | 0.0077 | 0.7513 | 0.7513 | 0.9059 | True |
| 0.8700 | 0.5900 | 0.0077 | 0.0077 | 0.7513 | 0.7513 | 0.9059 | True |
| 0.8700 | 0.5750 | 0.0077 | 0.0052 | 0.7539 | 0.7539 | 0.8827 | True |
| 0.8700 | 0.5700 | 0.0077 | 0.0039 | 0.7564 | 0.7564 | 0.8724 | True |

Calibration bins

| bin | count | avg_prob | frac_pos |
|---|---:|---:|---:|
| [0.00,0.10) | 0 | 0.0000 | 0.0000 |
| [0.10,0.20) | 0 | 0.0000 | 0.0000 |
| [0.20,0.30) | 0 | 0.0000 | 0.0000 |
| [0.30,0.40) | 0 | 0.0000 | 0.0000 |
| [0.40,0.50) | 0 | 0.0000 | 0.0000 |
| [0.50,0.60) | 0 | 0.0000 | 0.0000 |
| [0.60,0.70) | 9 | 0.6556 | 0.4444 |
| [0.70,0.80) | 40 | 0.7659 | 0.6500 |
| [0.80,0.90) | 97 | 0.8496 | 0.8969 |
| [0.90,1.00) | 10 | 0.9123 | 1.0000 |