# Stage1 Threshold Tuning (fever)

Generated: `2026-01-06T14:07:46.951780+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `200`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau: `0.8783919597989949`
- tau_source: `fit`

Baseline thresholds

- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.3350 |
| uncertain_rate | 0.6650 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0150 |
| ok_rate | 0.9850 |

Top 10 configs for budget <= 0.3

- feasible_configs: `0`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Recommended thresholds

- none (no configs met uncertain budget)

Top 10 configs for budget <= 0.4

- feasible_configs: `0`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Recommended thresholds

- none (no configs met uncertain budget)

Top 10 configs for budget <= 0.5

- feasible_configs: `2`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| 0.0808 | 0.1716 | 0.0050 | 0.4950 | 0.5000 | 0.0050 | 0.0300 | 0.9650 | grid | 0.8650 |
| 0.0808 | 0.1514 | 0.0300 | 0.4950 | 0.4750 | 0.0250 | 0.0300 | 0.9450 | grid | 0.8500 |

Recommended thresholds

- t_lower: `0.09183673469387754`
- t_upper: `0.1769387755102041`
- fp_accept_rate: `0.0`
- fn_reject_rate: `0.05`
- ok_rate: `0.95`
- uncertain_rate: `0.335`

Top 10 configs for budget <= 0.6

- feasible_configs: `43`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| 0.0808 | 0.1918 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.2120 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.2322 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.2524 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.2727 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.2929 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.3131 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.3333 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.3535 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |
| 0.0808 | 0.3737 | 0.0000 | 0.4950 | 0.5050 | 0.0000 | 0.0300 | 0.9700 | grid | 0.8690 |

Recommended thresholds

- t_lower: `0.08081632653061224`
- t_upper: `0.19183673469387755`
- fp_accept_rate: `0.0`
- fn_reject_rate: `0.03`
- ok_rate: `0.97`
- uncertain_rate: `0.505`

YAML output

- written_to_yaml: t_lower=0.09183673469387754 t_upper=0.1769387755102041 tau=0.8783919597989949 export_budget=0.5
- out_yaml_preexisted: True
- out_yaml_removed: False

Interpretation

Selection uses constraints if provided, then objective. Pareto objective uses the frontier of lower (fp+fn) and lower uncertain, then picks the highest ok_rate. Lower uncertain budgets reduce deferrals but can increase fp or fn; higher budgets allow more deferrals and typically improve ok_rate. Use the budget that matches expected stage2 capacity.
