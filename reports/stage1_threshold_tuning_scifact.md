# Stage1 Threshold Tuning (scifact)

Generated: `2026-01-06T14:08:02.635211+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- tau: `0.5618090452261306`
- tau_source: `fit`

Baseline thresholds

- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value |
|---|---|
| accept_rate | 0.8892 |
| reject_rate | 0.0026 |
| uncertain_rate | 0.1082 |
| fp_accept_rate | 0.1366 |
| fn_reject_rate | 0.0026 |
| ok_rate | 0.8608 |

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

- feasible_configs: `0`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Recommended thresholds

- none (no configs met uncertain budget)

Top 10 configs for budget <= 0.6

- feasible_configs: `0`
- baseline_candidate_feasible: `False`

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label | score |
|---|---|---|---|---|---|---|---|---|---|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Recommended thresholds

- none (no configs met uncertain budget)

YAML output

- written_to_yaml: none (no feasible config for export budget)
- out_yaml_preexisted: False
- out_yaml_removed: False

Interpretation

Selection uses constraints if provided, then objective. Pareto objective uses the frontier of lower (fp+fn) and lower uncertain, then picks the highest ok_rate. Lower uncertain budgets reduce deferrals but can increase fp or fn; higher budgets allow more deferrals and typically improve ok_rate. Use the budget that matches expected stage2 capacity.
