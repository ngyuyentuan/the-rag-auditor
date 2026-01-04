# Stage1 Threshold Tuning v2 (fever)

Generated: `2025-12-31T17:23:48.223706+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `5715`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau: `0.9957177012600404`
- mode: `accept_only`

Baseline thresholds

- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.1081 |
| uncertain_rate | 0.8919 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0052 |
| ok_rate | 0.9948 |

Constraints and best configs

Constraints fp<=0.02 fn<=0.0 uncertain<=0.3

- no feasible configs

Constraints fp<=0.02 fn<=0.0 uncertain<=0.5

- no feasible configs

Constraints fp<=0.02 fn<=0.0 uncertain<=0.7

- no feasible configs

Constraints fp<=0.02 fn<=0.01 uncertain<=0.3

- no feasible configs

Constraints fp<=0.02 fn<=0.01 uncertain<=0.5

- no feasible configs

Constraints fp<=0.02 fn<=0.01 uncertain<=0.7

- no feasible configs

Constraints fp<=0.05 fn<=0.0 uncertain<=0.3

- no feasible configs

Constraints fp<=0.05 fn<=0.0 uncertain<=0.5

- no feasible configs

Constraints fp<=0.05 fn<=0.0 uncertain<=0.7

- no feasible configs

Constraints fp<=0.05 fn<=0.01 uncertain<=0.3

- no feasible configs

Constraints fp<=0.05 fn<=0.01 uncertain<=0.5

- no feasible configs

Constraints fp<=0.05 fn<=0.01 uncertain<=0.7

- no feasible configs

Pareto view

- fp<=0.02 uncertain<=0.3: no feasible configs
- fp<=0.02 uncertain<=0.5: no feasible configs
- fp<=0.02 uncertain<=0.7: no feasible configs
- fp<=0.05 uncertain<=0.3: no feasible configs
- fp<=0.05 uncertain<=0.5: no feasible configs
- fp<=0.05 uncertain<=0.7: no feasible configs

Interpretation

accept_only is recommended for product safety because it avoids catastrophic false rejects by deferring most cases to Stage2. Choose lower max_fp_accept for higher safety and adjust max_uncertain to balance Stage2 capacity and user experience.
