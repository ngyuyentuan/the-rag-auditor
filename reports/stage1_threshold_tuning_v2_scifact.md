# Stage1 Threshold Tuning v2 (scifact)

Generated: `2025-12-31T17:23:33.362406+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- tau: `0.609483051351548`
- mode: `accept_only`

Baseline thresholds

- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value |
|---|---|
| accept_rate | 0.4536 |
| reject_rate | 0.0619 |
| uncertain_rate | 0.4845 |
| fp_accept_rate | 0.0425 |
| fn_reject_rate | 0.0322 |
| ok_rate | 0.9253 |

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

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|
| 0.0000 | 0.8010 | 0.4497 | 0.0000 | 0.5503 | 0.0425 | 0.0000 | 0.9575 |

Top 10 feasible configs

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|
| 0.0000 | 0.8010 | 0.4497 | 0.0000 | 0.5503 | 0.0425 | 0.0000 | 0.9575 |

Constraints fp<=0.05 fn<=0.01 uncertain<=0.3

- no feasible configs

Constraints fp<=0.05 fn<=0.01 uncertain<=0.5

- no feasible configs

Constraints fp<=0.05 fn<=0.01 uncertain<=0.7

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|
| 0.0000 | 0.8010 | 0.4497 | 0.0000 | 0.5503 | 0.0425 | 0.0000 | 0.9575 |

Top 10 feasible configs

| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|
| 0.0000 | 0.8010 | 0.4497 | 0.0000 | 0.5503 | 0.0425 | 0.0000 | 0.9575 |

Pareto view

- fp<=0.02 uncertain<=0.3: no feasible configs
- fp<=0.02 uncertain<=0.5: no feasible configs
- fp<=0.02 uncertain<=0.7: no feasible configs
- fp<=0.05 uncertain<=0.3: no feasible configs
- fp<=0.05 uncertain<=0.5: no feasible configs
- fp<=0.05 uncertain<=0.7: t_lower=0.0000 t_upper=0.8010 fp=0.0425 fn=0.0000 uncertain=0.5503

Interpretation

accept_only is recommended for product safety because it avoids catastrophic false rejects by deferring most cases to Stage2. Choose lower max_fp_accept for higher safety and adjust max_uncertain to balance Stage2 capacity and user experience.
