# Stage1 Action-CI Tuning (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `200`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau_source: `config`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `25`
- threshold_steps: `50`
- min_coverage: `0.3`
- min_accept_count: `50`
- min_reject_count: `20`
- max_fp_given_accept_upper95: `0.1`
- max_fn_given_reject_upper95: `0.15`
- certify: `reject_only`

- feasible_count: `1035`

Top candidates

| tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | accuracy_on_decided |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9957 | 0.2222 | 0.2322 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.2524 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.2727 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.2929 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.3131 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.3333 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.3535 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.3737 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.3939 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |
| 0.9957 | 0.2222 | 0.4141 | 1.0000 | 0 | 200 | 0.0000 | 0.1319 | N/A | PASS | 0.9150 |

Selected config

- accept_status: `N/A`
- reject_status: `PASS`
- tau: `0.9957177012600404`
- t_lower: `0.22224489795918367`
- t_upper: `0.23224489795918368`
- coverage: `1.0`
- accept_count: `0`
- reject_count: `200`
- fp_given_accept_upper95: `0.0`
- fn_given_reject_upper95: `0.13189688544235392`
- accuracy_on_decided: `0.915`

Repro command

```
scripts/stage1_action_ci_tune.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 200 --seed 14 --certify reject_only --min_reject_count 20 --min_coverage 0.30 --max_fn_given_reject_upper95 0.15 --out_md reports/stage1_action_ci_tune_fever_reject.md --out_yaml configs/thresholds_stage1_action_reject_fever.yaml
```
