# Stage1 Action-CI Tuning (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `1000`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau_source: `config`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `25`
- threshold_steps: `50`
- min_coverage: `0.3`
- min_accept_count: `10`
- min_reject_count: `200`
- max_fp_given_accept_upper95: `0.1`
- max_fn_given_reject_upper95: `0.15`
- max_reject_rate: `0.95`
- certify: `reject_only`

- feasible_count: `174`

Candidate feasibility

| candidate | tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_candidate | 0.9957 | 0.0722 | 1.0000 | 0.1080 | 0 | 108 | 0.0000 | 0.0506 | N/A | INSUFFICIENT_N | coverage,reject_count |

Top candidates

| tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | accuracy_on_decided |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9957 | 0.1616 | 0.1716 | 0.9710 | 39 | 932 | 0.8735 | 0.1091 | N/A | PASS | 0.8836 |
| 0.9957 | 0.1414 | 0.1514 | 0.9460 | 97 | 849 | 0.8876 | 0.1068 | N/A | PASS | 0.8383 |
| 0.9957 | 0.1616 | 0.1918 | 0.9420 | 10 | 932 | 0.8922 | 0.1091 | N/A | PASS | 0.9045 |
| 0.9957 | 0.1616 | 0.2120 | 0.9330 | 1 | 932 | 1.0000 | 0.1091 | N/A | PASS | 0.9100 |
| 0.9957 | 0.1616 | 0.2322 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |
| 0.9957 | 0.1616 | 0.2524 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |
| 0.9957 | 0.1616 | 0.2727 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |
| 0.9957 | 0.1616 | 0.2929 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |
| 0.9957 | 0.1616 | 0.3131 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |
| 0.9957 | 0.1616 | 0.3333 | 0.9320 | 0 | 932 | 0.0000 | 0.1091 | N/A | PASS | 0.9109 |

Selected config

- accept_status: `N/A`
- reject_status: `PASS`
- trivial_reject_all: `False`
- trivial_accept_none: `False`
- tau: `0.9957177012600404`
- t_lower: `0.16163265306122448`
- t_upper: `0.1716326530612245`
- coverage: `0.971`
- accept_count: `39`
- reject_count: `932`
- fp_given_accept_upper95: `0.8735433757121625`
- fn_given_reject_upper95: `0.10906920799917767`
- accuracy_on_decided: `0.8836251287332647`

Repro command

```
scripts/stage1_action_ci_tune.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --seed 14 --certify reject_only --min_reject_count 200 --min_accept_count 10 --min_coverage 0.30 --max_fn_given_reject_upper95 0.15 --out_md reports/stage1_action_ci_tune_fever_reject.md --out_yaml configs/thresholds_stage1_action_reject_fever.yaml
```
