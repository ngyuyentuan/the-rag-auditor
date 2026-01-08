# Stage1 Action-CI Tuning (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `200`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- tau_source: `config`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `25`
- threshold_steps: `50`
- min_coverage: `0.3`
- min_accept_count: `50`
- min_reject_count: `50`
- max_fp_given_accept_upper95: `0.15`
- max_fn_given_reject_upper95: `0.1`
- max_reject_rate: `0.95`
- certify: `accept_only`

- feasible_count: `1`

Candidate feasibility

| candidate | tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_candidate | 0.6095 | 0.7792 | 0.8009 | 0.5250 | 89 | 16 | 0.1394 | 0.5560 | PASS | N/A | ok |

Top candidates

| tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | accuracy_on_decided |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6095 | 0.7792 | 0.8009 | 0.5250 | 89 | 16 | 0.1394 | 0.5560 | PASS | N/A | 0.8952 |

Selected config

- accept_status: `PASS`
- reject_status: `N/A`
- trivial_reject_all: `False`
- trivial_accept_none: `False`
- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`
- coverage: `0.525`
- accept_count: `89`
- reject_count: `16`
- fp_given_accept_upper95: `0.139369447982017`
- fn_given_reject_upper95: `0.5559606813326594`
- accuracy_on_decided: `0.8952380952380953`

Repro command

```
scripts/stage1_action_ci_tune.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --n 200 --seed 14 --certify accept_only --min_accept_count 50 --min_coverage 0.30 --max_fp_given_accept_upper95 0.15 --out_md reports/stage1_action_ci_tune_scifact_accept.md --out_yaml configs/thresholds_stage1_action_accept_scifact.yaml
```
