# Stage1 Tau Regression Explainer (fever)

Generated: `2026-01-06T15:09:33.653452+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n_raw: `5715`
- n_used: `1000`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`

Baseline thresholds

- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.1080 |
| uncertain_rate | 0.8920 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0010 |
| ok_rate | 0.9990 |
| coverage | 0.1080 |
| accuracy_on_decided | 0.9907 |

Baseline cs_ret quantiles

| q | value |
|---:|---:|
| 0.00 | 0.040821 |
| 0.01 | 0.055743 |
| 0.10 | 0.071785 |
| 0.50 | 0.105770 |
| 0.90 | 0.150233 |
| 0.99 | 0.190938 |
| 1.00 | 0.213920 |

Candidate thresholds

- tau: `1.7559322033898306`
- t_lower: `0.24244897959183673`
- t_upper: `0.29285714285714287`

| metric | value |
|---|---|
| accept_rate | 0.0280 |
| reject_rate | 0.6460 |
| uncertain_rate | 0.3260 |
| fp_accept_rate | 0.0220 |
| fn_reject_rate | 0.0480 |
| ok_rate | 0.9300 |
| coverage | 0.6740 |
| accuracy_on_decided | 0.8961 |

Candidate cs_ret quantiles

| q | value |
|---:|---:|
| 0.00 | 0.143055 |
| 0.01 | 0.167343 |
| 0.10 | 0.189780 |
| 0.50 | 0.229613 |
| 0.90 | 0.272379 |
| 0.99 | 0.306020 |
| 1.00 | 0.323441 |

Delta (candidate - baseline)

| metric | delta |
|---|---:|
| accept_rate | 0.0280 |
| reject_rate | 0.5380 |
| uncertain_rate | -0.5660 |
| fp_accept_rate | 0.0220 |
| fn_reject_rate | 0.0470 |
| ok_rate | -0.0690 |
| coverage | 0.5660 |
| accuracy_on_decided | -0.0946 |

Why no feasible configs can happen

- baseline fp_accept_rate=0.0000, fn_reject_rate=0.0010, uncertain_rate=0.8920
- candidate fp_accept_rate=0.0220, fn_reject_rate=0.0480, uncertain_rate=0.3260
- if fp or fn increases while uncertain decreases, budgets with fp/fn caps can become infeasible.

Repro command

```
scripts/explain_stage1_tau_regression.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --baseline_yaml configs/thresholds.yaml --candidate_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --out_md reports/stage1_tau_regression_explainer_fever.md
```
