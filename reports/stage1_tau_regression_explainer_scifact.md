# Stage1 Tau Regression Explainer (scifact)

Generated: `2026-01-06T15:09:22.056059+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n_raw: `776`
- n_used: `776`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`

Baseline thresholds

- tau: `0.609483051351548`
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
| coverage | 0.5155 |
| accuracy_on_decided | 0.8550 |

Baseline cs_ret quantiles

| q | value |
|---:|---:|
| 0.00 | 0.761396 |
| 0.01 | 0.766566 |
| 0.10 | 0.782096 |
| 0.50 | 0.799348 |
| 0.90 | 0.812472 |
| 0.99 | 0.820004 |
| 1.00 | 0.824157 |

Candidate thresholds

- tau: `0.5661016949152542`
- t_lower: `0.7879591836734694`
- t_upper: `0.8181632653061224`

| metric | value |
|---|---|
| accept_rate | 0.4278 |
| reject_rate | 0.0258 |
| uncertain_rate | 0.5464 |
| fp_accept_rate | 0.0374 |
| fn_reject_rate | 0.0103 |
| ok_rate | 0.9523 |
| coverage | 0.4536 |
| accuracy_on_decided | 0.8949 |

Candidate cs_ret quantiles

| q | value |
|---:|---:|
| 0.00 | 0.777173 |
| 0.01 | 0.782473 |
| 0.10 | 0.798324 |
| 0.50 | 0.815800 |
| 0.90 | 0.828993 |
| 0.99 | 0.836522 |
| 1.00 | 0.840661 |

Delta (candidate - baseline)

| metric | delta |
|---|---:|
| accept_rate | -0.0258 |
| reject_rate | -0.0361 |
| uncertain_rate | 0.0619 |
| fp_accept_rate | -0.0052 |
| fn_reject_rate | -0.0219 |
| ok_rate | 0.0271 |
| coverage | -0.0619 |
| accuracy_on_decided | 0.0399 |

Why no feasible configs can happen

- baseline fp_accept_rate=0.0425, fn_reject_rate=0.0322, uncertain_rate=0.4845
- candidate fp_accept_rate=0.0374, fn_reject_rate=0.0103, uncertain_rate=0.5464
- if fp or fn increases while uncertain decreases, budgets with fp/fn caps can become infeasible.

Repro command

```
scripts/explain_stage1_tau_regression.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --n 1000 --baseline_yaml configs/thresholds.yaml --candidate_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --out_md reports/stage1_tau_regression_explainer_scifact.md
```
