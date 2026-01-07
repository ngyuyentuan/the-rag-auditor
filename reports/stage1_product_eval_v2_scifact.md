# Stage1 Product Eval V2 (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: 200
- seed: 14
- logit_col: `raw_max_top3`
- y_col: `y`
- claim_col: `None`
- passage_col: `None`
- cheap_checks: `off`

## baseline (checks off)

- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.4750 | [0.4150, 0.5500] |
| fp_accept_rate | 0.0300 | [0.0100, 0.0550] |
| fn_reject_rate | 0.0250 | [0.0050, 0.0500] |
| risk | 0.0550 | [0.0274, 0.0850] |
| ok_rate | 0.9450 | [0.9150, 0.9726] |
| coverage | 0.5250 | [0.4500, 0.5850] |
| accuracy_on_decided | 0.8952 | [0.8380, 0.9461] |

## joint_tuned (checks off)

- tau: `0.5661016949152542`
- t_lower: `0.7879591836734694`
- t_upper: `0.8181632653061224`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.5450 | [0.4800, 0.6126] |
| fp_accept_rate | 0.0250 | [0.0100, 0.0500] |
| fn_reject_rate | 0.0050 | [0.0000, 0.0150] |
| risk | 0.0300 | [0.0100, 0.0550] |
| ok_rate | 0.9700 | [0.9450, 0.9900] |
| coverage | 0.4550 | [0.3874, 0.5200] |
| accuracy_on_decided | 0.9341 | [0.8771, 0.9783] |

## product_tuned (checks off)

- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.4750 | [0.3974, 0.5400] |
| fp_accept_rate | 0.0300 | [0.0100, 0.0550] |
| fn_reject_rate | 0.0250 | [0.0050, 0.0500] |
| risk | 0.0550 | [0.0250, 0.0850] |
| ok_rate | 0.9450 | [0.9150, 0.9750] |
| coverage | 0.5250 | [0.4600, 0.6026] |
| accuracy_on_decided | 0.8952 | [0.8436, 0.9495] |

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use risk, coverage, and accuracy_on_decided together when choosing a product default.

Repro command

```
scripts/eval_stage1_product_v2.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --baseline_yaml configs/thresholds.yaml --joint_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --product_yaml configs/thresholds_stage1_product_scifact.yaml --n 200 --bootstrap 500 --cheap_checks off --out_md reports/stage1_product_eval_v2_scifact.md
```
