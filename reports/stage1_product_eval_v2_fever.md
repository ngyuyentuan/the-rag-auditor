# Stage1 Product Eval V2 (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: 200
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- claim_col: `None`
- passage_col: `None`
- cheap_checks: `off`

## baseline (checks off)

- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.8800 | [0.8300, 0.9200] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0050 | [0.0000, 0.0150] |
| risk | 0.0050 | [0.0000, 0.0150] |
| ok_rate | 0.9950 | [0.9850, 1.0000] |
| coverage | 0.1200 | [0.0800, 0.1700] |
| accuracy_on_decided | 0.9583 | [0.8571, 1.0000] |

## joint_tuned (checks off)

- tau: `0.6271186440677967`
- t_lower: `0.04040816326530612`
- t_upper: `0.07061224489795918`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.2450 | [0.1850, 0.3100] |
| fp_accept_rate | 0.0550 | [0.0250, 0.0876] |
| fn_reject_rate | 0.0550 | [0.0250, 0.0900] |
| risk | 0.1100 | [0.0700, 0.1550] |
| ok_rate | 0.8900 | [0.8450, 0.9300] |
| coverage | 0.7550 | [0.6900, 0.8150] |
| accuracy_on_decided | 0.8543 | [0.7922, 0.9091] |

## product_tuned (checks off)

- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.8800 | [0.8400, 0.9200] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0050 | [0.0000, 0.0150] |
| risk | 0.0050 | [0.0000, 0.0150] |
| ok_rate | 0.9950 | [0.9850, 1.0000] |
| coverage | 0.1200 | [0.0800, 0.1600] |
| accuracy_on_decided | 0.9583 | [0.8617, 1.0000] |

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use risk, coverage, and accuracy_on_decided together when choosing a product default.

Repro command

```
scripts/eval_stage1_product_v2.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --baseline_yaml configs/thresholds.yaml --joint_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --product_yaml configs/thresholds_stage1_product_fever.yaml --n 200 --bootstrap 500 --cheap_checks off --out_md reports/stage1_product_eval_v2_fever.md
```
