# Stage1 Product Eval (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`

Baseline thresholds

- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.8920 | [0.8720, 0.9110] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0010 | [0.0000, 0.0030] |
| ok_rate | 0.9990 | [0.9970, 1.0000] |
| coverage | 0.1080 | [0.0890, 0.1280] |
| accuracy_on_decided | 0.9907 | [0.9691, 1.0000] |

Tuned thresholds

- tau: `0.6271186440677967`
- t_lower: `0.04040816326530612`
- t_upper: `0.07061224489795918`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.2820 | [0.2550, 0.3100] |
| fp_accept_rate | 0.0390 | [0.0280, 0.0510] |
| fn_reject_rate | 0.0490 | [0.0360, 0.0630] |
| ok_rate | 0.9120 | [0.8940, 0.9290] |
| coverage | 0.7180 | [0.6900, 0.7450] |
| accuracy_on_decided | 0.8774 | [0.8524, 0.9013] |

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high even when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use these together to judge product readiness.

Repro command

```
scripts/eval_stage1_product.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --n 1000 --out_md reports/stage1_product_eval_fever_joint.md
```
