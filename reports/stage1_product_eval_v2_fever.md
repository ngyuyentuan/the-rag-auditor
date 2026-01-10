# Stage1 Product Eval V2 (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- runs_jsonl: `None`
- n: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- claim_col: `None`
- passage_col: `None`
- cheap_checks_requested: `auto`
- cheap_checks_effective: off (no text columns found)
- overlap_min: `0.2`


## baseline (checks off)

- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.8920 | [0.8720, 0.9110] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0010 | [0.0000, 0.0030] |
| risk | 0.0010 | [0.0000, 0.0030] |
| ok_rate | 0.9990 | [0.9970, 1.0000] |
| coverage | 0.1080 | [0.0890, 0.1280] |
| accuracy_on_decided | 0.9907 | [0.9691, 1.0000] |

## joint_tuned (checks off)

- tau: `1.1152542372881356`
- t_lower: `0.14142857142857143`
- t_upper: `0.19183673469387755`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.3110 | [0.2830, 0.3400] |
| fp_accept_rate | 0.0390 | [0.0280, 0.0510] |
| fn_reject_rate | 0.0470 | [0.0340, 0.0610] |
| risk | 0.0860 | [0.0690, 0.1040] |
| ok_rate | 0.9140 | [0.8960, 0.9310] |
| coverage | 0.6890 | [0.6600, 0.7170] |
| accuracy_on_decided | 0.8752 | [0.8491, 0.8994] |

## product_tuned (checks off)

- tau: `0.6271186440677967`
- t_lower: `0.0335593220338983`
- t_upper: `0.12745762711864406`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.4800 | [0.4490, 0.5100] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0360 | [0.0250, 0.0480] |
| risk | 0.0360 | [0.0250, 0.0480] |
| ok_rate | 0.9640 | [0.9520, 0.9750] |
| coverage | 0.5200 | [0.4900, 0.5510] |
| accuracy_on_decided | 0.9308 | [0.9084, 0.9513] |

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use risk, coverage, and accuracy_on_decided together when choosing a product default.

Repro command

```
scripts/eval_stage1_product_v2.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --baseline_yaml configs/thresholds.yaml --joint_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --product_yaml configs/thresholds_stage1_product_fever.yaml --n 1000 --bootstrap 2000 --cheap_checks auto --out_md reports/stage1_product_eval_v2_fever.md
```
