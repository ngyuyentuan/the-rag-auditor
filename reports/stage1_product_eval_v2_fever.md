# Stage1 Product Eval V2 (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- runs_jsonl: `None`
- n: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- claim_col: `None`
- passage_col: `None`
- cheap_checks_requested: `off`
- cheap_checks_effective: `off`
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

- tau: `0.9974683544303797`
- t_lower: `0.11278481012658227`
- t_upper: `0.24810126582278483`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.4180 | [0.3870, 0.4480] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0420 | [0.0310, 0.0550] |
| risk | 0.0420 | [0.0310, 0.0550] |
| ok_rate | 0.9580 | [0.9450, 0.9690] |
| coverage | 0.5820 | [0.5520, 0.6130] |
| accuracy_on_decided | 0.9278 | [0.9059, 0.9469] |

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use risk, coverage, and accuracy_on_decided together when choosing a product default.

Repro command

```
scripts/eval_stage1_product_v2.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --out_md reports/stage1_product_eval_v2_fever.md
```
