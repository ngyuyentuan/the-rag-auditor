# Stage1 Product Profiles (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- tau: 0.9957177012600404

## Profile: safety_first
- t_lower: 0.08081632653061224
- t_upper: 0.23224489795918368
- accept_rate: 0.0000
- reject_rate: 0.1750
- uncertain_rate: 0.8250
- fp_accept_rate: 0.0000
- fn_reject_rate: 0.0040
- ok_rate: 0.9960
Product interpretation: prioritizes minimizing wrong decisions; high defer if needed.

## Profile: coverage_first
- t_lower: 0.1010204081632653
- t_upper: 0.1716326530612245
- accept_rate: 0.0390
- reject_rate: 0.4400
- uncertain_rate: 0.5210
- fp_accept_rate: 0.0300
- fn_reject_rate: 0.0290
- ok_rate: 0.9410
Product interpretation: coverage-first under 0.05 fp/fn caps.

## Repro command
```
scripts/stage1_product_profile.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --out_md reports/stage1_product_profile_fever.md
```
