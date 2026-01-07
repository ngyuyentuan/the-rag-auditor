# Stage1 Product Profiles (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: 776
- seed: 14
- logit_col: `raw_max_top3`
- y_col: `y`
- tau: 0.609483051351548

## Profile: safety_first
- t_lower: 0.0
- t_upper: 0.7979591836734694
- accept_rate: 0.5541
- reject_rate: 0.0000
- uncertain_rate: 0.4459
- fp_accept_rate: 0.0644
- fn_reject_rate: 0.0000
- ok_rate: 0.9356
Product interpretation: prioritizes minimizing wrong decisions; high defer if needed.

## Profile: coverage_first
- t_lower: 0.7677551020408163
- t_upper: 0.8181632653061224
- accept_rate: 0.0193
- reject_rate: 0.0142
- uncertain_rate: 0.9665
- fp_accept_rate: 0.0000
- fn_reject_rate: 0.0077
- ok_rate: 0.9923
Product interpretation: coverage-first under 0.05 fp/fn caps.

## Repro command
```
scripts/stage1_product_profile.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --n 1000 --out_md reports/stage1_product_profile_scifact.md
```
