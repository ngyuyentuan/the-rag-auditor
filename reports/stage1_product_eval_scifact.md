# Stage1 Product Eval (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: 776
- seed: 14
- logit_col: `raw_max_top3`
- y_col: `y`

Baseline thresholds

- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value | 95% CI |
|---|---:|---:|
| uncertain_rate | 0.4845 | [0.4485, 0.5206] |
| fp_accept_rate | 0.0425 | [0.0284, 0.0567] |
| fn_reject_rate | 0.0322 | [0.0206, 0.0451] |
| ok_rate | 0.9253 | [0.9072, 0.9433] |
| coverage | 0.5155 | [0.4794, 0.5515] |
| accuracy_on_decided | 0.8550 | [0.8210, 0.8900] |

Tuned thresholds

- tuned: not available

Interpretation

ok_rate treats UNCERTAIN as safe deferral, so it can look high even when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use these together to judge product readiness.

Repro command

```
scripts/eval_stage1_product.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_tuned_scifact.yaml --n 1000 --out_md reports/stage1_product_eval_scifact.md
```
