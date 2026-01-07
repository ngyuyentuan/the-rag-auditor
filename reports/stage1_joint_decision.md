# Stage1 Joint Decision

SciFact summary

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided | ok_rate CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9253 | 0.0425 | 0.0322 | 0.4845 | 0.5155 | 0.8550 | [0.9072, 0.9433] |
| joint_tuned | 0.9523 | 0.0374 | 0.0103 | 0.5464 | 0.4536 | 0.8949 | [0.9369, 0.9678] |

FEVER summary

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided | ok_rate CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9990 | 0.0000 | 0.0010 | 0.8920 | 0.1080 | 0.9907 | [0.9970, 1.0000] |
| joint_tuned | 0.9120 | 0.0390 | 0.0490 | 0.2820 | 0.7180 | 0.8774 | [0.8940, 0.9290] |

Recommendation

- scifact: Stage1-only OK
- fever: Stage1-only OK

Risks & mitigations

- ok_rate can look strong while coverage remains low; use coverage and accuracy_on_decided for product readiness.
- Stage2 must report evidence hit and verdict accuracy to validate end-to-end correctness.
- Monitor fp_accept_rate and fn_reject_rate drift; alert when exceeding product gates.

Repro commands

```
wsl -e bash -lc "cd ~/the-rag-auditor && .venv/bin/python scripts/eval_stage1_product.py --track scifact --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet' --logit_col raw_max_top3 --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --n 1000 --out_md reports/stage1_product_eval_scifact_joint.md"
wsl -e bash -lc "cd ~/the-rag-auditor && .venv/bin/python scripts/eval_stage1_product.py --track fever --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet' --logit_col logit_platt --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --n 1000 --out_md reports/stage1_product_eval_fever_joint.md"
```

