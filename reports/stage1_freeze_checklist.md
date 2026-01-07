# Stage1 Freeze Checklist

## Repo state
- head: `84ed3ce72a884499816adf12833d87040a951e24`
- tag: `stage1_freeze`
- tag_object: `24b38babb67a6403f8646c47d5b8047edfc5b5b8`
- tag_target: `84ed3ce72a884499816adf12833d87040a951e24`
- tag_matches_head: `True`

## Environment
- python: `3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]`
- numpy: `2.4.0`
- pandas: `2.3.3`
- torch: `2.9.1+cu128`
- transformers: `4.57.3`

## Inputs
- scifact_in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- fever_in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `500`
- seed: `15`

## Thresholds
- scifact_source: `configs/thresholds_stage1_tuned_scifact.yaml`
- scifact_tau: `0.609483051351548`
- scifact_t_lower: `0.0`
- scifact_t_upper: `0.7777551020408163`
- fever_source: `configs/thresholds_stage1_tuned_fever.yaml`
- fever_tau: `0.9957177012600404`
- fever_t_lower: `0.14142857142857143`
- fever_t_upper: `0.2524489795918367`

## Stage1 distribution
### scifact
- accept_rate: `0.944`
- reject_rate: `0.0`
- uncertain_rate: `0.056`
- fp_accept_rate: `0.172`
- fn_reject_rate: `0.0`
- ok_rate: `0.8280000000000001`
### fever
- accept_rate: `0.0`
- reject_rate: `0.832`
- uncertain_rate: `0.168`
- fp_accept_rate: `0.0`
- fn_reject_rate: `0.072`
- ok_rate: `0.928`

## File hashes (sha256)
- configs/thresholds.yaml: `f14c6a1ac2cf17903ba8bb5b0ccce23fc74dc3e8721a054a8856411dfbd9b3b0`
- configs/thresholds_stage1_tuned_fever.yaml: `4d25ddeae513976ec53bf80dfdfa0dc0c29bfb80806dacfdc89a5c8a0f7ea452`
- configs/thresholds_stage1_tuned_scifact.yaml: `19c6a6b9196c2ffd8b2543826f5af073789d005baa386feb963fa0af45e465a2`
- scripts/day12_e2e_run_500.py: `1ab85ebc4e24de62be8c8d159f822881d9fff6d98d9f483797228c527a095b93`
- scripts/demo_stage1.py: `9b81fb221cce0920b77a9f563f4e66d8233fe860269249f2aea502b9edbfa79e`
- scripts/tune_stage1_thresholds.py: `535d116aa2c4bf831242780c451b37aa8ce3181f689df53402ecc751c1151923`
- src/utils/buffer.py: `2f5f15016f3981242ac9aca2cfad7d0a1207a477cc2ffc4715822de57832215f`

## Reproduce commands
```bash
.venv/bin/python scripts/demo_stage1.py --track scifact --n 500 --seed 15 --scifact_in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet' --logit_col raw_max_top3 --out_md reports/demo_stage1.md
.venv/bin/python scripts/demo_stage1.py --track fever --n 500 --seed 15 --fever_in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet' --logit_col logit_platt --out_md reports/demo_stage1.md
.venv/bin/python scripts/tune_stage1_thresholds.py --track scifact --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet' --logit_col raw_max_top3 --y_col y --budgets 0.30 0.40 0.50 0.60
.venv/bin/python scripts/tune_stage1_thresholds.py --track fever --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet' --logit_col logit_platt --y_col y --budgets 0.30 0.40 0.50 0.60
.venv/bin/python scripts/day12_e2e_run_500.py --track scifact --n 50 --seed 15 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_scifact_50_e2e_stage1only.jsonl --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet' --dry_run_stage2
.venv/bin/python scripts/day12_e2e_run_500.py --track fever --n 50 --seed 15 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_fever_50_e2e_stage1only.jsonl --in_path '/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet' --logit_col logit_platt --y_col y --dry_run_stage2
.venv/bin/python scripts/day12_e2e_stats.py --tracks scifact fever --runs_dir runs --out_md reports/day12_e2e_stats.md --out_json reports/day12_e2e_compare.json
```

## Expected outputs
- reports/demo_stage1.md
- reports/stage1_threshold_tuning_scifact.md
- reports/stage1_threshold_tuning_fever.md
- configs/thresholds_stage1_tuned_scifact.yaml
- configs/thresholds_stage1_tuned_fever.yaml
- runs/day12_scifact_50_e2e_stage1only.jsonl
- runs/day12_fever_50_e2e_stage1only.jsonl
- reports/day12_e2e_stats.md
- reports/day12_e2e_compare.json

## Freeze definition
Do not change the parquet inputs, their sha256, or the threshold sources after this tag. Stage1 routing logic and tuned thresholds are locked for Stage2 work.

## Decision memo linkage
- [ ] Stage1 decision memo generated: reports/stage1_decision_memo.md
- [ ] Product thresholds selected: scifact -> configs/thresholds_stage1_tuned_scifact.yaml; fever -> configs/thresholds_stage1_tuned_fever.yaml
