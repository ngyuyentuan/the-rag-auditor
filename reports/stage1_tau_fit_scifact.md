# Stage1 Tau Fit (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: 776
- seed: 14
- logit_col: `raw_max_top3`
- y_col: `y`
- metric: `nll`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `200`
- best_tau: `0.5618090452261306`
- best_metric: `0.47356045852348355`

| tau | metric |
|---:|---:|
| 0.561809 | 0.473560 |
| 0.552764 | 0.473561 |
| 0.570854 | 0.473645 |
| 0.543719 | 0.473654 |
| 0.579899 | 0.473808 |
| 0.534673 | 0.473847 |
| 0.588945 | 0.474044 |
| 0.525628 | 0.474149 |
| 0.597990 | 0.474345 |
| 0.516583 | 0.474568 |

## Repro command
```
scripts/fit_stage1_tau.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --n 1000 --seed 14
```
