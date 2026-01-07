# Stage1 Tau Fit (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- metric: `nll`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `200`
- best_tau: `0.9326633165829146`
- best_metric: `0.3147498161102037`

| tau | metric |
|---:|---:|
| 0.932663 | 0.314750 |
| 0.941709 | 0.314752 |
| 0.923618 | 0.314788 |
| 0.950754 | 0.314793 |
| 0.914573 | 0.314868 |
| 0.959799 | 0.314870 |
| 0.968844 | 0.314984 |
| 0.905528 | 0.314992 |
| 0.977889 | 0.315132 |
| 0.896482 | 0.315161 |

## Repro command
```
scripts/fit_stage1_tau.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --seed 14
```
