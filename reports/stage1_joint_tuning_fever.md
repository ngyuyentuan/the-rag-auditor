# Stage1 Joint Tuning (fever)

Generated: `2026-01-06T15:12:49.854978+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n_raw: `5715`
- n_used: `1000`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau_grid_min: `0.2`
- tau_grid_max: `2.0`
- tau_grid_steps: `60`
- grid_lower_steps: `50`
- grid_upper_steps: `50`
- max_fp_accept_rate: `0.05`
- max_fn_reject_rate: `0.05`
- objective: `feasible_best_ok`

Budget <= 0.3

- feasible_configs: `2`
- recommended: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918
  coverage=0.7180 accuracy_on_decided=0.8774 ok_rate=0.9120 fp=0.0390 fn=0.0490 uncertain=0.2820
- best_near_feasible: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 fp=0.0390 fn=0.0490 uncertain=0.2820

Budget <= 0.4

- feasible_configs: `556`
- recommended: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918
  coverage=0.7180 accuracy_on_decided=0.8774 ok_rate=0.9120 fp=0.0390 fn=0.0490 uncertain=0.2820
- best_near_feasible: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 fp=0.0390 fn=0.0490 uncertain=0.2820

Budget <= 0.5

- feasible_configs: `1348`
- recommended: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918
  coverage=0.7180 accuracy_on_decided=0.8774 ok_rate=0.9120 fp=0.0390 fn=0.0490 uncertain=0.2820
- best_near_feasible: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 fp=0.0390 fn=0.0490 uncertain=0.2820

Budget <= 0.6

- feasible_configs: `1989`
- recommended: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918
  coverage=0.7180 accuracy_on_decided=0.8774 ok_rate=0.9120 fp=0.0390 fn=0.0490 uncertain=0.2820
- best_near_feasible: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 fp=0.0390 fn=0.0490 uncertain=0.2820

YAML output

- written_to_yaml: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 export_budget=0.3

Repro command

```
scripts/joint_tune_stage1_tau_thresholds.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --n 1000 --budgets 0.30 0.40 0.50 0.60 --out_md reports/stage1_joint_tuning_fever.md --out_yaml configs/thresholds_stage1_joint_tuned_fever.yaml
```
