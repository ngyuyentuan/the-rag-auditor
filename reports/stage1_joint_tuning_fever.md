# Stage1 Joint Tuning (fever)

Generated: `2026-01-08T14:59:48.276281+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n_raw: `5715`
- n_used: `5715`
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

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=0.6271186440677967 t_lower=0.04040816326530612 t_upper=0.07061224489795918 fp=0.0486 fn=0.0511 uncertain=0.2810

Budget <= 0.4

- feasible_configs: `212`
- recommended: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755
  coverage=0.6847 accuracy_on_decided=0.8640 ok_rate=0.9069 fp=0.0446 fn=0.0485 uncertain=0.3153
- best_near_feasible: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755 fp=0.0446 fn=0.0485 uncertain=0.3153

Budget <= 0.5

- feasible_configs: `1109`
- recommended: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755
  coverage=0.6847 accuracy_on_decided=0.8640 ok_rate=0.9069 fp=0.0446 fn=0.0485 uncertain=0.3153
- best_near_feasible: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755 fp=0.0446 fn=0.0485 uncertain=0.3153

Budget <= 0.6

- feasible_configs: `1858`
- recommended: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755
  coverage=0.6847 accuracy_on_decided=0.8640 ok_rate=0.9069 fp=0.0446 fn=0.0485 uncertain=0.3153
- best_near_feasible: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755 fp=0.0446 fn=0.0485 uncertain=0.3153

YAML output

- written_to_yaml: tau=1.1152542372881356 t_lower=0.14142857142857143 t_upper=0.19183673469387755 export_budget=0.4

Repro command

```
scripts/joint_tune_stage1_tau_thresholds.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --budgets 0.30 0.40 0.50 0.60 --out_md reports/stage1_joint_tuning_fever.md --out_yaml configs/thresholds_stage1_joint_tuned_fever.yaml
```
