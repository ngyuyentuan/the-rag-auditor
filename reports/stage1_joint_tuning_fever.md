# Stage1 Joint Tuning (fever)

Generated: `2026-01-09T02:55:19.494289+00:00`

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
- grid_lower_steps: `60`
- grid_upper_steps: `60`
- max_fp_accept_rate: `0.05`
- max_fn_reject_rate: `0.05`
- objective: `feasible_best_ok`

Budget <= 0.3

- feasible_configs: `1`
- recommended: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644
  coverage=0.7011 accuracy_on_decided=0.8612 ok_rate=0.9027 fp=0.0476 fn=0.0497 uncertain=0.2989
- best_near_feasible: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644 fp=0.0476 fn=0.0497 uncertain=0.2989

Budget <= 0.4

- feasible_configs: `636`
- recommended: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644
  coverage=0.7011 accuracy_on_decided=0.8612 ok_rate=0.9027 fp=0.0476 fn=0.0497 uncertain=0.2989
- best_near_feasible: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644 fp=0.0476 fn=0.0497 uncertain=0.2989

Budget <= 0.5

- feasible_configs: `1735`
- recommended: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644
  coverage=0.7011 accuracy_on_decided=0.8612 ok_rate=0.9027 fp=0.0476 fn=0.0497 uncertain=0.2989
- best_near_feasible: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644 fp=0.0476 fn=0.0497 uncertain=0.2989

Budget <= 0.6

- feasible_configs: `2859`
- recommended: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644
  coverage=0.7011 accuracy_on_decided=0.8612 ok_rate=0.9027 fp=0.0476 fn=0.0497 uncertain=0.2989
- best_near_feasible: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644 fp=0.0476 fn=0.0497 uncertain=0.2989

YAML output

- written_to_yaml: tau=1.694915254237288 t_lower=0.23491525423728812 t_upper=0.27847457627118644 export_budget=0.3

Repro command

```
scripts/joint_tune_stage1_tau_thresholds.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --budgets 0.30 0.40 0.50 0.60 --tau_grid_steps 60 --grid_lower_steps 60 --grid_upper_steps 60 --out_md reports/stage1_joint_tuning_fever.md --out_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --progress
```
