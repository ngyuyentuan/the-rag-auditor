# Stage1 Joint Tuning (scifact)

Generated: `2026-01-09T02:55:19.778722+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n_raw: `776`
- n_used: `776`
- seed: `14`
- logit_col: `raw_max_top3`
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

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=0.993220338983051 t_lower=0.6879661016949152 t_upper=0.6979661016949152 fp=0.0696 fn=0.0657 uncertain=0.2964

Budget <= 0.4

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=0.41355932203389834 t_lower=0.8557627118644068 t_upper=0.882542372881356 fp=0.0670 fn=0.0103 uncertain=0.4046

Budget <= 0.5

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=1.8169491525423729 t_lower=0.6040677966101695 t_upper=0.6140677966101695 fp=0.0541 fn=0.0322 uncertain=0.4510

Budget <= 0.6

- feasible_configs: `146`
- recommended: tau=0.7186440677966102 t_lower=0.7383050847457626 t_upper=0.7650847457627118
  coverage=0.4794 accuracy_on_decided=0.8898 ok_rate=0.9472 fp=0.0425 fn=0.0103 uncertain=0.5206
- best_near_feasible: tau=0.7186440677966102 t_lower=0.7383050847457626 t_upper=0.7650847457627118 fp=0.0425 fn=0.0103 uncertain=0.5206

YAML output

- written_to_yaml: tau=0.7186440677966102 t_lower=0.7383050847457626 t_upper=0.7650847457627118 export_budget=0.6

Repro command

```
scripts/joint_tune_stage1_tau_thresholds.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --budgets 0.30 0.40 0.50 0.60 --tau_grid_steps 60 --grid_lower_steps 60 --grid_upper_steps 60 --out_md reports/stage1_joint_tuning_scifact.md --out_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --progress
```
