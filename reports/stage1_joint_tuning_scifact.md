# Stage1 Joint Tuning (scifact)

Generated: `2026-01-08T14:59:47.005508+00:00`

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
- grid_lower_steps: `50`
- grid_upper_steps: `50`
- max_fp_accept_rate: `0.05`
- max_fn_reject_rate: `0.05`
- objective: `feasible_best_ok`

Budget <= 0.3

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=0.26101694915254237 t_lower=0.9495918367346938 t_upper=0.9595918367346938 fp=0.0799 fn=0.0322 uncertain=0.3015

Budget <= 0.4

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=0.5050847457627119 t_lower=0.8081632653061224 t_upper=0.8383673469387755 fp=0.0709 fn=0.0077 uncertain=0.3969

Budget <= 0.5

- feasible_configs: `0`
- recommended: none (no feasible config)
- best_near_feasible: tau=1.7864406779661017 t_lower=0.6061224489795918 t_upper=0.6161224489795918 fp=0.0515 fn=0.0374 uncertain=0.4575

Budget <= 0.6

- feasible_configs: `112`
- recommended: tau=0.5661016949152542 t_lower=0.7879591836734694 t_upper=0.8181632653061224
  coverage=0.4536 accuracy_on_decided=0.8949 ok_rate=0.9523 fp=0.0374 fn=0.0103 uncertain=0.5464
- best_near_feasible: tau=0.5661016949152542 t_lower=0.7879591836734694 t_upper=0.8181632653061224 fp=0.0374 fn=0.0103 uncertain=0.5464

YAML output

- written_to_yaml: tau=0.5661016949152542 t_lower=0.7879591836734694 t_upper=0.8181632653061224 export_budget=0.6

Repro command

```
scripts/joint_tune_stage1_tau_thresholds.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --budgets 0.30 0.40 0.50 0.60 --out_md reports/stage1_joint_tuning_scifact.md --out_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml
```
