# SciFact Robust Verify Aggregate (Filtered v2)

Discovery summary

- in_dirs: ['runs']
- glob: `*scifact*.jsonl`
- recursive: True
- files_matched_total: 165
- files_kept_scifact: 165
- files_kept_after_tag_filter: 7
- files_with_budget: 0
- files_missing_budget: 7
- include_tag: `d2_lr0.1_leaf20_l20.0`
- dropped_tag_mismatch: 145
- unknown_tag_dropped: 96

Top 20 kept file paths

- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.30.jsonl

Budget summary

| tag | budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst | n_files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Ship recommendation

- no group met constraints

Unusable files (missing budget)

- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.22.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.32.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.26.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.35.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.28.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.24.jsonl
- runs/sweeps/20260103_123758_43c1ad/d2_lr0.1_leaf20_l20.0/scifact_router_v2_d2_lr0.1_leaf20_l20.0_budget_0.30.jsonl

Dropped by tag mismatch

- runs/day12_scifact_50_e2e_tuned.jsonl
- runs/day12_scifact_500_e2e_random.jsonl
- runs/day12_scifact_500_e2e_always.jsonl
- runs/day12_scifact_500_e2e.jsonl
- runs/day12_scifact_50_e2e_stage1only.jsonl
- runs/day12_scifact_50_e2e.jsonl
- runs/day12_scifact_50_e2e_prod.jsonl
- runs/product_proof_router_v2_sweep_scifact/scifact_baseline_uncertain_only_200_real.jsonl
- runs/product_proof_router_v2_sweep_scifact/scifact_router_v2_current_uncertain_only_200.jsonl
- runs/product_proof_router_v2_sweep_scifact/scifact_router_v2_min_route_hgb_uncertain_only_200.jsonl
- runs/product_proof_router_v2_sweep_scifact/scifact_router_v2_min_route_logreg_uncertain_only_200.jsonl
- runs/matrix/scifact_calibrated_always_50.jsonl
- runs/matrix/scifact_calibrated_uncertain_only_50_tuned.jsonl
- runs/matrix/scifact_calibrated_uncertain_only_50.jsonl
- runs/day12_matrix/scifact_calibrated_always_200.jsonl
- runs/day12_matrix/scifact_calibrated_uncertain_only_200.jsonl
- runs/product_proof_router_v2/scifact_baseline_uncertain_only_200_real.jsonl
- runs/product_proof_router_v2/scifact_router_v2_uncertain_only_200.jsonl
- runs/product_proof_router_v2/scifact_router_v2_uncertain_only_50.jsonl
- runs/product_proof_router_v2/scifact_baseline_always_200_real.jsonl
- runs/product_proof_router_v2/scifact_baseline_uncertain_only_50_real.jsonl
- runs/product_proof_router_v2/scifact_router_v2_always_200.jsonl
- runs/product_proof_router_v2/scifact_baseline_always_50_real.jsonl
- runs/product_proof_router_v2/scifact_router_v2_always_50.jsonl
- runs/product_proof_cost/scifact_baseline_uncertain_only_50_real.jsonl
- runs/product_proof_cost/scifact_cost_always_50.jsonl
- runs/product_proof_cost/scifact_baseline_always_50_real.jsonl
- runs/product_proof_cost/scifact_cost_uncertain_only_50.jsonl
- runs/product_proof_constrained/scifact_baseline_uncertain_only_50_real.jsonl
- runs/product_proof_constrained/scifact_constrained_uncertain_only_50.jsonl
- runs/product_proof_constrained/scifact_constrained_always_50.jsonl
- runs/product_proof_constrained/scifact_baseline_always_50_real.jsonl
- runs/product_proof/scifact_prod_uncertain_only_50.jsonl
- runs/product_proof/scifact_baseline_uncertain_only_50_dry.jsonl
- runs/product_proof/scifact_baseline_uncertain_only_50_real.jsonl
- runs/product_proof/scifact_prod_always_50.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed12_budget0.280.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed12_budget0.260.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed5_budget0.260.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed12_budget0.300.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed5_budget0.280.jsonl
- runs/sweeps/robust_verify_20260104_144622_bf0046/scifact_ens_seed5_budget0.300.jsonl
- runs/sweeps/robust_verify_20260104_145414_ebb839/scifact_ens_seed5_budget0.280.jsonl
- runs/sweeps/robust_verify_20260104_144835_04a664/scifact_ens_seed5_budget0.260.jsonl
- runs/sweeps/robust_verify_20260104_144835_04a664/scifact_ens_seed5_budget0.280.jsonl
- runs/sweeps/robust_verify_20260104_144835_04a664/scifact_ens_seed5_budget0.300.jsonl
- runs/sweeps/robust_verify_20260103_231813_da4e7e/scifact_ens_seed5_budget0.260.jsonl
- runs/sweeps/robust_verify_20260103_231813_da4e7e/scifact_ens_seed5_budget0.280.jsonl
- runs/sweeps/robust_verify_fast_20260104_151740_b6e034/scifact_ens_seed12_budget0.260.jsonl
- runs/sweeps/robust_verify_fast_20260104_151740_b6e034/scifact_ens_seed5_budget0.260.jsonl
