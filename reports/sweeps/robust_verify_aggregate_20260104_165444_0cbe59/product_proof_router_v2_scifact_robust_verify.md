# Product Proof Router v2 SciFact Robust Verify (aggregate-filtered)

- in_dirs: `runs/sweeps/robust_verify_20260103_214044_b1672a, runs/product_proof_router_v2`
- glob: `scifact*budget*.jsonl`
- include_tag: `d2_lr0.1_leaf20_l20.0`
- group_by: `tag,budget`
- strict_schema: true
- skip_partial: true
- files_found: 0
- files_used: 0
- files_skipped: 0

## Per-file metrics

| file | tag | seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Group summary

| group | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst | worst_fp_file | worst_fn_file | worst_ran_file | worst_cap_file | worst_p95_file |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|

## Ship recommendation

- no group met fp_worst<=0.01 fn_worst<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case