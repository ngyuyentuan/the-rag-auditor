# Stage1 Release Gate (fever)

- baseline_jsonl: `runs/day12_fever_50_e2e.jsonl`
- prod_jsonl: `runs/day12_fever_50_e2e_prod.jsonl`
- prod_yaml: `configs/thresholds_stage1_prod_fever.yaml`

| variant | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0600 | 0.9400 | 0.0000 | 0.0200 | 0.9800 |
| prod | 0.0000 | 0.0600 | 0.9400 | 0.0000 | 0.0200 | 0.9800 |

Release gate

- fp_accept_rate <= max_fp_accept: `True`
- fn_reject_rate <= max_fn_reject: `False`
- uncertain_rate <= max_uncertain: `False`

If fp_accept_rate fails, raise t_upper or tighten max_fp_accept. If fn_reject_rate fails, lower t_lower or switch to accept_only. If uncertain_rate fails, relax max_uncertain or lower t_upper.
