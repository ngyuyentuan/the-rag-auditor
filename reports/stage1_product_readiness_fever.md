# Stage1 Product Readiness (fever)

- baseline_jsonl: `runs/day12_fever_50_e2e.jsonl`
- tuned_jsonl: `runs/day12_fever_50_e2e_tuned.jsonl`

| variant | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0600 | 0.9400 | 0.0000 | 0.0200 | 0.9800 |
| tuned | 0.0000 | 0.8200 | 0.1800 | 0.0000 | 0.1600 | 0.8400 |

Conclusion

Tuned fp_accept_rate unchanged by 0.0000. Uncertain_rate decreased by 0.7600.
