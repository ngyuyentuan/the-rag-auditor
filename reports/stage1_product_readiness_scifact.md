# Stage1 Product Readiness (scifact)

- baseline_jsonl: `runs/day12_scifact_50_e2e.jsonl`
- tuned_jsonl: `runs/day12_scifact_50_e2e_tuned.jsonl`

| variant | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4400 | 0.0600 | 0.5000 | 0.0400 | 0.0000 | 0.9600 |
| tuned | 0.0600 | 0.8200 | 0.1200 | 0.0000 | 0.6000 | 0.4000 |

Conclusion

Tuned fp_accept_rate decreased by 0.0400. Uncertain_rate decreased by 0.3800.
