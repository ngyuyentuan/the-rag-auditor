# Stage1 Decision Memo

## SciFact summary

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate |
|---|---:|---:|---:|---:|
| baseline | 0.9253 | 0.0425 | 0.0322 | 0.4845 |
| tuned | 0.9278 | 0.0399 | 0.0322 | 0.5000 |
| profile_safety_first | 0.9356 | 0.0644 | 0.0000 | 0.4459 |
| profile_coverage_first | 0.9923 | 0.0000 | 0.0077 | 0.9665 |

### SciFact thresholds
- tuned_yaml: t_lower=0.7788163428551291, t_upper=0.8012982808953983, tau=0.609483051351548

### SciFact product tradeoff
Baseline keeps risk balanced; tuned raises uncertain slightly but improves ok_rate marginally; coverage_first profile defers less but would require Stage2 to manage risk.

## FEVER summary

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate |
|---|---:|---:|---:|---:|
| baseline | 0.9990 | 0.0000 | 0.0010 | 0.8920 |
| tuned | 0.9530 | 0.0000 | 0.0470 | 0.3570 |
| profile_safety_first | 0.9960 | 0.0000 | 0.0040 | 0.8250 |
| profile_coverage_first | 0.9410 | 0.0300 | 0.0290 | 0.5210 |

### FEVER thresholds
- tuned_yaml: t_lower=0.11775510204081632, t_upper=0.2357142857142857, tau=0.9957177012600404

### FEVER product tradeoff
Baseline has near-zero accepts; tuned raises coverage but increases fn; coverage_first profile reduces uncertain but adds risk and should be paired with strong Stage2.

## Recommendation
- SciFact: retain tuned weighted config (export budget=0.50) if Stage2 can handle similar deferral; ok_rate improves slightly and fp/fn stay bounded.
- FEVER: prefer baseline for safety; tuned raises fn; only ship tuned if Stage2 evidence is strong and monitored.

## Risks & mitigations
- ok_rate alone hides evidence quality; Stage2 must measure evidence hit and verdict accuracy.
- Monitor fp_accept and fn_reject drift over time; alert if exceeding product gates.
- Ensure abstention/UNCERTAIN flows route to human/Stage2 with clear policy.

## Repro commands
```
scripts/stage1_decision_memo.py --scifact_report reports/stage1_eval_scifact_large.md --fever_report reports/stage1_eval_fever_large.md --scifact_tuning reports/stage1_threshold_tuning_scifact.md --fever_tuning reports/stage1_threshold_tuning_fever.md --scifact_profile reports/stage1_product_profile_scifact.md --fever_profile reports/stage1_product_profile_fever.md --out_md reports/stage1_decision_memo.md
```
