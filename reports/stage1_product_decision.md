# Stage1 Product Decision

SciFact

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9253 | 0.0425 | 0.0322 | 0.4845 | 0.5155 | 0.8550 |
| joint_tuned | 0.9523 | 0.0374 | 0.0103 | 0.5464 | 0.4536 | 0.8949 |
| action_accept | 0.9253 | 0.0425 | 0.0322 | 0.4845 | 0.5155 | 0.8550 |
| product_safety | 0.9253 | 0.0425 | 0.0322 | 0.4845 | 0.5155 | 0.8550 |
| product_coverage | 0.9253 | 0.0425 | 0.0322 | 0.4845 | 0.5155 | 0.8550 |

- safety_selected: `baseline`
- coverage_selected: `baseline`

- action_accept_certify: `accept_only`
- action_accept_action_ci_status: `PASS`

FEVER

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9990 | 0.0000 | 0.0010 | 0.8920 | 0.1080 | 0.9907 |
| joint_tuned | 0.9140 | 0.0390 | 0.0470 | 0.3110 | 0.6890 | 0.8752 |
| action_reject | 0.8870 | 0.0300 | 0.0830 | 0.0290 | 0.9710 | 0.8836 |
| product_safety | 0.9640 | 0.0000 | 0.0360 | 0.4800 | 0.5200 | 0.9308 |
| product_coverage | 0.9640 | 0.0000 | 0.0360 | 0.4800 | 0.5200 | 0.9308 |

- safety_selected: `product_tuned`
- coverage_selected: `product_tuned`

- action_reject_certify: `reject_only`
- action_reject_action_ci_status: `PASS`

Risks & mitigations

- ok_rate can remain high while coverage is low; coverage and accuracy_on_decided are required for product readiness.
- fp_accept_rate and fn_reject_rate should be monitored for drift.
- Stage2 should report evidence hit and verdict accuracy to validate end-to-end correctness.

