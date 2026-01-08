# Stage1 Product Decision

SciFact

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9450 | 0.0300 | 0.0250 | 0.4750 | 0.5250 | 0.8952 |
| joint_tuned | 0.9700 | 0.0250 | 0.0050 | 0.5450 | 0.4550 | 0.9341 |
| action_accept | 0.9450 | 0.0300 | 0.0250 | 0.4750 | 0.5250 | 0.8952 |
| product_safety | 0.9700 | 0.0250 | 0.0050 | 0.5450 | 0.4550 | 0.9341 |
| product_coverage | 0.9450 | 0.0300 | 0.0250 | 0.4750 | 0.5250 | 0.8952 |

- safety_selected: `joint_tuned`
- coverage_selected: `baseline`

- action_accept_certify: `accept_only`
- action_accept_action_ci_status: `PASS`

FEVER

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9950 | 0.0000 | 0.0050 | 0.8800 | 0.1200 | 0.9583 |
| joint_tuned | 0.8900 | 0.0550 | 0.0550 | 0.2450 | 0.7550 | 0.8543 |
| action_reject | 0.9150 | 0.0000 | 0.0850 | 0.0000 | 1.0000 | 0.9150 |
| product_safety | 0.9950 | 0.0000 | 0.0050 | 0.8800 | 0.1200 | 0.9583 |
| product_coverage | 0.9950 | 0.0000 | 0.0050 | 0.8800 | 0.1200 | 0.9583 |

- safety_selected: `baseline`
- coverage_selected: `baseline`

- action_reject_certify: `reject_only`
- action_reject_action_ci_status: `PASS`

Risks & mitigations

- ok_rate can remain high while coverage is low; coverage and accuracy_on_decided are required for product readiness.
- fp_accept_rate and fn_reject_rate should be monitored for drift.
- Stage2 should report evidence hit and verdict accuracy to validate end-to-end correctness.

