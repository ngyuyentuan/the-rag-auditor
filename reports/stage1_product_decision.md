# Stage1 Product Decision

SciFact

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9450 | 0.0300 | 0.0250 | 0.4750 | 0.5250 | 0.8952 |
| joint_tuned | 0.9700 | 0.0250 | 0.0050 | 0.5450 | 0.4550 | 0.9341 |
| product_tuned | 0.9450 | 0.0300 | 0.0250 | 0.4750 | 0.5250 | 0.8952 |

- recommendation: `product`
- rationale: `utility weights fp=10.0 fn=10.0 unc=1.0`

FEVER

| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9950 | 0.0000 | 0.0050 | 0.8800 | 0.1200 | 0.9583 |
| joint_tuned | 0.8900 | 0.0550 | 0.0550 | 0.2450 | 0.7550 | 0.8543 |
| product_tuned | 0.9950 | 0.0000 | 0.0050 | 0.8800 | 0.1200 | 0.9583 |

- recommendation: `product`
- rationale: `utility weights fp=10.0 fn=10.0 unc=1.0`

Risks & mitigations

- ok_rate can remain high while coverage is low; coverage and accuracy_on_decided are required for product readiness.
- fp_accept_rate and fn_reject_rate should be monitored for drift.
- Stage2 should report evidence hit and verdict accuracy to validate end-to-end correctness.

