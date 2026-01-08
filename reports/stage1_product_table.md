# Stage1 Product Table

- n: `200`
- seed: `14`

## scifact

| config | certify | ok_rate | coverage | decided_rate | decided_count | accuracy_on_decided | fp | fn | fp_upper95 | fn_upper95 | fp_decided | fn_decided | fp_decided_upper95 | fn_decided_upper95 | ci_safe | decided_ci_safe | decided_ci_status | accept_count | reject_count | fp_given_accept | fn_given_reject | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_ci_status | reject_ci_status | action_ci_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline |  | 0.9450 | 0.5250 | 0.5250 | 105 | 0.8952 | 0.0300 | 0.0250 | 0.0639 | 0.0572 | 0.0571 | 0.0476 | 0.1191 | 0.1067 | False | False | FAIL | 89 | 16 | 0.0674 | 0.3125 | 0.1394 | 0.5560 | PASS | FAIL | FAIL |
| joint_tuned |  | 0.9700 | 0.4550 | 0.4550 | 91 | 0.9341 | 0.0250 | 0.0050 | 0.0572 | 0.0278 | 0.0549 | 0.0110 | 0.1222 | 0.0597 | False | False | FAIL | 83 | 8 | 0.0602 | 0.1250 | 0.1334 | 0.4709 | PASS | FAIL | FAIL |
| product_tuned |  | 0.9950 | 0.0700 | 0.0700 | 14 | 0.9286 | 0.0000 | 0.0050 | 0.0188 | 0.0278 | 0.0000 | 0.0714 | 0.2153 | 0.3147 | True | False | FAIL | 6 | 8 | 0.0000 | 0.1250 | 0.3903 | 0.4709 | FAIL | FAIL | FAIL |
| product_safety |  | 0.9700 | 0.4550 | 0.4550 | 91 | 0.9341 | 0.0250 | 0.0050 | 0.0572 | 0.0278 | 0.0549 | 0.0110 | 0.1222 | 0.0597 | False | False | FAIL | 83 | 8 | 0.0602 | 0.1250 | 0.1334 | 0.4709 | PASS | FAIL | FAIL |
| product_coverage |  | 0.9450 | 0.5250 | 0.5250 | 105 | 0.8952 | 0.0300 | 0.0250 | 0.0639 | 0.0572 | 0.0571 | 0.0476 | 0.1191 | 0.1067 | False | False | FAIL | 89 | 16 | 0.0674 | 0.3125 | 0.1394 | 0.5560 | PASS | FAIL | FAIL |
| product_2d | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_accept | accept_only | 0.9450 | 0.5250 | 0.5250 | 105 | 0.8952 | 0.0300 | 0.0250 | 0.0639 | 0.0572 | 0.0571 | 0.0476 | 0.1191 | 0.1067 | N/A | N/A | N/A | 89 | 16 | 0.0674 | 0.3125 | 0.1394 | 0.5560 | PASS | N/A | PASS |
| action_reject | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |

## fever

| config | certify | ok_rate | coverage | decided_rate | decided_count | accuracy_on_decided | fp | fn | fp_upper95 | fn_upper95 | fp_decided | fn_decided | fp_decided_upper95 | fn_decided_upper95 | ci_safe | decided_ci_safe | decided_ci_status | accept_count | reject_count | fp_given_accept | fn_given_reject | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_ci_status | reject_ci_status | action_ci_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline |  | 0.9950 | 0.1200 | 0.1200 | 24 | 0.9583 | 0.0000 | 0.0050 | 0.0188 | 0.0278 | 0.0000 | 0.0417 | 0.1380 | 0.2024 | True | False | FAIL | 0 | 24 | 0.0000 | 0.0417 | 0.0000 | 0.2024 | INSUFFICIENT_N | FAIL | INSUFFICIENT_N |
| joint_tuned |  | 0.8900 | 0.7550 | 0.7550 | 151 | 0.8543 | 0.0550 | 0.0550 | 0.0958 | 0.0958 | 0.0728 | 0.0728 | 0.1257 | 0.1257 | False | False | FAIL | 14 | 137 | 0.7857 | 0.0803 | 0.9243 | 0.1381 | FAIL | PASS | FAIL |
| product_tuned |  | 0.9950 | 0.2100 | 0.2100 | 42 | 0.9762 | 0.0000 | 0.0050 | 0.0188 | 0.0278 | 0.0000 | 0.0238 | 0.0838 | 0.1232 | True | False | FAIL | 0 | 42 | 0.0000 | 0.0238 | 0.0000 | 0.1232 | INSUFFICIENT_N | PASS | INSUFFICIENT_N |
| product_safety |  | 0.9950 | 0.1200 | 0.1200 | 24 | 0.9583 | 0.0000 | 0.0050 | 0.0188 | 0.0278 | 0.0000 | 0.0417 | 0.1380 | 0.2024 | True | False | FAIL | 0 | 24 | 0.0000 | 0.0417 | 0.0000 | 0.2024 | INSUFFICIENT_N | FAIL | INSUFFICIENT_N |
| product_coverage |  | 0.9950 | 0.1200 | 0.1200 | 24 | 0.9583 | 0.0000 | 0.0050 | 0.0188 | 0.0278 | 0.0000 | 0.0417 | 0.1380 | 0.2024 | True | False | FAIL | 0 | 24 | 0.0000 | 0.0417 | 0.0000 | 0.2024 | INSUFFICIENT_N | FAIL | INSUFFICIENT_N |
| product_2d | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_accept | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_reject | reject_only | 0.9150 | 1.0000 | 1.0000 | 200 | 0.9150 | 0.0000 | 0.0850 | 0.0188 | 0.1319 | 0.0000 | 0.0850 | 0.0188 | 0.1319 | N/A | N/A | N/A | 0 | 200 | 0.0000 | 0.0850 | 0.0000 | 0.1319 | N/A | PASS | PASS |

