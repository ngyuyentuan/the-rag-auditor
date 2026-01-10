# Stage1 Product Table

- n: `1000`
- seed: `14`

## scifact

| config | certify | ok_rate | coverage | decided_rate | decided_count | accuracy_on_decided | fp | fn | fp_upper95 | fn_upper95 | fp_decided | fn_decided | fp_decided_upper95 | fn_decided_upper95 | ci_safe | decided_ci_safe | decided_ci_status | accept_count | reject_count | fp_given_accept | fn_given_reject | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_ci_status | reject_ci_status | action_ci_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline |  | 0.9253 | 0.5155 | 0.5155 | 400 | 0.8550 | 0.0425 | 0.0322 | 0.0591 | 0.0471 | 0.0825 | 0.0625 | 0.1136 | 0.0906 | False | False | FAIL | 352 | 48 | 0.0938 | 0.5208 | 0.1287 | 0.6553 | PASS | FAIL | FAIL |
| joint_tuned |  | 0.9523 | 0.4536 | 0.4536 | 352 | 0.8949 | 0.0374 | 0.0103 | 0.0532 | 0.0202 | 0.0824 | 0.0227 | 0.1158 | 0.0442 | False | False | FAIL | 332 | 20 | 0.0873 | 0.4000 | 0.1226 | 0.6134 | PASS | FAIL | FAIL |
| product_tuned |  | 0.9858 | 0.2745 | 0.2745 | 213 | 0.9484 | 0.0142 | 0.0000 | 0.0252 | 0.0049 | 0.0516 | 0.0000 | 0.0901 | 0.0177 | True | True | PASS | 213 | 0 | 0.0516 | 0.0000 | 0.0901 | 0.0000 | PASS | INSUFFICIENT_N | INSUFFICIENT_N |
| product_safety |  | 0.9523 | 0.4536 | 0.4536 | 352 | 0.8949 | 0.0374 | 0.0103 | 0.0532 | 0.0202 | 0.0824 | 0.0227 | 0.1158 | 0.0442 | False | False | FAIL | 332 | 20 | 0.0873 | 0.4000 | 0.1226 | 0.6134 | PASS | FAIL | FAIL |
| product_coverage |  | 0.9253 | 0.5155 | 0.5155 | 400 | 0.8550 | 0.0425 | 0.0322 | 0.0591 | 0.0471 | 0.0825 | 0.0625 | 0.1136 | 0.0906 | False | False | FAIL | 352 | 48 | 0.0938 | 0.5208 | 0.1287 | 0.6553 | PASS | FAIL | FAIL |
| product_2d | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_accept | accept_only | 0.9253 | 0.5155 | 0.5155 | 400 | 0.8550 | 0.0425 | 0.0322 | 0.0591 | 0.0471 | 0.0825 | 0.0625 | 0.1136 | 0.0906 | N/A | N/A | N/A | 352 | 48 | 0.0938 | 0.5208 | 0.1287 | 0.6553 | PASS | N/A | PASS |
| action_reject | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |

## fever

| config | certify | ok_rate | coverage | decided_rate | decided_count | accuracy_on_decided | fp | fn | fp_upper95 | fn_upper95 | fp_decided | fn_decided | fp_decided_upper95 | fn_decided_upper95 | ci_safe | decided_ci_safe | decided_ci_status | accept_count | reject_count | fp_given_accept | fn_given_reject | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_ci_status | reject_ci_status | action_ci_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline |  | 0.9990 | 0.1080 | 0.1080 | 108 | 0.9907 | 0.0000 | 0.0010 | 0.0038 | 0.0056 | 0.0000 | 0.0093 | 0.0343 | 0.0506 | True | True | PASS | 0 | 108 | 0.0000 | 0.0093 | 0.0000 | 0.0506 | INSUFFICIENT_N | PASS | INSUFFICIENT_N |
| joint_tuned |  | 0.9140 | 0.6890 | 0.6890 | 689 | 0.8752 | 0.0390 | 0.0470 | 0.0529 | 0.0619 | 0.0566 | 0.0682 | 0.0764 | 0.0895 | False | True | PASS | 52 | 637 | 0.7500 | 0.0738 | 0.8477 | 0.0967 | FAIL | PASS | FAIL |
| product_tuned |  | 0.9640 | 0.5200 | 0.5200 | 520 | 0.9308 | 0.0000 | 0.0360 | 0.0038 | 0.0494 | 0.0000 | 0.0692 | 0.0073 | 0.0944 | True | True | PASS | 0 | 520 | 0.0000 | 0.0692 | 0.0000 | 0.0944 | INSUFFICIENT_N | PASS | INSUFFICIENT_N |
| product_safety |  | 0.9990 | 0.1080 | 0.1080 | 108 | 0.9907 | 0.0000 | 0.0010 | 0.0038 | 0.0056 | 0.0000 | 0.0093 | 0.0343 | 0.0506 | True | True | PASS | 0 | 108 | 0.0000 | 0.0093 | 0.0000 | 0.0506 | INSUFFICIENT_N | PASS | INSUFFICIENT_N |
| product_coverage |  | 0.9990 | 0.1080 | 0.1080 | 108 | 0.9907 | 0.0000 | 0.0010 | 0.0038 | 0.0056 | 0.0000 | 0.0093 | 0.0343 | 0.0506 | True | True | PASS | 0 | 108 | 0.0000 | 0.0093 | 0.0000 | 0.0506 | INSUFFICIENT_N | PASS | INSUFFICIENT_N |
| product_2d | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_accept | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| action_reject | reject_only | 0.8870 | 0.9710 | 0.9710 | 971 | 0.8836 | 0.0300 | 0.0830 | 0.0425 | 0.1017 | 0.0309 | 0.0855 | 0.0438 | 0.1047 | N/A | N/A | N/A | 39 | 932 | 0.7692 | 0.0891 | 0.8735 | 0.1091 | N/A | PASS | PASS |

