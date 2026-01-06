# Stage1 Regression Diagnosis (scifact)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n_total: 776
- seed: 14
- logit_col: `raw_max_top3`
- y_col: `y`
- pos_rate: 0.8131
- missing_y: 0
- unique_y_values: [0, 1]
- missing_logit: 0
- nonfinite_logit_count: 0

## cs_ret summary
| metric | value |
|---|---:|
| min | 0.7614 |
| p01 | 0.7666 |
| p10 | 0.7821 |
| p50 | 0.7993 |
| p90 | 0.8125 |
| p99 | 0.8200 |
| max | 0.8242 |

## cs_ret histogram (20 bins)
| bin_left | bin_right | count | fraction |
|---:|---:|---:|---:|
| 0.000 | 0.050 | 0 | 0.0000 |
| 0.050 | 0.100 | 0 | 0.0000 |
| 0.100 | 0.150 | 0 | 0.0000 |
| 0.150 | 0.200 | 0 | 0.0000 |
| 0.200 | 0.250 | 0 | 0.0000 |
| 0.250 | 0.300 | 0 | 0.0000 |
| 0.300 | 0.350 | 0 | 0.0000 |
| 0.350 | 0.400 | 0 | 0.0000 |
| 0.400 | 0.450 | 0 | 0.0000 |
| 0.450 | 0.500 | 0 | 0.0000 |
| 0.500 | 0.550 | 0 | 0.0000 |
| 0.550 | 0.600 | 0 | 0.0000 |
| 0.600 | 0.650 | 0 | 0.0000 |
| 0.650 | 0.700 | 0 | 0.0000 |
| 0.700 | 0.750 | 0 | 0.0000 |
| 0.750 | 0.800 | 404 | 0.5206 |
| 0.800 | 0.850 | 372 | 0.4794 |
| 0.850 | 0.900 | 0 | 0.0000 |
| 0.900 | 0.950 | 0 | 0.0000 |
| 0.950 | 1.000 | 0 | 0.0000 |

## Threshold proximity counts
| metric | value |
|---|---:|
| within_0.005_t_lower | 0.0876 |
| within_0.005_t_upper | 0.3157 |
| within_0.01_t_lower | 0.1881 |
| within_0.01_t_upper | 0.6198 |

## Baseline config
- thresholds_yaml: `configs/thresholds.yaml`
- t_lower: 0.7792245061204353
- t_upper: 0.8008901176300922
- tau: 0.609483051351548

| metric | value | 95% CI |
|---|---:|---:|
| accept_rate | 0.4536 | [0.4175, 0.4884] |
| reject_rate | 0.0619 | [0.0464, 0.0786] |
| uncertain_rate | 0.4845 | [0.4497, 0.5193] |
| fp_accept_rate | 0.0425 | [0.0296, 0.0567] |
| fn_reject_rate | 0.0322 | [0.0206, 0.0451] |
| ok_rate | 0.9253 | [0.9059, 0.9433] |

| count | value |
|---|---:|
| tp | 319 |
| fp | 33 |
| tn | 23 |
| fn | 25 |
| defer_pos | 287 |
| defer_neg | 89 |

## Tuned config
- tuned_thresholds_yaml: `configs/thresholds_stage1_tuned_scifact.yaml`
- t_lower: 0.8081632653061224
- t_upper: 0.8181632653061224
- tau: 0.609483051351548

| metric | value | 95% CI |
|---|---:|---:|
| accept_rate | 0.0193 | [0.0103, 0.0296] |
| reject_rate | 0.7861 | [0.7564, 0.8157] |
| uncertain_rate | 0.1946 | [0.1662, 0.2216] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.6070 | [0.5722, 0.6418] |
| ok_rate | 0.3930 | [0.3582, 0.4278] |

| count | value |
|---|---:|
| tp | 15 |
| fp | 0 |
| tn | 139 |
| fn | 471 |
| defer_pos | 145 |
| defer_neg | 6 |

## Route flips (tuned vs baseline)
- flips_total: 713
| qid | y | logit | cs_ret | baseline_decision | tuned_decision | baseline_reason | tuned_reason |
|---|---:|---:|---:|---|---|---|---|
| 590 | 1 | 0.848319 | 0.800890 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 732 | 1 | 0.768651 | 0.779225 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 314 | 0 | 0.848232 | 0.800867 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 491 | 0 | 0.848229 | 0.800867 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 67 | 1 | 0.876633 | 0.808195 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 790 | 1 | 0.876682 | 0.808207 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 86 | 1 | 0.848502 | 0.800938 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 85 | 1 | 0.848502 | 0.800938 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 425 | 1 | 0.876320 | 0.808115 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 45 | 0 | 0.848046 | 0.800819 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 672 | 1 | 0.876188 | 0.808082 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 977 | 0 | 0.848000 | 0.800806 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 525 | 1 | 0.916258 | 0.818072 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 37 | 1 | 0.876141 | 0.808070 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 563 | 1 | 0.876104 | 0.808060 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 485 | 1 | 0.876932 | 0.808271 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 909 | 1 | 0.769034 | 0.779332 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 775 | 1 | 0.847825 | 0.800761 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 245 | 1 | 0.877042 | 0.808299 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 141 | 1 | 0.848863 | 0.801032 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 484 | 1 | 0.848909 | 0.801044 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 100 | 1 | 0.847623 | 0.800708 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 814 | 1 | 0.877269 | 0.808356 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |
| 781 | 1 | 0.875733 | 0.807966 | ACCEPT | REJECT | cs_ret>=t_upper | cs_ret<t_lower |
| 217 | 0 | 0.877319 | 0.808369 | ACCEPT | UNCERTAIN | cs_ret>=t_upper | t_lower<=cs_ret<t_upper |

## Bootstrap delta CI (tuned - baseline)
| metric | 95% CI |
|---|---:|
| ok_rate | [-0.5735, -0.4897] |
| fp_accept_rate | [-0.0567, -0.0284] |
| fn_reject_rate | [0.5374, 0.6082] |
| uncertain_rate | [-0.3466, -0.2358] |

## Conclusion
Root cause: tuned thresholds push decision boundary such that fn_reject increased by 0.5747

## Repro command
```
scripts/diagnose_stage1_regression.py --track scifact --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet --logit_col raw_max_top3 --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_tuned_scifact.yaml --n 1000 --out_md reports/stage1_regression_scifact.md
```
