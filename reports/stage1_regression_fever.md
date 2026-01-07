# Stage1 Regression Diagnosis (fever)

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n_total: 1000
- seed: 14
- logit_col: `logit_platt`
- y_col: `y`
- pos_rate: 0.1000
- missing_y: 0
- unique_y_values: [0, 1]
- missing_logit: 0
- nonfinite_logit_count: 0

## cs_ret summary
| metric | value |
|---|---:|
| min | 0.0408 |
| p01 | 0.0557 |
| p10 | 0.0718 |
| p50 | 0.1058 |
| p90 | 0.1502 |
| p99 | 0.1909 |
| max | 0.2139 |

## cs_ret histogram (20 bins)
| bin_left | bin_right | count | fraction |
|---:|---:|---:|---:|
| 0.000 | 0.050 | 5 | 0.0050 |
| 0.050 | 0.100 | 418 | 0.4180 |
| 0.100 | 0.150 | 475 | 0.4750 |
| 0.150 | 0.200 | 99 | 0.0990 |
| 0.200 | 0.250 | 3 | 0.0030 |
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
| 0.750 | 0.800 | 0 | 0.0000 |
| 0.800 | 0.850 | 0 | 0.0000 |
| 0.850 | 0.900 | 0 | 0.0000 |
| 0.900 | 0.950 | 0 | 0.0000 |
| 0.950 | 1.000 | 0 | 0.0000 |

## Threshold proximity counts
| metric | value |
|---|---:|
| within_0.005_t_lower | 0.0790 |
| within_0.005_t_upper | 0.0000 |
| within_0.01_t_lower | 0.1490 |
| within_0.01_t_upper | 0.0000 |

## Baseline config
- thresholds_yaml: `configs/thresholds.yaml`
- t_lower: 0.0721672686678491
- t_upper: 1.0
- tau: 0.9957177012600404

| metric | value | 95% CI |
|---|---:|---:|
| accept_rate | 0.0000 | [0.0000, 0.0000] |
| reject_rate | 0.1080 | [0.0900, 0.1270] |
| uncertain_rate | 0.8920 | [0.8730, 0.9100] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0010 | [0.0000, 0.0030] |
| ok_rate | 0.9990 | [0.9970, 1.0000] |

| count | value |
|---|---:|
| tp | 0 |
| fp | 0 |
| tn | 107 |
| fn | 1 |
| defer_pos | 99 |
| defer_neg | 793 |

## Tuned config
- tuned_thresholds_yaml: `configs/thresholds_stage1_tuned_fever.yaml`
- t_lower: 0.14142857142857143
- t_upper: 0.2524489795918367
- tau: 0.9957177012600404

| metric | value | 95% CI |
|---|---:|---:|
| accept_rate | 0.0000 | [0.0000, 0.0000] |
| reject_rate | 0.8490 | [0.8270, 0.8710] |
| uncertain_rate | 0.1510 | [0.1290, 0.1730] |
| fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| fn_reject_rate | 0.0730 | [0.0580, 0.0890] |
| ok_rate | 0.9270 | [0.9110, 0.9420] |

| count | value |
|---|---:|
| tp | 0 |
| fp | 0 |
| tn | 776 |
| fn | 73 |
| defer_pos | 27 |
| defer_neg | 124 |

## Route flips (tuned vs baseline)
- flips_total: 741
| qid | y | logit | cs_ret | baseline_decision | tuned_decision | baseline_reason | tuned_reason |
|---|---:|---:|---:|---|---|---|---|
| 1995 | 1 | -2.542928 | 0.072167 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 1161 | 0 | -2.542051 | 0.072226 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 3982 | 1 | -2.541173 | 0.072285 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 1101 | 0 | -2.540120 | 0.072356 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 1157 | 0 | -1.797862 | 0.141171 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 2508 | 0 | -1.797862 | 0.141171 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 5250 | 0 | -2.539067 | 0.072427 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 925 | 0 | -2.536083 | 0.072629 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 1502 | 0 | -2.535732 | 0.072653 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 4634 | 0 | -2.535206 | 0.072688 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 3832 | 0 | -2.534504 | 0.072736 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 7879 | 0 | -2.532222 | 0.072891 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 4870 | 0 | -2.531871 | 0.072914 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 172 | 0 | -2.530818 | 0.072986 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 4176 | 0 | -2.529414 | 0.073081 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 6987 | 0 | -2.524324 | 0.073428 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 903 | 0 | -2.523797 | 0.073464 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 6382 | 0 | -2.522393 | 0.073560 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 2836 | 0 | -2.520813 | 0.073669 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 2503 | 1 | -2.520463 | 0.073693 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 7625 | 0 | -1.808919 | 0.139830 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 8373 | 0 | -1.809446 | 0.139767 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 6977 | 1 | -1.809972 | 0.139703 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 847 | 0 | -1.811201 | 0.139555 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |
| 5012 | 0 | -1.811376 | 0.139534 | UNCERTAIN | REJECT | t_lower<=cs_ret<t_upper | cs_ret<t_lower |

## Bootstrap delta CI (tuned - baseline)
| metric | 95% CI |
|---|---:|
| ok_rate | [-0.0880, -0.0560] |
| fp_accept_rate | [0.0000, 0.0000] |
| fn_reject_rate | [0.0560, 0.0880] |
| uncertain_rate | [-0.7680, -0.7140] |

## Conclusion
Root cause: tuned thresholds push decision boundary such that fn_reject increased by 0.0720

## Repro command
```
scripts/diagnose_stage1_regression.py --track fever --in_path /mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet --logit_col logit_platt --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_tuned_fever.yaml --n 1000 --out_md reports/stage1_regression_fever.md
```
