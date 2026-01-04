# Stage1 Risk Coverage (fever)

Generated: `2026-01-01T05:44:10.715476+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `200`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau: `0.9957177012600404`

Baseline thresholds

- t_lower: `0.0721672686678491`
- t_upper: `1.0`

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.1200 |
| uncertain_rate | 0.8800 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0050 |
| ok_rate | 0.9950 |

Best ok_rate under uncertain budgets

| max_uncertain | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.1629 | 0.2105 | 0.0000 | 0.9300 | 0.0700 | 0.0000 | 0.0700 | 0.9300 |
| 0.2000 | 0.1378 | 0.2105 | 0.0000 | 0.8150 | 0.1850 | 0.0000 | 0.0600 | 0.9400 |
| 0.3000 | 0.1253 | 0.2105 | 0.0000 | 0.7300 | 0.2700 | 0.0000 | 0.0550 | 0.9450 |
| 0.5000 | 0.1128 | 0.2105 | 0.0000 | 0.6150 | 0.3850 | 0.0000 | 0.0450 | 0.9550 |
| 0.7000 | 0.1003 | 0.2105 | 0.0000 | 0.4350 | 0.5650 | 0.0000 | 0.0250 | 0.9750 |

Best ok_rate under fp_accept caps

| max_fp_accept | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| 0.0100 | 0.0501 | 0.2105 | 0.0000 | 0.0050 | 0.9950 | 0.0000 | 0.0000 | 1.0000 |
| 0.0200 | 0.0501 | 0.2105 | 0.0000 | 0.0050 | 0.9950 | 0.0000 | 0.0000 | 1.0000 |
| 0.0500 | 0.0501 | 0.2105 | 0.0000 | 0.0050 | 0.9950 | 0.0000 | 0.0000 | 1.0000 |
| 0.1000 | 0.0501 | 0.2105 | 0.0000 | 0.0050 | 0.9950 | 0.0000 | 0.0000 | 1.0000 |

Recommended operating points

| profile | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| conservative_accept | 0.0501 | 0.2105 | 0.0000 | 0.0050 | 0.9950 | 0.0000 | 0.0000 | 1.0000 |
| balanced | 0.0000 | 0.2105 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| aggressive | 0.2130 | 0.2230 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0850 | 0.9150 |
