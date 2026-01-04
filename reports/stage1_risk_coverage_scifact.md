# Stage1 Risk Coverage (scifact)

Generated: `2026-01-01T05:43:22.037687+00:00`

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `200`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- tau: `0.609483051351548`

Baseline thresholds

- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

| metric | value |
|---|---|
| accept_rate | 0.4450 |
| reject_rate | 0.0800 |
| uncertain_rate | 0.4750 |
| fp_accept_rate | 0.0300 |
| fn_reject_rate | 0.0250 |
| ok_rate | 0.9450 |

Best ok_rate under uncertain budgets

| max_uncertain | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.0000 | 0.7744 | 0.9400 | 0.0000 | 0.0600 | 0.1650 | 0.0000 | 0.8350 |
| 0.2000 | 0.0000 | 0.7870 | 0.8100 | 0.0000 | 0.1900 | 0.1250 | 0.0000 | 0.8750 |
| 0.3000 | 0.0000 | 0.7870 | 0.8100 | 0.0000 | 0.1900 | 0.1250 | 0.0000 | 0.8750 |
| 0.5000 | 0.7770 | 0.7995 | 0.4850 | 0.0700 | 0.4450 | 0.0450 | 0.0150 | 0.9400 |
| 0.7000 | 0.0000 | 0.7995 | 0.4850 | 0.0000 | 0.5150 | 0.0450 | 0.0000 | 0.9550 |

Best ok_rate under fp_accept caps

| max_fp_accept | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| 0.0100 | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |
| 0.0200 | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |
| 0.0500 | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |
| 0.1000 | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |

Recommended operating points

| profile | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |
|---|---|---|---|---|---|---|---|---|
| conservative_accept | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |
| balanced | 0.0000 | 0.8120 | 0.0750 | 0.0000 | 0.9250 | 0.0000 | 0.0000 | 1.0000 |
| aggressive | 0.8271 | 0.8371 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.7850 | 0.2150 |
