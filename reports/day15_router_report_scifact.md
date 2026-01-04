# Day15 Router Report (scifact)

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `200`
- seed: `15`
- logit_col: `raw_max_top3`
- y_col: `y`
- thresholds_yaml: `configs/thresholds.yaml`
- tuned_thresholds_yaml: `None`
- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`
- guard_band: `0.02`
- stage2_mode: `guarded`
- runs_jsonl: `None`
- entail_min: `0.9`
- contradict_min: `0.9`
- neutral_max: `0.8`

Stage1 distribution

| metric | value |
|---|---|
| accept_rate | 0.4550 |
| reject_rate | 0.0550 |
| uncertain_rate | 0.4900 |
| fp_accept_rate | 0.0350 |
| fn_reject_rate | 0.0250 |
| ok_rate | 0.9400 |

Stage2 trigger

- trigger_rate: `0.9950`

Final distribution

| metric | value |
|---|---|
| final_accept_rate | 0.0050 |
| final_reject_rate | 0.0000 |
| final_uncertain_rate | 0.9950 |
| final_fp_accept_rate | 0.0000 |
| final_fn_reject_rate | 0.0000 |
| final_ok_rate | 1.0000 |

Confusion breakdown

| split | tp | fp | tn | fn | defer_pos | defer_neg |
|---|---:|---:|---:|---:|---:|---:|
| stage1 | 84 | 7 | 6 | 5 | 71 | 27 |
| final | 1 | 0 | 0 | 0 | 159 | 40 |

Risky cases (closest to thresholds)

| qid | y | logit | cs_ret | stage1 | final | why_stage2 |
|---|---:|---:|---:|---|---|---|
| 491 | 0 | 0.8482 | 0.8009 | UNCERTAIN | UNCERTAIN | uncertain |
| 85 | 1 | 0.8485 | 0.8009 | ACCEPT | UNCERTAIN | guard_band_upper |
| 909 | 1 | 0.7690 | 0.7793 | UNCERTAIN | UNCERTAIN | uncertain |
| 854 | 1 | 0.8472 | 0.8006 | UNCERTAIN | UNCERTAIN | uncertain |
| 680 | 0 | 0.8470 | 0.8005 | UNCERTAIN | UNCERTAIN | uncertain |
| 62 | 1 | 0.8497 | 0.8012 | ACCEPT | UNCERTAIN | guard_band_upper |
| 296 | 1 | 0.8499 | 0.8013 | ACCEPT | UNCERTAIN | guard_band_upper |
| 218 | 1 | 0.8459 | 0.8003 | UNCERTAIN | UNCERTAIN | uncertain |
| 331 | 0 | 0.7710 | 0.7799 | UNCERTAIN | UNCERTAIN | uncertain |
| 301 | 0 | 0.7662 | 0.7785 | REJECT | UNCERTAIN | guard_band_lower |
| 787 | 1 | 0.7657 | 0.7784 | REJECT | UNCERTAIN | guard_band_lower |
| 704 | 1 | 0.8519 | 0.8018 | ACCEPT | UNCERTAIN | guard_band_upper |
| 381 | 1 | 0.8445 | 0.7999 | UNCERTAIN | UNCERTAIN | uncertain |
| 379 | 1 | 0.8522 | 0.8019 | ACCEPT | UNCERTAIN | guard_band_upper |
| 401 | 1 | 0.8444 | 0.7999 | UNCERTAIN | UNCERTAIN | uncertain |
| 63 | 0 | 0.8439 | 0.7997 | UNCERTAIN | UNCERTAIN | uncertain |
| 77 | 1 | 0.8530 | 0.8021 | ACCEPT | UNCERTAIN | guard_band_upper |
| 661 | 1 | 0.8435 | 0.7996 | UNCERTAIN | UNCERTAIN | uncertain |
| 613 | 1 | 0.8432 | 0.7995 | UNCERTAIN | UNCERTAIN | uncertain |
| 364 | 0 | 0.8537 | 0.8023 | ACCEPT | UNCERTAIN | guard_band_upper |
