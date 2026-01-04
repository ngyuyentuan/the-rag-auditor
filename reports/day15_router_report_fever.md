# Day15 Router Report (fever)

Config

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `200`
- seed: `15`
- logit_col: `logit_platt`
- y_col: `y`
- thresholds_yaml: `configs/thresholds.yaml`
- tuned_thresholds_yaml: `None`
- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`
- guard_band: `0.02`
- stage2_mode: `guarded`
- runs_jsonl: `None`
- entail_min: `0.9`
- contradict_min: `0.9`
- neutral_max: `0.8`

Stage1 distribution

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.0850 |
| uncertain_rate | 0.9150 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0050 |
| ok_rate | 0.9950 |

Stage2 trigger

- trigger_rate: `0.9900`

Final distribution

| metric | value |
|---|---|
| final_accept_rate | 0.0000 |
| final_reject_rate | 0.0100 |
| final_uncertain_rate | 0.9900 |
| final_fp_accept_rate | 0.0000 |
| final_fn_reject_rate | 0.0000 |
| final_ok_rate | 1.0000 |

Confusion breakdown

| split | tp | fp | tn | fn | defer_pos | defer_neg |
|---|---:|---:|---:|---:|---:|---:|
| stage1 | 0 | 0 | 16 | 1 | 18 | 165 |
| final | 0 | 0 | 2 | 0 | 19 | 179 |

Risky cases (closest to thresholds)

| qid | y | logit | cs_ret | stage1 | final | why_stage2 |
|---|---:|---:|---:|---|---|---|
| 7608.0 | 0 | -2.5399 | 0.0724 | UNCERTAIN | UNCERTAIN | uncertain |
| 51.0 | 0 | -2.5529 | 0.0715 | REJECT | UNCERTAIN | guard_band_lower |
| 5058.0 | 0 | -2.5547 | 0.0714 | REJECT | UNCERTAIN | guard_band_lower |
| 6458.0 | 0 | -2.5549 | 0.0714 | REJECT | UNCERTAIN | guard_band_lower |
| 8843.0 | 0 | -2.5582 | 0.0711 | REJECT | UNCERTAIN | guard_band_lower |
| 3541.0 | 0 | -2.5236 | 0.0735 | UNCERTAIN | UNCERTAIN | uncertain |
| 6774.0 | 0 | -2.5224 | 0.0736 | UNCERTAIN | UNCERTAIN | uncertain |
| 3552.0 | 0 | -2.5145 | 0.0741 | UNCERTAIN | UNCERTAIN | uncertain |
| 3527.0 | 0 | -2.5859 | 0.0693 | REJECT | UNCERTAIN | guard_band_lower |
| 9326.0 | 0 | -2.4996 | 0.0751 | UNCERTAIN | UNCERTAIN | uncertain |
| 8470.0 | 0 | -2.4917 | 0.0757 | UNCERTAIN | UNCERTAIN | uncertain |
| 7938.0 | 0 | -2.6135 | 0.0676 | REJECT | UNCERTAIN | guard_band_lower |
| 1446.0 | 0 | -2.4596 | 0.0780 | UNCERTAIN | UNCERTAIN | uncertain |
| 5284.0 | 0 | -2.4481 | 0.0788 | UNCERTAIN | UNCERTAIN | uncertain |
| 8821.0 | 0 | -2.6503 | 0.0653 | REJECT | UNCERTAIN | guard_band_lower |
| 8171.0 | 0 | -2.4427 | 0.0792 | UNCERTAIN | UNCERTAIN | uncertain |
| 5306.0 | 0 | -2.4357 | 0.0797 | UNCERTAIN | UNCERTAIN | uncertain |
| 4814.0 | 0 | -2.4353 | 0.0797 | UNCERTAIN | UNCERTAIN | uncertain |
| 1541.0 | 1 | -2.4294 | 0.0802 | UNCERTAIN | UNCERTAIN | uncertain |
| 4529.0 | 0 | -2.4248 | 0.0805 | UNCERTAIN | UNCERTAIN | uncertain |
