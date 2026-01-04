# Day15 Router A/B Sweep

Stage2 policy is simulated from cs_ret probabilities to avoid collapse before running real stage2.

## Track: scifact

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `500`
- seed: `15`
- logit_col: `raw_max_top3`
- thresholds_yaml: `configs/thresholds.yaml`
- tuned_thresholds_yaml: `configs/thresholds_stage1_tuned_scifact.yaml`
- tau: `0.609483051351548`
- t_lower: `0.0`
- t_upper: `0.7777551020408163`

Stage1 baseline

| metric | value |
|---|---|
| accept_rate | 0.9440 |
| reject_rate | 0.0000 |
| uncertain_rate | 0.0560 |
| fp_accept_rate | 0.1720 |
| fn_reject_rate | 0.0000 |
| ok_rate | 0.8280 |

Top 10 configs

| guard_band | entail_min | contradict_min | neutral_max | stage2_trigger_rate | final_accept_rate | final_reject_rate | final_uncertain_rate | final_coverage | unsafe_accept_proxy | unsafe_reject_proxy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.7000 | 0.7000 | 0.6000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.7000 | 0.7000 | 0.8000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.7000 | 0.8000 | 0.6000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.7000 | 0.8000 | 0.8000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.7000 | 0.9000 | 0.6000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.7000 | 0.9000 | 0.8000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.8000 | 0.7000 | 0.6000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.8000 | 0.7000 | 0.8000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.8000 | 0.8000 | 0.6000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |
| 0.0000 | 0.8000 | 0.8000 | 0.8000 | 0.0560 | 0.9440 | 0.0000 | 0.0560 | 0.9440 | 0.1720 | 0.0000 |

Recommended config

- guard_band: `0.0`
- entail_min: `0.7`
- contradict_min: `0.7`
- neutral_max: `0.6`
- constraint_met: `True`

Interpretation

Prior collapse is driven by strict policy thresholds and wide guard bands that route most cases to stage2 but keep them UNCERTAIN. The selected config maximizes coverage while keeping proxy error within slack of stage1.

## Track: fever

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `500`
- seed: `15`
- logit_col: `logit_platt`
- thresholds_yaml: `configs/thresholds.yaml`
- tuned_thresholds_yaml: `configs/thresholds_stage1_tuned_fever.yaml`
- tau: `0.9957177012600404`
- t_lower: `0.14142857142857143`
- t_upper: `0.2524489795918367`

Stage1 baseline

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.8320 |
| uncertain_rate | 0.1680 |
| fp_accept_rate | 0.0000 |
| fn_reject_rate | 0.0720 |
| ok_rate | 0.9280 |

Top 10 configs

| guard_band | entail_min | contradict_min | neutral_max | stage2_trigger_rate | final_accept_rate | final_reject_rate | final_uncertain_rate | final_coverage | unsafe_accept_proxy | unsafe_reject_proxy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.7000 | 0.7000 | 0.6000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0000 | 0.7000 | 0.7000 | 0.8000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0000 | 0.8000 | 0.7000 | 0.6000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0000 | 0.8000 | 0.7000 | 0.8000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0000 | 0.9000 | 0.7000 | 0.6000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0000 | 0.9000 | 0.7000 | 0.8000 | 0.1680 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0100 | 0.7000 | 0.7000 | 0.6000 | 0.2460 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0100 | 0.7000 | 0.7000 | 0.8000 | 0.2460 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0100 | 0.8000 | 0.7000 | 0.6000 | 0.2460 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |
| 0.0100 | 0.8000 | 0.7000 | 0.8000 | 0.2460 | 0.0000 | 0.9680 | 0.0320 | 0.9680 | 0.0000 | 0.0800 |

Recommended config

- guard_band: `0.0`
- entail_min: `0.7`
- contradict_min: `0.7`
- neutral_max: `0.6`
- constraint_met: `True`

Interpretation

Prior collapse is driven by strict policy thresholds and wide guard bands that route most cases to stage2 but keep them UNCERTAIN. The selected config maximizes coverage while keeping proxy error within slack of stage1.
