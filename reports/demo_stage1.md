# Demo Stage1 Report

Generated: `2025-12-27T16:42:49.344220+00:00`

## Track: scifact

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/scifact_stage1_dev_train.parquet`
- n: `50`
- seed: `14`
- logit_col: `raw_max_top3`
- y_col: `y`
- tau: `0.609483051351548`
- t_lower: `0.7792245061204353`
- t_upper: `0.8008901176300922`

Routing distribution:

| metric | value |
|---|---|
| accept_rate | 0.4200 |
| reject_rate | 0.1000 |
| uncertain_rate | 0.4800 |

First 5 examples:

| qid | y | logit | cs_ret | decision | reason |
|---|---|---|---|---|---|
| 237 | 1 | 0.8606 | 0.8041 | ACCEPT | cs_ret>=t_upper |
| 222 | 1 | 0.7745 | 0.7809 | UNCERTAIN | t_lower<=cs_ret<t_upper |
| 797 | 1 | 0.8656 | 0.8054 | ACCEPT | cs_ret>=t_upper |
| 310 | 0 | 0.8492 | 0.8011 | ACCEPT | cs_ret>=t_upper |
| 703 | 1 | 0.8679 | 0.8060 | ACCEPT | cs_ret>=t_upper |

## Track: fever

- in_path: `/mnt/c/Users/nguye/Downloads/My projject/the-rag-auditor/data/calibration/fever_stage1_dev_train.parquet`
- n: `50`
- seed: `14`
- logit_col: `logit_platt`
- y_col: `y`
- tau: `0.9957177012600404`
- t_lower: `0.0721672686678491`
- t_upper: `1.0`

Routing distribution:

| metric | value |
|---|---|
| accept_rate | 0.0000 |
| reject_rate | 0.1000 |
| uncertain_rate | 0.9000 |

First 5 examples:

| qid | y | logit | cs_ret | decision | reason |
|---|---|---|---|---|---|
| 5980.0 | 0 | -1.7275 | 0.1500 | UNCERTAIN | t_lower<=cs_ret<t_upper |
| 8212.0 | 0 | -1.5156 | 0.1791 | UNCERTAIN | t_lower<=cs_ret<t_upper |
| 5969.0 | 0 | -2.1091 | 0.1073 | UNCERTAIN | t_lower<=cs_ret<t_upper |
| 6255.0 | 0 | -2.4992 | 0.0752 | UNCERTAIN | t_lower<=cs_ret<t_upper |
| 8983.0 | 0 | -2.0324 | 0.1150 | UNCERTAIN | t_lower<=cs_ret<t_upper |
