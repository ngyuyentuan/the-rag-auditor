# Day12 E2E Stats

- Generated: `2025-12-30T12:02:18.888090+00:00`
- Known limitations: train-only sanity check; cs_ret uses sigmoid(logit / tau) where tau<1 sharpens and tau>1 softens; FEVER is expected to be weak with current thresholds.

## Track: scifact

| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms | std_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 500 | 0.454 | 0.056 | 0.490 | 0.490 | 0.434 | 0.042 | 0.034 | 1313.96 | 3929.82 | 5020.21 | 6046.41 | 1479.27 |
| always_stage2 | 500 | 0.454 | 0.056 | 0.490 | 1.000 | 0.434 | 0.042 | 0.034 | 3610.71 | 6751.32 | 9959.44 | 13999.53 | 1739.24 |
| random | 500 | 0.448 | 0.062 | 0.490 | 0.490 | 0.386 | 0.074 | 0.050 | 1239.54 | 3968.56 | 4684.29 | 5365.46 | 1428.33 |

NLI label distribution (stage2 ran):
- calibrated:
  CONTRADICTION: 0.278
  ENTAILMENT: 0.269
  NEUTRAL: 0.453
- always_stage2:
  CONTRADICTION: 0.296
  ENTAILMENT: 0.268
  NEUTRAL: 0.436
- random:
  CONTRADICTION: 0.302
  ENTAILMENT: 0.253
  NEUTRAL: 0.445

## Track: fever

| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms | std_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 500 | 0.000 | 0.112 | 0.888 | 0.888 | 0.108 | 0.000 | 0.004 | 333.14 | 891.98 | 1122.16 | 1463.60 | 233.87 |
| always_stage2 | 500 | 0.000 | 0.112 | 0.888 | 1.000 | 0.108 | 0.000 | 0.004 | 351.07 | 808.08 | 1118.90 | 2520.05 | 211.64 |
| random | 500 | 0.000 | 0.118 | 0.882 | 0.882 | 0.110 | 0.000 | 0.008 | 747.24 | 1290.85 | 1622.92 | 1820.61 | 371.88 |

NLI label distribution (stage2 ran):
- calibrated:
  CONTRADICTION: 0.453
  ENTAILMENT: 0.189
  NEUTRAL: 0.358
- always_stage2:
  CONTRADICTION: 0.464
  ENTAILMENT: 0.188
  NEUTRAL: 0.348
- random:
  CONTRADICTION: 0.456
  ENTAILMENT: 0.193
  NEUTRAL: 0.351
