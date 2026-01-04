# Day12 E2E Stats

- Generated: `2025-12-31T10:52:16.629354+00:00`
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
