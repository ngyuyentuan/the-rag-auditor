# Product Proof v2

## scifact

Baseline vs tuned

| metric | baseline | tuned | delta |
|---|---:|---:|---:|
| accept_rate | 0.4400 | 0.4400 | 0.0000 |
| reject_rate | 0.0600 | 0.0600 | 0.0000 |
| uncertain_rate | 0.5000 | 0.5000 | 0.0000 |
| stage2_rate | 0.0000 | 0.0000 | 0.0000 |
| fp_accept_rate | 0.0400 | 0.0400 | 0.0000 |
| fn_reject_rate | 0.0000 | 0.0000 | 0.0000 |
| ok_rate_stage1 | 0.9600 | 0.9600 | 0.0000 |
| mean_ms | 0.03 | 0.03 | 0.00 |
| p95_ms | 0.05 | 0.05 | 0.00 |

- stage2_compute_count baseline: `0` tuned: `0`

- gate violation: `fp_accept_rate, uncertain_rate` (fp<=0.01, fn<=0.01, uncertain<=0.3)

Stage2 scenario

- stage1_only: stage2_rate=0.0000 mean_ms=0.03 p95_ms=0.04 rerank_ms_gt0=0 nli_ms_gt0=0
- stage2_enabled: stage2_rate=0.5000 mean_ms=4274.13 p95_ms=11136.50 rerank_ms_gt0=25 nli_ms_gt0=25

Policy comparisons

- prod thresholds: fallback baseline

- uncertain_only
  | metric | baseline | prod | delta |
  |---|---:|---:|---:|
  | accept_rate | 0.4400 | 0.4400 | 0.0000 |
  | reject_rate | 0.0600 | 0.0600 | 0.0000 |
  | uncertain_rate | 0.5000 | 0.5000 | 0.0000 |
  | stage2_rate | 0.5000 | 0.5000 | 0.0000 |
  | fp_accept_rate | 0.0400 | 0.0400 | 0.0000 |
  | fn_reject_rate | 0.0000 | 0.0000 | 0.0000 |
  | ok_rate_stage1 | 0.9600 | 0.9600 | 0.0000 |
  | mean_ms | 4274.13 | 4274.13 | 0.00 |
  | p95_ms | 11136.50 | 11136.50 | 0.00 |
- always
  | metric | baseline | prod | delta |
  |---|---:|---:|---:|
  | accept_rate | 0.4400 | 0.4400 | 0.0000 |
  | reject_rate | 0.0600 | 0.0600 | 0.0000 |
  | uncertain_rate | 0.5000 | 0.5000 | 0.0000 |
  | stage2_rate | 1.0000 | 1.0000 | 0.0000 |
  | fp_accept_rate | 0.0400 | 0.0400 | 0.0000 |
  | fn_reject_rate | 0.0000 | 0.0000 | 0.0000 |
  | ok_rate_stage1 | 0.9600 | 0.9600 | 0.0000 |
  | mean_ms | 13829.64 | 13829.64 | 0.00 |
  | p95_ms | 22005.67 | 22005.67 | 0.00 |

Ship recommendation

- consider uncertain_only if stage2_rate is acceptable and p95_ms stays within product latency budget

## fever

Baseline vs tuned

| metric | baseline | tuned | delta |
|---|---:|---:|---:|
| accept_rate | 0.0000 | 0.0000 | 0.0000 |
| reject_rate | 0.0600 | 0.0600 | 0.0000 |
| uncertain_rate | 0.9400 | 0.9400 | 0.0000 |
| stage2_rate | 0.0000 | 0.0000 | 0.0000 |
| fp_accept_rate | 0.0000 | 0.0000 | 0.0000 |
| fn_reject_rate | 0.0200 | 0.0200 | 0.0000 |
| ok_rate_stage1 | 0.9800 | 0.9800 | 0.0000 |
| mean_ms | 0.03 | 0.03 | 0.00 |
| p95_ms | 0.03 | 0.03 | 0.00 |

- stage2_compute_count baseline: `0` tuned: `0`

- gate violation: `fn_reject_rate, uncertain_rate` (fp<=0.01, fn<=0.01, uncertain<=0.3)

Stage2 scenario

- stage1_only: stage2_rate=0.0000 mean_ms=0.13 p95_ms=0.06 rerank_ms_gt0=0 nli_ms_gt0=0
- stage2_enabled: stage2_rate=0.9400 mean_ms=4076.18 p95_ms=6079.34 rerank_ms_gt0=47 nli_ms_gt0=47

Policy comparisons

- prod thresholds: fallback baseline

- uncertain_only
  | metric | baseline | prod | delta |
  |---|---:|---:|---:|
  | accept_rate | 0.0000 | 0.0000 | 0.0000 |
  | reject_rate | 0.0600 | 0.0600 | 0.0000 |
  | uncertain_rate | 0.9400 | 0.9400 | 0.0000 |
  | stage2_rate | 0.9400 | 0.9400 | 0.0000 |
  | fp_accept_rate | 0.0000 | 0.0000 | 0.0000 |
  | fn_reject_rate | 0.0200 | 0.0200 | 0.0000 |
  | ok_rate_stage1 | 0.9800 | 0.9800 | 0.0000 |
  | mean_ms | 4076.18 | 4076.18 | 0.00 |
  | p95_ms | 6079.34 | 6079.34 | 0.00 |
- always
  | metric | baseline | prod | delta |
  |---|---:|---:|---:|
  | accept_rate | 0.0000 | 0.0000 | 0.0000 |
  | reject_rate | 0.0600 | 0.0600 | 0.0000 |
  | uncertain_rate | 0.9400 | 0.9400 | 0.0000 |
  | stage2_rate | 1.0000 | 1.0000 | 0.0000 |
  | fp_accept_rate | 0.0000 | 0.0000 | 0.0000 |
  | fn_reject_rate | 0.0200 | 0.0200 | 0.0000 |
  | ok_rate_stage1 | 0.9800 | 0.9800 | 0.0000 |
  | mean_ms | 3821.39 | 3821.39 | 0.00 |
  | p95_ms | 5633.21 | 5633.21 | 0.00 |

Ship recommendation

- consider uncertain_only if stage2_rate is acceptable and p95_ms stays within product latency budget
