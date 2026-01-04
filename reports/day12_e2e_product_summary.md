# Day12 E2E Product Summary

Included runs:
- `runs/day12_scifact_500_e2e.jsonl`
- `runs/day12_scifact_500_e2e_always.jsonl`
- `runs/day12_scifact_500_e2e_random.jsonl`
- `runs/day12_fever_500_e2e.jsonl`
- `runs/day12_fever_500_e2e_always.jsonl`
- `runs/day12_fever_500_e2e_random.jsonl`

## Track: scifact

| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 500 | 0.4540 | 0.0560 | 0.4900 | 0.4900 | 0.9240 | 0.0420 | 0.0340 | 1313.96 | 3929.82 | 5020.21 | 6046.41 |
| always_stage2 | 500 | 0.4540 | 0.0560 | 0.4900 | 1.0000 | 0.9240 | 0.0420 | 0.0340 | 3610.71 | 6751.32 | 9959.44 | 13999.53 |
| random | 500 | 0.4480 | 0.0620 | 0.4900 | 0.4900 | 0.8760 | 0.0740 | 0.0500 | 1239.54 | 3968.56 | 4684.29 | 5365.46 |

Product interpretation

UNCERTAIN is a deferral to stage2 and is not counted as error in ok_rate.
ok_rate reflects stage1 label-only correctness on accept/reject decisions, not end-to-end factual correctness without gold evidence/verdict.
always_stage2 vs calibrated: delta ok_rate `0.0000`, delta mean_ms `2296.75`.

## Track: fever

| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 500 | 0.0000 | 0.1120 | 0.8880 | 0.8880 | 0.9960 | 0.0000 | 0.0040 | 333.14 | 891.98 | 1122.16 | 1463.60 |
| always_stage2 | 500 | 0.0000 | 0.1120 | 0.8880 | 1.0000 | 0.9960 | 0.0000 | 0.0040 | 351.07 | 808.08 | 1118.90 | 2520.05 |
| random | 500 | 0.0000 | 0.1180 | 0.8820 | 0.8820 | 0.9920 | 0.0000 | 0.0080 | 747.24 | 1290.85 | 1622.92 | 1820.61 |

Product interpretation

UNCERTAIN is a deferral to stage2 and is not counted as error in ok_rate.
ok_rate reflects stage1 label-only correctness on accept/reject decisions, not end-to-end factual correctness without gold evidence/verdict.
always_stage2 vs calibrated: delta ok_rate `0.0000`, delta mean_ms `17.93`.
