# Router Eval Large-N (scifact)

- thresholds_yaml: `configs/thresholds.yaml`
- t_lower: 0.7792245061204353
- t_upper: 0.8008901176300922
- tau: 0.609483051351548
- guard_band: 0.0
- entail_min: 0.7
- contradict_min: 0.7
- neutral_max: 0.6

- runs_jsonl: `runs/day12_scifact_500_e2e.jsonl`

## Stage1 distribution
| metric | value |
|---|---:|
| accept_rate | 0.4540 |
| reject_rate | 0.0560 |
| uncertain_rate | 0.4900 |

## Policy A: pure stage1
| metric | value | 95% CI |
|---|---:|---:|
| final_accept_rate | 0.4540 | |
| final_reject_rate | 0.0560 | |
| final_uncertain_rate | 0.4900 | [0.4460, 0.5340] |
| final_fp_accept_rate | 0.0420 | [0.0260, 0.0600] |
| final_fn_reject_rate | 0.0340 | [0.0180, 0.0500] |
| final_ok_rate | 0.9240 | [0.9000, 0.9480] |

## Policy B: guarded router
| metric | value | 95% CI |
|---|---:|---:|
| final_accept_rate | 0.5460 | |
| final_reject_rate | 0.1720 | |
| final_uncertain_rate | 0.2820 | [0.2440, 0.3220] |
| trigger_rate | 0.4900 | |
| final_fp_accept_rate | 0.0560 | [0.0360, 0.0760] |
| final_fn_reject_rate | 0.1300 | [0.1020, 0.1620] |
| final_ok_rate | 0.8140 | [0.7800, 0.8480] |

## Bootstrap delta (B - A)
| metric | 95% CI |
|---|---:|
| ok_rate | [-0.1380, -0.0840] |
| fp_accept_rate | [0.0040, 0.0260] |
| fn_reject_rate | [0.0700, 0.1240] |
| uncertain_rate | [-0.2440, -0.1740] |

## Stage2 and latency
| metric | value |
|---|---:|
| stage2_ran_rate | 0.4900 |
| mean_total_ms | 1313.96 |
| p95_total_ms | 3929.82 |
| p99_total_ms | 5020.21 |
| rerank_ms_gt0_rate | 0.4900 |
| nli_ms_gt0_rate | 0.4900 |

## Over-engineering check
router does not materially improve ok_rate or trigger_rate vs baseline

## Repro command
```
scripts/eval_router_large_n.py --track scifact --runs_jsonl runs/day12_scifact_500_e2e.jsonl --logit_col raw_max_top3 --out_md reports/router_eval_scifact_large.md
```
