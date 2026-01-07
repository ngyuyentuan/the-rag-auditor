# Router Eval Large-N (fever)

- thresholds_yaml: `configs/thresholds.yaml`
- t_lower: 0.0721672686678491
- t_upper: 1.0
- tau: 0.9957177012600404
- guard_band: 0.0
- entail_min: 0.7
- contradict_min: 0.7
- neutral_max: 0.6

- runs_jsonl: `runs/day12_fever_500_e2e.jsonl`

## Stage1 distribution
| metric | value |
|---|---:|
| accept_rate | 0.0000 |
| reject_rate | 0.1120 |
| uncertain_rate | 0.8880 |

## Policy A: pure stage1
| metric | value | 95% CI |
|---|---:|---:|
| final_accept_rate | 0.0000 | |
| final_reject_rate | 0.1120 | |
| final_uncertain_rate | 0.8880 | [0.8580, 0.9160] |
| final_fp_accept_rate | 0.0000 | [0.0000, 0.0000] |
| final_fn_reject_rate | 0.0040 | [0.0000, 0.0100] |
| final_ok_rate | 0.9960 | [0.9900, 1.0000] |

## Policy B: guarded router
| metric | value | 95% CI |
|---|---:|---:|
| final_accept_rate | 0.1580 | |
| final_reject_rate | 0.4940 | |
| final_uncertain_rate | 0.3480 | [0.3080, 0.3900] |
| trigger_rate | 0.8880 | |
| final_fp_accept_rate | 0.1240 | [0.0980, 0.1540] |
| final_fn_reject_rate | 0.0440 | [0.0260, 0.0640] |
| final_ok_rate | 0.8320 | [0.8000, 0.8640] |

## Bootstrap delta (B - A)
| metric | 95% CI |
|---|---:|
| ok_rate | [-0.1960, -0.1320] |
| fp_accept_rate | [0.0980, 0.1540] |
| fn_reject_rate | [0.0240, 0.0580] |
| uncertain_rate | [-0.5820, -0.4980] |

## Stage2 and latency
| metric | value |
|---|---:|
| stage2_ran_rate | 0.8880 |
| mean_total_ms | 333.14 |
| p95_total_ms | 891.98 |
| p99_total_ms | 1122.16 |
| rerank_ms_gt0_rate | 0.8880 |
| nli_ms_gt0_rate | 0.8880 |

## Over-engineering check
router does not materially improve ok_rate or trigger_rate vs baseline
## FEVER verdict
negative result: accept_rate remains near zero or uncertain is very high; stage1-only is weak

## Repro command
```
scripts/eval_router_large_n.py --track fever --runs_jsonl runs/day12_fever_500_e2e.jsonl --logit_col logit_platt --out_md reports/router_eval_fever_large.md
```
