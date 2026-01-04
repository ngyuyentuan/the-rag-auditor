# Product Proof Router v2 SciFact Sweep

| name | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.4800 | 0.0850 | 0.4350 | 0.4350 | 0.4350 | 0.0000 | 0.0400 | 0.0450 | 0.9150 | 1633.85 | 4768.73 |
| router_v2_current | 0.2600 | 0.0000 | 0.7400 | 0.7400 | 0.2300 | 0.2900 | 0.0100 | 0.0000 | 0.9900 | 884.19 | 4356.65 |
| router_v2_min_route_logreg | 0.2600 | 0.0300 | 0.7100 | 0.7100 | 0.1750 | 0.2600 | 0.0100 | 0.0100 | 0.9800 | 677.63 | 4297.08 |
| router_v2_min_route_hgb | 0.4550 | 0.0750 | 0.4700 | 0.4700 | 0.2700 | 0.0200 | 0.0100 | 0.0050 | 0.9850 | 1013.69 | 4539.23 |

Ship recommendation

- no candidate met fp<=0.01 fn<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20