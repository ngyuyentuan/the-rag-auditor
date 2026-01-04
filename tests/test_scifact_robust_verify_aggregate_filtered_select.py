from scripts.run_product_proof_router_v2_scifact_robust_verify_aggregate_filtered import aggregate_group, select_ship


def test_filtered_select():
    rows = [
        {"ok_rate_stage1": 0.98, "fp_accept_rate": 0.009, "fn_reject_rate": 0.008, "stage2_ran_rate": 0.22, "capped_rate": 0.10, "p95_ms": 900.0, "file": "a.jsonl"},
        {"ok_rate_stage1": 0.97, "fp_accept_rate": 0.010, "fn_reject_rate": 0.007, "stage2_ran_rate": 0.21, "capped_rate": 0.11, "p95_ms": 910.0, "file": "b.jsonl"},
    ]
    agg = aggregate_group(rows)
    agg_map = {("tag", 0.26): agg}
    best = select_ship(agg_map, 0.01, 0.01, 0.25, 0.20)
    assert best is not None
