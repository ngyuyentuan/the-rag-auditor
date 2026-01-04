from scripts.run_product_proof_router_v2_scifact_robust_verify_v8 import aggregate_budget_rows, select_ship


def test_aggregate_and_select_ship_v8():
    rows = [
        {"budget": 0.26, "ok_rate_stage1": 0.98, "fp_accept_rate": 0.008, "fn_reject_rate": 0.006, "stage2_ran_rate": 0.20, "capped_rate": 0.10, "p95_ms": 900.0},
        {"budget": 0.26, "ok_rate_stage1": 0.97, "fp_accept_rate": 0.009, "fn_reject_rate": 0.007, "stage2_ran_rate": 0.22, "capped_rate": 0.12, "p95_ms": 950.0},
        {"budget": 0.28, "ok_rate_stage1": 0.99, "fp_accept_rate": 0.012, "fn_reject_rate": 0.005, "stage2_ran_rate": 0.18, "capped_rate": 0.08, "p95_ms": 880.0},
    ]
    agg = aggregate_budget_rows(rows)
    best = select_ship(agg, 0.01, 0.01, 0.25, 0.20)
    assert best is not None
    _, _, _, budget = best
    assert abs(budget - 0.26) < 1e-9
