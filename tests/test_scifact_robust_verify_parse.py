from scripts.run_product_proof_router_v2_scifact_robust_verify import aggregate_budget_rows, select_ship


def test_aggregate_budget_rows_and_select_ship():
    rows = [
        {
            "budget": 0.26,
            "fp_accept_rate": 0.01,
            "fn_reject_rate": 0.01,
            "stage2_ran_rate": 0.24,
            "capped_rate": 0.18,
            "ok_rate_stage1": 0.98,
            "p95_ms": 1000.0,
        },
        {
            "budget": 0.26,
            "fp_accept_rate": 0.02,
            "fn_reject_rate": 0.01,
            "stage2_ran_rate": 0.26,
            "capped_rate": 0.19,
            "ok_rate_stage1": 0.97,
            "p95_ms": 1100.0,
        },
        {
            "budget": 0.28,
            "fp_accept_rate": 0.01,
            "fn_reject_rate": 0.01,
            "stage2_ran_rate": 0.25,
            "capped_rate": 0.20,
            "ok_rate_stage1": 0.985,
            "p95_ms": 900.0,
        },
        {
            "budget": 0.28,
            "fp_accept_rate": 0.01,
            "fn_reject_rate": 0.00,
            "stage2_ran_rate": 0.22,
            "capped_rate": 0.15,
            "ok_rate_stage1": 0.99,
            "p95_ms": 950.0,
        },
    ]
    agg = aggregate_budget_rows(rows)
    assert agg[0.26]["fp_worst"] == 0.02
    assert agg[0.26]["ran_worst"] == 0.26
    assert agg[0.28]["fn_worst"] == 0.01
    best = select_ship(agg, 0.01, 0.01, 0.25, 0.20)
    assert best is not None
    _, _, _, budget = best
    assert budget == 0.28
