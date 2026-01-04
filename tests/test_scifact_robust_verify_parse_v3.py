from scripts.run_product_proof_router_v2_scifact_robust_verify_v3 import aggregate_budget_rows, select_ship, average_probs


def test_aggregate_and_select_ship_v3():
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
            "fp_accept_rate": 0.008,
            "fn_reject_rate": 0.008,
            "stage2_ran_rate": 0.22,
            "capped_rate": 0.15,
            "ok_rate_stage1": 0.985,
            "p95_ms": 900.0,
        },
        {
            "budget": 0.28,
            "fp_accept_rate": 0.007,
            "fn_reject_rate": 0.006,
            "stage2_ran_rate": 0.21,
            "capped_rate": 0.14,
            "ok_rate_stage1": 0.99,
            "p95_ms": 950.0,
        },
    ]
    agg = aggregate_budget_rows(rows)
    assert agg[0.26]["fp_worst"] == 0.02
    assert agg[0.28]["fn_worst"] == 0.008
    best = select_ship(agg, 0.008, 0.008, 0.22, 0.15)
    assert best is not None
    _, _, _, budget = best
    assert budget == 0.28


def test_average_probs_flip():
    probs = [[0.2, 0.6], [0.4, 0.8]]
    flips = [False, True]
    avg = average_probs(probs, flips)
    assert abs(avg[0] - 0.4) < 1e-6
