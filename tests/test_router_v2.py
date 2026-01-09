import math

import pytest

from scripts.train_stage1_router_v2 import select_best_candidate
from scripts.run_product_proof_router_v2_matrix import summarize
from src.stage1.router_v2 import decide_router


def test_router_decision_rule():
    assert decide_router(0.9, 0.8, 0.2) == "ACCEPT"
    assert decide_router(0.1, 0.8, 0.2) == "REJECT"
    assert decide_router(0.5, 0.8, 0.2) == "UNCERTAIN"
    assert decide_router(0.95, 0.8, 0.2) != "REJECT"


def test_router_decision_requires_ordered_thresholds():
    with pytest.raises(ValueError):
        decide_router(0.5, 0.2, 0.2)


def test_min_route_selection():
    results = [
        {"fp_accept_rate": 0.01, "fn_reject_rate": 0.01, "uncertain_rate": 0.4, "expected_cost": 1.0, "ok_rate": 0.98},
        {"fp_accept_rate": 0.01, "fn_reject_rate": 0.01, "uncertain_rate": 0.2, "expected_cost": 2.0, "ok_rate": 0.97},
        {"fp_accept_rate": 0.02, "fn_reject_rate": 0.01, "uncertain_rate": 0.1, "expected_cost": 0.5, "ok_rate": 0.99},
    ]
    best, feasible = select_best_candidate(results, 0.01, 0.01, "min_route")
    assert feasible is True
    assert best["uncertain_rate"] == 0.2


def test_capped_count_budget_formula():
    stage2_budget = 0.45
    n = 20
    uncertain_count = 12
    budget_k = int(math.floor(stage2_budget * n))
    expected_capped = max(0, uncertain_count - budget_k)
    expected_should_run = uncertain_count - expected_capped
    expected_missing_outputs = 1
    rows = []
    accept_count = n - uncertain_count
    for i in range(n):
        if i < accept_count:
            decision = "ACCEPT"
            budget_capped = False
            stage2 = {}
        else:
            decision = "UNCERTAIN"
            unc_idx = i - accept_count
            budget_capped = unc_idx < expected_capped
            if budget_capped:
                stage2 = {"ran": False}
            elif unc_idx == expected_capped:
                stage2 = {"ran": False, "rerank": {"reason": "missing_stage2_outputs"}}
            else:
                stage2 = {"ran": True}
        rows.append({
            "stage1": {
                "route_decision": decision,
                "router": {
                    "budget_capped": budget_capped,
                    "stage2_budget": stage2_budget,
                },
            },
            "stage2": stage2,
            "timing_ms": {},
            "pred": {},
            "ground_truth": {"y": 1},
        })
    summary = summarize(rows, "uncertain_only")
    assert summary["budget_k"] == budget_k
    assert summary["capped_count"] == expected_capped
    assert summary["stage2_should_run"] == expected_should_run
    assert summary["stage2_should_run"] == min(uncertain_count, summary["budget_k"])
    expected_missing = summary["stage2_should_run"] - summary["stage2_ran_count"]
    if expected_missing < 0:
        expected_missing = 0
    assert summary["missing_stage2_outputs"] == expected_missing
    assert summary["missing_stage2_outputs"] == expected_missing_outputs
    assert summary["missing_stage2_outputs"] == summary["missing_stage2_reason_count"]
    assert summary["missing_stage2_outputs_negative_count"] == 0
