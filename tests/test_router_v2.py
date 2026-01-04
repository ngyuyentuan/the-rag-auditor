import pytest

from scripts.train_stage1_router_v2 import select_best_candidate
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
