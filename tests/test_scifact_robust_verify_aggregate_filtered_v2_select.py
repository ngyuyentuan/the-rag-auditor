import json
from pathlib import Path

from scripts.run_product_proof_router_v2_scifact_robust_verify_aggregate_filtered_v2 import summarize_groups


def test_select_ship_tiebreak():
    constraints = {"fp": 0.01, "fn": 0.01, "ran": 0.25, "cap": 0.20}
    groups = {
        ("tagA", 0.28): [
            {"file": "a1", "fp_accept_rate": 0.005, "fn_reject_rate": 0.005, "stage2_ran_rate": 0.20, "capped_rate": 0.10, "p95_ms": 100.0, "ok_rate_stage1": 0.99},
            {"file": "a2", "fp_accept_rate": 0.006, "fn_reject_rate": 0.006, "stage2_ran_rate": 0.21, "capped_rate": 0.11, "p95_ms": 120.0, "ok_rate_stage1": 0.98},
        ],
        ("tagB", 0.30): [
            {"file": "b1", "fp_accept_rate": 0.004, "fn_reject_rate": 0.004, "stage2_ran_rate": 0.21, "capped_rate": 0.12, "p95_ms": 80.0, "ok_rate_stage1": 0.97},
        ],
    }
    summary, pick, closest = summarize_groups(groups, constraints)
    assert closest is None
    assert pick["tag"] == "tagA"
    assert pick["budget"] == 0.28


def test_unknown_and_missing_budget_handling(tmp_path):
    rows = [
        {"stage1": {"route_decision": "UNCERTAIN"}, "stage2": {"ran": False}, "timing_ms": {"total_ms": 1.0}, "ground_truth": {"y": 1}},
    ]
    p = tmp_path / "scifact_unknown.jsonl"
    with open(p, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert p.exists()
