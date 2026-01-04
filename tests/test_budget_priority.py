import json
from pathlib import Path

import numpy as np

from scripts.run_product_proof_router_v2_hgb_param_sweep_scifact import apply_budget


def test_budget_priority_deterministic(tmp_path):
    rows = [
        {"metadata": {"qid": "1"}, "stage1": {"route_decision": "UNCERTAIN", "cs_ret": 0.5}, "stage2": {"ran": True}, "timing_ms": {"rerank_ms": 1.0, "nli_ms": 1.0, "total_ms": 2.0}, "ground_truth": {"y": 1}},
        {"metadata": {"qid": "2"}, "stage1": {"route_decision": "UNCERTAIN", "cs_ret": 0.9}, "stage2": {"ran": True}, "timing_ms": {"rerank_ms": 1.0, "nli_ms": 1.0, "total_ms": 2.0}, "ground_truth": {"y": 0}},
    ]
    prepared = [
        {"row": rows[0], "qid": "1", "p_hat": 0.5, "decision": "UNCERTAIN", "features_present": {"f1": True}},
        {"row": rows[1], "qid": "2", "p_hat": 0.9, "decision": "UNCERTAIN", "features_present": {"f1": True}},
    ]
    out_path = tmp_path / "out.jsonl"
    apply_budget(prepared, out_path, type("R", (), {"t_accept": 0.8, "t_reject": 0.2})(), 0.5)
    out = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ran = [r["metadata"]["qid"] for r in out if r["stage2"]["ran"]]
    assert ran == ["1"]
