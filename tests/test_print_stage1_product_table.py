from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import print_stage1_product_table as mod


def test_metrics_basic(tmp_path):
    df = pd.DataFrame({"logit": [2.0, -2.0, 0.0], "y": [1, 0, 1]})
    conf = {"tau": 1.0, "t_lower": 0.2, "t_upper": 0.8}
    m = mod.evaluate(df, "logit", "y", conf)
    assert np.isfinite(m["ok_rate"])
    assert 0.0 <= m["fp_upper95"] <= 1.0
    assert 0.0 <= m["fn_upper95"] <= 1.0
    assert 0.0 <= m["fp_decided_upper95"] <= 1.0
    assert 0.0 <= m["fn_decided_upper95"] <= 1.0
    assert m["decided_count"] == m["n"] - int(m["uncertain_rate"] * m["n"])


def test_decided_differs_from_total():
    df = pd.DataFrame({"logit": [2.0, -2.0, 0.0], "y": [1, 0, 1]})
    conf = {"tau": 1.0, "t_lower": 0.0, "t_upper": 1.5}
    m = mod.evaluate(df, "logit", "y", conf)
    assert m["coverage"] < 1.0
    assert m["fp_decided_rate"] <= m["fp_accept_rate"] or m["fn_decided_rate"] <= m["fn_reject_rate"]


def test_insufficient_decided_status():
    df = pd.DataFrame({"logit": [0.0, 0.0, 0.0], "y": [1, 0, 1]})
    conf = {"tau": 1.0, "t_lower": -1.0, "t_upper": 1.0}
    m = mod.evaluate(df, "logit", "y", conf)
    decided_status = "INSUFFICIENT_N" if m["decided_count"] < 10 else "PASS"
    assert decided_status == "INSUFFICIENT_N"


def test_accept_status_insufficient():
    df = pd.DataFrame({"logit": [0.0, 0.0], "y": [1, 0]})
    conf = {"tau": 1.0, "t_lower": -1.0, "t_upper": 1.0}
    m = mod.evaluate(df, "logit", "y", conf)
    accept_status = "INSUFFICIENT_N" if m["accept_count"] < 5 else "PASS"
    assert accept_status == "INSUFFICIENT_N"


def test_action_accept_sets_na():
    df = pd.DataFrame({"logit": [2.0, 2.0, 2.0], "y": [1, 1, 1]})
    conf = {"tau": 1.0, "t_lower": -1.0, "t_upper": 1.0}
    m = mod.evaluate(df, "logit", "y", conf)
    class Args:
        min_decided_count = 0
        max_fp_decided_upper95 = 0.10
        max_fn_decided_upper95 = 0.10
        min_accept_count = 1
        min_reject_count = 1
        max_fp_given_accept_upper95 = 0.10
        max_fn_given_reject_upper95 = 0.10
    decided_status, accept_status, reject_status, action_status, ci_safe, decided_ci_safe = mod.compute_statuses(m, Args, "accept_only", {})
    assert reject_status == "N/A"
    assert action_status == accept_status
    assert decided_status == "N/A"
    assert accept_status in ("PASS", "INSUFFICIENT_N", "FAIL")
    assert ci_safe == "N/A"
    assert decided_ci_safe == "N/A"


def test_action_reject_status_mapping():
    df = pd.DataFrame({"logit": [-2.0, -2.0, -2.0], "y": [0, 0, 0]})
    conf = {"tau": 1.0, "t_lower": -1.0, "t_upper": 1.0}
    m = mod.evaluate(df, "logit", "y", conf)
    class Args:
        min_decided_count = 0
        max_fp_decided_upper95 = 0.10
        max_fn_decided_upper95 = 0.10
        min_accept_count = 1
        min_reject_count = 1
        max_fp_given_accept_upper95 = 0.10
        max_fn_given_reject_upper95 = 0.10
    decided_status, accept_status, reject_status, action_status, ci_safe, decided_ci_safe = mod.compute_statuses(m, Args, "reject_only", {})
    assert accept_status == "N/A"
    assert action_status == reject_status
    assert decided_status == "N/A"
    assert reject_status in ("PASS", "INSUFFICIENT_N", "FAIL")
