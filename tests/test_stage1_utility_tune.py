from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage1_utility_tune as sut


def test_lambda_fp_monotonic():
    candidates = [
        {
            "t_lower": 0.2,
            "t_upper": 0.8,
            "fp_accept_rate": 0.04,
            "fn_reject_rate": 0.01,
            "uncertain_rate": 0.3,
            "ok_rate": 0.95,
            "coverage": 0.7,
            "accuracy_on_decided": 0.9,
            "n": 100,
            "fp_count": 4,
            "fn_count": 1,
            "decided": 70,
            "fp_decided_rate": 4 / 70,
            "fn_decided_rate": 1 / 70,
        },
        {
            "t_lower": 0.3,
            "t_upper": 0.9,
            "fp_accept_rate": 0.02,
            "fn_reject_rate": 0.02,
            "uncertain_rate": 0.3,
            "ok_rate": 0.96,
            "coverage": 0.7,
            "accuracy_on_decided": 0.91,
            "n": 100,
            "fp_count": 2,
            "fn_count": 2,
            "decided": 70,
            "fp_decided_rate": 2 / 70,
            "fn_decided_rate": 2 / 70,
        },
    ]
    pick_lo, _ = sut.select_best(candidates, 0.05, 0.05, 0.1, 1.0, 0.0, "none", 0.0, 0.1, 0.1, "both", False)
    pick_hi, _ = sut.select_best(candidates, 0.05, 0.05, 10.0, 1.0, 0.0, "none", 0.0, 0.1, 0.1, "both", False)
    assert pick_hi["fp_accept_rate"] <= pick_lo["fp_accept_rate"]


def test_lambda_unc_prefers_coverage_when_risk_tied():
    candidates = [
        {"t_lower": 0.1, "t_upper": 0.6, "fp_accept_rate": 0.01, "fn_reject_rate": 0.01, "uncertain_rate": 0.4, "ok_rate": 0.98, "coverage": 0.6, "accuracy_on_decided": 0.9},
        {"t_lower": 0.2, "t_upper": 0.7, "fp_accept_rate": 0.01, "fn_reject_rate": 0.01, "uncertain_rate": 0.2, "ok_rate": 0.98, "coverage": 0.8, "accuracy_on_decided": 0.9},
    ]
    for c in candidates:
        c["n"] = 100
        c["fp_count"] = int(c["fp_accept_rate"] * c["n"])
        c["fn_count"] = int(c["fn_reject_rate"] * c["n"])
        c["decided"] = int((1 - c["uncertain_rate"]) * c["n"])
        c["fp_decided_rate"] = c["fp_count"] / c["decided"] if c["decided"] else 0.0
        c["fn_decided_rate"] = c["fn_count"] / c["decided"] if c["decided"] else 0.0
    pick_low_unc, _ = sut.select_best(candidates, 0.05, 0.05, 1.0, 1.0, 5.0, "none", 0.0, 0.1, 0.1, "both", False)
    pick_high_unc, _ = sut.select_best(candidates, 0.05, 0.05, 1.0, 1.0, 0.1, "none", 0.0, 0.1, 0.1, "both", False)
    assert pick_low_unc["uncertain_rate"] <= pick_high_unc["uncertain_rate"]


def test_joint_grid_tau_differs():
    # simulate scaling effect on tau preference
    logits = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]
    y = [1, 1, 1, 0, 0, 0]
    # config tau list only uses base tau=1.0
    base = sut.compute_metrics([sut.compute_cs_ret_from_logit(x, 1.0) for x in logits], y, 0.2, 0.8)
    # joint grid should consider different tau and can find better utility with lower fp/fn
    tau_grid = [0.5, 1.0, 1.5]
    candidates = []
    for tau in tau_grid:
        cs = [sut.compute_cs_ret_from_logit(x, tau) for x in logits]
        for tl in (0.2, 0.3):
            for tu in (0.7, 0.8, 0.9):
                if tl >= tu:
                    continue
                m = sut.compute_metrics(cs, y, tl, tu)
                m["tau"] = tau
                m["t_lower"] = tl
                m["t_upper"] = tu
                m["coverage"] = 1 - m["uncertain_rate"]
                m["n"] = len(y)
                m["fp_count"] = int(m["fp_accept_rate"] * m["n"])
                m["fn_count"] = int(m["fn_reject_rate"] * m["n"])
                m["decided"] = int((1 - m["uncertain_rate"]) * m["n"])
                m["fp_decided_rate"] = m["fp_count"] / m["decided"] if m["decided"] else 0.0
                m["fn_decided_rate"] = m["fn_count"] / m["decided"] if m["decided"] else 0.0
                candidates.append(m)
    pick, _ = sut.select_best(candidates, 0.05, 0.05, 1.0, 1.0, 1.0, "none", 0.0, 0.1, 0.1, "both", False)
    assert pick["tau"] != 1.0
