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
        },
    ]
    pick_lo, _ = sut.select_best(candidates, 0.05, 0.05, 0.1, 1.0, 0.0)
    pick_hi, _ = sut.select_best(candidates, 0.05, 0.05, 10.0, 1.0, 0.0)
    assert pick_hi["fp_accept_rate"] <= pick_lo["fp_accept_rate"]
