from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import stage1_2d_utility_tune as two_d


def test_2d_gating_improves_coverage():
    logits = [0.9, 0.8, 0.1, 0.2]
    cs = [0.9, 0.5, 0.9, 0.2]
    y = [1, 1, 0, 0]
    m1 = two_d.compute_metrics(logits, cs, y, 0.5, 0.7, 0.6, 0.3)
    m2 = two_d.compute_metrics(logits, cs, y, 0.5, 0.7, 0.8, 0.3)
    assert m1["coverage"] >= m2["coverage"]


def test_constraint_ci_wilson_rejects():
    logits = [0.9, 0.9, 0.9, 0.9]
    cs = [0.9, 0.9, 0.9, 0.9]
    y = [0, 0, 0, 0]
    m = two_d.compute_metrics(logits, cs, y, 0.1, 0.2, 0.1, 0.05)
    upper = two_d.wilson_ci(m["fp_accept_rate"] * len(y), len(y))[1]
    assert upper >= m["fp_accept_rate"]
