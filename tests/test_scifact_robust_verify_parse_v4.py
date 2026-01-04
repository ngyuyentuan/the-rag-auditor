from scripts.run_product_proof_router_v2_scifact_robust_verify_v4 import fit_thresholds
import numpy as np


def test_fit_thresholds_deterministic():
    p_hat = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
    y = np.array([0, 0, 1, 1, 1, 0])
    a_list = [0.2, 0.5]
    b_list = [0.2, 0.5]
    r1 = fit_thresholds(p_hat, y, a_list, b_list, "min_route")
    r2 = fit_thresholds(p_hat, y, a_list, b_list, "min_route")
    assert r1["t_accept"] == r2["t_accept"]
    assert r1["t_reject"] == r2["t_reject"]
