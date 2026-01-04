import math
import random

import numpy as np

from src.utils.buffer import (
    compute_cs_ret_from_logit,
    decide_route,
    legacy_ok_fp_fn,
    routing_distribution,
    sample_stratified_random_decision,
    sample_uniform_random_decision,
    stage1_outcomes,
)


def test_temperature_scaling_direction():
    logit_pos = 2.0
    logit_neg = -2.0
    v_tau_small_pos = compute_cs_ret_from_logit(logit_pos, 0.5)
    v_tau_large_pos = compute_cs_ret_from_logit(logit_pos, 2.0)
    assert v_tau_small_pos > v_tau_large_pos
    v_tau_small_neg = compute_cs_ret_from_logit(logit_neg, 0.5)
    v_tau_large_neg = compute_cs_ret_from_logit(logit_neg, 2.0)
    assert v_tau_small_neg < v_tau_large_neg


def test_decide_route_nan_and_equal_thresholds():
    d, reason = decide_route(float("nan"), 0.2, 0.8)
    assert d == "UNCERTAIN"
    assert reason == "cs_ret_invalid"
    d2, reason2 = decide_route(0.5, 0.5, 0.5)
    assert d2 == "ACCEPT"
    assert reason2 == "cs_ret>=t_equal"
    d3, reason3 = decide_route(0.49, 0.5, 0.5)
    assert d3 == "REJECT"
    assert reason3 == "cs_ret<t_equal"


def test_routing_distribution_sums_to_one():
    dist_empty = routing_distribution([], 0.2, 0.8)
    assert math.isclose(sum(dist_empty.values()), 1.0)
    dist = routing_distribution([0.1, 0.5, 0.9], 0.2, 0.8)
    assert math.isclose(sum(dist.values()), 1.0)


def test_uniform_random_decision_is_roughly_uniform():
    rng = random.Random(0)
    n = 6000
    counts = {"ACCEPT": 0, "REJECT": 0, "UNCERTAIN": 0}
    for _ in range(n):
        counts[sample_uniform_random_decision(rng)] += 1
    for v in counts.values():
        assert abs(v / n - 1.0 / 3.0) < 0.08


def test_stratified_random_matches_distribution():
    rng = random.Random(1)
    dist = {"ACCEPT": 0.2, "REJECT": 0.3, "UNCERTAIN": 0.5}
    n = 8000
    counts = {"ACCEPT": 0, "REJECT": 0, "UNCERTAIN": 0}
    for _ in range(n):
        counts[sample_stratified_random_decision(dist, rng)] += 1
    for k, target in dist.items():
        assert abs(counts[k] / n - target) < 0.04


def test_stage1_outcomes_one_count_per_case():
    for decision in ["ACCEPT", "REJECT", "UNCERTAIN"]:
        for y in [0, 1]:
            out = stage1_outcomes(decision, y)
            assert sum(out.values()) == 1


def test_legacy_ok_fp_fn_uncertain_is_ok():
    ok, fp, fn = legacy_ok_fp_fn("UNCERTAIN", 1)
    assert (ok, fp, fn) == (1, 0, 0)
    ok, fp, fn = legacy_ok_fp_fn("ACCEPT", 1)
    assert (ok, fp, fn) == (1, 0, 0)
    ok, fp, fn = legacy_ok_fp_fn("REJECT", 1)
    assert (ok, fp, fn) == (0, 0, 1)
    ok, fp, fn = legacy_ok_fp_fn("ACCEPT", 0)
    assert (ok, fp, fn) == (0, 1, 0)
    ok, fp, fn = legacy_ok_fp_fn("REJECT", 0)
    assert (ok, fp, fn) == (1, 0, 0)


def test_sample_stratified_invalid_fallback():
    rng = random.Random(2)
    assert sample_stratified_random_decision({"ACCEPT": 0.0, "REJECT": 0.0, "UNCERTAIN": 0.0}, rng) == "UNCERTAIN"
    assert sample_stratified_random_decision({"ACCEPT": float("nan")}, rng) == "UNCERTAIN"
