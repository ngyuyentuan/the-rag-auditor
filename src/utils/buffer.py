import random
from typing import Literal

import numpy as np

Decision = Literal["ACCEPT", "REJECT", "UNCERTAIN"]


def stable_sigmoid(x: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    expx = np.exp(arr[~pos])
    out[~pos] = expx / (1.0 + expx)
    if out.shape == ():
        return float(out)
    return out


def compute_cs_ret_from_logit(logit: float, tau: float) -> float:
    tau_val = float(tau)
    if not np.isfinite(tau_val) or tau_val <= 0:
        tau_val = 1.0
    val = stable_sigmoid(np.asarray(logit, dtype=np.float64) / tau_val)
    if isinstance(val, np.ndarray):
        if val.shape == ():
            val = float(val)
        else:
            val = float(val.reshape(-1)[0])
    val = float(val)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def decide_route(cs_ret: float, t_lower: float, t_upper: float) -> tuple[Decision, str]:
    if not np.isfinite(cs_ret):
        return "UNCERTAIN", "cs_ret_invalid"
    tl = float(t_lower)
    tu = float(t_upper)
    if not np.isfinite(tl) or not np.isfinite(tu):
        return "UNCERTAIN", "threshold_invalid"
    if abs(tu - tl) <= 1e-12:
        if cs_ret >= tu:
            return "ACCEPT", "cs_ret>=t_equal"
        return "REJECT", "cs_ret<t_equal"
    if cs_ret >= tu:
        return "ACCEPT", "cs_ret>=t_upper"
    if cs_ret < tl:
        return "REJECT", "cs_ret<t_lower"
    return "UNCERTAIN", "t_lower<=cs_ret<t_upper"


def routing_distribution(scores: list[float] | np.ndarray, t_lower: float, t_upper: float) -> dict[str, float]:
    counts = {"ACCEPT": 0, "REJECT": 0, "UNCERTAIN": 0}
    if scores is None or len(scores) == 0:
        return {"ACCEPT": 0.0, "REJECT": 0.0, "UNCERTAIN": 1.0}
    for s in scores:
        decision, _ = decide_route(float(s), t_lower, t_upper)
        counts[decision] += 1
    total = counts["ACCEPT"] + counts["REJECT"] + counts["UNCERTAIN"]
    if total <= 0:
        return {"ACCEPT": 0.0, "REJECT": 0.0, "UNCERTAIN": 1.0}
    return {
        "ACCEPT": counts["ACCEPT"] / total,
        "REJECT": counts["REJECT"] / total,
        "UNCERTAIN": counts["UNCERTAIN"] / total,
    }


def sample_uniform_random_decision(rng: random.Random) -> Decision:
    r = rng.random()
    if r < 1.0 / 3.0:
        return "ACCEPT"
    if r < 2.0 / 3.0:
        return "REJECT"
    return "UNCERTAIN"


def sample_stratified_random_decision(dist: dict[str, float], rng: random.Random) -> Decision:
    p_accept = float(dist.get("ACCEPT", 0.0))
    p_reject = float(dist.get("REJECT", 0.0))
    p_uncertain = float(dist.get("UNCERTAIN", 0.0))
    if any(not np.isfinite(v) or v < 0 for v in [p_accept, p_reject, p_uncertain]):
        return "UNCERTAIN"
    total = p_accept + p_reject + p_uncertain
    if total <= 0:
        return "UNCERTAIN"
    p_accept /= total
    p_reject /= total
    r = rng.random()
    if r < p_accept:
        return "ACCEPT"
    if r < p_accept + p_reject:
        return "REJECT"
    return "UNCERTAIN"


def stage1_outcomes(decision: Decision, y: int) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    if y not in (0, 1):
        return out
    if decision == "UNCERTAIN":
        if y == 1:
            out["defer_pos"] = 1
        else:
            out["defer_neg"] = 1
        return out
    if decision == "ACCEPT":
        if y == 1:
            out["tp"] = 1
        else:
            out["fp"] = 1
        return out
    if decision == "REJECT":
        if y == 0:
            out["tn"] = 1
        else:
            out["fn"] = 1
        return out
    return out


def legacy_ok_fp_fn(decision: Decision, y: int) -> tuple[int, int, int]:
    ok = 0
    fp = 0
    fn = 0
    if y not in (0, 1):
        return ok, fp, fn
    if decision == "UNCERTAIN":
        ok += 1
    elif y == 1 and decision == "REJECT":
        fn += 1
    elif y == 0 and decision == "ACCEPT":
        fp += 1
    else:
        ok += 1
    return ok, fp, fn
