import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def quantiles(values, qs):
    if not values:
        return {q: None for q in qs}
    vals = sorted(values)
    out = {}
    n = len(vals)
    for q in qs:
        k = int(round((n - 1) * q))
        out[q] = vals[k]
    return out


def summarize(path):
    counts = {"ACCEPT": 0, "REJECT": 0, "UNCERTAIN": 0}
    p_vals = []
    missing_features = 0
    total = 0
    stage2_ran = 0
    stage2_route = 0
    capped = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    route_count = 0
    for row in iter_jsonl(path):
        total += 1
        stage1 = row.get("stage1") or {}
        decision = stage1.get("router_decision") or stage1.get("route_decision")
        if decision in counts:
            counts[decision] += 1
        if decision == "UNCERTAIN":
            route_count += 1
        router = stage1.get("router") or {}
        p_hat = router.get("p_hat")
        if p_hat is not None:
            p_vals.append(float(p_hat))
        features_present = router.get("features_present") or {}
        if features_present:
            if not all(bool(v) for v in features_present.values()):
                missing_features += 1
        stage2 = row.get("stage2") or {}
        if stage2.get("route_requested") is True:
            stage2_route += 1
        if stage2.get("capped") is True:
            capped += 1
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms") or {}
        if float(timing.get("rerank_ms", 0.0)) > 0:
            rerank_gt0 += 1
        if float(timing.get("nli_ms", 0.0)) > 0:
            nli_gt0 += 1
    qs = quantiles(p_vals, [0.1, 0.5, 0.9])
    return {
        "total": total,
        "counts": counts,
        "p_hat_q10": qs[0.1],
        "p_hat_q50": qs[0.5],
        "p_hat_q90": qs[0.9],
        "missing_features": missing_features,
        "stage2_ran": stage2_ran,
        "stage2_route": stage2_route,
        "capped": capped,
        "rerank_gt0": rerank_gt0,
        "nli_gt0": nli_gt0,
        "route_count": route_count,
    }


def analyze_rejects(path, limit=10):
    p_vals = []
    thresholds = []
    samples = []
    for row in iter_jsonl(path):
        stage1 = row.get("stage1") or {}
        decision = stage1.get("router_decision") or stage1.get("route_decision")
        if decision != "REJECT":
            continue
        router = stage1.get("router") or {}
        p_hat = router.get("p_hat")
        t_accept = router.get("t_accept")
        t_reject = router.get("t_reject")
        if p_hat is not None:
            p_vals.append(float(p_hat))
        thresholds.append((t_accept, t_reject))
        y = (row.get("ground_truth") or {}).get("y")
        if y == 1 and len(samples) < limit:
            debug = stage1.get("router_debug") or {}
            samples.append({
                "qid": (row.get("metadata") or {}).get("qid"),
                "p_hat": p_hat,
                "t_accept": t_accept,
                "t_reject": t_reject,
                "cs_ret": stage1.get("cs_ret"),
                "features": debug.get("features"),
            })
    qs = quantiles(p_vals, [0.1, 0.5, 0.9])
    return {
        "p_hat_q10": qs[0.1],
        "p_hat_q50": qs[0.5],
        "p_hat_q90": qs[0.9],
        "thresholds": thresholds[:5],
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--router_jsonl", required=True)
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    args = ap.parse_args()

    base = summarize(args.baseline_jsonl)
    router = summarize(args.router_jsonl)

    print("baseline_total", base["total"])
    print("baseline_counts", base["counts"])
    print("baseline_p_hat_q10", base["p_hat_q10"])
    print("baseline_p_hat_q50", base["p_hat_q50"])
    print("baseline_p_hat_q90", base["p_hat_q90"])
    print("baseline_route_count", base["route_count"])
    print("baseline_stage2_ran", base["stage2_ran"])
    print("baseline_rerank_gt0", base["rerank_gt0"])
    print("baseline_nli_gt0", base["nli_gt0"])
    print("router_total", router["total"])
    print("router_counts", router["counts"])
    print("router_p_hat_q10", router["p_hat_q10"])
    print("router_p_hat_q50", router["p_hat_q50"])
    print("router_p_hat_q90", router["p_hat_q90"])
    print("router_route_count", router["route_count"])
    print("router_stage2_ran", router["stage2_ran"])
    print("router_stage2_route", router["stage2_route"])
    print("router_capped", router["capped"])
    print("router_rerank_gt0", router["rerank_gt0"])
    print("router_nli_gt0", router["nli_gt0"])
    if args.track == "fever":
        print("router_missing_features", router["missing_features"])
    reject_info = analyze_rejects(args.router_jsonl)
    print("router_reject_p_hat_q10", reject_info["p_hat_q10"])
    print("router_reject_p_hat_q50", reject_info["p_hat_q50"])
    print("router_reject_p_hat_q90", reject_info["p_hat_q90"])
    print("router_reject_thresholds_sample", reject_info["thresholds"])
    print("router_false_reject_samples", reject_info["samples"])


if __name__ == "__main__":
    main()
