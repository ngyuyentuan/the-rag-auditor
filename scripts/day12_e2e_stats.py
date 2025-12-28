import argparse
import json
import os
import sys
from datetime import datetime, timezone

P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [P]

import numpy as np
from src.utils.buffer import stage1_outcomes


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def summarize(rows):
    if not rows:
        return {}
    stage2_ran = [1 if r.get("stage2", {}).get("ran") else 0 for r in rows]
    total_ms = [float(r["timing_ms"]["total_ms"]) for r in rows]
    rerank_ms = [float(r["timing_ms"]["rerank_ms"]) for r in rows]
    nli_ms = [float(r["timing_ms"]["nli_ms"]) for r in rows]
    stage2_ms = [a + b for a, b in zip(rerank_ms, nli_ms)]
    decisions = [r.get("stage1", {}).get("route_decision") for r in rows]
    y_vals = [r.get("ground_truth", {}).get("y") for r in rows]

    labels = []
    for r in rows:
        if not r.get("stage2", {}).get("ran"):
            continue
        nli = (r.get("stage2") or {}).get("nli") or {}
        label = nli.get("top_label")
        if label:
            labels.append(label)

    if labels:
        vals, counts = np.unique(labels, return_counts=True)
        label_counts = {str(v): int(c) for v, c in zip(vals, counts)}
    else:
        label_counts = {}
    if labels:
        total_labels = sum(label_counts.values())
        label_dist = {k: v / total_labels for k, v in label_counts.items()}
    else:
        label_dist = {}

    decision_counts = {k: int(sum(1 for d in decisions if d == k)) for k in ["ACCEPT", "REJECT", "UNCERTAIN"]}
    n = len(rows)
    routing_dist = {
        "accept_rate": decision_counts["ACCEPT"] / n,
        "reject_rate": decision_counts["REJECT"] / n,
        "uncertain_rate": decision_counts["UNCERTAIN"] / n,
    }

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    valid = 0
    for d, y in zip(decisions, y_vals):
        if y not in (0, 1):
            continue
        valid += 1
        out = stage1_outcomes(d, int(y))
        for k in counts:
            counts[k] += int(out[k])

    ok = counts["tp"] + counts["tn"]
    fp_accept = counts["fp"]
    fn_reject = counts["fn"]
    uncertain = counts["defer_pos"] + counts["defer_neg"]

    correctness = {}
    if valid > 0:
        correctness = {
            "ok_rate": ok / valid,
            "fp_accept_rate": fp_accept / valid,
            "fn_reject_rate": fn_reject / valid,
            "uncertain_rate": uncertain / valid,
        }

    return {
        "n": len(rows),
        "stage2_call_rate": float(np.mean(stage2_ran)),
        "routing_distribution": routing_dist,
        "routing_correctness": correctness,
        "latency_total_ms": {
            "mean": float(np.mean(total_ms)),
            "p95": float(np.percentile(total_ms, 95)),
            "p99": float(np.percentile(total_ms, 99)),
            "max": float(np.max(total_ms)),
            "std": float(np.std(total_ms)),
        },
        "latency_stage2_ms": {
            "mean": float(np.mean(stage2_ms)),
            "p95": float(np.percentile(stage2_ms, 95)),
            "p99": float(np.percentile(stage2_ms, 99)),
            "max": float(np.max(stage2_ms)),
            "std": float(np.std(stage2_ms)),
        },
        "nli_label_dist": label_dist,
    }


def build_md(summary):
    lines = []
    lines.append("# Day12 E2E Stats")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("- Known limitations: train-only sanity check; cs_ret uses sigmoid(logit / tau) where tau<1 sharpens and tau>1 softens; FEVER is expected to be weak with current thresholds.")
    lines.append("")
    for track, info in summary["tracks"].items():
        lines.append(f"## Track: {track}")
        lines.append("")
        lines.append("| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms | std_ms |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in info.get("baselines", []):
            s = b.get("summary") or {}
            if not s:
                continue
            rd = s.get("routing_distribution", {})
            rc = s.get("routing_correctness", {})
            lt = s.get("latency_total_ms", {})
            lines.append(
                "| {name} | {n} | {acc:.3f} | {rej:.3f} | {unc:.3f} | {s2:.3f} | {ok:.3f} | {fp:.3f} | {fn:.3f} | {mean:.2f} | {p95:.2f} | {p99:.2f} | {max:.2f} | {std:.2f} |".format(
                    name=b.get("name", ""),
                    n=s.get("n", 0),
                    acc=rd.get("accept_rate", 0.0),
                    rej=rd.get("reject_rate", 0.0),
                    unc=rd.get("uncertain_rate", 0.0),
                    s2=s.get("stage2_call_rate", 0.0),
                    ok=rc.get("ok_rate", 0.0),
                    fp=rc.get("fp_accept_rate", 0.0),
                    fn=rc.get("fn_reject_rate", 0.0),
                    mean=lt.get("mean", 0.0),
                    p95=lt.get("p95", 0.0),
                    p99=lt.get("p99", 0.0),
                    max=lt.get("max", 0.0),
                    std=lt.get("std", 0.0),
                )
            )
        lines.append("")
        lines.append("NLI label distribution (stage2 ran):")
        for b in info.get("baselines", []):
            s = b.get("summary") or {}
            lines.append(f"- {b.get('name', '')}:")
            dist = s.get("nli_label_dist") or {}
            if dist:
                for k, v in sorted(dist.items(), key=lambda x: x[0]):
                    lines.append(f"  {k}: {v:.3f}")
            else:
                lines.append("  n/a")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    summary = {"generated": datetime.now(timezone.utc).isoformat(), "tracks": {}}
    for track in args.tracks:
        baselines = []
        base_files = {
            "calibrated": f"day12_{track}_500_e2e.jsonl",
            "always_stage2": f"day12_{track}_500_e2e_always.jsonl",
            "random": f"day12_{track}_500_e2e_random.jsonl",
        }
        for name, fname in base_files.items():
            path = os.path.join(args.runs_dir, fname)
            rows = list(iter_jsonl(path)) if os.path.exists(path) else []
            baselines.append({"name": name, "path": path, "summary": summarize(rows)})
        summary["tracks"][track] = {"baselines": baselines}

    md = build_md(summary)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[ok] wrote:", args.out_md)
    print("[ok] wrote:", args.out_json)


if __name__ == "__main__":
    main()
