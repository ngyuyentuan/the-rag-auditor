import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.policy import finalize_decision


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(rows, entail_min, contradict_min, neutral_max):
    totals = {
        "n": 0,
        "final_accept": 0,
        "final_reject": 0,
        "final_uncertain": 0,
        "stage2_ran": 0,
        "unsafe_accept": 0,
        "total_ms": [],
    }
    for row in rows:
        totals["n"] += 1
        stage2 = row.get("stage2", {}) or {}
        stage1_decision = row.get("stage1", {}).get("route_decision", "UNCERTAIN")
        final, _ = finalize_decision(
            stage1_decision,
            stage2,
            entail_min=entail_min,
            contradict_min=contradict_min,
            neutral_max=neutral_max,
        )
        if final == "ACCEPT":
            totals["final_accept"] += 1
        elif final == "REJECT":
            totals["final_reject"] += 1
        else:
            totals["final_uncertain"] += 1
        if stage2.get("ran", False):
            totals["stage2_ran"] += 1
        nli_probs = (stage2.get("nli") or {}).get("label_probs") or {}
        p_entail = float(nli_probs.get("ENTAILMENT", 0.0)) if isinstance(nli_probs, dict) else 0.0
        if final == "ACCEPT" and p_entail < entail_min:
            totals["unsafe_accept"] += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms = timing.get("total_ms")
        if total_ms is not None:
            totals["total_ms"].append(float(total_ms))
    n = totals["n"] or 1
    total_ms = totals["total_ms"]
    latency = {
        "mean_ms": float(np.mean(total_ms)) if total_ms else None,
        "p95_ms": float(np.percentile(total_ms, 95)) if total_ms else None,
        "p99_ms": float(np.percentile(total_ms, 99)) if total_ms else None,
        "max_ms": float(np.max(total_ms)) if total_ms else None,
    }
    return {
        "n": totals["n"],
        "final_accept_rate": totals["final_accept"] / n,
        "final_reject_rate": totals["final_reject"] / n,
        "final_uncertain_rate": totals["final_uncertain"] / n,
        "stage2_call_rate": totals["stage2_ran"] / n,
        "verdict_rate": (totals["final_accept"] + totals["final_reject"]) / n,
        "unsafe_accept_proxy_rate": totals["unsafe_accept"] / n,
        "latency": latency,
    }


def format_float(x):
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def format_ms(x):
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--out_md", default="reports/day12_policy_report.md")
    ap.add_argument("--entail_min", type=float, default=0.90)
    ap.add_argument("--contradict_min", type=float, default=0.90)
    ap.add_argument("--neutral_max", type=float, default=0.80)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    baselines = [
        ("calibrated", "day12_{track}_500_e2e.jsonl"),
        ("always_stage2", "day12_{track}_500_e2e_always.jsonl"),
        ("random", "day12_{track}_500_e2e_random.jsonl"),
    ]

    lines = []
    lines.append("# Day12 Policy Report")
    lines.append("")
    lines.append(f"- entail_min: `{args.entail_min}`")
    lines.append(f"- contradict_min: `{args.contradict_min}`")
    lines.append(f"- neutral_max: `{args.neutral_max}`")
    lines.append("")

    for track in args.tracks:
        lines.append(f"## Track: {track}")
        lines.append("")
        for label, pattern in baselines:
            path = runs_dir / pattern.format(track=track)
            if not path.exists():
                lines.append(f"- {label}: missing {path}")
                lines.append("")
                continue
            rows = list(read_jsonl(path))
            metrics = compute_metrics(
                rows,
                entail_min=args.entail_min,
                contradict_min=args.contradict_min,
                neutral_max=args.neutral_max,
            )
            lines.append(f"### Baseline: {label}")
            lines.append("")
            lines.append(f"- path: `{path}`")
            lines.append(f"- n: `{metrics['n']}`")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("|---|---|")
            lines.append(f"| final_accept_rate | {format_float(metrics['final_accept_rate'])} |")
            lines.append(f"| final_reject_rate | {format_float(metrics['final_reject_rate'])} |")
            lines.append(f"| final_uncertain_rate | {format_float(metrics['final_uncertain_rate'])} |")
            lines.append(f"| stage2_call_rate | {format_float(metrics['stage2_call_rate'])} |")
            lines.append(f"| verdict_rate | {format_float(metrics['verdict_rate'])} |")
            lines.append(f"| unsafe_accept_proxy_rate | {format_float(metrics['unsafe_accept_proxy_rate'])} |")
            lines.append(f"| latency_mean_ms | {format_ms(metrics['latency']['mean_ms'])} |")
            lines.append(f"| latency_p95_ms | {format_ms(metrics['latency']['p95_ms'])} |")
            lines.append(f"| latency_p99_ms | {format_ms(metrics['latency']['p99_ms'])} |")
            lines.append(f"| latency_max_ms | {format_ms(metrics['latency']['max_ms'])} |")
            lines.append("")
        lines.append("Policy interpretation")
        lines.append("")
        lines.append("UNCERTAIN is the default safety fallback. Higher thresholds reduce unsafe accepts but increase deferrals.")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
