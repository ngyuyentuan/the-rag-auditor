import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def verdict_to_binary(verdict):
    if verdict == "SUPPORTS":
        return 1
    if verdict == "REFUTES":
        return 0
    return None


def compute_metrics(rows):
    total = 0
    gold_verdict_count = 0
    gold_evidence_count = 0
    verdict_correct = 0
    evidence_hit = 0
    verdict_total = 0
    evidence_total = 0
    end_to_end_correct = 0
    fp = 0
    fn = 0
    timing = []
    for row in rows:
        total += 1
        t = (row.get("timing_ms") or {}).get("total_ms")
        if t is not None:
            timing.append(float(t))
        gold = row.get("gold") or {}
        pred = row.get("pred") or {}
        gold_verdict = gold.get("gold_verdict")
        gold_doc_ids = gold.get("gold_evidence_doc_ids") or []
        pred_verdict = pred.get("pred_verdict")
        pred_doc_id = pred.get("pred_doc_id")
        has_gold_verdict = gold_verdict is not None
        has_gold_evidence = len(gold_doc_ids) > 0
        if has_gold_verdict:
            gold_verdict_count += 1
        if has_gold_evidence:
            gold_evidence_count += 1
        if has_gold_verdict and pred_verdict is not None:
            verdict_total += 1
            if pred_verdict == gold_verdict:
                verdict_correct += 1
        if has_gold_evidence and pred_doc_id is not None:
            evidence_total += 1
            if pred_doc_id in set(gold_doc_ids):
                evidence_hit += 1
        gold_bin = verdict_to_binary(gold_verdict) if has_gold_verdict else None
        pred_bin = verdict_to_binary(pred_verdict) if pred_verdict is not None else None
        if gold_bin is not None and pred_bin is not None:
            if pred_bin == gold_bin:
                pass
            elif pred_bin == 1:
                fp += 1
            else:
                fn += 1
        if has_gold_verdict and pred_verdict is not None:
            if pred_verdict == gold_verdict:
                if has_gold_evidence:
                    if pred_doc_id is not None and pred_doc_id in set(gold_doc_ids):
                        end_to_end_correct += 1
                else:
                    end_to_end_correct += 1
    verdict_accuracy = verdict_correct / verdict_total if verdict_total > 0 else 0.0
    evidence_hit_rate = evidence_hit / evidence_total if evidence_total > 0 else 0.0
    coverage_gold_verdict = gold_verdict_count / total if total > 0 else 0.0
    coverage_gold_evidence = gold_evidence_count / total if total > 0 else 0.0
    end_to_end_correct_rate = end_to_end_correct / total if total > 0 else 0.0
    mean_ms = statistics.mean(timing) if timing else 0.0
    if len(timing) >= 100:
        q = statistics.quantiles(timing, n=100)
        p95 = q[94]
        p99 = q[98]
    else:
        p95 = max(timing) if timing else 0.0
        p99 = max(timing) if timing else 0.0
    return {
        "total": total,
        "verdict_accuracy": verdict_accuracy,
        "evidence_hit": evidence_hit_rate,
        "coverage_gold_verdict": coverage_gold_verdict,
        "coverage_gold_evidence": coverage_gold_evidence,
        "end_to_end_correct_rate": end_to_end_correct_rate,
        "fp": fp,
        "fn": fn,
        "mean_ms": mean_ms,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max(timing) if timing else 0.0,
    }


def format_float(v, digits=4):
    return f"{v:.{digits}f}"


def build_table(metrics):
    headers = [
        "verdict_accuracy",
        "evidence_hit",
        "coverage_gold_verdict",
        "coverage_gold_evidence",
        "end_to_end_correct_rate",
        "fp",
        "fn",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    ]
    row = [
        format_float(metrics["verdict_accuracy"]),
        format_float(metrics["evidence_hit"]),
        format_float(metrics["coverage_gold_verdict"]),
        format_float(metrics["coverage_gold_evidence"]),
        format_float(metrics["end_to_end_correct_rate"]),
        str(metrics["fp"]),
        str(metrics["fn"]),
        format_float(metrics["mean_ms"], 2),
        format_float(metrics["p95_ms"], 2),
        format_float(metrics["p99_ms"], 2),
        format_float(metrics["max_ms"], 2),
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def stratify(rows):
    by_route = {"ACCEPT": [], "REJECT": [], "UNCERTAIN": []}
    by_stage2 = {"stage2_ran": [], "stage2_not_ran": []}
    for row in rows:
        decision = (row.get("stage1") or {}).get("route_decision")
        if decision in by_route:
            by_route[decision].append(row)
        stage2_ran = (row.get("stage2") or {}).get("ran")
        if stage2_ran:
            by_stage2["stage2_ran"].append(row)
        else:
            by_stage2["stage2_not_ran"].append(row)
    return by_route, by_stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = list(iter_jsonl(args.in_jsonl))
    overall = compute_metrics(rows)
    by_route, by_stage2 = stratify(rows)

    strat = {
        "route_decision": {k: compute_metrics(v) for k, v in by_route.items()},
        "stage2_ran": {k: compute_metrics(v) for k, v in by_stage2.items()},
    }

    out_json = {
        "input": args.in_jsonl,
        "generated": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "stratified": strat,
    }

    Path(os.path.dirname(args.out_md) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Day12 Stage2 End-to-End Evaluation")
    lines.append("")
    lines.append(f"Generated: `{out_json['generated']}`")
    lines.append("")
    lines.append(f"Input: `{args.in_jsonl}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(build_table(overall))
    lines.append("")
    lines.append("## Stratified by route_decision")
    lines.append("")
    for key, metrics in strat["route_decision"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(build_table(metrics))
        lines.append("")
    lines.append("## Stratified by stage2 ran")
    lines.append("")
    for key, metrics in strat["stage2_ran"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(build_table(metrics))
        lines.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("wrote:", args.out_md)
    print("wrote:", args.out_json)


if __name__ == "__main__":
    main()
