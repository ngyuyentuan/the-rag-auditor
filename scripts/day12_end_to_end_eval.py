import argparse
import json
import math
from pathlib import Path


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[int(f)] * (c - k)
    d1 = vals[int(c)] * (k - f)
    return d0 + d1


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    n = 0
    stage2_ran = 0
    accept = reject = uncertain = 0
    gold_total = 0
    gold_correct = 0
    ev_total = 0
    hit1 = 0
    hit5 = 0
    total_ms = []

    for row in iter_jsonl(Path(args.in_jsonl)):
        n += 1
        stage1 = row.get("stage1", {}) or {}
        decision = stage1.get("route_decision")
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        gold = row.get("gold", {}) or {}
        pred = row.get("pred", {}) or {}
        gold_verdict = gold.get("gold_verdict")
        pred_verdict = pred.get("pred_verdict")
        if gold_verdict is not None:
            gold_total += 1
            if pred_verdict == gold_verdict:
                gold_correct += 1
        gold_docs = gold.get("gold_evidence_doc_ids") or []
        pred_docs = (stage2.get("rerank", {}) or {}).get("doc_ids") or []
        if gold_docs:
            ev_total += 1
            gold_set = set(map(str, gold_docs))
            if pred_docs:
                if str(pred_docs[0]) in gold_set:
                    hit1 += 1
                if any(str(d) in gold_set for d in pred_docs[:5]):
                    hit5 += 1

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Day12 End-to-End Eval")
    lines.append("")
    lines.append(f"- in_jsonl: `{args.in_jsonl}`")
    lines.append(f"- rows: `{n}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| stage2_rate | {format_float(stage2_ran / n if n else 0.0)} |")
    lines.append(f"| accept_rate | {format_float(accept / n if n else 0.0)} |")
    lines.append(f"| reject_rate | {format_float(reject / n if n else 0.0)} |")
    lines.append(f"| uncertain_rate | {format_float(uncertain / n if n else 0.0)} |")
    lines.append(f"| mean_ms | {format_float(sum(total_ms) / n if n else 0.0)} |")
    lines.append(f"| p95_ms | {format_float(percentile(total_ms, 95) or 0.0)} |")
    lines.append("")
    lines.append("Gold evaluation")
    lines.append("")
    if gold_total:
        lines.append(f"- coverage_gold_verdict: `{format_float(gold_total / n)}`")
        lines.append(f"- verdict_accuracy: `{format_float(gold_correct / gold_total)}`")
    else:
        lines.append("- coverage_gold_verdict: `0.0000`")
        lines.append("- verdict_accuracy: `n/a`")
    if ev_total:
        lines.append(f"- coverage_gold_evidence: `{format_float(ev_total / n)}`")
        lines.append(f"- evidence_hit_at_1: `{format_float(hit1 / ev_total)}`")
        lines.append(f"- evidence_hit_at_5: `{format_float(hit5 / ev_total)}`")
    else:
        lines.append("- coverage_gold_evidence: `0.0000`")
        lines.append("- evidence_hit_at_1: `n/a`")
        lines.append("- evidence_hit_at_5: `n/a`")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
