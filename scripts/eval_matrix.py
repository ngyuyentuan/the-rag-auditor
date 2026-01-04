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


def summarize(rows):
    n = 0
    accept = 0
    reject = 0
    uncertain = 0
    fp = 0
    fn = 0
    stage2_ran = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    total_ms = []
    gold_total = 0
    gold_correct = 0
    ev_total = 0
    hit1 = 0
    hit5 = 0

    for row in rows:
        n += 1
        stage1 = row.get("stage1", {}) or {}
        decision = stage1.get("route_decision")
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        y = (row.get("ground_truth") or {}).get("y")
        if decision == "ACCEPT" and y == 0:
            fp += 1
        if decision == "REJECT" and y == 1:
            fn += 1
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        if float(timing.get("rerank_ms", 0.0)) > 0:
            rerank_gt0 += 1
        if float(timing.get("nli_ms", 0.0)) > 0:
            nli_gt0 += 1
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

    n = n or 1
    return {
        "n": n,
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "stage2_rate": stage2_ran / n,
        "fp_accept_rate": fp / n,
        "fn_reject_rate": fn / n,
        "ok_rate_stage1": 1.0 - (fp / n) - (fn / n),
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95) or 0.0,
        "rerank_ms_gt0": rerank_gt0,
        "nli_ms_gt0": nli_gt0,
        "gold_total": gold_total,
        "gold_acc": (gold_correct / gold_total) if gold_total else None,
        "ev_total": ev_total,
        "hit1": (hit1 / ev_total) if ev_total else None,
        "hit5": (hit5 / ev_total) if ev_total else None,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/matrix")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = set()
    for path in runs_dir.glob("*.jsonl"):
        name = path.name
        if "_" in name:
            tracks.add(name.split("_", 1)[0])

    for track in sorted(tracks):
        rows_by_file = {}
        for path in runs_dir.glob(f"{track}_*.jsonl"):
            rows = list(iter_jsonl(path))
            if rows:
                rows_by_file[path.name] = summarize(rows)
        md_path = out_dir / f"day12_matrix_{track}.md"
        json_path = out_dir / f"day12_matrix_{track}.json"
        lines = []
        lines.append(f"# Matrix Stats ({track})")
        lines.append("")
        if not rows_by_file:
            lines.append("- no runs found")
            md_path.write_text("\n".join(lines), encoding="utf-8")
            json_path.write_text(json.dumps({}), encoding="utf-8")
            continue
        lines.append("| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, s in sorted(rows_by_file.items()):
            lines.append("| {f} | {a} | {r} | {u} | {s2} | {fp} | {fn} | {ok} | {mean} | {p95} | {rg} | {ng} |".format(
                f=name,
                a=format_float(s["accept_rate"]),
                r=format_float(s["reject_rate"]),
                u=format_float(s["uncertain_rate"]),
                s2=format_float(s["stage2_rate"]),
                fp=format_float(s["fp_accept_rate"]),
                fn=format_float(s["fn_reject_rate"]),
                ok=format_float(s["ok_rate_stage1"]),
                mean=f"{s['mean_ms']:.2f}",
                p95=f"{s['p95_ms']:.2f}",
                rg=s["rerank_ms_gt0"],
                ng=s["nli_ms_gt0"],
            ))
        lines.append("")
        lines.append("Gold evaluation")
        lines.append("")
        for name, s in sorted(rows_by_file.items()):
            if s["gold_total"]:
                lines.append(f"- {name}: verdict_accuracy={format_float(s['gold_acc'])} (n={s['gold_total']})")
            else:
                lines.append(f"- {name}: gold unavailable")
            if s["ev_total"]:
                lines.append(f"  evidence_hit@1={format_float(s['hit1'])} evidence_hit@5={format_float(s['hit5'])} (n={s['ev_total']})")
        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        json_path.write_text(json.dumps(rows_by_file, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
