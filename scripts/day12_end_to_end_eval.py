import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def map_nli_to_verdict(label):
    if label == "ENTAILMENT":
        return "SUPPORTS"
    if label == "CONTRADICTION":
        return "REFUTES"
    if label == "NEUTRAL":
        return "NEI"
    return None


def load_scifact_qrels(scifact_dir: Path):
    qrels_dir = scifact_dir / "qrels"
    for name in ["test.tsv", "dev.tsv", "train.tsv", "qrels.tsv"]:
        path = qrels_dir / name
        if path.exists():
            break
    else:
        return {}
    gold = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                rel_val = float(rel)
            except Exception:
                rel_val = 0.0
            if rel_val <= 0:
                continue
            gold.setdefault(str(qid), set()).add(str(doc_id))
    return {k: {"verdict": None, "doc_ids": sorted(v)} for k, v in gold.items()}


def load_fever_gold(dev_jsonl: Path):
    gold = {}
    if not dev_jsonl.exists():
        return gold
    for idx, row in enumerate(iter_jsonl(dev_jsonl)):
        qid = row.get("id", row.get("_id"))
        if qid is None:
            qid = row.get("qid", row.get("claim_id", row.get("i", idx)))
        qid = str(qid)
        label = row.get("label", row.get("verdict"))
        verdict = str(label) if isinstance(label, str) and label.strip() else None
        doc_ids = set()
        evidence = row.get("evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, list) and len(item) >= 3:
                    doc_id = item[2]
                    if doc_id is not None:
                        doc_ids.add(str(doc_id))
                elif isinstance(item, list):
                    for ev in item:
                        if isinstance(ev, list) and len(ev) >= 3:
                            doc_id = ev[2]
                            if doc_id is not None:
                                doc_ids.add(str(doc_id))
        gold[qid] = {"verdict": verdict, "doc_ids": sorted(doc_ids)}
    return gold


def compute_metrics(rows, gold_map):
    total = 0
    verdict_total = 0
    verdict_correct = 0
    evidence_total = 0
    evidence_hit = 0
    for row in rows:
        total += 1
        metadata = row.get("metadata", {})
        qid = metadata.get("qid")
        gold_entry = gold_map.get(str(qid)) if qid is not None else None
        gold_verdict = None
        gold_doc_ids = []
        if gold_entry:
            gold_verdict = gold_entry.get("verdict")
            gold_doc_ids = gold_entry.get("doc_ids", [])
        pred = row.get("pred", {})
        pred_verdict = pred.get("pred_verdict")
        pred_doc_id = pred.get("pred_doc_id")
        if pred_verdict is None:
            nli = (row.get("stage2") or {}).get("nli") or {}
            pred_verdict = map_nli_to_verdict(nli.get("top_label"))
        if pred_verdict is None:
            decision = (row.get("stage1") or {}).get("route_decision")
            if decision == "ACCEPT":
                pred_verdict = "SUPPORTS"
            elif decision == "REJECT":
                pred_verdict = "REFUTES"
        if gold_verdict is not None and pred_verdict is not None:
            verdict_total += 1
            if pred_verdict == gold_verdict:
                verdict_correct += 1
        if gold_doc_ids and pred_doc_id is not None:
            evidence_total += 1
            if str(pred_doc_id) in set(gold_doc_ids):
                evidence_hit += 1
    return {
        "n": total,
        "coverage_gold_verdict": verdict_total / total if total else 0.0,
        "coverage_gold_evidence": evidence_total / total if total else 0.0,
        "verdict_accuracy": verdict_correct / verdict_total if verdict_total else 0.0,
        "evidence_hit_rate": evidence_hit / evidence_total if evidence_total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scifact_dir", default="data/beir_scifact/scifact")
    ap.add_argument("--fever_dev_jsonl", default="data/fever_splits/dev.jsonl")
    ap.add_argument("--out_md", default="reports/day12_end_to_end_eval.md")
    ap.add_argument("--out_json", default="reports/day12_end_to_end_eval.json")
    args = ap.parse_args()

    scifact_gold = load_scifact_qrels(Path(args.scifact_dir))
    fever_gold = load_fever_gold(Path(args.fever_dev_jsonl))

    rows_out = []
    for run_path in args.runs:
        path = Path(run_path)
        rows = list(iter_jsonl(path))
        track = None
        if rows:
            track = (rows[0].get("metadata") or {}).get("track")
        gold_map = scifact_gold if track == "scifact" else fever_gold
        metrics = compute_metrics(rows, gold_map)
        rows_out.append({
            "path": str(path),
            "track": track,
            "metrics": metrics,
        })

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Day12 End-to-End Evaluation")
    lines.append("")
    lines.append("| track | run_path | n | coverage_gold_verdict | coverage_gold_evidence | verdict_accuracy | evidence_hit_rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows_out:
        m = row["metrics"]
        lines.append("| {track} | `{path}` | {n} | {cgv:.4f} | {cge:.4f} | {va:.4f} | {ehr:.4f} |".format(
            track=row["track"],
            path=row["path"],
            n=m["n"],
            cgv=m["coverage_gold_verdict"],
            cge=m["coverage_gold_evidence"],
            va=m["verdict_accuracy"],
            ehr=m["evidence_hit_rate"],
        ))
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
