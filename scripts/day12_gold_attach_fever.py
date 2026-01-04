import argparse
import json
from pathlib import Path


def load_fever_gold(dev_jsonl):
    gold = {}
    with dev_jsonl.open("r", encoding="utf-8") as f:
        for idx, row in enumerate(f):
            row = row.strip()
            if not row:
                continue
            r = json.loads(row)
            qid = r.get("id", r.get("_id"))
            if qid is None:
                qid = r.get("qid", r.get("claim_id", r.get("i", idx)))
            qid = str(qid)
            label = r.get("label", r.get("verdict"))
            verdict = str(label) if isinstance(label, str) and label.strip() else None
            doc_ids = set()
            evidence = r.get("evidence")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    dev_path = data_root / "dev.jsonl"
    if not dev_path.exists():
        raise SystemExit(f"missing dev.jsonl under {data_root}")
    gold_map = load_fever_gold(dev_path)

    total = 0
    covered = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as f_in, open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            qid = str((row.get("metadata") or {}).get("qid"))
            entry = gold_map.get(qid)
            doc_ids = entry.get("doc_ids") if entry else []
            verdict = entry.get("verdict") if entry else None
            if doc_ids or verdict:
                covered += 1
            gold = row.get("gold", {}) or {}
            gold["gold_verdict"] = verdict
            gold["gold_evidence_doc_ids"] = doc_ids
            gold["gold_has_evidence"] = bool(doc_ids)
            row["gold"] = gold
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("rows", total, "gold_with_verdict_or_evidence", covered)


if __name__ == "__main__":
    main()
