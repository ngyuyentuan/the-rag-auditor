import argparse
import json
from pathlib import Path


def load_qrels(qrels_path):
    gold = {}
    with qrels_path.open("r", encoding="utf-8") as f:
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
    return {k: sorted(v) for k, v in gold.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    qrels_candidates = [
        data_root / "qrels" / "test.tsv",
        data_root / "qrels" / "dev.tsv",
        data_root / "qrels" / "train.tsv",
        data_root / "qrels" / "qrels.tsv",
    ]
    qrels_path = next((p for p in qrels_candidates if p.exists()), None)
    gold_map = load_qrels(qrels_path) if qrels_path else {}

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
            doc_ids = gold_map.get(qid, [])
            if doc_ids:
                covered += 1
            gold = row.get("gold", {}) or {}
            gold["gold_verdict"] = gold.get("gold_verdict")
            gold["gold_evidence_doc_ids"] = doc_ids
            gold["gold_has_evidence"] = bool(doc_ids)
            row["gold"] = gold
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("rows", total, "gold_with_evidence", covered)


if __name__ == "__main__":
    main()
