import argparse, json, os, time, hashlib
from typing import Any, Iterable, List, Sequence, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.utils.normalize import normalize_text


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def extract_evidence_sentences(evidence_field: Any) -> List[str]:
    """
    Support BOTH formats:
    A) Our split format:
       evidence = [ ["Page", "Line", "Sentence"], ... ]
    B) Classic nested FEVER:
       evidence = [ [ ["Page","Line","Sentence"], ... ], [ ... ] ]
    """
    out: List[str] = []
    if not isinstance(evidence_field, list) or not evidence_field:
        return out

    # Case A: list of triples directly (your data)
    if (
        isinstance(evidence_field[0], list)
        and len(evidence_field[0]) >= 3
        and isinstance(evidence_field[0][2], str)
    ):
        for item in evidence_field:
            if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                out.append(item[2])
        return out

    # Case B: nested
    for group in evidence_field:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                out.append(item[2])
    return out




def stable_text_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/fever_splits/train.jsonl")
    ap.add_argument("--collection", default="fever_corpus")
    ap.add_argument("--qdrant_path", default="artifacts/qdrant_db")
    ap.add_argument("--manifest_path", default="artifacts/index_manifest.json")
    ap.add_argument("--emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--emb_revision", default=None, help="Optional HF revision/sha for reproducibility")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_sentences", type=int, default=50000, help="limit for CPU speed (student-friendly)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.manifest_path), exist_ok=True)
    os.makedirs(args.qdrant_path, exist_ok=True)

    t0 = time.time()

    # 1) Build corpus (dedup by normalized)
    seen = set()
    corpus: List[str] = []

    for ex in tqdm(iter_jsonl(args.train_jsonl), desc="[build_index] read train"):
        ev_sents = extract_evidence_sentences(ex.get("evidence"))
        for s in ev_sents:
            if not isinstance(s, str):
                continue
            ns = normalize_text(s)
            if not ns:
                continue
            if ns in seen:
                continue
            seen.add(ns)
            corpus.append(s)

            if args.max_sentences and len(corpus) >= args.max_sentences:
                break
        if args.max_sentences and len(corpus) >= args.max_sentences:
            break

    print(f"[build_index] corpus sentences (deduped): {len(corpus)}")

    # 2) Qdrant (local)
    client = QdrantClient(path=args.qdrant_path)

    existing = []
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print("[build_index] WARN: get_collections failed:", repr(e))

    if args.collection not in existing:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[build_index] created collection: {args.collection}")
    else:
        print(f"[build_index] collection exists: {args.collection}")

    # 3) Embed + upsert
    model = SentenceTransformer(args.emb_model, revision=args.emb_revision)

    total_upserted = 0
    for i in tqdm(range(0, len(corpus), args.batch_size), desc="[build_index] embed+upsert"):
        batch = corpus[i : i + args.batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)

        points = []
        for s, v in zip(batch, vecs):
            sid = stable_text_id(normalize_text(s))
            points.append(PointStruct(id=sid, vector=v.tolist(), payload={"text": s}))

        client.upsert(collection_name=args.collection, points=points)
        total_upserted += len(points)

    elapsed = time.time() - t0
    count = client.count(collection_name=args.collection, exact=True).count
    print(f"[build_index] upserted={total_upserted} elapsed={elapsed:.1f}s qdrant_count={count}")

    manifest = {
        "created_ts": time.time(),
        "train_jsonl": args.train_jsonl,
        "collection": args.collection,
        "qdrant_path": args.qdrant_path,
        "emb_model": args.emb_model,
        "emb_revision": args.emb_revision,
        "max_sentences": args.max_sentences,
        "batch_size": args.batch_size,
        "dedup_norm": True,
        "corpus_size": len(corpus),
        "upserted": total_upserted,
        "qdrant_count": count,
        "elapsed_sec": elapsed,
    }
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[build_index] wrote manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()
