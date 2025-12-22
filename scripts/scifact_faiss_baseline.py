
import os, json, time, argparse, random, platform
from typing import Dict, List, Tuple

import numpy as np
import torch
import faiss
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer

# BEIR loader + SciFact download URL (official BEIR list)
from beir import util
from beir.datasets.data_loader import GenericDataLoader

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2norm(x: np.ndarray) -> np.ndarray:
    # normalize to unit length for cosine via dot product
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype(np.float32)


def build_faiss_index(doc_embs: np.ndarray) -> faiss.Index:
    d = doc_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(doc_embs)
    return index


def success_at_k(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int) -> float:
    # %queries that retrieve >=1 relevant doc in top-k
    hit = 0
    total = 0
    for qid, rels in qrels.items():
        if not rels:
            continue
        total += 1
        topk = rankings.get(qid, [])[:k]
        if any(docid in rels and rels[docid] > 0 for docid in topk):
            hit += 1
    return hit / total if total else 0.0


def recall_at_k(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int) -> float:
    # macro average recall: retrieved_relevant / total_relevant
    vals = []
    for qid, rels in qrels.items():
        rel_docs = {d for d, v in rels.items() if v > 0}
        if not rel_docs:
            continue
        topk = rankings.get(qid, [])[:k]
        got = len([d for d in topk if d in rel_docs])
        vals.append(got / len(rel_docs))
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/beir_scifact")
    ap.add_argument("--split", default="test")  # BEIR SciFact typically evaluated on "test" :contentReference[oaicite:1]{index=1}
    ap.add_argument("--emb_model", default="sentence-transformers/allenai-specter")  # science-friendly
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", nargs="+", type=int, default=[1,5,10,20,50])

    ap.add_argument("--index_path", default="artifacts/scifact.faiss.index")
    ap.add_argument("--meta_path", default="artifacts/scifact_faiss_meta.json")
    ap.add_argument("--report_path", default="reports/retrieval_scifact_baseline.md")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1) Download SciFact (once)
    dataset_dir = os.path.join(args.data_root, "scifact")
    if not os.path.exists(dataset_dir):
        os.makedirs(args.data_root, exist_ok=True)
        util.download_and_unzip(SCIFACT_URL, args.data_root)

    # 2) Load corpus/queries/qrels
    corpus, queries, qrels = GenericDataLoader(dataset_dir).load(split=args.split)

    # BEIR corpus format: {docid: {"title":..., "text":...}}
    doc_ids = list(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        title = corpus[did].get("title", "") or ""
        text = corpus[did].get("text", "") or ""
        doc_texts.append((title + "\n" + text).strip())

    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.emb_model, device=device)

    # 3) Embed corpus
    t0 = time.time()
    doc_embs = model.encode(
        doc_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    doc_embs = l2norm(doc_embs)

    # 4) Build FAISS
    index = build_faiss_index(doc_embs)
    faiss.write_index(index, args.index_path)

    # 5) Retrieve
    rankings: Dict[str, List[str]] = {}
    q_embs = model.encode(
        q_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    q_embs = l2norm(q_embs)

    max_k = max(args.ks)
    for i, qid in enumerate(tqdm(q_ids, desc="[day7-science] search")):
        q = q_embs[i:i+1]
        _, I = index.search(q, max_k)
        rankings[qid] = [doc_ids[j] for j in I[0] if 0 <= j < len(doc_ids)]

    # 6) Metrics
    ks = sorted(set(args.ks))
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "success_at_k": success_at_k(rankings, qrels, k),
            "recall_at_k": recall_at_k(rankings, qrels, k),
        })

    elapsed = time.time() - t0

    meta = {
        "created_ts": time.time(),
        "dataset": "BEIR/SciFact",
        "split": args.split,
        "scifact_url": SCIFACT_URL,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "emb_model": args.emb_model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_docs": len(doc_ids),
        "n_queries": len(q_ids),
        "ks": ks,
        "metrics": rows,
        "paths": {
            "index_path": args.index_path,
            "report_path": args.report_path,
            "meta_path": args.meta_path,
        },
        "elapsed_sec": elapsed,
    }

    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 7) Report MD
    lines = []
    lines.append("# Day7-SCIENCE — SciFact (BEIR) Bi-encoder + FAISS")
    lines.append("")
    lines.append("SciFact is a scientific claim verification dataset with a document corpus and claim queries (IR-style evaluation).")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Dataset: `{meta['dataset']}` split=`{meta['split']}`")
    lines.append(f"- Embedding model: `{meta['emb_model']}`")
    lines.append(f"- Device: `{meta['device']}`")
    lines.append(f"- Docs: `{meta['n_docs']}` | Queries: `{meta['n_queries']}`")
    lines.append(f"- Elapsed: `{meta['elapsed_sec']:.1f}s`")
    lines.append("")
    lines.append("## Results")
    lines.append("| k | Success@k | Recall@k (macro) |")
    lines.append("|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['k']} | {r['success_at_k']:.4f} | {r['recall_at_k']:.4f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- FAISS IndexFlatIP + L2-normalized embeddings = cosine similarity via dot product.")
    lines.append("- Success@k = %queries with at least one relevant doc in top-k.")
    lines.append("- Recall@k (macro) = average fraction of relevant docs retrieved in top-k.")
    lines.append("")

    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[day7-science] index:  {args.index_path}")
    print(f"[day7-science] meta:   {args.meta_path}")
    print(f"[day7-science] report: {args.report_path}")
    for r in rows:
        print(f"[day7-science] k={r['k']}: success={r['success_at_k']:.4f}, recall={r['recall_at_k']:.4f}")


if __name__ == "__main__":
    main()
