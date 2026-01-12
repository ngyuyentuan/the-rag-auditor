"""
Build Real Calibration Data from BEIR SciFact/FEVER

This script:
1. Downloads BEIR SciFact dataset (if not exists)
2. Builds FAISS index with bi-encoder embeddings
3. Retrieves top-k candidates for each query
4. Reranks with cross-encoder to get calibration scores
5. Creates ground-truth labels (y) based on qrels
6. Outputs parquet file for Stage1 tuning

This creates REAL calibration data for testing the Stage1 brain.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, CrossEncoder

# Try to import BEIR
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("[warn] beir not installed. Install with: pip install beir")

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype(np.float32)


def truncate_text(text: str, max_chars: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    pos = cut.rfind(". ")
    if pos > max_chars * 0.7:
        return cut[:pos + 1]
    return cut + "..."


def main():
    ap = argparse.ArgumentParser(description="Build real calibration data from BEIR SciFact")
    ap.add_argument("--data_root", default="data/beir_scifact")
    ap.add_argument("--split", default="test", help="BEIR split to use")
    ap.add_argument("--bi_encoder", default="sentence-transformers/allenai-specter")
    ap.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--rerank_topk", type=int, default=50, help="Candidates for reranking")
    ap.add_argument("--rerank_keep", type=int, default=5, help="Top docs to keep after rerank")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_queries", type=int, default=0, help="Limit queries (0=all)")
    ap.add_argument("--out_parquet", default="data/calibration/scifact_stage1_real.parquet")
    ap.add_argument("--out_index", default="artifacts/scifact.faiss.index")
    ap.add_argument("--out_meta", default="artifacts/scifact_faiss_meta.json")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    if not BEIR_AVAILABLE:
        print("[ERROR] beir package required. Install with: pip install beir")
        return

    set_seed(args.seed)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")

    # 1) Download SciFact
    dataset_dir = os.path.join(args.data_root, "scifact")
    if not os.path.exists(dataset_dir):
        print("[info] Downloading BEIR SciFact dataset...")
        os.makedirs(args.data_root, exist_ok=True)
        util.download_and_unzip(SCIFACT_URL, args.data_root)
    
    # 2) Load corpus/queries/qrels
    print("[info] Loading dataset...")
    corpus, queries, qrels = GenericDataLoader(dataset_dir).load(split=args.split)
    
    doc_ids = list(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        title = corpus[did].get("title", "") or ""
        text = corpus[did].get("text", "") or ""
        doc_texts.append((title + "\n" + text).strip())
    
    q_ids = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    
    # Filter to queries with qrels
    valid_qids = [qid for qid in q_ids if qid in qrels and qrels[qid]]
    if args.max_queries > 0:
        valid_qids = valid_qids[:args.max_queries]
    
    print(f"[info] Corpus: {len(doc_ids)} docs, Queries: {len(valid_qids)} (with qrels)")
    
    # 3) Load models
    print(f"[info] Loading bi-encoder: {args.bi_encoder}")
    bi_encoder = SentenceTransformer(args.bi_encoder, device=device)
    
    print(f"[info] Loading cross-encoder: {args.cross_encoder}")
    cross_encoder = CrossEncoder(args.cross_encoder, device=device)
    
    # 4) Embed corpus and build FAISS index
    print("[info] Embedding corpus...")
    doc_embs = bi_encoder.encode(
        doc_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    doc_embs = l2norm(doc_embs)
    
    print("[info] Building FAISS index...")
    d = doc_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(doc_embs)
    
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    faiss.write_index(index, args.out_index)
    print(f"[info] Saved index: {args.out_index}")
    
    # 5) Process queries: retrieve + rerank
    print("[info] Processing queries with retrieval and reranking...")
    results = []
    
    for qid in tqdm(valid_qids, desc="Queries"):
        query = queries[qid]
        gold_docs = set(d for d, v in qrels.get(qid, {}).items() if v > 0)
        
        # Bi-encoder retrieval
        q_emb = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        _, I = index.search(q_emb, args.rerank_topk)
        cand_idx = [int(i) for i in I[0] if 0 <= i < len(doc_ids)]
        
        if not cand_idx:
            continue
        
        # Cross-encoder reranking
        pairs = [(query, truncate_text(doc_texts[i])) for i in cand_idx]
        try:
            scores = cross_encoder.predict(pairs, batch_size=args.batch_size)
        except Exception as e:
            print(f"[warn] Error processing qid={qid}: {e}")
            continue
        
        scores = [float(s) for s in scores]
        
        # Sort by score
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        
        # Get top scores for features
        top_scores = [scores[order[i]] for i in range(min(len(order), 5))]
        raw_max_top3 = max(top_scores[:3]) if len(top_scores) >= 1 else 0.0
        
        # Check if any gold doc is in top-k retrieved
        retrieved_docids = [doc_ids[cand_idx[j]] for j in order[:args.rerank_keep]]
        has_gold = any(did in gold_docs for did in retrieved_docids)
        
        # Ground truth: y=1 if gold evidence found in top-k
        y = 1 if has_gold else 0
        
        # Build row
        row = {
            "qid": qid,
            "query": query[:200],  # Truncate for storage
            "y": y,
            "raw_max_top3": raw_max_top3,
            "top1": top_scores[0] if len(top_scores) > 0 else 0.0,
            "top2": top_scores[1] if len(top_scores) > 1 else 0.0,
            "top3": top_scores[2] if len(top_scores) > 2 else 0.0,
            "gap12": (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else 0.0,
            "mean_top3": np.mean(top_scores[:3]) if len(top_scores) >= 1 else 0.0,
            "std_top3": np.std(top_scores[:3]) if len(top_scores) >= 3 else 0.0,
            "n_gold": len(gold_docs),
            "n_retrieved": min(len(order), args.rerank_keep),
        }
        results.append(row)
    
    # 6) Save parquet
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    
    print(f"\n[SUCCESS] Created calibration data: {args.out_parquet}")
    print(f"  Rows: {len(df)}")
    print(f"  Positive rate (y=1): {df['y'].mean():.2%}")
    print(f"  raw_max_top3 range: [{df['raw_max_top3'].min():.3f}, {df['raw_max_top3'].max():.3f}]")
    
    # Save meta
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "BEIR/SciFact",
        "split": args.split,
        "bi_encoder": args.bi_encoder,
        "cross_encoder": args.cross_encoder,
        "n_queries": len(df),
        "positive_rate": float(df["y"].mean()),
        "rerank_topk": args.rerank_topk,
        "rerank_keep": args.rerank_keep,
        "emb_model": args.bi_encoder,
    }
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta: {args.out_meta}")


if __name__ == "__main__":
    main()
