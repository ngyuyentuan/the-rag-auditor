import argparse, csv, json, os, time
from typing import Dict, Iterable, List, Tuple, Set, Any

import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from src.retrieval.rerank import rerank_topk


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_scifact_corpus(corpus_jsonl: str) -> Tuple[List[str], List[str]]:
    doc_ids, doc_texts = [], []
    for row in iter_jsonl(corpus_jsonl):
        did = str(row.get("_id"))
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_ids.append(did)
        doc_texts.append((title + "\n" + text).strip())
    return doc_ids, doc_texts


def load_scifact_queries(queries_jsonl: str) -> Tuple[List[str], List[str]]:
    qids, queries = [], []
    for row in iter_jsonl(queries_jsonl):
        qids.append(str(row.get("_id")))
        queries.append(str(row.get("text", "")))
    return qids, queries


def load_qrels_tsv(qrels_tsv: str) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    with open(qrels_tsv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)
        for qid, did, score, *_ in reader:
            try:
                s = float(score)
            except Exception:
                s = 0.0
            if s > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def success_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]).intersection(gold) else 0.0


def recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    # standard IR recall@k
    if not gold:
        return 0.0
    return len(set(retrieved[:k]).intersection(gold)) / len(gold)


def ndcg_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(retrieved[:k]):
        if did in gold:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg > 0 else 0.0


def write_md(path: str, info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ks = info["ks"]

    lines = []
    lines.append("# Day 8 — Cross-encoder Rerank Sanity (SciFact)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Timestamp: `{info['created_ts']}`")
    lines.append(f"- Device: `{info['device']}`")
    lines.append(f"- Bi-encoder: `{info['bi_encoder']}`")
    lines.append(f"- Cross-encoder: `{info['cross_encoder']}`")
    lines.append(f"- Candidate top_k (bi-encoder): `{info['top_k']}`")
    lines.append(f"- Rerank top_n: `{info['rerank_top_n']}`")
    lines.append(f"- Temperature τ: `{info['temperature']}`")
    lines.append("")
    lines.append("## Metrics (mean over queries)")
    lines.append("| k | baseline success@k | baseline recall@k | baseline ndcg@k | rerank success@k | rerank recall@k | rerank ndcg@k |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k in ks:
        b = info["baseline"][k]
        r = info["rerank"][k]
        lines.append(
            f"| {k} | {b['success']:.4f} | {b['recall']:.4f} | {b['ndcg']:.4f} | "
            f"{r['success']:.4f} | {r['recall']:.4f} | {r['ndcg']:.4f} |"
        )

    lines.append("")
    lines.append("## CS_ret")
    lines.append(r"- Definition: $CS_{ret}=\sigma(\max_{\text{top3}} score / \tau)$. Since scores are sorted desc, this equals $\sigma(score_{top1}/\tau)$.")
    lines.append(r"- Note: Cross-encoder scores are **uncalibrated**. Day 9/10 will calibrate (tune τ / thresholds).")
    lines.append("")
    lines.append("## Important note")
    lines.append("- Rerank is **upper-bounded** by whether the gold doc appears in the bi-encoder top_k candidates.")
    lines.append("")
    lines.append("## CS_ret distribution (sanity)")
    lines.append(f"- mean={info['cs_stats']['mean']:.4f}, std={info['cs_stats']['std']:.4f}, median={info['cs_stats']['median']:.4f}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/beir_scifact/scifact")
    ap.add_argument("--index_path", default="artifacts/scifact.faiss.index")
    ap.add_argument("--bi_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--rerank_top_n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--report_path", default="reports/rerank_sanity.md")
    ap.add_argument("--run_path", default="runs/day8_rerank_scifact.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus_jsonl = os.path.join(args.data_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(args.data_dir, "queries.jsonl")
    qrels_tsv = os.path.join(args.data_dir, "qrels", "test.tsv")

    doc_ids, doc_texts = load_scifact_corpus(corpus_jsonl)
    qids, queries = load_scifact_queries(queries_jsonl)
    qrels = load_qrels_tsv(qrels_tsv)

    kept = [(qid, q) for qid, q in zip(qids, queries) if qid in qrels]
    if args.max_queries and args.max_queries > 0:
        kept = kept[: args.max_queries]
    qids = [x[0] for x in kept]
    queries = [x[1] for x in kept]

    index = faiss.read_index(args.index_path)
    bienc = SentenceTransformer(args.bi_encoder, device=device)

    ks = sorted(args.ks)
    base_sum = {k: {"success": 0.0, "recall": 0.0, "ndcg": 0.0} for k in ks}
    rr_sum = {k: {"success": 0.0, "recall": 0.0, "ndcg": 0.0} for k in ks}

    cs_vals: List[float] = []
    n = 0

    os.makedirs(os.path.dirname(args.run_path), exist_ok=True)
    with open(args.run_path, "w", encoding="utf-8") as out:
        for qid, query in tqdm(list(zip(qids, queries)), desc="[day8] rerank eval"):
            gold = qrels.get(qid, set())
            if not gold:
                continue

            qemb = bienc.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            _, I = index.search(qemb, args.top_k)
            cand_idx = [int(i) for i in I[0] if 0 <= int(i) < len(doc_ids)]
            cand_docids = [doc_ids[i] for i in cand_idx]

            for k in ks:
                base_sum[k]["success"] += success_at_k(cand_docids, gold, k)
                base_sum[k]["recall"] += recall_at_k(cand_docids, gold, k)
                base_sum[k]["ndcg"] += ndcg_at_k(cand_docids, gold, k)

            candidates = [(i, doc_texts[i]) for i in cand_idx]
            rr = rerank_topk(
                query=query,
                candidates=candidates,
                cross_encoder_model=args.cross_encoder,
                top_n=args.rerank_top_n,
                device=device,
                temperature=args.temperature,
            )
            reranked_docids = [doc_ids[i] for i in rr.doc_indices]
            merged = reranked_docids + [d for d in cand_docids if d not in set(reranked_docids)]

            for k in ks:
                rr_sum[k]["success"] += success_at_k(merged, gold, k)
                rr_sum[k]["recall"] += recall_at_k(merged, gold, k)
                rr_sum[k]["ndcg"] += ndcg_at_k(merged, gold, k)

            cs_vals.append(rr.cs_ret)

            out.write(json.dumps({
                "qid": qid,
                "query": query,
                "gold_docids": sorted(list(gold)),
                "baseline_top10": cand_docids[:10],
                "rerank_top5": merged[:5],
                "max_top3_score": rr.max_top3_score,
                "cs_ret": rr.cs_ret,
                "temperature": rr.temperature,
            }, ensure_ascii=False) + "\n")

            n += 1

    baseline = {k: {m: base_sum[k][m] / n for m in base_sum[k]} for k in ks}
    rerank = {k: {m: rr_sum[k][m] / n for m in rr_sum[k]} for k in ks}

    cs_arr = np.array(cs_vals, dtype=np.float32) if cs_vals else np.array([0.0], dtype=np.float32)
    info = {
        "created_ts": time.time(),
        "device": device,
        "bi_encoder": args.bi_encoder,
        "cross_encoder": args.cross_encoder,
        "top_k": args.top_k,
        "rerank_top_n": args.rerank_top_n,
        "temperature": args.temperature,
        "n_queries": n,
        "ks": ks,
        "baseline": baseline,
        "rerank": rerank,
        "cs_stats": {"mean": float(cs_arr.mean()), "std": float(cs_arr.std()), "median": float(np.median(cs_arr))},
    }

    write_md(args.report_path, info)
    print("[day8] wrote report:", args.report_path)
    for k in ks:
        print(
            f"[day8] k={k} baseline success={baseline[k]['success']:.4f} recall={baseline[k]['recall']:.4f} ndcg={baseline[k]['ndcg']:.4f} | "
            f"rerank success={rerank[k]['success']:.4f} recall={rerank[k]['recall']:.4f} ndcg={rerank[k]['ndcg']:.4f}"
        )


if __name__ == "__main__":
    main()
