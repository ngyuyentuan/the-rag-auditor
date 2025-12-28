import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [P]

from src.utils.buffer import compute_cs_ret_from_logit, decide_route
import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_thresholds(path, track):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    t = (data.get("thresholds") or {}).get(track)
    if not t:
        raise SystemExit(f"missing thresholds for track={track}")
    return t


def select_logit_col(df, track, explicit=None):
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"logit_col not found: {explicit}")
        return explicit
    if track.lower() == "fever" and "logit_platt" in df.columns:
        return "logit_platt"
    if "raw_max_top3" in df.columns:
        return "raw_max_top3"
    if "logit_platt" in df.columns:
        return "logit_platt"
    raise SystemExit("no logit_col found (expected raw_max_top3 or logit_platt)")


def sanitize(df, logit_col, y_col):
    cols = [logit_col, y_col]
    if "qid" in df.columns:
        cols.append("qid")
    d = df[cols].copy()
    d[logit_col] = pd.to_numeric(d[logit_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[logit_col, y_col])
    d = d[np.isfinite(d[logit_col].to_numpy(np.float64))]
    d = d[(d[y_col] == 0) | (d[y_col] == 1)]
    d[y_col] = d[y_col].astype(int)
    d = d.reset_index(drop=True)
    d["row_idx"] = np.arange(len(d), dtype=np.int64)
    return d


def load_sample(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_sample(sample_path, track, n, seed):
    if sample_path and os.path.exists(sample_path):
        return sample_path
    if not sample_path:
        sample_path = f"runs/day12_{track}_e2e_sample.jsonl"
    cmd = [
        sys.executable,
        "scripts/day12_sample_500.py",
        "--track",
        track,
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--out",
        sample_path,
    ]
    subprocess.check_call(cmd)
    return sample_path


def load_scifact_queries(path):
    out = {}
    for row in iter_jsonl(path):
        qid = str(row.get("_id"))
        out[qid] = str(row.get("text", ""))
    return out


def load_scifact_corpus(path):
    doc_ids = []
    doc_texts = []
    for row in iter_jsonl(path):
        did = str(row.get("_id"))
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_ids.append(did)
        doc_texts.append((title + "\n" + text).strip())
    return doc_ids, doc_texts


def load_fever_claims(path):
    claims = []
    for row in iter_jsonl(path):
        claims.append(str(row.get("claim", "")))
    return claims


def load_fever_corpus(path):
    doc_ids = []
    doc_texts = []
    for row in iter_jsonl(path):
        did = row.get("doc_id", row.get("i"))
        doc_ids.append(str(did))
        doc_texts.append(str(row.get("text", "")))
    return doc_ids, doc_texts


def load_fever_gold(path):
    gold = {}
    if not path or not os.path.exists(path):
        return gold
    for idx, row in enumerate(iter_jsonl(path)):
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


def load_scifact_qrels(queries_jsonl):
    if not queries_jsonl:
        return {}
    base = os.path.abspath(os.path.join(os.path.dirname(queries_jsonl), ".."))
    candidates = [
        os.path.join(base, "qrels", "test.tsv"),
        os.path.join(base, "qrels", "dev.tsv"),
        os.path.join(base, "qrels", "train.tsv"),
        os.path.join(base, "qrels", "qrels.tsv"),
    ]
    qrels_path = next((p for p in candidates if os.path.exists(p)), None)
    if not qrels_path:
        return {}
    gold = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
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


def map_nli_to_verdict(label):
    if label == "ENTAILMENT":
        return "SUPPORTS"
    if label == "CONTRADICTION":
        return "REFUTES"
    if label == "NEUTRAL":
        return "NEI"
    return None


def get_gold_entry(gold_map, qid):
    qid_str = str(qid)
    if qid_str in gold_map:
        return gold_map[qid_str]
    try:
        qid_int = str(int(float(qid_str)))
    except Exception:
        return None
    return gold_map.get(qid_int)


def load_emb_model(meta_path, fallback):
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m = meta.get("emb_model")
        if isinstance(m, str) and m.strip():
            return m
    return fallback


def pick_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 7:
                    return "cuda"
            except Exception:
                pass
        return "cpu"
    return device


def nli_infer(pairs, model, tokenizer, device, batch_size):
    all_probs = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs.tolist())
    return all_probs


def truncate_passage(text, max_chars=2000):
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    pos = -1
    for d in [". ", "! ", "? ", "\n\n", "\n"]:
        p = cut.rfind(d)
        if p > pos:
            pos = p
    if pos >= int(max_chars * 0.7):
        return cut[:pos + 1].strip()
    parts = cut.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0].strip() + "..."
    return cut.strip() + "..."


def rerank_with_model(query, candidates, model, top_n, batch_size, max_chars=2000):
    if not candidates:
        return [], []
    idxs = []
    pairs = []
    for i, txt in candidates:
        idxs.append(int(i))
        pairs.append((query, truncate_passage(txt, max_chars=max_chars)))
    scores = model.predict(pairs, batch_size=batch_size)
    scores = [float(s) for s in scores]
    order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
    top_order = order[:top_n]
    top_idxs = [idxs[j] for j in top_order]
    top_scores = [scores[j] for j in top_order]
    return top_idxs, top_scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--stage2_policy", default="uncertain_only", choices=["uncertain_only", "always", "never"])
    ap.add_argument("--baseline_mode", default="calibrated", choices=["calibrated", "always_stage2", "random"])
    ap.add_argument("--rerank_topk", type=int, default=50)
    ap.add_argument("--rerank_keep", type=int, default=5)
    ap.add_argument("--nli_topn", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--logit_col", default=None)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--in_path", default=None)
    ap.add_argument("--index_path", default=None)
    ap.add_argument("--corpus_jsonl", default=None)
    ap.add_argument("--queries_jsonl", default=None)
    ap.add_argument("--fever_dev_jsonl", default="data/fever_splits/dev.jsonl")
    ap.add_argument("--meta_path", default=None)
    ap.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--nli_model", default="typeform/distilbert-base-uncased-mnli")
    ap.add_argument("--dry_run_stage2", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)
    in_path = args.in_path or f"data/calibration/{args.track}_stage1_dev_train.parquet"
    df = pd.read_parquet(in_path)
    logit_col = select_logit_col(df, args.track, args.logit_col)
    df = sanitize(df, logit_col, args.y_col)
    df_idx = df.set_index("row_idx")

    t = load_thresholds(args.thresholds, args.track)
    tau = float(t.get("tau"))
    t_lower = float(t.get("t_lower"))
    t_upper = float(t.get("t_upper"))

    sample_path = ensure_sample(args.sample, args.track, args.n, args.seed)
    sample_rows = load_sample(sample_path)

    rng = np.random.default_rng(args.seed)

    queries = {}
    doc_ids = []
    doc_texts = []
    emb_model_name = ""
    index_path = ""

    need_stage2_data = (not args.dry_run_stage2) and (args.stage2_policy != "never" or args.baseline_mode == "always_stage2")

    if need_stage2_data:
        if args.track == "scifact":
            queries_jsonl = args.queries_jsonl or "data/beir_scifact/scifact/queries.jsonl"
            corpus_jsonl = args.corpus_jsonl or "data/beir_scifact/scifact/corpus.jsonl"
            index_path = args.index_path or "artifacts/scifact.faiss.index"
            meta_path = args.meta_path or "artifacts/scifact_faiss_meta.json"
            queries = load_scifact_queries(queries_jsonl)
            doc_ids, doc_texts = load_scifact_corpus(corpus_jsonl)
            emb_model_name = load_emb_model(meta_path, "sentence-transformers/allenai-specter")
        else:
            corpus_jsonl = args.corpus_jsonl or "artifacts/fever_corpus.jsonl"
            index_path = args.index_path or "artifacts/fever.faiss.index"
            meta_path = args.meta_path or "artifacts/fever_faiss_meta.json"
            queries = load_fever_claims(args.fever_dev_jsonl)
            doc_ids, doc_texts = load_fever_corpus(corpus_jsonl)
            emb_model_name = load_emb_model(meta_path, "sentence-transformers/all-MiniLM-L6-v2")

    if args.track == "fever":
        gold_map = load_fever_gold(args.fever_dev_jsonl)
    else:
        gold_map = load_scifact_qrels(args.queries_jsonl or "data/beir_scifact/scifact/queries.jsonl")

    index = None
    bienc = None
    cross_encoder = None
    nli_model = None
    nli_tokenizer = None

    if need_stage2_data:
        index = faiss.read_index(index_path)
        try:
            bienc = SentenceTransformer(emb_model_name, device=device)
            cross_encoder = CrossEncoder(args.cross_encoder, device=device)
            nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
            nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
            nli_model.eval()
        except Exception as e:
            if device == "cuda" and ("no kernel image" in str(e).lower() or "cuda" in str(e).lower()):
                device = "cpu"
                bienc = SentenceTransformer(emb_model_name, device=device)
                cross_encoder = CrossEncoder(args.cross_encoder, device=device)
                nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
                nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
                nli_model.eval()
            else:
                raise

    calibrated_decisions = []
    for item in sample_rows:
        row_idx = item.get("row_idx")
        if row_idx is None:
            qid = item.get("qid")
            if qid is None:
                continue
            rows = df[df["qid"] == qid]
            if rows.empty:
                continue
            row = rows.iloc[0]
        else:
            if int(row_idx) not in df_idx.index:
                continue
            row = df_idx.loc[int(row_idx)]
        raw_logit = float(row[logit_col])
        cs_ret = compute_cs_ret_from_logit(raw_logit, tau)
        decision, _ = decide_route(cs_ret, t_lower, t_upper)
        calibrated_decisions.append(decision)

    n_cal = len(calibrated_decisions)
    if n_cal == 0:
        raise SystemExit("no valid samples found")
    p_accept = calibrated_decisions.count("ACCEPT") / n_cal
    p_reject = calibrated_decisions.count("REJECT") / n_cal
    p_uncertain = calibrated_decisions.count("UNCERTAIN") / n_cal

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for item in sample_rows:
            row_idx = item.get("row_idx")
            if row_idx is None:
                qid = item.get("qid")
                if qid is None:
                    continue
                rows = df[df["qid"] == qid]
                if rows.empty:
                    continue
                row = rows.iloc[0]
            else:
                if int(row_idx) not in df_idx.index:
                    continue
                row = df_idx.loc[int(row_idx)]

            qid = row["qid"] if "qid" in row else int(row["row_idx"])
            y = int(row[args.y_col])
            raw_logit = float(row[logit_col])

            t0 = time.perf_counter()
            cs_ret = compute_cs_ret_from_logit(raw_logit, tau)
            decision, reason = decide_route(cs_ret, t_lower, t_upper)
            if args.baseline_mode == "random":
                decision = rng.choice(["ACCEPT", "REJECT", "UNCERTAIN"], p=[p_accept, p_reject, p_uncertain])
                reason = "random_baseline(p_accept={:.4f},p_reject={:.4f},p_uncertain={:.4f})".format(
                    p_accept, p_reject, p_uncertain
                )
            stage1_ms = (time.perf_counter() - t0) * 1000.0

            if args.baseline_mode == "always_stage2":
                stage2_should_run = True
            elif args.stage2_policy == "always":
                stage2_should_run = True
            elif args.stage2_policy == "never":
                stage2_should_run = False
            else:
                stage2_should_run = decision == "UNCERTAIN"

            rerank_ms = 0.0
            nli_ms = 0.0
            rerank_info = {}
            nli_info = {}
            stage2_ran = False
            top_doc_id = None

            if stage2_should_run:
                if args.dry_run_stage2:
                    stage2_ran = True
                    rerank_info = {"skipped": True}
                    nli_info = {"skipped": True}
                else:
                    query = queries[qid] if isinstance(queries, dict) else queries[int(qid)]
                    t1 = time.perf_counter()
                    qemb = bienc.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
                    _, I = index.search(qemb, args.rerank_topk)
                    cand_idx = [int(i) for i in I[0] if 0 <= int(i) < len(doc_texts)]
                    candidates = [(i, doc_texts[i]) for i in cand_idx]
                    try:
                        rr_idxs, rr_scores = rerank_with_model(
                            query=query,
                            candidates=candidates,
                            model=cross_encoder,
                            top_n=args.rerank_keep,
                            batch_size=args.batch_size,
                        )
                    except RuntimeError as e:
                        if device == "cuda" and "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            device = "cpu"
                            cross_encoder = CrossEncoder(args.cross_encoder, device=device)
                            rr_idxs, rr_scores = rerank_with_model(
                                query=query,
                                candidates=candidates,
                                model=cross_encoder,
                                top_n=args.rerank_keep,
                                batch_size=args.batch_size,
                            )
                        else:
                            raise
                    rerank_ms = (time.perf_counter() - t1) * 1000.0
                    stage2_ran = True

                    docid_map = {i: doc_ids[i] for i in cand_idx if i < len(doc_ids)}
                    reranked_docids = [docid_map.get(i, str(i)) for i in rr_idxs]
                    reranked_scores = [float(s) for s in rr_scores]
                    top_doc_id = reranked_docids[0] if reranked_docids else None
                    top_score = reranked_scores[0] if reranked_scores else None
                    rerank_info = {
                        "top_k": args.rerank_topk,
                        "keep": args.rerank_keep,
                        "model": args.cross_encoder,
                        "doc_ids": reranked_docids,
                        "scores": reranked_scores,
                        "top_doc_id": top_doc_id,
                        "top_score": top_score,
                    }

                    t2 = time.perf_counter()
                    topn = max(1, min(args.nli_topn, len(reranked_docids)))
                    evidence_texts = [doc_texts[i] for i in rr_idxs[:topn]] if topn > 0 else []
                    pairs = [(evidence_texts[i], query) for i in range(len(evidence_texts))]
                    try:
                        probs = nli_infer(pairs, nli_model, nli_tokenizer, device, args.batch_size) if pairs else []
                    except RuntimeError as e:
                        if device == "cuda" and "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            device = "cpu"
                            nli_model = nli_model.to(device)
                            probs = nli_infer(pairs, nli_model, nli_tokenizer, device, args.batch_size) if pairs else []
                        else:
                            raise
                    nli_ms = (time.perf_counter() - t2) * 1000.0

                    id2label = getattr(nli_model.config, "id2label", None) or {}
                    labels = [id2label.get(i, str(i)) for i in range(len(probs[0]))] if probs else []
                    top_label = None
                    top_prob = None
                    if probs:
                        p0 = probs[0]
                        k = int(np.argmax(p0))
                        top_label = labels[k] if labels else str(k)
                        top_prob = float(p0[k])
                    label_probs = {}
                    if probs and labels:
                        label_probs = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
                    nli_info = {
                        "topn": topn,
                        "labels": labels,
                        "probs": probs[0] if probs else [],
                        "top_label": top_label,
                        "top_prob": top_prob,
                        "label_probs": label_probs,
                        "model": args.nli_model,
                    }

            gold_entry = get_gold_entry(gold_map, qid)
            gold_verdict = None
            if "label" in row and isinstance(row["label"], str) and row["label"].strip():
                gold_verdict = row["label"]
            elif "verdict" in row and isinstance(row["verdict"], str) and row["verdict"].strip():
                gold_verdict = row["verdict"]
            if gold_verdict is None and gold_entry and gold_entry.get("verdict"):
                gold_verdict = gold_entry.get("verdict")
            gold_doc_ids = gold_entry.get("doc_ids", []) if gold_entry else []
            gold_has_evidence = bool(gold_doc_ids)
            gold_missing = gold_verdict is None and not gold_doc_ids

            pred_verdict = map_nli_to_verdict(nli_info.get("top_label")) if nli_info else None
            pred_doc_id = top_doc_id if stage2_ran else None
            if gold_missing:
                pred_has_evidence = None
            else:
                pred_has_evidence = pred_doc_id in set(gold_doc_ids) if pred_doc_id is not None else False

            total_ms = stage1_ms + rerank_ms + nli_ms
            payload = {
                "baseline": {"name": args.baseline_mode},
                "metadata": {
                    "qid": qid,
                    "track": args.track,
                    "split": "train",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": int(args.seed),
                },
                "stage1": {
                    "cs_ret": cs_ret,
                    "route_decision": decision,
                    "route_reason": reason,
                    "t_lower": t_lower,
                    "t_upper": t_upper,
                    "tau": tau,
                },
                "stage2": {
                    "ran": stage2_ran,
                    "policy": args.stage2_policy,
                    "rerank": rerank_info,
                    "nli": nli_info,
                },
                "timing_ms": {
                    "stage1_ms": stage1_ms,
                    "rerank_ms": rerank_ms,
                    "nli_ms": nli_ms,
                    "total_ms": total_ms,
                },
                "ground_truth": {"y": y},
                "gold": {
                    "gold_verdict": gold_verdict,
                    "gold_evidence_doc_ids": gold_doc_ids,
                    "gold_has_evidence": gold_has_evidence,
                },
                "pred": {
                    "pred_verdict": pred_verdict,
                    "pred_doc_id": pred_doc_id,
                    "pred_has_evidence": pred_has_evidence,
                },
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("[ok] wrote:", args.out)


if __name__ == "__main__":
    main()
