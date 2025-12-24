import os, sys, json, math, argparse, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.normalize import normalize_text
from sentence_transformers import SentenceTransformer
import faiss

def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def load_scifact_qrels(scifact_dir: str):
    qrels = {}
    for name in ["train.tsv", "test.tsv"]:
        fp = os.path.join(scifact_dir, "qrels", name)
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, sep="\t")
        for _, r in df.iterrows():
            qid = str(r["query-id"])
            did = str(r["corpus-id"])
            score = float(r["score"])
            if score <= 0:
                continue
            qrels.setdefault(qid, set()).add(did)
    return qrels

def load_doc_ids(meta_path: str):
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    rows = meta.get("rows", None)
    if rows is None:
        x = meta.get("id2doc", None)
        if isinstance(x, dict):
            ks=list(x.keys())
            if ks and all(str(k).isdigit() for k in ks):
                ks=sorted(ks, key=lambda z:int(z))
                rows=[x[k] for k in ks]
            else:
                rows=list(x.values())
        else:
            rows=x
    if isinstance(rows, dict):
        ks=list(rows.keys())
        if ks and all(str(k).isdigit() for k in ks):
            ks=sorted(ks, key=lambda z:int(z))
            rows=[rows[k] for k in ks]
        else:
            rows=list(rows.values())
    if not isinstance(rows, list):
        rows=[]
    out=[]
    for r in rows:
        if isinstance(r, dict):
            out.append(str(r.get("doc_id") or r.get("id") or r.get("corpus_id") or r.get("_id") or ""))
        else:
            out.append(str(r))
    return out, meta.get("emb_model", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--index_meta", required=True)
    ap.add_argument("--scifact_dir", required=True)
    ap.add_argument("--bi_encoder", required=True)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    index = faiss.read_index(args.index_path)
    doc_ids, emb_model = load_doc_ids(args.index_meta)
    if len(doc_ids) == 0:
        raise SystemExit("empty doc_ids")
    if index.ntotal != len(doc_ids):
        raise SystemExit(f"index.ntotal={index.ntotal} != len(doc_ids)={len(doc_ids)}")

    qmap = {}
    for r in iter_jsonl(os.path.join(args.scifact_dir, "queries.jsonl")):
        qid = str(r.get("_id") or r.get("id") or r.get("query_id") or r.get("qid"))
        qtext = r.get("text") or r.get("query") or ""
        qmap[qid] = normalize_text(qtext)

    qrels = load_scifact_qrels(args.scifact_dir)
    qids = sorted(qmap.keys(), key=lambda x: int(x) if x.isdigit() else x)
    n = len(qids)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    tr_qids = qids[:n_tr]
    va_qids = qids[n_tr:n_tr + n_va]
    te_qids = qids[n_tr + n_va:]

    bi = SentenceTransformer(args.bi_encoder, device="cuda")

    def search(qtexts):
        emb = bi.encode(qtexts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        if emb.shape[1] != index.d:
            raise SystemExit(f"emb_dim {emb.shape[1]} != index.d {index.d}")
        D, I = index.search(emb, args.top_k)
        return D, I

    def build_split(name, split_qids):
        rows=[]
        bs=64
        for i in tqdm(range(0, len(split_qids), bs), desc=f"[day9] scifact {name}"):
            batch=split_qids[i:i+bs]
            qtexts=[qmap[q] for q in batch]
            D,I=search(qtexts)
            for qi,qid in enumerate(batch):
                idxs=[j for j in I[qi].tolist() if 0 <= j < len(doc_ids)]
                cand=[doc_ids[j] for j in idxs]
                gold=qrels.get(str(qid), set())
                num_gold=sum(1 for d in cand if d in gold)
                y=int(num_gold>0)

                s=D[qi].astype(np.float64)
                top3=np.sort(s)[-3:] if s.size>=3 else (s if s.size else np.array([],dtype=np.float64))
                top5=np.sort(s)[-5:] if s.size>=5 else (s if s.size else np.array([],dtype=np.float64))
                raw_max_top3=float(top3.max()) if top3.size else 0.0
                mean_top3=float(top3.mean()) if top3.size else 0.0
                std_top3=float(top3.std()) if top3.size else 0.0
                gap12=float(top3[-1]-top3[-2]) if top3.size>=2 else 0.0
                p=np.exp(top5-top5.max()) if top5.size else np.array([],dtype=np.float64)
                p=p/p.sum() if p.size and p.sum()>0 else p
                entropy_top5=entropy(p) if p.size else 0.0

                bi_top1=float(s.max()) if s.size else 0.0
                bi_mean_top5=float(top5.mean()) if top5.size else 0.0
                cs_ret=float(stable_sigmoid(raw_max_top3 / max(args.temperature,1e-6)))

                rows.append({
                    "qid": str(qid),
                    "cs_ret": cs_ret,
                    "raw_max_top3": raw_max_top3,
                    "mean_top3": mean_top3,
                    "std_top3": std_top3,
                    "gap12": gap12,
                    "entropy_top5": entropy_top5,
                    "bi_top1": bi_top1,
                    "bi_mean_top5": bi_mean_top5,
                    "y": y,
                    "num_gold": int(num_gold),
                })
        return pd.DataFrame(rows)

    df_tr=build_split("train", tr_qids)
    df_va=build_split("val", va_qids)
    df_te=build_split("test", te_qids)

    base=args.out_base[:-8] if args.out_base.endswith(".parquet") else args.out_base
    out_tr=base+"_train.parquet"
    out_va=base+"_val.parquet"
    out_te=base+"_test.parquet"
    os.makedirs(os.path.dirname(out_tr), exist_ok=True)
    df_tr.to_parquet(out_tr, index=False)
    df_va.to_parquet(out_va, index=False)
    df_te.to_parquet(out_te, index=False)

    stats={
        "mode":"scifact",
        "n_train": int(len(df_tr)),
        "pos_train": float(df_tr["y"].mean()) if len(df_tr) else 0.0,
        "n_val": int(len(df_va)),
        "pos_val": float(df_va["y"].mean()) if len(df_va) else 0.0,
        "n_test": int(len(df_te)),
        "pos_test": float(df_te["y"].mean()) if len(df_te) else 0.0,
        "bi_encoder": args.bi_encoder,
        "index_d": int(index.d),
        "top_k": int(args.top_k),
        "doc_ids_n": int(len(doc_ids)),
        "emb_model_meta": str(emb_model),
    }
    os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
    json.dump(stats, open(args.stats_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print("saved", out_tr)
    print("saved", out_va)
    print("saved", out_te)
    print("stats", args.stats_path)
    print(stats)

if __name__=="__main__":
    main()
