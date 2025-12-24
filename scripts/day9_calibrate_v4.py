import os, sys, json, time, math, argparse, random, inspect
from typing import Any, Dict, List, Tuple, Optional

P=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
R=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path[:0]=[P,R]

import numpy as np
import pandas as pd
import faiss
from tqdm.auto import tqdm

from src.utils.normalize import normalize_text
try:
    from src.utils.evidence_match import check_sufficiency_substring
except Exception:
    check_sufficiency_substring = None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def softmax_np(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m=float(np.max(x))
    e=np.exp(x-m)
    s=float(np.sum(e))
    if s==0.0:
        return np.zeros_like(x)
    return e/s

def spearman_corr_rank(a: np.ndarray, b: np.ndarray) -> float:
    n=min(len(a),len(b))
    if n<=1:
        return float("nan")
    ar=np.argsort(np.argsort(a[:n]))
    br=np.argsort(np.argsort(b[:n]))
    da=ar-ar.mean()
    db=br-br.mean()
    denom=float(np.sqrt((da*da).sum())*np.sqrt((db*db).sum()))
    if denom==0.0:
        return float("nan")
    return float((da*db).sum()/denom)

def truncate_chars(s: str, max_chars: int) -> str:
    if not isinstance(s,str):
        return ""
    if max_chars<=0:
        return s
    if len(s)<=max_chars:
        return s
    cut=s[:max_chars]
    sp=cut.rfind(" ")
    if sp>0:
        return cut[:sp]
    return cut

def load_scifact(scifact_dir: str):
    corpus_path=os.path.join(scifact_dir,"corpus.jsonl")
    queries_path=os.path.join(scifact_dir,"queries.jsonl")
    qrels_path=os.path.join(scifact_dir,"qrels","test.tsv")
    if not os.path.exists(qrels_path):
        qrels_path=os.path.join(scifact_dir,"qrels","train.tsv")

    dids=[]
    dtexts=[]
    for row in iter_jsonl(corpus_path):
        did=str(row.get("_id"))
        title=row.get("title","")
        text=row.get("text","")
        dids.append(did)
        dtexts.append((title+" "+text).strip())

    qids=[]
    qtexts=[]
    for row in iter_jsonl(queries_path):
        qid=str(row.get("_id"))
        q=qrow=row.get("text","")
        qids.append(qid)
        qtexts.append(q)

    gold={}
    with open(qrels_path,"r",encoding="utf-8") as f:
        header=f.readline()
        for line in f:
            parts=line.strip().split("\t")
            if len(parts)<4:
                continue
            qid,_,did,rel=parts[0],parts[1],parts[2],parts[3]
            if int(rel)>0:
                gold.setdefault(str(qid),set()).add(str(did))

    return dids,dtexts,qids,qtexts,gold

def extract_evidence_sentences(evidence_field: Any) -> List[str]:
    out=[]
    if not isinstance(evidence_field,list) or not evidence_field:
        return out
    if isinstance(evidence_field[0],list) and len(evidence_field[0])>=3 and isinstance(evidence_field[0][2],str):
        for item in evidence_field:
            if isinstance(item,list) and len(item)>=3 and isinstance(item[2],str):
                out.append(item[2])
        return out
    for group in evidence_field:
        if not isinstance(group,list):
            continue
        for item in group:
            if isinstance(item,list) and len(item)>=3 and isinstance(item[2],str):
                out.append(item[2])
    return out

def load_fever_dev(dev_jsonl: str):
    claims=[]
    gold_sents=[]
    for ex in iter_jsonl(dev_jsonl):
        c=ex.get("claim","")
        if not isinstance(c,str) or not c.strip():
            continue
        ev=extract_evidence_sentences(ex.get("evidence"))
        ev=[e for e in ev if isinstance(e,str) and normalize_text(e)]
        if not ev:
            continue
        claims.append(c)
        gold_sents.append(ev)
    return claims,gold_sents

def load_faiss_corpus_jsonl(path: str):
    texts=[]
    for row in iter_jsonl(path):
        t=row.get("text","")
        texts.append(t)
    return texts

def rerank_one(ce, query: str, cand_ids: List[str], cand_texts: List[str], ce_batch_size: int, rerank_top_n: int, temperature: float, max_doc_chars: int, bi_scores: Optional[np.ndarray]):
    n=len(cand_ids)
    if n==0:
        return [],[],float("-inf"),0.0,float("nan"),float("nan"),float("nan"),float("nan")
    pairs=[]
    for i in range(n):
        txt=truncate_chars(cand_texts[i], max_doc_chars)
        pairs.append((query, txt))
    scores=ce.predict(pairs, batch_size=ce_batch_size)
    scores=np.asarray(scores,dtype=np.float32)
    order=np.argsort(-scores)
    reranked=[cand_ids[int(j)] for j in order[:rerank_top_n]]
    reranked_scores=[float(scores[int(j)]) for j in order[:rerank_top_n]]
    top3=scores[order[:min(3,len(order))]]
    max_top3=float(np.max(top3)) if top3.size>0 else float("-inf")
    mean_top3=float(np.mean(top3)) if top3.size>0 else float("nan")
    std_top3=float(np.std(top3)) if top3.size>0 else float("nan")
    gap12=float(scores[int(order[0])]-scores[int(order[1])]) if len(order)>=2 else float("nan")
    cs_ret=stable_sigmoid(max_top3/temperature) if max_top3!=float("-inf") else 0.0
    entropy_top5=float("nan")
    top5=scores[order[:min(5,len(order))]]
    if top5.size>0:
        p=softmax_np(top5.astype(np.float64))
        eps=1e-12
        entropy_top5=float(-np.sum(p*np.log(np.clip(p,eps,1.0))))
    ce_bi_corr=float("nan")
    if bi_scores is not None and len(bi_scores)==len(scores):
        ce_bi_corr=spearman_corr_rank(-bi_scores.astype(np.float64), -scores.astype(np.float64))
    return reranked, reranked_scores, max_top3, cs_ret, mean_top3, std_top3, gap12, entropy_top5, ce_bi_corr

def split_indices(n: int, seed: int):
    rng=np.random.RandomState(seed)
    idx=np.arange(n)
    rng.shuffle(idx)
    n_train=int(0.6*n)
    n_val=int(0.2*n)
    train=idx[:n_train]
    val=idx[n_train:n_train+n_val]
    test=idx[n_train+n_val:]
    return train,val,test

def save_parquet(base_path: str, split_name: str, rows: List[Dict[str,Any]]):
    out_path=base_path.replace(".parquet", f"_{split_name}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print("saved", out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scifact","fever"], required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--index_path", required=True)
    ap.add_argument("--index_meta", default="")

    ap.add_argument("--bi_encoder", default="")
    ap.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    ap.add_argument("--scifact_dir", default="")
    ap.add_argument("--fever_dev_jsonl", default="")
    ap.add_argument("--corpus_jsonl", default="")

    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--rerank_top_n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce_batch_size", type=int, default=16)
    ap.add_argument("--max_doc_chars", type=int, default=2000)
    ap.add_argument("--max_queries", type=int, default=0)

    ap.add_argument("--out_base", required=True)
    ap.add_argument("--stats_path", required=True)
    args=ap.parse_args()

    set_seed(args.seed)

    if (not args.bi_encoder) and args.index_meta and os.path.exists(args.index_meta):
        meta=read_json(args.index_meta)
        m=meta.get("emb_model","")
        if isinstance(m,str) and m.strip():
            args.bi_encoder=m

    if not args.bi_encoder:
        raise RuntimeError("bi_encoder is required (or provide --index_meta with emb_model).")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    bi=SentenceTransformer(args.bi_encoder)
    ce=CrossEncoder(args.cross_encoder)

    index=faiss.read_index(args.index_path)

    rows=[]
    stats={"mode":args.mode,"seed":args.seed,"top_k":args.top_k,"rerank_top_n":args.rerank_top_n,"temperature":args.temperature,"count":0}

    if args.mode=="scifact":
        dids,dtexts,qids,qtexts,gold=load_scifact(args.scifact_dir)
        if index.ntotal!=len(dids):
            stats["warn_ntotal_mismatch"]={"index_ntotal":int(index.ntotal),"docs":len(dids)}
        n=len(qids)
        train_idx,val_idx,test_idx=split_indices(n,args.seed)

        def run_split(name, idxs):
            out=[]
            for j in tqdm(idxs, desc=f"[day9] scifact {name}"):
                qid=qids[int(j)]
                qtext=qtexts[int(j)]
                qemb=bi.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
                D,I=index.search(qemb,args.top_k)
                I=I[0].tolist()
                D=D[0].astype(np.float32)
                cand_ids=[]
                cand_texts=[]
                bi_scores=[]
                for rank,pos in enumerate(I):
                    if pos<0 or pos>=len(dids):
                        continue
                    cand_ids.append(dids[pos])
                    cand_texts.append(dtexts[pos])
                    bi_scores.append(float(D[rank]))
                bi_scores_np=np.asarray(bi_scores,dtype=np.float32) if bi_scores else None
                rr_ids, rr_scores, raw_max_top3, cs_ret, mean_top3, std_top3, gap12, entropy_top5, ce_bi_corr = rerank_one(
                    ce=ce, query=qtext, cand_ids=cand_ids, cand_texts=cand_texts,
                    ce_batch_size=args.ce_batch_size, rerank_top_n=args.rerank_top_n,
                    temperature=args.temperature, max_doc_chars=args.max_doc_chars,
                    bi_scores=bi_scores_np
                )
                g=gold.get(str(qid),set())
                y=1 if (g and any(d in g for d in rr_ids)) else 0
                out.append({
                    "qid":str(qid),
                    "cs_ret":float(cs_ret),
                    "raw_max_top3":float(raw_max_top3),
                    "mean_top3":float(mean_top3) if not math.isnan(mean_top3) else float("nan"),
                    "std_top3":float(std_top3) if not math.isnan(std_top3) else float("nan"),
                    "gap12":float(gap12) if not math.isnan(gap12) else float("nan"),
                    "entropy_top5":float(entropy_top5) if not math.isnan(entropy_top5) else float("nan"),
                    "ce_bi_corr":float(ce_bi_corr) if not math.isnan(ce_bi_corr) else float("nan"),
                    "bi_top1":float(bi_scores[0]) if bi_scores else float("nan"),
                    "bi_mean_top5":float(np.mean(bi_scores[:5])) if len(bi_scores)>=5 else float("nan"),
                    "y":int(y),
                    "num_gold":int(len(g)),
                })
                if args.max_queries and len(out)>=args.max_queries:
                    break
            return out

        train_rows=run_split("train",train_idx)
        val_rows=run_split("val",val_idx)
        test_rows=run_split("test",test_idx)

        save_parquet(args.out_base,"train",train_rows)
        save_parquet(args.out_base,"val",val_rows)
        save_parquet(args.out_base,"test",test_rows)

        all_rows=train_rows+val_rows+test_rows
        stats["count"]=len(all_rows)
        stats["pos_rate"]=float(np.mean([r["y"] for r in all_rows])) if all_rows else 0.0

    else:
        if check_sufficiency_substring is None:
            raise RuntimeError("check_sufficiency_substring missing; fix src/utils/evidence_match.py for FEVER.")
        corpus=load_faiss_corpus_jsonl(args.corpus_jsonl)
        claims,golds=load_fever_dev(args.fever_dev_jsonl)
        n=len(claims)
        train_idx,val_idx,test_idx=split_indices(n,args.seed)

        sig=inspect.signature(check_sufficiency_substring).parameters
        use_min_ratio=("min_ratio" in sig)
        use_min_overlap=("min_overlap_ratio" in sig)

        def run_split(name, idxs):
            out=[]
            for j in tqdm(idxs, desc=f"[day9] fever {name}"):
                claim=claims[int(j)]
                gold=golds[int(j)]
                qemb=bi.encode([claim], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
                D,I=index.search(qemb,args.top_k)
                I=I[0].tolist()
                D=D[0].astype(np.float32)
                cand_texts=[]
                bi_scores=[]
                for rank,pos in enumerate(I):
                    if pos<0 or pos>=len(corpus):
                        continue
                    cand_texts.append(corpus[pos])
                    bi_scores.append(float(D[rank]))
                cand_ids=[str(i) for i in range(len(cand_texts))]
                bi_scores_np=np.asarray(bi_scores,dtype=np.float32) if bi_scores else None

                rr_ids, rr_scores, raw_max_top3, cs_ret, mean_top3, std_top3, gap12, entropy_top5, ce_bi_corr = rerank_one(
                    ce=ce, query=claim, cand_ids=cand_ids, cand_texts=cand_texts,
                    ce_batch_size=args.ce_batch_size, rerank_top_n=args.rerank_top_n,
                    temperature=args.temperature, max_doc_chars=args.max_doc_chars,
                    bi_scores=bi_scores_np
                )

                rr_chunks=[]
                for did in rr_ids:
                    try:
                        rr_chunks.append(cand_texts[int(did)])
                    except Exception:
                        pass

                kwargs={}
                if use_min_ratio:
                    kwargs["min_ratio"]=0.9
                if use_min_overlap:
                    kwargs["min_overlap_ratio"]=0.9
                res=check_sufficiency_substring(retrieved_chunks=rr_chunks, gold_evidence_sentences=gold, **kwargs)
                y=1 if bool(getattr(res,"sufficient",res)) else 0

                out.append({
                    "qid":int(j),
                    "cs_ret":float(cs_ret),
                    "raw_max_top3":float(raw_max_top3),
                    "mean_top3":float(mean_top3) if not math.isnan(mean_top3) else float("nan"),
                    "std_top3":float(std_top3) if not math.isnan(std_top3) else float("nan"),
                    "gap12":float(gap12) if not math.isnan(gap12) else float("nan"),
                    "entropy_top5":float(entropy_top5) if not math.isnan(entropy_top5) else float("nan"),
                    "ce_bi_corr":float(ce_bi_corr) if not math.isnan(ce_bi_corr) else float("nan"),
                    "bi_top1":float(bi_scores[0]) if bi_scores else float("nan"),
                    "bi_mean_top5":float(np.mean(bi_scores[:5])) if len(bi_scores)>=5 else float("nan"),
                    "y":int(y),
                })
                if args.max_queries and len(out)>=args.max_queries:
                    break
            return out

        train_rows=run_split("train",train_idx)
        val_rows=run_split("val",val_idx)
        test_rows=run_split("test",test_idx)

        save_parquet(args.out_base,"train",train_rows)
        save_parquet(args.out_base,"val",val_rows)
        save_parquet(args.out_base,"test",test_rows)

        all_rows=train_rows+val_rows+test_rows
        stats["count"]=len(all_rows)
        stats["pos_rate"]=float(np.mean([r["y"] for r in all_rows])) if all_rows else 0.0

    os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
    with open(args.stats_path,"w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)
    print("stats", args.stats_path)

if __name__=="__main__":
    main()
