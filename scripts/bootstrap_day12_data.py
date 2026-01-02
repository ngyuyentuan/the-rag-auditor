import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
FEVER_URLS = []
if os.environ.get("FEVER_DEV_URL"):
    FEVER_URLS.append(os.environ.get("FEVER_DEV_URL"))
FEVER_URLS.extend([
    "https://fever.ai/download/fever-data/fever-data.zip",
    "https://fever.ai/download/fever-data/fever-splits.zip",
    "https://fever.ai/download/fever-data/fever-dev.jsonl",
])


def l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype(np.float32)


def pick_device():
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def download_zip(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        urlretrieve(url, tmp_path)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(out_dir)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return out_dir


def parse_demo_in_paths(report_path: Path):
    if not report_path.exists():
        return None, None
    lines = report_path.read_text(encoding="utf-8").splitlines()
    paths = []
    for line in lines:
        s = line.strip()
        if s.startswith("- in_path:"):
            after = s.split(":", 1)[1].strip()
            after = after.strip(chr(96)).strip()
            if after:
                paths.append(after)
    if len(paths) >= 2:
        return paths[0], paths[1]
    return None, None


def build_scifact_index(corpus_path: Path, index_path: Path, meta_path: Path, emb_model: str):
    doc_ids = []
    doc_texts = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("_id", row.get("doc_id", "")))
            title = row.get("title", "") or ""
            text = row.get("text", "") or ""
            doc_ids.append(did)
            doc_texts.append((title + "\n" + text).strip())
    device = pick_device()
    model = SentenceTransformer(emb_model, device=device)
    try:
        embs = model.encode(doc_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    except Exception:
        device = "cpu"
        model = SentenceTransformer(emb_model, device=device)
        embs = model.encode(doc_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    embs = l2norm(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta = {
        "emb_model": emb_model,
        "n_docs": len(doc_ids),
        "corpus_path": str(corpus_path),
        "index_path": str(index_path),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_scifact(
    data_root: Path,
    artifacts_root: Path,
    emb_model: str,
    queries_jsonl: str | None,
    corpus_jsonl: str | None,
    index_path: str | None,
    meta_path: str | None,
) -> list:
    required = [
        Path(corpus_jsonl) if corpus_jsonl else data_root / "beir_scifact" / "scifact" / "corpus.jsonl",
        Path(queries_jsonl) if queries_jsonl else data_root / "beir_scifact" / "scifact" / "queries.jsonl",
        Path(index_path) if index_path else artifacts_root / "scifact.faiss.index",
        Path(meta_path) if meta_path else artifacts_root / "scifact_faiss_meta.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        download_zip(SCIFACT_URL, data_root / "beir_scifact")
    index_out = Path(index_path) if index_path else artifacts_root / "scifact.faiss.index"
    meta_out = Path(meta_path) if meta_path else artifacts_root / "scifact_faiss_meta.json"
    if not index_out.exists() or not meta_out.exists():
        corpus_path = Path(corpus_jsonl) if corpus_jsonl else data_root / "beir_scifact" / "scifact" / "corpus.jsonl"
        build_scifact_index(
            corpus_path,
            index_out,
            meta_out,
            emb_model,
        )
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing SciFact artifacts: " + ", ".join(str(p) for p in missing))
    return [str(p) for p in required]


def read_fever_dev(dev_path: Path):
    claims = []
    evidence = []
    with dev_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            claims.append(str(row.get("claim", "")))
            evidence.append(row.get("evidence"))
    return claims, evidence


def extract_fever_pages(evidence_field):
    pages = {}
    if not isinstance(evidence_field, list):
        return pages
    for group in evidence_field:
        if isinstance(group, list) and group and isinstance(group[0], list):
            for item in group:
                if isinstance(item, list) and len(item) >= 3:
                    page = item[2]
                    sent = item[3] if len(item) > 3 else None
                    if page is not None:
                        pages.setdefault(str(page), []).append(str(sent) if sent is not None else "")
        elif isinstance(group, list) and len(group) >= 3:
            page = group[2]
            sent = group[3] if len(group) > 3 else None
            if page is not None:
                pages.setdefault(str(page), []).append(str(sent) if sent is not None else "")
    return pages


def build_fever_corpus(dev_path: Path, corpus_path: Path):
    _, evidence = read_fever_dev(dev_path)
    page_to_sents = {}
    for ev in evidence:
        pages = extract_fever_pages(ev)
        for page, sents in pages.items():
            page_to_sents.setdefault(page, []).extend(sents)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as f:
        for page, sents in page_to_sents.items():
            text = "\n".join([s for s in sents if s]).strip()
            f.write(json.dumps({"doc_id": page, "text": text}, ensure_ascii=False) + "\n")


def build_fever_index(corpus_path: Path, index_path: Path, meta_path: Path, emb_model: str):
    doc_ids = []
    doc_texts = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_ids.append(str(row.get("doc_id")))
            doc_texts.append(str(row.get("text", "")))
    device = pick_device()
    model = SentenceTransformer(emb_model, device=device)
    try:
        embs = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    except Exception:
        device = "cpu"
        model = SentenceTransformer(emb_model, device=device)
        embs = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    embs = l2norm(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta = {
        "emb_model": emb_model,
        "n_docs": len(doc_ids),
        "corpus_path": str(corpus_path),
        "index_path": str(index_path),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_fever(data_root: Path, artifacts_root: Path, emb_model: str, dev_jsonl: str | None) -> list:
    dev_path = Path(dev_jsonl) if dev_jsonl else data_root / "fever_splits" / "dev.jsonl"
    corpus_path = artifacts_root / "fever_corpus.jsonl"
    index_path = artifacts_root / "fever.faiss.index"
    meta_path = artifacts_root / "fever_faiss_meta.json"
    required = [dev_path, corpus_path, index_path, meta_path]
    if not dev_path.exists():
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        last_err = None
        for url in FEVER_URLS:
            try:
                if url.endswith(".jsonl"):
                    urlretrieve(url, dev_path)
                    break
                tmp_dir = Path(tempfile.mkdtemp())
                download_zip(url, tmp_dir)
                candidates = list(tmp_dir.rglob("dev.jsonl"))
                if candidates:
                    dev_path.write_text(candidates[0].read_text(encoding="utf-8"), encoding="utf-8")
                    break
            except Exception as e:
                last_err = e
        if not dev_path.exists():
            raise FileNotFoundError("missing FEVER dev.jsonl; download failed: " + (str(last_err) if last_err else "unknown error"))
    if not corpus_path.exists():
        build_fever_corpus(dev_path, corpus_path)
    if not index_path.exists() or not meta_path.exists():
        build_fever_index(corpus_path, index_path, meta_path, emb_model)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing FEVER artifacts: " + ", ".join(str(p) for p in missing))
    return [str(p) for p in required]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["scifact", "fever"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--artifacts_root", default="artifacts")
    ap.add_argument("--scifact_emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--fever_emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--scifact_in_path", default=None)
    ap.add_argument("--fever_in_path", default=None)
    ap.add_argument("--scifact_queries_jsonl", default=None)
    ap.add_argument("--scifact_corpus_jsonl", default=None)
    ap.add_argument("--scifact_index_path", default=None)
    ap.add_argument("--scifact_meta_path", default=None)
    ap.add_argument("--fever_dev_jsonl", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    artifacts_root = Path(args.artifacts_root)

    scifact_from_demo, fever_from_demo = parse_demo_in_paths(Path("reports/demo_stage1.md"))
    scifact_in = args.scifact_in_path or scifact_from_demo
    fever_in = args.fever_in_path or fever_from_demo

    produced = []
    if scifact_in:
        print("scifact_in_path:", scifact_in)
    if fever_in:
        print("fever_in_path:", fever_in)

    if "scifact" in args.tracks:
        try:
            produced.extend(
                ensure_scifact(
                    data_root,
                    artifacts_root,
                    args.scifact_emb_model,
                    args.scifact_queries_jsonl,
                    args.scifact_corpus_jsonl,
                    args.scifact_index_path,
                    args.scifact_meta_path,
                )
            )
        except FileNotFoundError as e:
            print(str(e))

    if "fever" in args.tracks:
        try:
            produced.extend(ensure_fever(data_root, artifacts_root, args.fever_emb_model, args.fever_dev_jsonl))
        except FileNotFoundError as e:
            print(str(e))
            print("fever stage2 artifacts missing; stage1-only mode available")

    if produced:
        print("ready paths:")
        for p in produced:
            print(p)


if __name__ == "__main__":
    main()
