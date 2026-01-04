import argparse
import os
import shutil
from pathlib import Path


def find_files(roots, targets, max_depth=8):
    found = {}
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= max_depth:
                dirnames[:] = []
            for name in filenames:
                if name in targets and name not in found:
                    found[name] = os.path.join(dirpath, name)
            if len(found) == len(targets):
                return found
    return found


def copy_if_found(found, name, dest):
    src = found.get(name)
    if not src:
        return False
    dest = Path(dest)
    if os.path.abspath(src) == os.path.abspath(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search_roots", nargs="+", default=["/mnt/c/Users/nguye/Downloads", "/mnt/c/Users/nguye/Documents"])
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--allow_download", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.out_dir
    artifacts_dir = repo_root / "artifacts"

    scifact_targets = [
        "queries.jsonl",
        "corpus.jsonl",
        "scifact.faiss.index",
        "scifact_faiss_meta.json",
        "qrels.tsv",
        "test.tsv",
        "dev.tsv",
        "train.tsv",
    ]
    fever_targets = [
        "dev.jsonl",
        "fever_corpus.jsonl",
        "corpus.jsonl",
        "fever.faiss.index",
        "fever_faiss_meta.json",
    ]

    roots = [str(repo_root)] + args.search_roots
    scifact_found = find_files(roots, scifact_targets)
    fever_found = find_files(roots, fever_targets)

    scifact_ok = {
        "queries_jsonl": copy_if_found(scifact_found, "queries.jsonl", out_dir / "beir_scifact" / "scifact" / "queries.jsonl"),
        "corpus_jsonl": copy_if_found(scifact_found, "corpus.jsonl", out_dir / "beir_scifact" / "scifact" / "corpus.jsonl"),
        "index": copy_if_found(scifact_found, "scifact.faiss.index", artifacts_dir / "scifact.faiss.index"),
        "meta": copy_if_found(scifact_found, "scifact_faiss_meta.json", artifacts_dir / "scifact_faiss_meta.json"),
        "qrels": False,
    }
    for qrels_name in ["qrels.tsv", "test.tsv", "dev.tsv", "train.tsv"]:
        if copy_if_found(scifact_found, qrels_name, out_dir / "beir_scifact" / "qrels" / qrels_name):
            scifact_ok["qrels"] = True
            break

    fever_ok = {
        "dev": copy_if_found(fever_found, "dev.jsonl", out_dir / "fever" / "dev.jsonl"),
        "corpus": False,
        "index": copy_if_found(fever_found, "fever.faiss.index", artifacts_dir / "fever.faiss.index"),
        "meta": copy_if_found(fever_found, "fever_faiss_meta.json", artifacts_dir / "fever_faiss_meta.json"),
    }
    if copy_if_found(fever_found, "fever_corpus.jsonl", artifacts_dir / "fever_corpus.jsonl"):
        fever_ok["corpus"] = True
    elif copy_if_found(fever_found, "corpus.jsonl", artifacts_dir / "fever_corpus.jsonl"):
        fever_ok["corpus"] = True

    missing = []
    for k, v in scifact_ok.items():
        if not v:
            missing.append(f"scifact:{k}")
    for k, v in fever_ok.items():
        if not v:
            missing.append(f"fever:{k}")

    print("Scifact artifacts:", scifact_ok)
    print("Fever artifacts:", fever_ok)
    if missing:
        print("Missing:", ", ".join(missing))
        if not args.allow_download:
            print("Downloads disabled; provide missing files under search roots.")


if __name__ == "__main__":
    main()
