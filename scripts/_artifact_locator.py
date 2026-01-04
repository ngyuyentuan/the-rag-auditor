import os


def locate_scifact_artifacts(repo_root, extra_roots):
    out = {}
    repo_root = os.path.abspath(repo_root)
    candidates = {
        "queries_jsonl": os.path.join(repo_root, "data", "beir_scifact", "scifact", "queries.jsonl"),
        "corpus_jsonl": os.path.join(repo_root, "data", "beir_scifact", "scifact", "corpus.jsonl"),
        "index_path": os.path.join(repo_root, "artifacts", "scifact.faiss.index"),
        "meta_path": os.path.join(repo_root, "artifacts", "scifact_faiss_meta.json"),
    }
    for k, p in candidates.items():
        if os.path.exists(p):
            out[k] = p
    if len(out) == 4:
        return out

    roots = [repo_root] + [r for r in (extra_roots or []) if r]
    max_depth = 8
    for root in roots:
        if len(out) == 4:
            break
        if not os.path.exists(root):
            continue
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= max_depth:
                dirnames[:] = []
            low = dirpath.lower()
            for name in filenames:
                lname = name.lower()
                path = os.path.join(dirpath, name)
                if "queries_jsonl" not in out and lname == "queries.jsonl" and "scifact" in low:
                    out["queries_jsonl"] = path
                if "corpus_jsonl" not in out and lname == "corpus.jsonl" and "scifact" in low:
                    out["corpus_jsonl"] = path
                if "index_path" not in out and lname.endswith(".faiss.index") and "scifact" in lname:
                    out["index_path"] = path
                if "meta_path" not in out and "faiss_meta" in lname and "scifact" in lname:
                    out["meta_path"] = path
                if len(out) == 4:
                    return out
    return out
