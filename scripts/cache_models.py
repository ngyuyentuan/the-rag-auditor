import os

CACHE_DIR = os.path.abspath("artifacts/hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

def main():
    print("Cache dir:", CACHE_DIR)

    # Embedding (Stage 1)
    from sentence_transformers import SentenceTransformer
    emb_id = "sentence-transformers/all-MiniLM-L6-v2"
    SentenceTransformer(emb_id).encode(["hello", "xin chào"])
    print("Embedding cached:", emb_id)

    # Cross-encoder reranker (Stage 1)
    from sentence_transformers import CrossEncoder
    ce_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CrossEncoder(ce_id).predict([("q", "d")])
    print("Cross-encoder cached:", ce_id)

    # NLI Judge (Stage 2) — public, không cần HF token
    nli_id = "cross-encoder/nli-deberta-v3-base"
    CrossEncoder(nli_id).predict([("A cat sits on a mat.", "An animal is on a mat.")])
    print("NLI cached:", nli_id)

    print("DONE")

if __name__ == "__main__":
    main()
