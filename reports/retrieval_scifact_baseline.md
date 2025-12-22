# Day7-SCIENCE — SciFact (BEIR) Bi-encoder + FAISS

SciFact is a scientific claim verification dataset with a document corpus and claim queries (IR-style evaluation).

## Setup
- Dataset: `BEIR/SciFact` split=`test`
- Embedding model: `sentence-transformers/allenai-specter`
- Device: `cpu`
- Docs: `5183` | Queries: `300`
- Elapsed: `4373.7s`

## Results
| k | Success@k | Recall@k (macro) |
|---:|---:|---:|
| 1 | 0.3233 | 0.3056 |
| 5 | 0.5833 | 0.5708 |
| 10 | 0.6567 | 0.6373 |
| 20 | 0.7333 | 0.7259 |
| 50 | 0.8133 | 0.8099 |

## Notes
- FAISS IndexFlatIP + L2-normalized embeddings = cosine similarity via dot product.
- Success@k = %queries with at least one relevant doc in top-k.
- Recall@k (macro) = average fraction of relevant docs retrieved in top-k.
