# Day 8 — Cross-encoder Rerank Sanity (SciFact)

## Setup
- Timestamp: `1766322453.3358252`
- Device: `cpu`
- Bi-encoder: `sentence-transformers/allenai-specter`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Candidate top_k (bi-encoder): `50`
- Rerank top_n: `5`
- Temperature τ: `1.0`

## Metrics (mean over queries)
| k | baseline success@k | baseline recall@k | baseline ndcg@k | rerank success@k | rerank recall@k | rerank ndcg@k |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3200 | 0.2950 | 0.3200 | 0.6600 | 0.6190 | 0.6600 |
| 5 | 0.5800 | 0.5680 | 0.4596 | 0.7600 | 0.7470 | 0.7063 |
| 10 | 0.6400 | 0.6180 | 0.4766 | 0.8200 | 0.8020 | 0.7246 |
| 20 | 0.7600 | 0.7460 | 0.5097 | 0.8200 | 0.8060 | 0.7261 |

## CS_ret
- Definition: $CS_{ret}=\sigma(\max_{\text{top3}} score / \tau)$. Since scores are sorted desc, this equals $\sigma(score_{top1}/\tau)$.
- Note: Cross-encoder scores are **uncalibrated**. Day 9/10 will calibrate (tune τ / thresholds).

## Important note
- Rerank is **upper-bounded** by whether the gold doc appears in the bi-encoder top_k candidates.

## CS_ret distribution (sanity)
- mean=0.7944, std=0.3342, median=0.9954
