import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from sentence_transformers import CrossEncoder

def sigmoid_stable(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def truncate_passage(text: str, max_chars: int = 2000) -> str:
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

@dataclass
class RerankResult:
    doc_indices: List[int]
    doc_scores: List[float]
    max_top3_score: float
    cs_ret: float
    temperature: float

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def pick_batch_size(device: str) -> int:
    return 16 if device == "cuda" else 8

def rerank_topk(
    query: str,
    candidates: List[Tuple[int, str]],
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_chars: int = 2000,
    temperature: float = 1.0,
) -> RerankResult:
    if device is None:
        device = pick_device()
    if batch_size is None:
        batch_size = pick_batch_size(device)
    if not candidates:
        return RerankResult([], [], float("-inf"), 0.0, float(temperature))
    ce = CrossEncoder(cross_encoder_model, device=device)
    idxs = []
    pairs = []
    for i, txt in candidates:
        idxs.append(int(i))
        pairs.append((query, truncate_passage(txt, max_chars=max_chars)))
    if device == "cuda":
        torch.cuda.empty_cache()
    scores = ce.predict(pairs, batch_size=batch_size)
    scores = [float(s) for s in scores]
    order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
    top_order = order[:top_n]
    top_idxs = [idxs[j] for j in top_order]
    top_scores = [scores[j] for j in top_order]
    max_top3 = scores[order[0]]
    cs_ret = sigmoid_stable(max_top3 / float(temperature))
    return RerankResult(top_idxs, top_scores, max_top3, cs_ret, float(temperature))
