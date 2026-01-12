"""
RAG Auditor Multilingual - Cross-lingual Claim Verification

Features:
1. Multilingual NLI using XLM-RoBERTa or mDeBERTa
2. Cross-lingual semantic search using multilingual embeddings
3. Support: English, Vietnamese, Chinese, French, German, Spanish, etc.
"""
import sys
import re
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AuditResult:
    claim: str
    verdict: str
    confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class MultilingualAuditor:
    """
    Multilingual RAG Auditor supporting cross-lingual verification.
    
    Supports:
    - English knowledge + Vietnamese/Chinese/etc queries
    - Multilingual NLI (XLM-RoBERTa based)
    - Multilingual semantic embeddings for retrieval
    """
    
    # Multilingual refutation patterns (EN, VI, ZH)
    REFUTE_PATTERNS = [
        # English
        r'\bnot\b', r'\bno\b', r'\bfalse\b', r'\bnever\b', r'\bincorrect\b',
        r'\bno\s+(evidence|link|connection)\b',
        # Vietnamese
        r'\bkhông\b', r'\bchưa\b', r'\bsai\b', r'\bkhông có\b', r'\bkhông đúng\b',
        r'\bkhông phải\b', r'\bchống lại\b',
        # Chinese
        r'不是', r'没有', r'错误', r'不正确',
    ]
    
    SUPPORT_PATTERNS = [
        # English
        r'\bconfirms?\b', r'\bproves?\b', r'\bshows?\b',
        # Vietnamese
        r'\bxác nhận\b', r'\bchứng minh\b', r'\bcho thấy\b', r'\bđúng\b',
        # Chinese
        r'证明', r'确认', r'显示',
    ]
    
    NEI_PATTERNS = [
        # English
        r'\bmixed\b', r'\bdepends\b', r'\bunclear\b', r'\binconclusive\b',
        # Vietnamese
        r'\bcó thể\b', r'\btùy thuộc\b', r'\bkhông rõ\b', r'\bchưa chắc\b',
        # Chinese
        r'可能', r'取决于', r'不清楚',
    ]
    
    def __init__(
        self,
        # Use multilingual model for cross-lingual support
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto",
        enable_cache: bool = True,
    ):
        self.nli_model_name = nli_model
        self.embedding_model_name = embedding_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        
        self.nli_model = None
        self.nli_tokenizer = None
        self.embedder = None
        
        self._nli_loaded = False
        self._embedder_loaded = False
        self._cache = {}
    
    def _load_nli(self):
        if self._nli_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[Multilingual] Loading NLI: {self.nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._nli_loaded = True
        print(f"[Multilingual] NLI loaded on {self.device}")
    
    def _load_embedder(self):
        if self._embedder_loaded:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[Multilingual] Loading embedder: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        self._embedder_loaded = True
        print("[Multilingual] Embedder loaded")
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _check_patterns(self, text: str, patterns: List[str]) -> int:
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                count += 1
        return count
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        """Run multilingual NLI inference."""
        if not self._nli_loaded:
            self._load_nli()
        
        evidence = evidence[:1500] if len(evidence) > 1500 else evidence
        claim = claim[:500] if len(claim) > 500 else claim
        
        inputs = self.nli_tokenizer(
            evidence, claim,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        id2label = self.nli_model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute multilingual semantic similarity."""
        if not self._embedder_loaded:
            self._load_embedder()
        
        embeddings = self.embedder.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def _decide(self, probs: Dict[str, float], evidence: str, claim: str) -> Tuple[str, str, float]:
        """Make decision with optimized thresholds for higher accuracy."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        refute_ptn = self._check_patterns(evidence, self.REFUTE_PATTERNS)
        support_ptn = self._check_patterns(evidence, self.SUPPORT_PATTERNS)
        nei_ptn = self._check_patterns(evidence, self.NEI_PATTERNS)
        
        # Stronger pattern weights for higher accuracy
        support_score = p_entail + (support_ptn * 0.15)
        refute_score = p_contra + (refute_ptn * 0.18)
        nei_score = p_neutral + (nei_ptn * 0.12)
        
        signals = f"S={support_score:.2f}, R={refute_score:.2f}, N={nei_score:.2f}"
        
        # Priority 1: Strong NLI contradiction (catch hallucinations)
        if p_contra > 0.55:
            return "REFUTES", f"strong_contra ({signals})", min(p_contra, 1.0)
        
        # Priority 2: Strong NLI entailment
        if p_entail > 0.55:
            return "SUPPORTS", f"strong_entail ({signals})", min(p_entail, 1.0)
        
        # Priority 3: Pattern + moderate NLI
        if refute_ptn >= 1 and p_contra > 0.30:
            return "REFUTES", f"pattern_contra ({signals})", min(refute_score, 1.0)
        
        if support_ptn >= 1 and p_entail > 0.30:
            return "SUPPORTS", f"pattern_entail ({signals})", min(support_score, 1.0)
        
        # Priority 4: NEI patterns (only if strong)
        if nei_ptn >= 2 and p_neutral > 0.35:
            return "NEI", f"nei_patterns ({signals})", min(nei_score, 1.0)
        
        # Priority 5: Margin-based with tighter thresholds
        margin = 0.12
        if p_entail > p_contra + margin and p_entail > 0.35:
            return "SUPPORTS", f"entail_margin ({signals})", min(p_entail, 1.0)
        if p_contra > p_entail + margin and p_contra > 0.30:
            return "REFUTES", f"contra_margin ({signals})", min(p_contra, 1.0)
        
        # Priority 6: High neutral = NEI
        if p_neutral > 0.45 and p_neutral > max(p_entail, p_contra):
            return "NEI", f"high_neutral ({signals})", min(p_neutral, 1.0)
        
        # Final: Argmax with score boost
        if support_score >= refute_score and support_score >= nei_score:
            return "SUPPORTS", f"score ({signals})", min(support_score, 1.0)
        elif refute_score > support_score and refute_score >= nei_score:
            return "REFUTES", f"score ({signals})", min(refute_score, 1.0)
        else:
            return "NEI", f"score ({signals})", min(nei_score, 1.0)
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        """Audit claim against evidence (multilingual)."""
        start = time.time()
        
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        probs = self._nli_inference(claim, evidence)
        verdict, reason, confidence = self._decide(probs, evidence, claim)
        
        latency = (time.time() - start) * 1000
        
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=reason,
            signals={"probs": probs},
            latency_ms=latency,
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result
    
    def search_multilingual(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Multilingual semantic search."""
        if not self._embedder_loaded:
            self._load_embedder()
        
        if not documents:
            return []
        
        query_emb = self.embedder.encode([query])[0]
        doc_embs = self.embedder.encode(documents)
        
        scores = []
        for i, doc_emb in enumerate(doc_embs):
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            scores.append((documents[i], float(sim)))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


def main():
    print("="*60)
    print("MULTILINGUAL RAG AUDITOR")
    print("="*60)
    
    auditor = MultilingualAuditor(device="cpu")
    
    # Test cases: Cross-lingual
    tests = [
        # English-English
        ("Vaccines are safe.", "Research confirms vaccines are safe and effective.", "SUPPORTS"),
        ("Earth is flat.", "Scientific evidence proves Earth is a sphere.", "REFUTES"),
        # Vietnamese query + English evidence  
        ("Vắc-xin an toàn.", "Research confirms vaccines are safe and effective.", "SUPPORTS"),
        ("Trái đất phẳng.", "Scientific evidence proves Earth is a sphere.", "REFUTES"),
        # Mixed
        ("Covid có nguồn gốc từ phòng thí nghiệm.", "The origin of COVID-19 remains unclear and debated.", "NEI"),
    ]
    
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = "✓" if result.verdict == expected else "✗"
        print(f"{match} [{expected:8}→{result.verdict:8}] {claim[:40]}")
    
    print("\nMultilingual search test:")
    docs = [
        "Vaccines are safe and effective.",
        "Earth is round, not flat.",
        "Coffee has mixed health effects.",
    ]
    results = auditor.search_multilingual("Vắc-xin có an toàn không?", docs)
    for doc, score in results:
        print(f"  {score:.2f}: {doc[:50]}")


if __name__ == "__main__":
    main()
