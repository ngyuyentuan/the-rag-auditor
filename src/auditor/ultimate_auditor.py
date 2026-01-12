import sys
import re
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
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
    calibrated_confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class UltimateAuditor:
    
    SUPPORT_WORDS = [
        'confirms', 'confirm', 'proves', 'prove', 'shows', 'show',
        'demonstrates', 'establishes', 'verifies', 'supports', 'validates',
        'evidence shows', 'research shows', 'studies show', 'data shows',
        'is true', 'is correct', 'is accurate', 'is valid', 'is right',
        'indeed', 'certainly', 'definitely', 'clearly', 'obviously',
    ]
    
    NEGATION_WORDS = [
        'not', 'no', 'never', 'none', "n't", 'neither', 'nor', 'nothing',
        'nobody', 'nowhere', 'hardly', 'barely', 'without', 'lack',
        'fail', 'deny', 'denied', 'refuse', 'reject', 'rejected',
        'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
        "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    ]
    
    CONTRADICTION_WORDS = [
        'false', 'incorrect', 'wrong', 'untrue', 'inaccurate', 'mistaken',
        'disprove', 'refute', 'contradict', 'debunk', 'myth', 'hoax',
        'no evidence', 'no proof', 'no link', 'no connection',
    ]
    
    NEI_WORDS = [
        'unclear', 'uncertain', 'unknown', 'debated', 'controversial',
        'may', 'might', 'could', 'possibly', 'perhaps', 'maybe',
        'mixed', 'inconclusive', 'insufficient', 'limited',
        'depends', 'varies', 'some', 'partially',
        'appears', 'seems', 'likely', 'unlikely', 'probable',
        'suggests', 'indicates', 'potential', 'potentially',
        'debatable', 'questionable', 'speculative', 'tentative',
        'complex', 'complicated', 'nuanced', 'multifaceted',
        'further research', 'more studies', 'remains to be seen',
        'however', 'although', 'but', 'yet', 'while',
    ]
    
    ANTONYM_PAIRS = [
        ('safe', 'dangerous'), ('safe', 'unsafe'), ('healthy', 'unhealthy'),
        ('good', 'bad'), ('true', 'false'), ('correct', 'incorrect'),
        ('beneficial', 'harmful'), ('help', 'harm'), ('positive', 'negative'),
        ('effective', 'ineffective'), ('real', 'fake'),
    ]
    
    def __init__(
        self,
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device: str = "auto",
        enable_cache: bool = True,
    ):
        self.nli_model_name = nli_model
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
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._nli_loaded = True
    
    def _load_embedder(self):
        if self._embedder_loaded:
            return
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self._embedder_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _count_words(self, text: str, words: List[str]) -> int:
        text_lower = text.lower()
        return sum(1 for w in words if w in text_lower)
    
    def _check_antonyms(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        score = 0.0
        for w1, w2 in self.ANTONYM_PAIRS:
            if (w1 in claim_lower and w2 in evidence_lower) or \
               (w2 in claim_lower and w1 in evidence_lower):
                score += 0.25
        return min(score, 0.5)
    
    def _semantic_similarity(self, claim: str, evidence: str) -> float:
        if not self._embedder_loaded:
            self._load_embedder()
        embeddings = self.embedder.encode([claim, evidence])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return float(similarity)
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._nli_loaded:
            self._load_nli()
        evidence = evidence[:1500]
        claim = claim[:500]
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
    
    def _decide(self, probs: Dict[str, float], evidence: str, claim: str, similarity: float) -> Tuple[str, str, float]:
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        support_words = self._count_words(evidence, self.SUPPORT_WORDS)
        negation_words = self._count_words(evidence, self.NEGATION_WORDS)
        contra_words = self._count_words(evidence, self.CONTRADICTION_WORDS)
        nei_words = self._count_words(evidence, self.NEI_WORDS)
        antonym_score = self._check_antonyms(claim, evidence)
        
        sig = f"E={p_entail:.2f},C={p_contra:.2f},N={p_neutral:.2f},sim={similarity:.2f}"
        
        if similarity < 0.45 and nei_words >= 1:
            return "NEI", f"low_sim_nei ({sig})", min(p_neutral + 0.2, 1.0)
        
        if nei_words >= 3:
            return "NEI", f"multi_nei_words ({sig})", min(p_neutral + 0.15, 1.0)
        
        if p_entail > 0.65:
            return "SUPPORTS", f"strong_entail ({sig})", min(p_entail, 1.0)
        
        if p_contra > 0.55:
            return "REFUTES", f"strong_contra ({sig})", min(p_contra, 1.0)
        
        if p_neutral > 0.50 and p_neutral > p_entail and p_neutral > p_contra:
            return "NEI", f"strong_neutral ({sig})", min(p_neutral, 1.0)
        
        if support_words >= 2 and p_entail > 0.40:
            return "SUPPORTS", f"pattern_support ({sig})", min(p_entail + 0.1, 1.0)
        
        if (contra_words >= 1 or antonym_score >= 0.25) and p_contra > 0.35:
            return "REFUTES", f"pattern_contra ({sig})", min(p_contra + 0.15, 1.0)
        
        if negation_words >= 2 and p_contra > 0.30:
            return "REFUTES", f"negation_contra ({sig})", min(p_contra + 0.1, 1.0)
        
        if nei_words >= 2 and p_neutral > 0.28:
            return "NEI", f"pattern_nei ({sig})", min(p_neutral + 0.15, 1.0)
        
        if nei_words >= 1 and p_neutral > 0.40:
            return "NEI", f"single_nei ({sig})", min(p_neutral + 0.1, 1.0)
        
        margin = 0.12
        
        if p_entail > p_contra + margin and p_entail > p_neutral + margin:
            if p_entail > 0.40:
                return "SUPPORTS", f"margin_entail ({sig})", min(p_entail, 1.0)
        
        if p_contra > p_entail + margin and p_contra > p_neutral + margin:
            if p_contra > 0.30:
                return "REFUTES", f"margin_contra ({sig})", min(p_contra, 1.0)
        
        if p_neutral > p_entail + 0.05 and p_neutral > p_contra + 0.05:
            return "NEI", f"margin_neutral ({sig})", min(p_neutral, 1.0)
        
        if support_words >= 1 and p_entail > p_contra:
            return "SUPPORTS", f"mild_support ({sig})", min(p_entail, 1.0)
        
        if p_entail >= p_contra and p_entail >= p_neutral:
            return "SUPPORTS", f"argmax ({sig})", min(p_entail, 1.0)
        elif p_contra > p_entail and p_contra >= p_neutral:
            return "REFUTES", f"argmax ({sig})", min(p_contra, 1.0)
        else:
            return "NEI", f"argmax ({sig})", min(p_neutral, 1.0)
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        start = time.time()
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return AuditResult(
                claim=cached.claim,
                verdict=cached.verdict,
                confidence=cached.confidence,
                calibrated_confidence=cached.calibrated_confidence,
                explanation=cached.explanation,
                signals=cached.signals,
                latency_ms=0.0
            )
        probs = self._nli_inference(claim, evidence)
        similarity = self._semantic_similarity(claim, evidence)
        verdict, reason, confidence = self._decide(probs, evidence, claim, similarity)
        calibrated = min(confidence * 1.1, 1.0)
        latency = (time.time() - start) * 1000
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            calibrated_confidence=calibrated,
            explanation=reason,
            signals={"probs": probs, "similarity": similarity},
            latency_ms=latency,
        )
        if self.enable_cache:
            self._cache[cache_key] = result
        return result


ProductionAuditor = UltimateAuditor
BalancedAuditor = UltimateAuditor


if __name__ == "__main__":
    print("=" * 60)
    print("ULTIMATE AUDITOR TEST")
    print("=" * 60)
    auditor = UltimateAuditor(device="cpu")
    tests = [
        ("Vaccines are safe.", "Research confirms vaccines are safe and effective.", "SUPPORTS"),
        ("Exercise improves health.", "Studies show exercise improves health.", "SUPPORTS"),
        ("Vaccines cause autism.", "No evidence links vaccines to autism.", "REFUTES"),
        ("Earth is flat.", "Earth is spherical, not flat.", "REFUTES"),
        ("AI will replace humans.", "The impact of AI on jobs is debated.", "NEI"),
        ("Coffee is good for health.", "Studies show mixed results on coffee.", "NEI"),
    ]
    correct = 0
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        correct += 1 if match else 0
        print(f"{'Y' if match else 'N'} [{expected}->{result.verdict}] sim={result.signals.get('similarity', 0):.2f} | {claim[:40]}")
    print(f"\nAccuracy: {correct}/{len(tests)}")
