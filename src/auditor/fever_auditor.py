import sys
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
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


class FeverOptimizedAuditor:
    VERSION = "3.0.0"
    
    SUPPORTS_BOOST_WORDS = [
        'is a', 'was a', 'is an', 'was an', 'are', 'were',
        'born', 'died', 'released', 'founded', 'made',
        'played', 'starred', 'appeared', 'directed',
        'published', 'created', 'produced', 'won',
    ]
    
    REFUTES_PATTERNS = [
        (r'\bnot\b', 0.15),
        (r"\bn't\b", 0.15),
        (r'\bnever\b', 0.20),
        (r'\bno\b', 0.10),
        (r'\bonly\b.*\b(american|british|english|german|french|italian|spanish)\b', 0.25),
        (r'\bexclusively\b', 0.20),
        (r'\bincorrect\b', 0.25),
        (r'\bwrong\b', 0.20),
        (r'\bfalse\b', 0.25),
        (r'\buntrue\b', 0.25),
        (r'\bdid not\b', 0.20),
        (r'\bwas not\b', 0.15),
        (r'\bis not\b', 0.15),
        (r'\bcannot\b', 0.15),
        (r'\brefuse\b', 0.15),
        (r'\bdeny\b', 0.15),
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
        self._nli_loaded = False
        self._cache = {}
        self._refutes_re = [(re.compile(p, re.IGNORECASE), w) for p, w in self.REFUTES_PATTERNS]
        
        self.T_ENTAIL_HIGH = 0.65
        self.T_CONTRA_HIGH = 0.60
        self.T_NEUTRAL_HIGH = 0.70
        self.T_ENTAIL_MED = 0.45
        self.T_CONTRA_MED = 0.40
    
    def _load_nli(self):
        if self._nli_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._nli_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.sha256(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _calculate_refutes_boost(self, claim: str) -> float:
        boost = 0.0
        for pattern, weight in self._refutes_re:
            if pattern.search(claim):
                boost += weight
        return min(boost, 0.40)
    
    def _calculate_supports_boost(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        boost = 0.0
        for word in self.SUPPORTS_BOOST_WORDS:
            if word in claim_lower and word in evidence_lower:
                boost += 0.05
        return min(boost, 0.25)
    
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
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[str, str, float]:
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        refutes_boost = self._calculate_refutes_boost(claim)
        supports_boost = self._calculate_supports_boost(claim, evidence)
        
        p_contra_adj = min(p_contra + refutes_boost, 1.0)
        p_entail_adj = min(p_entail + supports_boost, 1.0)
        
        sig = f"E={p_entail:.2f}({p_entail_adj:.2f}),C={p_contra:.2f}({p_contra_adj:.2f}),N={p_neutral:.2f}"
        
        if p_contra_adj > self.T_CONTRA_HIGH:
            return "REFUTES", f"high_contra ({sig})", p_contra_adj
        
        if p_entail_adj > self.T_ENTAIL_HIGH:
            return "SUPPORTS", f"high_entail ({sig})", p_entail_adj
        
        if p_neutral > self.T_NEUTRAL_HIGH:
            return "NEI", f"high_neutral ({sig})", p_neutral
        
        if p_contra_adj > self.T_CONTRA_MED and p_contra_adj > p_entail_adj:
            return "REFUTES", f"med_contra ({sig})", p_contra_adj
        
        if p_entail_adj > self.T_ENTAIL_MED and p_entail_adj > p_contra_adj:
            return "SUPPORTS", f"med_entail ({sig})", p_entail_adj
        
        if p_neutral > 0.50:
            return "NEI", f"med_neutral ({sig})", p_neutral
        
        if p_entail_adj >= p_contra_adj and p_entail_adj >= p_neutral:
            return "SUPPORTS", f"argmax ({sig})", p_entail_adj
        elif p_contra_adj > p_entail_adj and p_contra_adj >= p_neutral:
            return "REFUTES", f"argmax ({sig})", p_contra_adj
        else:
            return "NEI", f"argmax ({sig})", p_neutral
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        start = time.time()
        
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return AuditResult(
                claim=cached.claim, verdict=cached.verdict,
                confidence=cached.confidence,
                calibrated_confidence=cached.calibrated_confidence,
                explanation=cached.explanation, signals=cached.signals,
                latency_ms=0.0
            )
        
        probs = self._nli_inference(claim, evidence)
        verdict, reason, confidence = self._decide(probs, claim, evidence)
        latency = (time.time() - start) * 1000
        
        result = AuditResult(
            claim=claim, verdict=verdict, confidence=confidence,
            calibrated_confidence=min(confidence * 1.05, 1.0),
            explanation=reason, signals={"probs": probs},
            latency_ms=latency
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result


if __name__ == "__main__":
    print("FEVER-OPTIMIZED AUDITOR TEST")
    auditor = FeverOptimizedAuditor(device="cpu")
    tests = [
        ("Telemundo is an English-language network.", "Spanish-language network", "REFUTES"),
        ("Damon Albarn's debut album was released in 2011.", "Released in 2014", "REFUTES"),
        ("Fox 2000 Pictures released Soul Food.", "Fox 2000 Pictures produced", "SUPPORTS"),
        ("Alexandra Daddario is Canadian.", "American actress", "REFUTES"),
        ("Tim Roth is an English actor.", "English actor", "SUPPORTS"),
    ]
    correct = 0
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        correct += 1 if match else 0
        print(f"{'Y' if match else 'N'} [{expected:8}->{result.verdict:8}] {result.explanation[:50]}")
    print(f"\nAccuracy: {correct}/{len(tests)}")
