import sys
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import torch
import numpy as np

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


class EnsembleAuditor:
    VERSION = "4.0.0"
    
    REFUTES_CLAIM_PATTERNS = [
        (r'\bnot\s+(?:a|an)?\s*\w+', 0.15),
        (r"\bn't\b", 0.15),
        (r'\bonly\s+\w+', 0.20),
        (r'\bexclusively\b', 0.20),
        (r'\bjust\s+(?:a|an)?\s*\w+', 0.10),
        (r'\bincorrect\b', 0.25),
        (r'\bwrong\b', 0.20),
        (r'\bfalse\b', 0.25),
    ]
    
    NATIONALITY_PAIRS = [
        ('american', 'british'), ('american', 'english'), ('american', 'german'),
        ('american', 'french'), ('american', 'italian'), ('american', 'spanish'),
        ('american', 'japanese'), ('american', 'chinese'), ('american', 'korean'),
        ('american', 'australian'), ('american', 'canadian'), ('american', 'indian'),
        ('british', 'american'), ('british', 'german'), ('british', 'french'),
        ('english', 'spanish'), ('english', 'german'), ('english', 'french'),
    ]
    
    YEAR_RE = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    
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
        self._refutes_re = [(re.compile(p, re.IGNORECASE), w) for p, w in self.REFUTES_CLAIM_PATTERNS]
    
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
    
    def _claim_refutes_boost(self, claim: str) -> float:
        boost = 0.0
        for pattern, weight in self._refutes_re:
            if pattern.search(claim):
                boost += weight
        return min(boost, 0.35)
    
    def _nationality_contradiction(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        for nat1, nat2 in self.NATIONALITY_PAIRS:
            if nat1 in claim_lower and nat2 in evidence_lower:
                if 'only' in claim_lower or 'exclusively' in claim_lower:
                    return 0.35
                return 0.25
        return 0.0
    
    def _year_contradiction(self, claim: str, evidence: str) -> float:
        claim_years = set(self.YEAR_RE.findall(claim))
        evidence_years = set(self.YEAR_RE.findall(evidence))
        if claim_years and evidence_years and claim_years.isdisjoint(evidence_years):
            return 0.30
        return 0.0
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._nli_loaded:
            self._load_nli()
        inputs = self.nli_tokenizer(
            evidence[:1500], claim[:500],
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        id2label = self.nli_model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def _ensemble_decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[str, str, float]:
        p_e = probs.get('entailment', 0)
        p_c = probs.get('contradiction', 0)
        p_n = probs.get('neutral', 0)
        
        refutes_boost = self._claim_refutes_boost(claim)
        nat_boost = self._nationality_contradiction(claim, evidence)
        year_boost = self._year_contradiction(claim, evidence)
        
        total_refutes_boost = refutes_boost + nat_boost + year_boost
        
        p_c_adj = min(p_c + total_refutes_boost, 1.0)
        
        sig = f"E={p_e:.2f},C={p_c:.2f}(+{total_refutes_boost:.2f}={p_c_adj:.2f}),N={p_n:.2f}"
        
        has_strong_pattern = total_refutes_boost > 0.25
        
        if p_c_adj > 0.55 or (has_strong_pattern and p_c_adj > 0.40):
            return "REFUTES", f"refutes ({sig})", p_c_adj
        
        if p_e > 0.60:
            return "SUPPORTS", f"supports ({sig})", p_e
        
        if p_n > 0.65:
            return "NEI", f"nei ({sig})", p_n
        
        if p_c_adj > 0.40 and p_c_adj > p_e:
            return "REFUTES", f"med_refutes ({sig})", p_c_adj
        
        if p_e > 0.40 and p_e > p_c_adj:
            return "SUPPORTS", f"med_supports ({sig})", p_e
        
        if p_n > 0.45:
            return "NEI", f"med_nei ({sig})", p_n
        
        scores = [(p_e, "SUPPORTS"), (p_c_adj, "REFUTES"), (p_n, "NEI")]
        best = max(scores, key=lambda x: x[0])
        return best[1], f"argmax ({sig})", best[0]
    
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
        verdict, reason, confidence = self._ensemble_decide(probs, claim, evidence)
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
    print("ENSEMBLE AUDITOR TEST")
    auditor = EnsembleAuditor(device="cpu")
    tests = [
        ("Telemundo is an English-language network.", "Spanish-language network", "REFUTES"),
        ("Alexandra Daddario is Canadian.", "American actress", "REFUTES"),
        ("Savages was exclusively a German film.", "American crime thriller", "REFUTES"),
        ("Tim Roth is an English actor.", "English actor", "SUPPORTS"),
    ]
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        print(f"{'Y' if match else 'N'} [{expected}->{result.verdict}] {result.explanation[:60]}")
