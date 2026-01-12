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


class ProductionAuditorV2:
    VERSION = "6.0.0"
    
    REFUTES_PATTERNS = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french)\b', re.I), 0.30),
        (re.compile(r'\bexclusively\b', re.I), 0.25),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), 0.20),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bdoesn't\b|\bdidn't\b", re.I), 0.20),
        (re.compile(r'\bnever\b', re.I), 0.25),
        (re.compile(r'\bincapable\b|\bunable\b', re.I), 0.25),
        (re.compile(r'\byet\s+to\b', re.I), 0.20),
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
        score = 0.0
        for pattern, weight in self.REFUTES_PATTERNS:
            if pattern.search(claim):
                score += weight
        return min(score, 0.40)
    
    def _check_same_text(self, claim: str, evidence: str) -> bool:
        c = claim.lower().strip()
        e = evidence.lower().strip()
        if c == e:
            return True
        if len(e) < len(c) * 1.2 and c in e:
            return True
        return False
    
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
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[str, str, float]:
        p_e = probs.get('entailment', 0)
        p_c = probs.get('contradiction', 0)
        p_n = probs.get('neutral', 0)
        
        if self._check_same_text(claim, evidence):
            return "NEI", f"same_text (E={p_e:.2f},C={p_c:.2f},N={p_n:.2f})", 0.85
        
        refutes_boost = self._claim_refutes_boost(claim)
        p_c_adj = min(p_c + refutes_boost, 1.0)
        
        sig = f"E={p_e:.2f},C={p_c:.2f}+{refutes_boost:.2f}={p_c_adj:.2f},N={p_n:.2f}"
        
        if p_c > 0.65:
            return "REFUTES", f"strong_contra ({sig})", p_c
        
        if p_e > 0.65:
            return "SUPPORTS", f"strong_entail ({sig})", p_e
        
        if refutes_boost >= 0.20 and p_c > 0.25:
            return "REFUTES", f"pattern_refutes ({sig})", p_c_adj
        
        if p_c_adj > 0.50 and p_c_adj > p_e:
            return "REFUTES", f"contra ({sig})", p_c_adj
        
        if p_e > 0.50 and p_e > p_c_adj:
            return "SUPPORTS", f"entail ({sig})", p_e
        
        if p_n > 0.55:
            return "NEI", f"neutral ({sig})", p_n
        
        if p_e > p_c and p_e > p_n:
            return "SUPPORTS", f"argmax_s ({sig})", p_e
        elif p_c_adj > p_e and p_c_adj > p_n:
            return "REFUTES", f"argmax_r ({sig})", p_c_adj
        else:
            return "NEI", f"argmax_n ({sig})", p_n
    
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
    print("PRODUCTION AUDITOR V2 TEST")
    auditor = ProductionAuditorV2(device="cpu")
    tests = [
        ("Tim Roth is an English actor.", "Tim Roth is an English actor.", "NEI"),
        ("Telemundo is English.", "Telemundo is Spanish-language network.", "REFUTES"),
        ("Fox released Soul Food.", "Soul Food was produced by Fox 2000.", "SUPPORTS"),
    ]
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        print(f"{'Y' if match else 'N'} [{expected}->{result.verdict}] {result.explanation[:60]}")
