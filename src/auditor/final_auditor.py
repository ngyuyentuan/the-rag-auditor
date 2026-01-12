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


class FinalAuditor:
    VERSION = "7.0.0"
    
    REFUTES_PATTERNS = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french|italian|spanish|chinese|japanese)\b', re.I), 0.30),
        (re.compile(r'\bexclusively\b', re.I), 0.25),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), 0.18),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bdoesn't\b|\bdidn't\b", re.I), 0.18),
        (re.compile(r'\bnever\b', re.I), 0.22),
        (re.compile(r'\bincapable\b|\bunable\b', re.I), 0.22),
        (re.compile(r'\byet\s+to\b', re.I), 0.18),
        (re.compile(r'\bwithout\b', re.I), 0.12),
    ]
    
    SUPPORTS_PHRASES = [
        'is a', 'is an', 'is the', 'was a', 'was an', 'was the',
        'are', 'were', 'has', 'have', 'had',
        'born', 'died', 'released', 'founded', 'made', 'created',
        'played', 'starred', 'appeared', 'directed', 'produced',
        'published', 'written', 'located', 'based in',
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
        return min(score, 0.35)
    
    def _claim_supports_boost(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        score = 0.0
        for phrase in self.SUPPORTS_PHRASES:
            if phrase in claim_lower:
                score += 0.03
        
        claim_years = set(self.YEAR_RE.findall(claim))
        evidence_years = set(self.YEAR_RE.findall(evidence))
        if claim_years and claim_years & evidence_years:
            score += 0.10
        
        claim_words = set(re.findall(r'\b[A-Z][a-z]+\b', claim))
        evidence_words = set(re.findall(r'\b[A-Z][a-z]+\b', evidence))
        overlap = len(claim_words & evidence_words)
        if overlap >= 2:
            score += min(0.05 * overlap, 0.15)
        
        return min(score, 0.25)
    
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
            return "NEI", f"same_text", 0.90
        
        refutes_boost = self._claim_refutes_boost(claim)
        supports_boost = self._claim_supports_boost(claim, evidence)
        
        p_c_adj = min(p_c + refutes_boost, 1.0)
        p_e_adj = min(p_e + supports_boost, 1.0)
        
        sig = f"E={p_e:.2f}+{supports_boost:.2f}={p_e_adj:.2f},C={p_c:.2f}+{refutes_boost:.2f}={p_c_adj:.2f}"
        
        if p_c > 0.60:
            return "REFUTES", f"strong_c ({sig})", p_c
        
        if p_e > 0.55:
            return "SUPPORTS", f"strong_e ({sig})", p_e
        
        if refutes_boost >= 0.18 and p_c > 0.20:
            return "REFUTES", f"pattern_r ({sig})", p_c_adj
        
        if p_c_adj > 0.48 and p_c_adj > p_e_adj:
            return "REFUTES", f"adj_c ({sig})", p_c_adj
        
        if p_e_adj > 0.48 and p_e_adj > p_c_adj:
            return "SUPPORTS", f"adj_e ({sig})", p_e_adj
        
        if p_n > 0.50:
            return "NEI", f"neutral ({sig})", p_n
        
        scores = [(p_e_adj, "SUPPORTS"), (p_c_adj, "REFUTES"), (p_n, "NEI")]
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
