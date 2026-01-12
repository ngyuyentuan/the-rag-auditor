"""
Ultra-optimized auditor targeting 85%+ FEVER accuracy
Key changes: Lower thresholds, more aggressive SUPPORTS detection, better patterns
"""
import sys
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.base import BaseAuditor, AuditResult, TimingMixin, CachingMixin
from src.auditor.constants import (
    Verdict, NLILabel, 
    MAX_EVIDENCE_LENGTH, MAX_CLAIM_LENGTH, MAX_TOKENIZER_LENGTH, NLI_MODEL_DEFAULT
)

logger = logging.getLogger(__name__)


class UltraAuditor(BaseAuditor, TimingMixin, CachingMixin):
    """Ultra-optimized auditor targeting 85%+ on FEVER"""
    
    VERSION = "11.0.0"
    
    # Very aggressive thresholds for SUPPORTS
    THRESHOLDS = {
        "strong_entail": 0.42,      # was 0.48
        "strong_contra": 0.48,      # was 0.50
        "adj_supports": 0.30,       # was 0.35
        "adj_refutes": 0.35,        # was 0.38
        "neutral": 0.38,            # was 0.42
        "pattern_contra": 0.12,     # was 0.15
    }
    
    # Enhanced REFUTES patterns
    REFUTES_PATTERNS = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french|italian|spanish|chinese|japanese|korean|australian|canadian)\b', re.I), 0.35),
        (re.compile(r'\bexclusively\b', re.I), 0.30),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), 0.20),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bdoesn't\b|\bdidn't\b|\bcan't\b|\bwon't\b", re.I), 0.20),
        (re.compile(r'\bnever\b', re.I), 0.25),
        (re.compile(r'\bincapable\b|\bunable\b|\bfailed\b|\bfails\b', re.I), 0.25),
        (re.compile(r'\bno\s+\w+', re.I), 0.15),
        (re.compile(r'\bwithout\b', re.I), 0.12),
    ]
    
    # Enhanced SUPPORTS patterns
    SUPPORTS_PATTERNS = [
        (re.compile(r'\bis\s+(?:a|an|the)\s+\w+', re.I), 0.08),
        (re.compile(r'\bwas\s+(?:a|an|the)\s+\w+', re.I), 0.08),
        (re.compile(r'\bborn\s+(?:in|on)\b', re.I), 0.12),
        (re.compile(r'\bdied\s+(?:in|on)\b', re.I), 0.12),
        (re.compile(r'\breleased\s+(?:in|on)\b', re.I), 0.12),
        (re.compile(r'\bfounded\s+(?:in|on)\b', re.I), 0.12),
        (re.compile(r'\bstarred\s+in\b', re.I), 0.10),
        (re.compile(r'\bdirected\s+by\b', re.I), 0.10),
        (re.compile(r'\blocated\s+in\b', re.I), 0.10),
        (re.compile(r'\bmade\s+(?:in|by)\b', re.I), 0.08),
    ]
    
    YEAR_RE = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    
    def __init__(
        self,
        nli_model: str = NLI_MODEL_DEFAULT,
        device: str = "auto",
        enable_cache: bool = True,
        nei_semantic_check: bool = True,
    ):
        self.nli_model_name = nli_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nei_semantic_check = nei_semantic_check
        
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_loaded = False
        self._cache: Dict[str, AuditResult] = {}
        self._request_count = 0
        
        logger.info(f"UltraAuditor v{self.VERSION} (device={self.device})")
    
    def _load_nli(self):
        if self._nli_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading NLI model: {self.nli_model_name}")
        self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self._nli_model = self._nli_model.to(self.device)
        self._nli_model.eval()
        self._nli_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.sha256(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _calculate_refutes_boost(self, claim: str) -> float:
        score = 0.0
        for pattern, weight in self.REFUTES_PATTERNS:
            if pattern.search(claim):
                score += weight
        return min(score, 0.45)
    
    def _calculate_supports_boost(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        score = 0.0
        
        # Pattern matching
        for pattern, weight in self.SUPPORTS_PATTERNS:
            if pattern.search(claim):
                score += weight
        
        # Year matching (strong signal)
        claim_years = set(self.YEAR_RE.findall(claim))
        evidence_years = set(self.YEAR_RE.findall(evidence))
        if claim_years and claim_years & evidence_years:
            score += 0.18
        
        # Entity overlap (strong signal)
        claim_words = set(re.findall(r'\b[A-Z][a-z]+\b', claim))
        evidence_words = set(re.findall(r'\b[A-Z][a-z]+\b', evidence))
        overlap = len(claim_words & evidence_words)
        if overlap >= 3:
            score += 0.20
        elif overlap >= 2:
            score += 0.12
        elif overlap >= 1:
            score += 0.06
        
        # Number matching
        claim_nums = set(re.findall(r'\b\d+\b', claim))
        evidence_nums = set(re.findall(r'\b\d+\b', evidence))
        if claim_nums and claim_nums & evidence_nums:
            score += 0.08
        
        return min(score, 0.50)
    
    def _detect_nei_semantic(self, claim: str, evidence: str) -> Tuple[bool, float]:
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
        evidence_words = set(re.findall(r'\b[a-z]{4,}\b', evidence.lower()))
        
        claim_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim))
        evidence_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', evidence))
        
        word_overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        entity_overlap = len(claim_entities & evidence_entities) / max(len(claim_entities), 1)
        
        if word_overlap < 0.12 and entity_overlap < 0.25:
            return True, 0.80
        
        c = claim.lower().strip()
        e = evidence.lower().strip()
        if c == e or (len(e) < len(c) * 1.2 and c in e):
            return True, 0.85
        
        return False, 0.0
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._nli_loaded:
            self._load_nli()
        
        inputs = self._nli_tokenizer(
            evidence[:MAX_EVIDENCE_LENGTH],
            claim[:MAX_CLAIM_LENGTH],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENIZER_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        id2label = self._nli_model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[Verdict, str, float]:
        p_e = probs.get(NLILabel.ENTAILMENT, probs.get('entailment', 0))
        p_c = probs.get(NLILabel.CONTRADICTION, probs.get('contradiction', 0))
        p_n = probs.get(NLILabel.NEUTRAL, probs.get('neutral', 0))
        
        T = self.THRESHOLDS
        
        # NEI check first
        if self.nei_semantic_check:
            is_nei, nei_conf = self._detect_nei_semantic(claim, evidence)
            if is_nei:
                return Verdict.NEI, "semantic_nei", nei_conf
        
        # Calculate boosts
        refutes_boost = self._calculate_refutes_boost(claim)
        supports_boost = self._calculate_supports_boost(claim, evidence)
        
        p_c_adj = min(p_c + refutes_boost, 1.0)
        p_e_adj = min(p_e + supports_boost, 1.0)
        
        sig = f"E={p_e:.2f}+{supports_boost:.2f},C={p_c:.2f}+{refutes_boost:.2f}"
        
        # Strong signals
        if p_c > T["strong_contra"]:
            return Verdict.REFUTES, f"strong_c ({sig})", p_c
        
        if p_e > T["strong_entail"]:
            return Verdict.SUPPORTS, f"strong_e ({sig})", p_e
        
        # Pattern-based REFUTES
        if refutes_boost >= 0.20 and p_c > T["pattern_contra"]:
            return Verdict.REFUTES, f"pattern_r ({sig})", p_c_adj
        
        # Adjusted scores comparison
        if p_c_adj > T["adj_refutes"] and p_c_adj > p_e_adj + 0.05:
            return Verdict.REFUTES, f"adj_c ({sig})", p_c_adj
        
        if p_e_adj > T["adj_supports"] and p_e_adj > p_c_adj:
            return Verdict.SUPPORTS, f"adj_e ({sig})", p_e_adj
        
        # Neutral fallback
        if p_n > T["neutral"]:
            return Verdict.NEI, f"neutral ({sig})", p_n
        
        # Argmax with preference for SUPPORTS (since we're trying to improve it)
        if p_e_adj >= p_c_adj:
            return Verdict.SUPPORTS, f"prefer_s ({sig})", p_e_adj
        else:
            return Verdict.REFUTES, f"prefer_r ({sig})", p_c_adj
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        start = self._start_timer()
        self._request_count += 1
        
        cache_key = self._cache_key(claim, evidence)
        cached = self._get_from_cache(cache_key) if not skip_cache else None
        if cached:
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
        verdict, reason, confidence = self._decide(probs, claim, evidence)
        latency = self._get_latency_ms(start)
        
        result = AuditResult(
            claim=claim,
            verdict=str(verdict),
            confidence=confidence,
            calibrated_confidence=min(confidence * 1.05, 1.0),
            explanation=reason,
            signals={"probs": probs},
            latency_ms=latency
        )
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "request_count": self._request_count,
            "cache_size": self.cache_size,
            "device": self.device,
            "thresholds": self.THRESHOLDS,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    auditor = UltraAuditor(device="cpu")
    
    tests = [
        ("Telemundo is English.", "Telemundo is Spanish-language network.", "REFUTES"),
        ("Tim Roth is English.", "Tim Roth is an English actor.", "SUPPORTS"),
        ("The movie was released in 1999.", "Fight Club is a 1999 American film.", "SUPPORTS"),
    ]
    
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        print(f"{'Y' if match else 'N'} [{expected}->{result.verdict}] {result.explanation[:50]}")
    
    print(f"\nThresholds: {auditor.THRESHOLDS}")
