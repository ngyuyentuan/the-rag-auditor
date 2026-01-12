import sys
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.base import BaseAuditor, AuditResult, TimingMixin, CachingMixin
from src.auditor.constants import (
    Verdict, NLILabel, ThresholdConfig, PatternWeights,
    NLI_MODEL_DEFAULT, MAX_EVIDENCE_LENGTH, MAX_CLAIM_LENGTH, MAX_TOKENIZER_LENGTH
)

logger = logging.getLogger(__name__)


class ProductionAuditor(BaseAuditor, TimingMixin, CachingMixin):
    VERSION = "8.0.0"
    
    REFUTES_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french|italian|spanish|chinese|japanese)\b', re.I), PatternWeights.ONLY_NATIONALITY),
        (re.compile(r'\bexclusively\b', re.I), PatternWeights.EXCLUSIVELY),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), PatternWeights.NOT_PHRASE),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bdoesn't\b|\bdidn't\b", re.I), PatternWeights.CONTRACTION),
        (re.compile(r'\bnever\b', re.I), PatternWeights.NEVER),
        (re.compile(r'\bincapable\b|\bunable\b', re.I), PatternWeights.INCAPABLE),
        (re.compile(r'\byet\s+to\b', re.I), PatternWeights.YET_TO),
        (re.compile(r'\bwithout\b', re.I), PatternWeights.WITHOUT),
    ]
    
    SUPPORTS_PHRASES = [
        'is a', 'is an', 'is the', 'was a', 'was an', 'was the',
        'are', 'were', 'has', 'have', 'had',
        'born', 'died', 'released', 'founded', 'made', 'created',
        'played', 'starred', 'appeared', 'directed', 'produced',
    ]
    
    YEAR_RE = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    
    def __init__(
        self,
        nli_model: str = NLI_MODEL_DEFAULT,
        device: str = "auto",
        enable_cache: bool = True,
        nei_same_text_check: bool = False,
    ):
        self.nli_model_name = nli_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nei_same_text_check = nei_same_text_check
        
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_loaded = False
        self._cache: Dict[str, AuditResult] = {}
        self._request_count = 0
        
        logger.info(f"ProductionAuditor v{self.VERSION} initialized (device={self.device})")
    
    def _load_nli(self) -> None:
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
        return min(score, ThresholdConfig.REFUTES_BOOST_MAX)
    
    def _calculate_supports_boost(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        score = 0.0
        
        for phrase in self.SUPPORTS_PHRASES:
            if phrase in claim_lower:
                score += 0.03
        
        claim_years = set(self.YEAR_RE.findall(claim))
        evidence_years = set(self.YEAR_RE.findall(evidence))
        if claim_years and claim_years & evidence_years:
            score += PatternWeights.YEAR_MATCH
        
        claim_words = set(re.findall(r'\b[A-Z][a-z]+\b', claim))
        evidence_words = set(re.findall(r'\b[A-Z][a-z]+\b', evidence))
        overlap = len(claim_words & evidence_words)
        if overlap >= 2:
            score += min(PatternWeights.ENTITY_OVERLAP * overlap, 0.15)
        
        return min(score, ThresholdConfig.SUPPORTS_BOOST_MAX)
    
    def _is_same_text(self, claim: str, evidence: str) -> bool:
        c = claim.lower().strip()
        e = evidence.lower().strip()
        if c == e:
            return True
        if len(e) < len(c) * ThresholdConfig.SAME_TEXT_THRESHOLD and c in e:
            return True
        return False
    
    def _detect_nei_semantic(self, claim: str, evidence: str) -> Tuple[bool, float]:
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
        evidence_words = set(re.findall(r'\b[a-z]{4,}\b', evidence.lower()))
        
        claim_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim))
        evidence_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', evidence))
        
        word_overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        entity_overlap = len(claim_entities & evidence_entities) / max(len(claim_entities), 1)
        
        if word_overlap < 0.15 and entity_overlap < 0.30:
            return True, 0.75
        
        if self._is_same_text(claim, evidence):
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
        p_e = probs.get(NLILabel.ENTAILMENT, 0)
        p_c = probs.get(NLILabel.CONTRADICTION, 0)
        p_n = probs.get(NLILabel.NEUTRAL, 0)
        
        if self.nei_same_text_check:
            is_nei, nei_conf = self._detect_nei_semantic(claim, evidence)
            if is_nei:
                return Verdict.NEI, "semantic_nei", nei_conf
        
        refutes_boost = self._calculate_refutes_boost(claim)
        supports_boost = self._calculate_supports_boost(claim, evidence)
        
        p_c_adj = min(p_c + refutes_boost, 1.0)
        p_e_adj = min(p_e + supports_boost, 1.0)
        
        sig = f"E={p_e:.2f}+{supports_boost:.2f},C={p_c:.2f}+{refutes_boost:.2f}"
        
        if p_c > ThresholdConfig.CONTRA_HIGH:
            return Verdict.REFUTES, f"strong_c ({sig})", p_c
        
        if p_e > ThresholdConfig.ENTAIL_HIGH:
            return Verdict.SUPPORTS, f"strong_e ({sig})", p_e
        
        if refutes_boost >= PatternWeights.NOT_PHRASE and p_c > 0.20:
            return Verdict.REFUTES, f"pattern_r ({sig})", p_c_adj
        
        if p_c_adj > ThresholdConfig.CONTRA_MED and p_c_adj > p_e_adj:
            return Verdict.REFUTES, f"adj_c ({sig})", p_c_adj
        
        if p_e_adj > ThresholdConfig.ENTAIL_MED and p_e_adj > p_c_adj:
            return Verdict.SUPPORTS, f"adj_e ({sig})", p_e_adj
        
        if p_n > ThresholdConfig.NEUTRAL_HIGH:
            return Verdict.NEI, f"neutral ({sig})", p_n
        
        scores = [(p_e_adj, Verdict.SUPPORTS), (p_c_adj, Verdict.REFUTES), (p_n, Verdict.NEI)]
        best = max(scores, key=lambda x: x[0])
        return best[1], f"argmax ({sig})", best[0]
    
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
            "nei_same_text_check": self.nei_same_text_check,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    auditor = ProductionAuditor(device="cpu", nei_same_text_check=False)
    
    tests = [
        ("Telemundo is English.", "Telemundo is Spanish-language network.", Verdict.REFUTES),
        ("Fox released Soul Food.", "Soul Food was produced by Fox 2000.", Verdict.SUPPORTS),
        ("Tim Roth is English.", "Tim Roth is an English actor.", Verdict.SUPPORTS),
    ]
    
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == str(expected)
        print(f"{'✓' if match else '✗'} [{expected.value}->{result.verdict}] {result.explanation[:50]}")
    
    print(f"\nStats: {auditor.get_stats()}")
