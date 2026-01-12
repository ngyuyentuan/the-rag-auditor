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
    Verdict, NLILabel, ThresholdConfig, PatternWeights,
    MAX_EVIDENCE_LENGTH, MAX_CLAIM_LENGTH, MAX_TOKENIZER_LENGTH
)

logger = logging.getLogger(__name__)


class EnsembleAuditorV2(BaseAuditor, TimingMixin, CachingMixin):
    VERSION = "10.0.0"
    
    NLI_MODELS = [
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", 0.50),
        ("facebook/bart-large-mnli", 0.30),
        ("cross-encoder/nli-deberta-v3-base", 0.20),
    ]
    
    REFUTES_PATTERNS = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french|italian|spanish)\b', re.I), 0.30),
        (re.compile(r'\bexclusively\b', re.I), 0.25),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), 0.18),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bdoesn't\b|\bdidn't\b", re.I), 0.18),
        (re.compile(r'\bnever\b', re.I), 0.22),
        (re.compile(r'\bincapable\b|\bunable\b', re.I), 0.22),
    ]
    
    YEAR_RE = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    
    def __init__(
        self,
        device: str = "auto",
        enable_cache: bool = True,
        nei_semantic_check: bool = True,
        use_single_model: bool = True,
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nei_semantic_check = nei_semantic_check
        self.use_single_model = use_single_model
        
        self._models = {}
        self._tokenizers = {}
        self._loaded = False
        self._cache: Dict[str, AuditResult] = {}
        self._request_count = 0
        
        logger.info(f"EnsembleAuditorV2 v{self.VERSION} (device={self.device})")
    
    def _load_models(self):
        if self._loaded:
            return
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        models_to_load = self.NLI_MODELS[:1] if self.use_single_model else self.NLI_MODELS
        
        for model_name, weight in models_to_load:
            try:
                logger.info(f"Loading: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model = model.to(self.device)
                model.eval()
                self._models[model_name] = (model, weight)
                self._tokenizers[model_name] = tokenizer
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        
        self._loaded = True
    
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
        score = 0.0
        
        phrases = ['is a', 'is an', 'was a', 'was an', 'born', 'died', 'released', 'founded']
        for phrase in phrases:
            if phrase in claim_lower:
                score += 0.04
        
        claim_years = set(self.YEAR_RE.findall(claim))
        evidence_years = set(self.YEAR_RE.findall(evidence))
        if claim_years and claim_years & evidence_years:
            score += PatternWeights.YEAR_MATCH
        
        claim_words = set(re.findall(r'\b[A-Z][a-z]+\b', claim))
        evidence_words = set(re.findall(r'\b[A-Z][a-z]+\b', evidence))
        overlap = len(claim_words & evidence_words)
        if overlap >= 2:
            score += min(PatternWeights.ENTITY_OVERLAP * overlap, 0.20)
        
        return min(score, ThresholdConfig.SUPPORTS_BOOST_MAX)
    
    def _detect_nei_semantic(self, claim: str, evidence: str) -> Tuple[bool, float]:
        claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
        evidence_words = set(re.findall(r'\b[a-z]{4,}\b', evidence.lower()))
        
        claim_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim))
        evidence_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', evidence))
        
        word_overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        entity_overlap = len(claim_entities & evidence_entities) / max(len(claim_entities), 1)
        
        if word_overlap < 0.15 and entity_overlap < 0.30:
            return True, 0.75
        
        c = claim.lower().strip()
        e = evidence.lower().strip()
        if c == e or (len(e) < len(c) * 1.2 and c in e):
            return True, 0.85
        
        return False, 0.0
    
    def _ensemble_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._loaded:
            self._load_models()
        
        aggregated = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        total_weight = 0.0
        
        for model_name, (model, weight) in self._models.items():
            tokenizer = self._tokenizers[model_name]
            
            inputs = tokenizer(
                evidence[:MAX_EVIDENCE_LENGTH],
                claim[:MAX_CLAIM_LENGTH],
                padding=True,
                truncation=True,
                max_length=MAX_TOKENIZER_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            id2label = model.config.id2label
            for i, prob in enumerate(probs):
                label = id2label[i].lower()
                if label in aggregated:
                    aggregated[label] += float(prob) * weight
            
            total_weight += weight
        
        if total_weight > 0:
            for k in aggregated:
                aggregated[k] /= total_weight
        
        return aggregated
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[Verdict, str, float]:
        p_e = probs.get(NLILabel.ENTAILMENT, probs.get('entailment', 0))
        p_c = probs.get(NLILabel.CONTRADICTION, probs.get('contradiction', 0))
        p_n = probs.get(NLILabel.NEUTRAL, probs.get('neutral', 0))
        
        if self.nei_semantic_check:
            is_nei, nei_conf = self._detect_nei_semantic(claim, evidence)
            if is_nei:
                return Verdict.NEI, "semantic_nei", nei_conf
        
        refutes_boost = self._calculate_refutes_boost(claim)
        supports_boost = self._calculate_supports_boost(claim, evidence)
        
        p_c_adj = min(p_c + refutes_boost, 1.0)
        p_e_adj = min(p_e + supports_boost, 1.0)
        
        sig = f"E={p_e:.2f}+{supports_boost:.2f},C={p_c:.2f}+{refutes_boost:.2f}"
        
        if p_c > 0.50:
            return Verdict.REFUTES, f"strong_c ({sig})", p_c
        
        if p_e > 0.48:
            return Verdict.SUPPORTS, f"strong_e ({sig})", p_e
        
        if refutes_boost >= 0.18 and p_c > 0.15:
            return Verdict.REFUTES, f"pattern_r ({sig})", p_c_adj
        
        if p_c_adj > 0.38 and p_c_adj > p_e_adj:
            return Verdict.REFUTES, f"adj_c ({sig})", p_c_adj
        
        if p_e_adj > 0.35 and p_e_adj > p_c_adj:
            return Verdict.SUPPORTS, f"adj_e ({sig})", p_e_adj
        
        if p_n > 0.42:
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
        
        probs = self._ensemble_inference(claim, evidence)
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
            "models_loaded": len(self._models),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    auditor = EnsembleAuditorV2(device="cpu", use_single_model=True)
    
    tests = [
        ("Telemundo is English.", "Telemundo is Spanish-language network.", "REFUTES"),
        ("Tim Roth is English.", "Tim Roth is an English actor.", "SUPPORTS"),
    ]
    
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        print(f"{'✓' if match else '✗'} [{expected}->{result.verdict}] {result.explanation[:50]}")
    
    print(f"\nStats: {auditor.get_stats()}")
