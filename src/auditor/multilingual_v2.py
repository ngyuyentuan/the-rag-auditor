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
from src.auditor.vietnamese import (
    VIETNAMESE_REFUTES_PATTERNS, VIETNAMESE_SUPPORTS_PHRASES,
    VIETNAMESE_NEI_PATTERNS, NATIONALITY_VI
)

logger = logging.getLogger(__name__)


class MultilingualAuditorV2(BaseAuditor, TimingMixin, CachingMixin):
    VERSION = "9.0.0"
    
    EN_REFUTES_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r'\bonly\b.*\b(american|british|english|german|french|italian|spanish|chinese|japanese)\b', re.I), PatternWeights.ONLY_NATIONALITY),
        (re.compile(r'\bexclusively\b', re.I), PatternWeights.EXCLUSIVELY),
        (re.compile(r'\bnot\s+(?:a|an|the)?\s*\w+', re.I), PatternWeights.NOT_PHRASE),
        (re.compile(r"\bisn't\b|\bwasn't\b|\baren't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bdoesn't\b|\bdidn't\b", re.I), PatternWeights.CONTRACTION),
        (re.compile(r'\bnever\b', re.I), PatternWeights.NEVER),
        (re.compile(r'\bincapable\b|\bunable\b', re.I), PatternWeights.INCAPABLE),
    ]
    
    VI_REFUTES_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(p, re.I), 0.20) for p in VIETNAMESE_REFUTES_PATTERNS
    ]
    
    SUPPORTS_PHRASES_EN = [
        'is a', 'is an', 'is the', 'was a', 'was an', 'was the',
        'born', 'died', 'released', 'founded', 'made', 'created',
    ]
    
    YEAR_RE = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    
    def __init__(
        self,
        nli_model: str = NLI_MODEL_DEFAULT,
        device: str = "auto",
        enable_cache: bool = True,
        nei_semantic_check: bool = True,
        use_persistent_cache: bool = False,
    ):
        self.nli_model_name = nli_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nei_semantic_check = nei_semantic_check
        self.use_persistent_cache = use_persistent_cache
        
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_loaded = False
        self._cache: Dict[str, AuditResult] = {}
        self._request_count = 0
        
        self._persistent_cache = None
        if use_persistent_cache:
            from src.utils.persistent_cache import NLICache
            self._persistent_cache = NLICache()
        
        logger.info(f"MultilingualAuditorV2 v{self.VERSION} (device={self.device}, persistent={use_persistent_cache})")
    
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
    
    def _detect_language(self, text: str) -> str:
        vi_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        text_lower = text.lower()
        if any(c in vi_chars for c in text_lower):
            return "vi"
        return "en"
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.sha256(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _calculate_refutes_boost(self, claim: str, lang: str) -> float:
        patterns = self.EN_REFUTES_PATTERNS if lang == "en" else self.VI_REFUTES_PATTERNS
        score = 0.0
        for pattern, weight in patterns:
            if pattern.search(claim):
                score += weight
        return min(score, ThresholdConfig.REFUTES_BOOST_MAX)
    
    def _calculate_supports_boost(self, claim: str, evidence: str, lang: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        score = 0.0
        
        phrases = self.SUPPORTS_PHRASES_EN if lang == "en" else VIETNAMESE_SUPPORTS_PHRASES
        for phrase in phrases:
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
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if self._persistent_cache:
            cached = self._persistent_cache.get_nli_result(claim, evidence)
            if cached:
                return cached
        
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
        result = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        
        if self._persistent_cache:
            self._persistent_cache.set_nli_result(claim, evidence, result)
        
        return result
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str, lang: str) -> Tuple[Verdict, str, float]:
        p_e = probs.get(NLILabel.ENTAILMENT, 0)
        p_c = probs.get(NLILabel.CONTRADICTION, 0)
        p_n = probs.get(NLILabel.NEUTRAL, 0)
        
        if self.nei_semantic_check:
            is_nei, nei_conf = self._detect_nei_semantic(claim, evidence)
            if is_nei:
                return Verdict.NEI, "semantic_nei", nei_conf
        
        refutes_boost = self._calculate_refutes_boost(claim, lang)
        supports_boost = self._calculate_supports_boost(claim, evidence, lang)
        
        p_c_adj = min(p_c + refutes_boost, 1.0)
        p_e_adj = min(p_e + supports_boost, 1.0)
        
        sig = f"E={p_e:.2f}+{supports_boost:.2f},C={p_c:.2f}+{refutes_boost:.2f},lang={lang}"
        
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
        
        lang = self._detect_language(claim + " " + evidence)
        
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
        verdict, reason, confidence = self._decide(probs, claim, evidence, lang)
        latency = self._get_latency_ms(start)
        
        result = AuditResult(
            claim=claim,
            verdict=str(verdict),
            confidence=confidence,
            calibrated_confidence=min(confidence * 1.05, 1.0),
            explanation=reason,
            signals={"probs": probs, "language": lang},
            latency_ms=latency
        )
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "version": self.VERSION,
            "request_count": self._request_count,
            "cache_size": self.cache_size,
            "device": self.device,
            "nei_semantic_check": self.nei_semantic_check,
            "use_persistent_cache": self.use_persistent_cache,
        }
        if self._persistent_cache:
            stats["persistent_cache"] = self._persistent_cache.get_stats()
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    auditor = MultilingualAuditorV2(device="cpu", use_persistent_cache=True)
    
    tests = [
        ("Telemundo là kênh tiếng Anh.", "Telemundo là kênh tiếng Tây Ban Nha.", "REFUTES"),
        ("Tim Roth là diễn viên Anh.", "Tim Roth là diễn viên người Anh.", "SUPPORTS"),
        ("Telemundo is English.", "Telemundo is Spanish-language.", "REFUTES"),
        ("Tim Roth is English.", "Tim Roth is an English actor.", "SUPPORTS"),
    ]
    
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        lang = result.signals.get("language", "?")
        print(f"{'✓' if match else '✗'} [{lang}] [{expected}->{result.verdict}] {result.explanation[:40]}")
    
    print(f"\nStats: {auditor.get_stats()}")
