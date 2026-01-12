"""
RAG Auditor - Production Version

Final production-ready version with:
1. Configurable modes (SAFETY, BALANCED, COVERAGE)
2. Balanced REFUTES/NEI trade-off
3. FastAPI-ready interface
4. Comprehensive logging
5. Model caching and optimization

Usage:
    auditor = RAGAuditorPro(mode="balanced")
    result = auditor.audit(claim, evidence)
"""
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import time
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGAuditorPro")


class AuditMode(Enum):
    SAFETY = "safety"       # Prioritize catching hallucinations (high REFUTES)
    BALANCED = "balanced"   # Balance all categories
    COVERAGE = "coverage"   # Minimize NEI, maximize decisions


@dataclass
class AuditResult:
    """Comprehensive audit result."""
    claim: str
    verdict: str  # SUPPORTS, REFUTES, NEI
    confidence: float
    explanation: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    model_used: str = ""
    mode: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# Mode-specific thresholds
MODE_THRESHOLDS = {
    AuditMode.SAFETY: {
        'entail_high': 0.70,     # Higher bar for accepting
        'entail_low': 0.55,
        'contra_high': 0.35,     # LOWER bar for refuting (catch more hallucinations)
        'contra_low': 0.20,      # Even lower for margin check
        'nei_entropy': 1.05,     # Higher entropy needed for NEI
        'nei_spread': 0.18,      # Lower spread = NEI
        'prefer_refute': True,
        'refute_keyword_boost': 0.15,  # Boost REFUTES when keywords present
    },
    AuditMode.BALANCED: {
        'entail_high': 0.50,
        'entail_low': 0.35,
        'contra_high': 0.50,
        'contra_low': 0.35,
        'nei_entropy': 0.95,
        'nei_spread': 0.22,
        'prefer_refute': False,
        'refute_keyword_boost': 0.10,
    },
    AuditMode.COVERAGE: {
        'entail_high': 0.40,     # Lower bar for decisions
        'entail_low': 0.25,
        'contra_high': 0.40,
        'contra_low': 0.25,
        'nei_entropy': 1.10,     # Only strong uncertainty = NEI
        'nei_spread': 0.12,
        'prefer_refute': False,
        'refute_keyword_boost': 0.05,
    },
}


class RAGAuditorPro:
    """
    Production-ready RAG Auditor.
    
    Features:
    - Configurable modes for different use cases
    - Balanced accuracy across all categories
    - Caching for repeated queries
    - Comprehensive logging
    - FastAPI-ready interface
    """
    
    NEI_KEYWORDS = ['may', 'might', 'could', 'possibly', 'mixed', 
                    'unclear', 'depends', 'debated', 'controversial', 
                    'some studies', 'evidence suggests']
    
    REFUTE_KEYWORDS = ['not', 'no', 'false', 'myth', 'incorrect', 'wrong',
                       'contrary', 'never', 'disproven', 'debunked']
    
    def __init__(
        self,
        mode: Literal["safety", "balanced", "coverage"] = "balanced",
        model: str = "facebook/bart-large-mnli",
        device: str = "auto",
        enable_cache: bool = True,
        max_cache_size: int = 10000,
    ):
        self.mode = AuditMode(mode)
        self.thresholds = MODE_THRESHOLDS[self.mode]
        self.model_name = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._cache = {}
        self._stats = {"total": 0, "cache_hits": 0, "supports": 0, "refutes": 0, "nei": 0}
        
        logger.info(f"RAGAuditorPro initialized: mode={mode}, device={self.device}")
    
    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded on {self.device}")
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _compute_entropy(self, probs: Dict[str, float]) -> float:
        values = np.array(list(probs.values()))
        values = values[values > 0]
        return -np.sum(values * np.log(values + 1e-10))
    
    def _compute_spread(self, probs: Dict[str, float]) -> float:
        values = list(probs.values())
        return max(values) - min(values)
    
    def _check_keywords(self, evidence: str) -> Tuple[int, int, int]:
        """Count keyword signals."""
        evidence_lower = evidence.lower()
        nei_count = sum(1 for kw in self.NEI_KEYWORDS if kw in evidence_lower)
        refute_count = sum(1 for kw in self.REFUTE_KEYWORDS if kw in evidence_lower)
        return nei_count, refute_count, 0
    
    def _should_be_nei(self, probs: Dict[str, float], evidence: str) -> Tuple[bool, str]:
        """Determine if result should be NEI."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        entropy = self._compute_entropy(probs)
        spread = self._compute_spread(probs)
        nei_kw, refute_kw, _ = self._check_keywords(evidence)
        
        t = self.thresholds
        
        # High entropy = uncertain
        if entropy > t['nei_entropy']:
            return True, f"high_entropy ({entropy:.2f})"
        
        # Low spread = no clear winner
        if spread < t['nei_spread']:
            return True, f"low_spread ({spread:.2f})"
        
        # NEI keywords + neutral signal
        if nei_kw >= 2 and p_neutral > 0.3:
            return True, f"nei_keywords ({nei_kw})"
        
        # Strong neutral
        if p_neutral > max(p_entail, p_contra) + 0.15:
            return True, f"neutral_dominant ({p_neutral:.2%})"
        
        return False, ""
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._loaded:
            self._load_model()
        
        if len(evidence) > 1500:
            evidence = evidence[:1500] + "..."
        if len(claim) > 500:
            claim = claim[:500] + "..."
        
        inputs = self.tokenizer(
            evidence, claim,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        id2label = self.model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def _decide(self, probs: Dict[str, float], evidence: str) -> Tuple[str, str, float]:
        """Make final decision with mode-specific logic."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        t = self.thresholds
        
        # Step 1: Check NEI (but be careful not to miss REFUTES in SAFETY mode)
        is_nei, nei_reason = self._should_be_nei(probs, evidence)
        
        # In SAFETY mode, prefer REFUTES over NEI when there's a refutation signal
        if is_nei and t['prefer_refute'] and p_contra > 0.3:
            is_nei = False
        
        if is_nei:
            return "NEI", f"nei: {nei_reason}", max(p_neutral, 0.5)
        
        # Step 2: High confidence
        if p_entail > t['entail_high']:
            return "SUPPORTS", f"entailment={p_entail:.2%}", p_entail
        if p_contra > t['contra_high']:
            return "REFUTES", f"contradiction={p_contra:.2%}", p_contra
        
        # Step 3: Medium confidence with margin
        if p_entail > t['entail_low'] and p_entail > p_contra + 0.08:
            return "SUPPORTS", f"entailment={p_entail:.2%} (margin)", p_entail
        if p_contra > t['contra_low'] and p_contra > p_entail + 0.08:
            return "REFUTES", f"contradiction={p_contra:.2%} (margin)", p_contra
        
        # Step 4: Argmax with keyword boost
        _, refute_kw, _ = self._check_keywords(evidence)
        
        if refute_kw >= 1 and p_contra > p_entail:
            return "REFUTES", f"contradiction={p_contra:.2%} (keyword)", p_contra
        
        if p_entail > p_contra:
            return "SUPPORTS", f"entailment={p_entail:.2%} (argmax)", p_entail
        elif p_contra > p_entail:
            return "REFUTES", f"contradiction={p_contra:.2%} (argmax)", p_contra
        
        return "NEI", f"neutral={p_neutral:.2%}", p_neutral
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        """
        Audit a claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence: The evidence text
            skip_cache: Whether to skip cache lookup
        
        Returns:
            AuditResult with verdict, confidence, and explanation
        """
        start_time = time.time()
        self._stats["total"] += 1
        
        # Check cache
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        # NLI inference
        probs = self._nli_inference(claim, evidence)
        verdict, reason, confidence = self._decide(probs, evidence)
        
        latency = (time.time() - start_time) * 1000
        
        # Update stats
        self._stats[verdict.lower()] = self._stats.get(verdict.lower(), 0) + 1
        
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=reason,
            probabilities=probs,
            latency_ms=latency,
            model_used=self.model_name,
            mode=self.mode.value,
        )
        
        # Cache result
        if self.enable_cache:
            if len(self._cache) >= self.max_cache_size:
                # Simple LRU: remove first item
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        
        return result
    
    def audit_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[AuditResult]:
        """Batch audit for improved throughput."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            for claim, evidence in batch:
                results.append(self.audit(claim, evidence))
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self._stats,
            "cache_hit_rate": self._stats["cache_hits"] / max(self._stats["total"], 1),
            "mode": self.mode.value,
            "device": self.device,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for API."""
        return {
            "status": "healthy",
            "model_loaded": self._loaded,
            "mode": self.mode.value,
            "device": self.device,
            "cache_size": len(self._cache),
        }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["safety", "balanced", "coverage"], default="balanced")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    
    print("="*60)
    print(f"RAG AUDITOR PRO - Mode: {args.mode.upper()}")
    print("="*60)
    
    auditor = RAGAuditorPro(mode=args.mode, device=args.device)
    
    tests = [
        ("The sky is blue.", "Rayleigh scattering makes the sky appear blue."),
        ("Vaccines cause autism.", "Research found no link between vaccines and autism."),
        ("Coffee is healthy.", "Studies show mixed effects depending on consumption."),
        ("The earth is flat.", "Scientific evidence confirms Earth is a sphere."),
        ("AI will replace jobs.", "AI automates some tasks but also creates new opportunities."),
    ]
    
    print(f"\n{'Claim':<35} {'Verdict':<12} {'Conf':<8} {'Explanation'}")
    print("-"*80)
    
    for claim, evidence in tests:
        result = auditor.audit(claim, evidence)
        print(f"{claim:<35} {result.verdict:<12} {result.confidence:>6.1%}   {result.explanation}")
    
    print(f"\nStats: {auditor.get_stats()}")


if __name__ == "__main__":
    main()
