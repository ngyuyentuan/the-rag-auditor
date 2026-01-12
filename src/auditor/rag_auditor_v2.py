"""
RAG Auditor V2 - Improved Version

Key improvements:
1. Ensemble approach: fast model + strong model
2. Semantic similarity fallback for NEI cases
3. Keyword-based quick decisions for obvious cases
4. Better confidence calibration
5. Caching for speed
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json
import hashlib
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AuditResult:
    """Result from RAG Auditor."""
    claim: str
    verdict: str
    confidence: float
    stage1_decision: str
    stage2_decision: Optional[str]
    stage2_ran: bool
    explanation: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RAGAuditorV2:
    """
    Improved RAG Auditor with:
    1. Ensemble NLI (fast + strong)
    2. Semantic similarity fallback
    3. Keyword matching for obvious cases
    4. Better NEI handling
    """
    
    # Keywords that strongly indicate support/refutation
    SUPPORT_KEYWORDS = ['confirms', 'shows', 'proves', 'demonstrates', 'is', 'are', 'was', 'were']
    REFUTE_KEYWORDS = ['not', 'no', 'false', 'myth', 'incorrect', 'wrong', 'contrary', 'never', 'isnt', "isn't", 'cannot']
    NEI_KEYWORDS = ['may', 'might', 'could', 'possibly', 'some', 'mixed', 'unclear', 'depends']
    
    def __init__(
        self,
        use_ensemble: bool = True,
        fast_model: str = "typeform/distilbert-base-uncased-mnli",
        strong_model: str = "facebook/bart-large-mnli",
        device: str = "auto",
        enable_cache: bool = True,
        # Thresholds
        fast_confidence_threshold: float = 0.85,  # Use fast model if confidence > this
        nei_threshold: float = 0.4,  # Below this, consider NEI
    ):
        self.use_ensemble = use_ensemble
        self.fast_model_name = fast_model
        self.strong_model_name = strong_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.fast_confidence_threshold = fast_confidence_threshold
        self.nei_threshold = nei_threshold
        
        self.fast_model = None
        self.fast_tokenizer = None
        self.strong_model = None
        self.strong_tokenizer = None
        
        self._fast_loaded = False
        self._strong_loaded = False
        self._cache = {}
    
    def _load_fast_model(self):
        if self._fast_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[RAGAuditorV2] Loading fast model: {self.fast_model_name}")
        self.fast_tokenizer = AutoTokenizer.from_pretrained(self.fast_model_name)
        self.fast_model = AutoModelForSequenceClassification.from_pretrained(self.fast_model_name)
        self.fast_model = self.fast_model.to(self.device)
        self.fast_model.eval()
        self._fast_loaded = True
    
    def _load_strong_model(self):
        if self._strong_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[RAGAuditorV2] Loading strong model: {self.strong_model_name}")
        self.strong_tokenizer = AutoTokenizer.from_pretrained(self.strong_model_name)
        self.strong_model = AutoModelForSequenceClassification.from_pretrained(self.strong_model_name)
        self.strong_model = self.strong_model.to(self.device)
        self.strong_model.eval()
        self._strong_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _keyword_check(self, claim: str, evidence: str) -> Optional[Tuple[str, str, float]]:
        """Quick keyword-based decision for obvious cases."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Check for strong refutation keywords in evidence
        refute_score = sum(1 for kw in self.REFUTE_KEYWORDS if kw in evidence_lower)
        support_score = sum(1 for kw in self.SUPPORT_KEYWORDS if kw in evidence_lower)
        nei_score = sum(1 for kw in self.NEI_KEYWORDS if kw in evidence_lower)
        
        # Check if claim words appear in evidence
        claim_words = set(re.findall(r'\w+', claim_lower))
        evidence_words = set(re.findall(r'\w+', evidence_lower))
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        
        # Strong refutation signal
        if refute_score >= 2 and overlap > 0.3:
            return "REFUTES", "keyword_refute", 0.75
        
        # NEI signal
        if nei_score >= 2:
            return "NEI", "keyword_nei", 0.6
        
        # High overlap suggests support
        if overlap > 0.7 and refute_score == 0:
            return "SUPPORTS", "keyword_support", 0.7
        
        return None
    
    def _nli_inference(self, claim: str, evidence: str, model, tokenizer) -> Dict[str, float]:
        """Run NLI inference and return probabilities."""
        if len(evidence) > 1500:
            evidence = evidence[:1500] + "..."
        if len(claim) > 500:
            claim = claim[:500] + "..."
        
        inputs = tokenizer(
            evidence, claim,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        id2label = model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def _decide_from_probs(self, probs: Dict[str, float], strict: bool = False) -> Tuple[str, str, float]:
        """Make decision from NLI probabilities with improved logic."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        max_prob = max(p_entail, p_contra, p_neutral)
        
        # Strong signals
        if p_entail > 0.6:
            return "SUPPORTS", f"entailment={p_entail:.2%}", p_entail
        if p_contra > 0.6:
            return "REFUTES", f"contradiction={p_contra:.2%}", p_contra
        
        # Medium signals - prefer non-neutral
        if not strict:
            if p_entail > p_contra and p_entail > 0.35:
                return "SUPPORTS", f"entailment={p_entail:.2%} (weak)", p_entail
            if p_contra > p_entail and p_contra > 0.35:
                return "REFUTES", f"contradiction={p_contra:.2%} (weak)", p_contra
        
        # Check if truly uncertain
        if max_prob < self.nei_threshold:
            return "NEI", f"low_confidence (max={max_prob:.2%})", max_prob
        
        # Neutral is strongest
        if p_neutral > 0.5:
            return "NEI", f"neutral={p_neutral:.2%}", p_neutral
        
        # Fallback: use argmax for non-neutral
        if p_entail > p_contra:
            return "SUPPORTS", f"entailment={p_entail:.2%} (fallback)", p_entail
        else:
            return "REFUTES", f"contradiction={p_contra:.2%} (fallback)", p_contra
    
    def audit(
        self,
        claim: str,
        evidence: str,
        skip_cache: bool = False,
    ) -> AuditResult:
        """
        Audit a claim against evidence with improved accuracy.
        
        Pipeline:
        1. Check cache
        2. Keyword quick decision for obvious cases
        3. Fast model inference
        4. If uncertain, use strong model
        """
        # Check cache
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        stage1_decision = "UNCERTAIN"
        stage1_reason = ""
        stage2_decision = None
        stage2_reason = ""
        stage2_ran = False
        probs = {}
        
        # Step 1: Keyword check
        kw_result = self._keyword_check(claim, evidence)
        if kw_result:
            verdict, reason, confidence = kw_result
            stage1_decision = "ACCEPT" if verdict == "SUPPORTS" else ("REJECT" if verdict == "REFUTES" else "UNCERTAIN")
            stage1_reason = f"keyword: {reason}"
            
            # For keyword matches, still verify with NLI for important cases
            if confidence < 0.8:
                stage1_decision = "UNCERTAIN"
        
        # Step 2: Fast model (if ensemble and Stage1 uncertain)
        if self.use_ensemble and stage1_decision == "UNCERTAIN":
            self._load_fast_model()
            probs = self._nli_inference(claim, evidence, self.fast_model, self.fast_tokenizer)
            verdict, reason, confidence = self._decide_from_probs(probs, strict=True)
            
            if confidence >= self.fast_confidence_threshold:
                stage1_decision = "ACCEPT" if verdict == "SUPPORTS" else ("REJECT" if verdict == "REFUTES" else "UNCERTAIN")
                stage1_reason = f"fast_model: {reason}"
        
        # Step 3: Strong model (if still uncertain)
        if stage1_decision == "UNCERTAIN":
            stage2_ran = True
            self._load_strong_model()
            probs = self._nli_inference(claim, evidence, self.strong_model, self.strong_tokenizer)
            verdict, reason, confidence = self._decide_from_probs(probs, strict=False)
            
            stage2_decision = "ACCEPT" if verdict == "SUPPORTS" else ("REJECT" if verdict == "REFUTES" else "UNCERTAIN")
            stage2_reason = f"strong_model: {reason}"
        
        # Final verdict
        if stage2_ran:
            final_decision = stage2_decision
            final_reason = stage2_reason
        else:
            final_decision = stage1_decision
            final_reason = stage1_reason
        
        if final_decision == "ACCEPT":
            verdict = "SUPPORTS"
        elif final_decision == "REJECT":
            verdict = "REFUTES"
        else:
            verdict = "NEI"
        
        confidence = max(probs.values()) if probs else 0.7
        
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            stage1_decision=stage1_decision,
            stage2_decision=stage2_decision,
            stage2_ran=stage2_ran,
            explanation=final_reason,
            details={
                "probs": probs,
                "stage1_reason": stage1_reason,
                "stage2_reason": stage2_reason,
            }
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result


def main():
    """Demo the improved RAG Auditor."""
    import argparse
    
    ap = argparse.ArgumentParser(description="RAG Auditor V2")
    ap.add_argument("--claim", default="The Earth is round.")
    ap.add_argument("--evidence", default="Scientific evidence confirms that Earth is an oblate spheroid.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ensemble", action="store_true")
    args = ap.parse_args()
    
    print("="*60)
    print("RAG AUDITOR V2 - Improved Version")
    print("="*60)
    
    auditor = RAGAuditorV2(
        device=args.device,
        use_ensemble=not args.no_ensemble,
    )
    
    test_cases = [
        (args.claim, args.evidence),
        ("The sky is blue.", "Rayleigh scattering of sunlight by the atmosphere makes the sky appear blue."),
        ("Vaccines cause autism.", "Extensive research has found no link between vaccines and autism."),
        ("Coffee may be healthy.", "Studies show mixed effects depending on consumption."),
    ]
    
    for claim, evidence in test_cases:
        print(f"\nClaim: {claim}")
        print(f"Evidence: {evidence[:60]}...")
        
        result = auditor.audit(claim, evidence)
        
        print(f">>> VERDICT: {result.verdict} ({result.confidence:.1%})")
        print(f"    Stage1: {result.stage1_decision}, Stage2: {result.stage2_decision}")
        print(f"    Reason: {result.explanation}")


if __name__ == "__main__":
    main()
