"""
RAG Auditor Ultimate - Push for 85%+ Accuracy

Advanced techniques:
1. Semantic overlap scoring (word/phrase matching)
2. Better REFUTES detection with negation patterns
3. Multi-signal fusion from NLI + keywords + overlap
4. Claim-evidence alignment analysis
"""
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
import hashlib
import time
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AuditResult:
    claim: str
    verdict: str
    confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class RAGAuditorUltimate:
    """
    Ultimate RAG Auditor targeting 85%+ accuracy.
    
    Multi-signal approach:
    1. NLI probabilities
    2. Semantic overlap
    3. Negation patterns
    4. Keyword signals
    5. Claim-evidence alignment
    """
    
    # Strong refutation patterns
    REFUTE_PATTERNS = [
        r'\bnot\b.*\b(true|correct|accurate|real)\b',
        r'\bno\b.*\b(evidence|link|connection|proof)\b',
        r'\bfalse\b',
        r'\bmyth\b',
        r'\bcontrary\b.*\b(to|belief)\b',
        r'\bdisproven\b',
        r'\bdebunked\b',
        r'\bincorrect\b',
        r'\bwrong\b',
        r'\bnever\b',
        r'\bcannot\b',
        r'\bno\s+link\b',
        r'\bno\s+evidence\b',
        r'\bno\s+connection\b',
    ]
    
    # Strong support patterns
    SUPPORT_PATTERNS = [
        r'\bconfirms?\b',
        r'\bproves?\b',
        r'\bdemonstrates?\b',
        r'\bshows?\b.*\bthat\b',
        r'\bestablished\b',
        r'\bverified\b',
        r'\bfact\b',
        r'\bevidence\s+(shows?|confirms?|proves?)\b',
    ]
    
    # NEI patterns
    NEI_PATTERNS = [
        r'\bmixed\b.*\b(results?|evidence|findings?)\b',
        r'\bdepends?\b',
        r'\bvaries?\b',
        r'\bunclear\b',
        r'\bsome\b.*\bwhile\b.*\bother\b',
        r'\bcontroversial\b',
        r'\binconclusive\b',
    ]
    
    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: str = "auto",
        enable_cache: bool = True,
    ):
        self.model_name = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._cache = {}
    
    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[Ultimate] Loading: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into words."""
        return set(re.findall(r'\w+', text.lower()))
    
    def _compute_overlap(self, claim: str, evidence: str) -> float:
        """Compute semantic overlap between claim and evidence."""
        claim_tokens = self._tokenize(claim)
        evidence_tokens = self._tokenize(evidence)
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'that', 'this', 'it', 'its'}
        
        claim_tokens = claim_tokens - stopwords
        evidence_tokens = evidence_tokens - stopwords
        
        if not claim_tokens:
            return 0.0
        
        overlap = len(claim_tokens & evidence_tokens)
        return overlap / len(claim_tokens)
    
    def _check_patterns(self, text: str, patterns: List[str]) -> int:
        """Count pattern matches in text."""
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                count += 1
        return count
    
    def _analyze_negation(self, claim: str, evidence: str) -> Tuple[bool, str]:
        """Check if evidence negates the claim."""
        # Extract key content words from claim
        claim_words = self._tokenize(claim)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        claim_keywords = claim_words - stopwords
        
        evidence_lower = evidence.lower()
        
        # Check for negation + claim keyword
        negation_words = ['not', 'no', 'never', 'cannot', "can't", "isn't", "aren't", "wasn't", "weren't"]
        
        for neg in negation_words:
            if neg in evidence_lower:
                # Check if any claim keyword appears near negation
                for kw in claim_keywords:
                    if kw in evidence_lower:
                        # Found negation + claim keyword
                        return True, f"negation_pattern: {neg}...{kw}"
        
        return False, ""
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._loaded:
            self._load_model()
        
        evidence = evidence[:1500] if len(evidence) > 1500 else evidence
        claim = claim[:500] if len(claim) > 500 else claim
        
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
    
    def _fuse_signals(
        self,
        probs: Dict[str, float],
        overlap: float,
        refute_patterns: int,
        support_patterns: int,
        nei_patterns: int,
        negation: bool,
    ) -> Tuple[str, str, float]:
        """Fuse all signals to make final decision."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        # Compute adjusted scores with stronger weights
        support_score = p_entail + (support_patterns * 0.15) + (overlap * 0.25)
        refute_score = p_contra + (refute_patterns * 0.18) + (0.25 if negation else 0)
        nei_score = p_neutral + (nei_patterns * 0.20) - (overlap * 0.1)  # High overlap = less NEI
        
        # Cap scores at 1.0
        support_score = min(support_score, 1.0)
        refute_score = min(refute_score, 1.0)
        nei_score = max(min(nei_score, 1.0), 0)
        
        signals = f"S={support_score:.2f}, R={refute_score:.2f}, N={nei_score:.2f}"
        
        # Strong REFUTES: pattern match or negation
        if refute_patterns >= 1 and p_contra > 0.25:
            return "REFUTES", f"refute_pattern ({signals})", max(refute_score, p_contra)
        
        if negation and p_contra > 0.2:
            return "REFUTES", f"negation ({signals})", max(refute_score, p_contra)
        
        # Strong NLI signals
        if p_contra > 0.6:
            return "REFUTES", f"nli_strong ({signals})", p_contra
        if p_entail > 0.6:
            return "SUPPORTS", f"nli_strong ({signals})", p_entail
        
        # High overlap + support signal
        if overlap > 0.5 and p_entail > 0.3 and p_entail > p_contra:
            return "SUPPORTS", f"overlap_support ({signals})", support_score
        
        # NEI patterns strong signal
        if nei_patterns >= 2:
            return "NEI", f"nei_patterns ({signals})", nei_score
        
        # Medium NLI with margin
        if p_entail > p_contra + 0.15:
            return "SUPPORTS", f"entail_margin ({signals})", p_entail
        if p_contra > p_entail + 0.1:
            return "REFUTES", f"contra_margin ({signals})", p_contra
        
        # Final: score-based
        if support_score > refute_score and support_score > nei_score:
            return "SUPPORTS", f"score ({signals})", support_score
        elif refute_score > support_score and refute_score > nei_score:
            return "REFUTES", f"score ({signals})", refute_score
        else:
            return "NEI", f"score ({signals})", nei_score
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        """Audit with multi-signal fusion."""
        start = time.time()
        
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute all signals
        probs = self._nli_inference(claim, evidence)
        overlap = self._compute_overlap(claim, evidence)
        refute_patterns = self._check_patterns(evidence, self.REFUTE_PATTERNS)
        support_patterns = self._check_patterns(evidence, self.SUPPORT_PATTERNS)
        nei_patterns = self._check_patterns(evidence, self.NEI_PATTERNS)
        negation, neg_reason = self._analyze_negation(claim, evidence)
        
        # Fuse signals
        verdict, reason, confidence = self._fuse_signals(
            probs, overlap, refute_patterns, support_patterns, nei_patterns, negation
        )
        
        latency = (time.time() - start) * 1000
        
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=reason,
            signals={
                "nli_probs": probs,
                "overlap": overlap,
                "refute_patterns": refute_patterns,
                "support_patterns": support_patterns,
                "nei_patterns": nei_patterns,
                "negation": negation,
            },
            latency_ms=latency,
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result


def main():
    print("="*60)
    print("RAG AUDITOR ULTIMATE - Testing")
    print("="*60)
    
    auditor = RAGAuditorUltimate(device="cpu")
    
    tests = [
        ("The earth is flat.", "Scientific evidence confirms Earth is a sphere.", "REFUTES"),
        ("Vaccines cause autism.", "Research found no link between vaccines and autism.", "REFUTES"),
        ("The sky is blue.", "Rayleigh scattering makes the sky appear blue.", "SUPPORTS"),
        ("Coffee is healthy.", "Studies show mixed effects depending on consumption.", "NEI"),
        ("Python is a language.", "Python is a programming language created by Guido.", "SUPPORTS"),
    ]
    
    correct = 0
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = "✓" if result.verdict == expected else "✗"
        if result.verdict == expected:
            correct += 1
        print(f"{match} {claim[:30]:<30} Expected:{expected:<10} Got:{result.verdict:<10} ({result.confidence:.1%})")
    
    print(f"\nAccuracy: {correct}/{len(tests)} ({correct/len(tests):.1%})")


if __name__ == "__main__":
    main()
