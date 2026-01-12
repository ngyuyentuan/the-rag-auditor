"""
RAG Auditor V3 - Final Optimized Version

Improvements over V2:
1. Dedicated NEI detection using probability spread
2. Calibrated thresholds
3. Batch inference for speed
4. Better uncertainty handling
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


class RAGAuditorV3:
    """
    Final optimized RAG Auditor with:
    1. Dedicated NEI detection using probability entropy
    2. Calibrated thresholds from benchmark data
    3. Batch inference for speed
    4. Better uncertainty quantification
    """
    
    # Calibrated thresholds (from benchmark analysis)
    THRESHOLDS = {
        'entail_high': 0.55,      # High confidence support
        'entail_low': 0.35,       # Low confidence support  
        'contra_high': 0.55,      # High confidence refute
        'contra_low': 0.35,       # Low confidence refute
        'nei_entropy': 0.9,       # Entropy threshold for NEI
        'nei_spread': 0.25,       # Max prob spread for NEI
    }
    
    NEI_KEYWORDS = ['may', 'might', 'could', 'possibly', 'some', 'mixed', 
                    'unclear', 'depends', 'debated', 'controversial']
    
    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: str = "auto",
        enable_cache: bool = True,
        use_nei_detection: bool = True,
    ):
        self.model_name = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.use_nei_detection = use_nei_detection
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._cache = {}
    
    def _load_model(self):
        if self._loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[RAGAuditorV3] Loading model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _compute_entropy(self, probs: Dict[str, float]) -> float:
        """Compute entropy of probability distribution."""
        values = np.array(list(probs.values()))
        values = values[values > 0]
        return -np.sum(values * np.log(values + 1e-10))
    
    def _compute_spread(self, probs: Dict[str, float]) -> float:
        """Compute spread (max - min) of probabilities."""
        values = list(probs.values())
        return max(values) - min(values)
    
    def _detect_nei(self, probs: Dict[str, float], evidence: str) -> Tuple[bool, str]:
        """
        Dedicated NEI detection using multiple signals:
        1. High entropy (uncertainty)
        2. Low probability spread (no clear winner)
        3. NEI keywords in evidence
        """
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        entropy = self._compute_entropy(probs)
        spread = self._compute_spread(probs)
        
        # Check for high entropy (uncertainty)
        if entropy > self.THRESHOLDS['nei_entropy']:
            return True, f"high_entropy ({entropy:.2f})"
        
        # Check for low spread (no clear winner)
        if spread < self.THRESHOLDS['nei_spread']:
            return True, f"low_spread ({spread:.2f})"
        
        # Check for NEI keywords
        evidence_lower = evidence.lower()
        nei_keyword_count = sum(1 for kw in self.NEI_KEYWORDS if kw in evidence_lower)
        if nei_keyword_count >= 2 and p_neutral > 0.25:
            return True, f"nei_keywords ({nei_keyword_count})"
        
        # Neutral is highest by significant margin
        if p_neutral > p_entail + 0.1 and p_neutral > p_contra + 0.1:
            return True, f"neutral_dominant ({p_neutral:.2%})"
        
        return False, ""
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        """Run NLI inference."""
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
    
    def _nli_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Batch NLI inference for speed."""
        if not self._loaded:
            self._load_model()
        
        # Prepare inputs
        claims = [p[0][:500] for p in pairs]
        evidences = [p[1][:1500] for p in pairs]
        
        inputs = self.tokenizer(
            evidences, claims,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            all_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        id2label = self.model.config.id2label
        results = []
        for probs in all_probs:
            results.append({id2label[i].lower(): float(probs[i]) for i in range(len(probs))})
        return results
    
    def _decide(self, probs: Dict[str, float], evidence: str) -> Tuple[str, str, float]:
        """Make final decision with calibrated thresholds."""
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        # Step 1: Check for NEI
        if self.use_nei_detection:
            is_nei, nei_reason = self._detect_nei(probs, evidence)
            if is_nei:
                return "NEI", f"nei: {nei_reason}", max(p_neutral, 0.5)
        
        # Step 2: Check high confidence cases
        if p_entail > self.THRESHOLDS['entail_high']:
            return "SUPPORTS", f"entailment={p_entail:.2%}", p_entail
        if p_contra > self.THRESHOLDS['contra_high']:
            return "REFUTES", f"contradiction={p_contra:.2%}", p_contra
        
        # Step 3: Check medium confidence with margin
        if p_entail > self.THRESHOLDS['entail_low'] and p_entail > p_contra + 0.1:
            return "SUPPORTS", f"entailment={p_entail:.2%} (margin)", p_entail
        if p_contra > self.THRESHOLDS['contra_low'] and p_contra > p_entail + 0.1:
            return "REFUTES", f"contradiction={p_contra:.2%} (margin)", p_contra
        
        # Step 4: Argmax fallback
        if p_entail > p_contra and p_entail > p_neutral:
            return "SUPPORTS", f"entailment={p_entail:.2%} (argmax)", p_entail
        if p_contra > p_entail and p_contra > p_neutral:
            return "REFUTES", f"contradiction={p_contra:.2%} (argmax)", p_contra
        
        return "NEI", f"neutral={p_neutral:.2%}", p_neutral
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        """Audit a claim against evidence."""
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        probs = self._nli_inference(claim, evidence)
        verdict, reason, confidence = self._decide(probs, evidence)
        
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            stage1_decision=verdict,
            stage2_decision=None,
            stage2_ran=False,
            explanation=reason,
            details={"probs": probs}
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result
    
    def audit_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[AuditResult]:
        """Batch audit for speed."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            
            # Check cache first
            to_process = []
            cached_indices = []
            for j, (claim, evidence) in enumerate(batch):
                cache_key = self._cache_key(claim, evidence)
                if self.enable_cache and cache_key in self._cache:
                    cached_indices.append((j, self._cache[cache_key]))
                else:
                    to_process.append((j, claim, evidence))
            
            # Batch inference for uncached
            if to_process:
                batch_probs = self._nli_batch([(c, e) for _, c, e in to_process])
                
                for (j, claim, evidence), probs in zip(to_process, batch_probs):
                    verdict, reason, confidence = self._decide(probs, evidence)
                    result = AuditResult(
                        claim=claim, verdict=verdict, confidence=confidence,
                        stage1_decision=verdict, stage2_decision=None,
                        stage2_ran=False, explanation=reason, details={"probs": probs}
                    )
                    if self.enable_cache:
                        self._cache[self._cache_key(claim, evidence)] = result
                    cached_indices.append((j, result))
            
            # Sort by original order
            cached_indices.sort(key=lambda x: x[0])
            results.extend([r for _, r in cached_indices])
        
        return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="The sky is blue.")
    ap.add_argument("--evidence", default="Rayleigh scattering makes the sky appear blue.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    
    print("="*60)
    print("RAG AUDITOR V3 - Final Optimized")
    print("="*60)
    
    auditor = RAGAuditorV3(device=args.device)
    
    tests = [
        (args.claim, args.evidence),
        ("Coffee is healthy.", "Studies show mixed effects depending on consumption."),
        ("AI will replace all jobs.", "AI automates some tasks but also creates new jobs."),
        ("The earth is flat.", "Scientific evidence confirms Earth is a sphere."),
    ]
    
    for claim, evidence in tests:
        result = auditor.audit(claim, evidence)
        print(f"\nClaim: {claim}")
        print(f">>> {result.verdict} ({result.confidence:.1%}): {result.explanation}")


if __name__ == "__main__":
    main()
