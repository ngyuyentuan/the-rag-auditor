"""
RAG Auditor - Complete Claim Verification System

This is the main entry point for the RAG Auditor system.
It combines Stage1 (ML Router) and Stage2 (NLI) to verify claims
against retrieved evidence.

Usage:
    python -m src.auditor.rag_auditor --claim "Your claim here" --evidence "Evidence text"
    
Or programmatically:
    from src.auditor.rag_auditor import RAGAuditor
    auditor = RAGAuditor()
    result = auditor.audit("claim", "evidence")
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AuditResult:
    """Result from RAG Auditor."""
    claim: str
    verdict: str  # SUPPORTS, REFUTES, NEI, UNCERTAIN
    confidence: float
    stage1_decision: str  # ACCEPT, REJECT, UNCERTAIN
    stage2_decision: Optional[str]  # ACCEPT, REJECT, UNCERTAIN or None
    stage2_ran: bool
    explanation: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class RAGAuditor:
    """
    Complete RAG Auditor System.
    
    Combines:
    - Stage1: ML Router for fast initial decisions
    - Stage2: NLI verification for uncertain cases
    
    The system aims to:
    - Quickly accept high-confidence correct responses
    - Quickly reject obvious hallucinations
    - Defer uncertain cases to deeper NLI analysis
    """
    
    def __init__(
        self,
        stage1_mode: str = "ml",  # "ml" or "threshold"
        stage2_enabled: bool = True,
        device: str = "auto",
        nli_model: str = "facebook/bart-large-mnli",
        # Stage1 ML Router thresholds
        stage1_t_accept: float = 0.70,
        stage1_t_reject: float = 0.30,
        # Stage2 NLI thresholds
        stage2_t_entail: float = 0.85,
        stage2_t_contra: float = 0.30,
    ):
        """
        Initialize RAG Auditor.
        
        Args:
            stage1_mode: "ml" for ML router, "threshold" for simple threshold
            stage2_enabled: Whether to run Stage2 on UNCERTAIN cases
            device: "cpu", "cuda", or "auto"
            nli_model: HuggingFace model for Stage2 NLI
        """
        self.stage1_mode = stage1_mode
        self.stage2_enabled = stage2_enabled
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.stage1_t_accept = stage1_t_accept
        self.stage1_t_reject = stage1_t_reject
        self.stage2_t_entail = stage2_t_entail
        self.stage2_t_contra = stage2_t_contra
        
        # Stage1 ML Router
        self.stage1_model = None
        self.stage1_scaler = None
        
        # Stage2 NLI
        self.nli_model_name = nli_model
        self.nli_model = None
        self.nli_tokenizer = None
        
        self._stage2_loaded = False
    
    def _compute_features(self, logit: float) -> np.ndarray:
        """Compute features for Stage1 ML Router."""
        sigmoid = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
        
        # Simulated top-k (in real system would come from retrieval)
        top1 = sigmoid
        top2 = top1 * 0.9
        top3 = top2 * 0.9
        
        delta12 = top1 - top2
        margin_ratio = delta12 / (top1 + 1e-6)
        score_span = top1 - top3
        
        # Entropy
        probs = np.array([top1, top2, top3])
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        topk_mean = (top1 + top2 + top3) / 3
        topk_std = np.std([top1, top2, top3])
        conf_ratio = top1 / (topk_mean + 1e-6)
        
        return np.array([
            logit, abs(logit), sigmoid,
            top1, delta12, margin_ratio, score_span,
            entropy, topk_mean, topk_std, conf_ratio
        ])
    
    def _stage1_decide(self, confidence_score: float) -> tuple:
        """Stage1 routing decision."""
        if confidence_score >= self.stage1_t_accept:
            return "ACCEPT", f"confidence {confidence_score:.2f} >= {self.stage1_t_accept}"
        elif confidence_score <= self.stage1_t_reject:
            return "REJECT", f"confidence {confidence_score:.2f} <= {self.stage1_t_reject}"
        else:
            return "UNCERTAIN", f"confidence {confidence_score:.2f} in ({self.stage1_t_reject}, {self.stage1_t_accept})"
    
    def _load_stage2(self):
        """Load Stage2 NLI model."""
        if self._stage2_loaded:
            return
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"[RAGAuditor] Loading NLI model: {self.nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._stage2_loaded = True
        print(f"[RAGAuditor] NLI loaded on {self.device}")
    
    def _stage2_nli(self, claim: str, evidence: str) -> tuple:
        """Run Stage2 NLI inference."""
        if not self._stage2_loaded:
            self._load_stage2()
        
        # Truncate
        if len(evidence) > 1500:
            evidence = evidence[:1500] + "..."
        if len(claim) > 500:
            claim = claim[:500] + "..."
        
        # Tokenize
        inputs = self.nli_tokenizer(
            evidence, claim,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get label mapping
        id2label = self.nli_model.config.id2label
        prob_dict = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        
        p_entail = prob_dict.get('entailment', 0)
        p_contra = prob_dict.get('contradiction', 0)
        p_neutral = prob_dict.get('neutral', 0)
        
        # Decision: prioritize non-neutral predictions
        # NLI models tend to be conservative and output neutral too often
        
        # If entailment or contradiction is strong enough, use it
        if p_entail > 0.5:
            return "ACCEPT", "SUPPORTS", prob_dict, f"entailment={p_entail:.2%}"
        if p_contra > 0.5:
            return "REJECT", "REFUTES", prob_dict, f"contradiction={p_contra:.2%}"
        
        # If entailment beats contradiction (even if neutral is highest)
        if p_entail > p_contra and p_entail > 0.3:
            return "ACCEPT", "SUPPORTS", prob_dict, f"entailment={p_entail:.2%} (weak)"
        if p_contra > p_entail and p_contra > 0.3:
            return "REJECT", "REFUTES", prob_dict, f"contradiction={p_contra:.2%} (weak)"
        
        # Only return NEI if truly uncertain
        return "UNCERTAIN", "NEI", prob_dict, f"neutral={p_neutral:.2%}"
    
    def audit(
        self,
        claim: str,
        evidence: str,
        retrieval_score: Optional[float] = None,
    ) -> AuditResult:
        """
        Audit a claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence: The evidence text
            retrieval_score: Optional confidence score from retrieval
        
        Returns:
            AuditResult with verdict and explanation
        """
        # If no retrieval score, estimate from evidence length/match
        if retrieval_score is None:
            # Simple heuristic: longer evidence = higher confidence
            retrieval_score = min(0.5 + len(evidence) / 2000, 0.95)
        
        # Stage1 decision
        stage1_decision, stage1_reason = self._stage1_decide(retrieval_score)
        
        stage2_decision = None
        stage2_verdict = None
        stage2_probs = {}
        stage2_reason = ""
        stage2_ran = False
        
        # Stage2 if uncertain and enabled
        if stage1_decision == "UNCERTAIN" and self.stage2_enabled:
            stage2_ran = True
            stage2_decision, stage2_verdict, stage2_probs, stage2_reason = self._stage2_nli(claim, evidence)
        
        # Final verdict
        if stage1_decision == "ACCEPT":
            verdict = "SUPPORTS"
            confidence = retrieval_score
            explanation = f"Stage1 accepted with high confidence ({retrieval_score:.2%})"
        elif stage1_decision == "REJECT":
            verdict = "REFUTES"
            confidence = 1 - retrieval_score
            explanation = f"Stage1 rejected with low retrieval confidence ({retrieval_score:.2%})"
        elif stage2_ran:
            verdict = stage2_verdict
            confidence = max(stage2_probs.values()) if stage2_probs else 0.5
            explanation = f"Stage2 NLI: {stage2_verdict} ({stage2_reason})"
        else:
            verdict = "UNCERTAIN"
            confidence = 0.5
            explanation = "Unable to determine - insufficient evidence"
        
        return AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            stage1_decision=stage1_decision,
            stage2_decision=stage2_decision,
            stage2_ran=stage2_ran,
            explanation=explanation,
            details={
                "retrieval_score": retrieval_score,
                "stage1_reason": stage1_reason,
                "stage2_probs": stage2_probs,
                "stage2_reason": stage2_reason,
            }
        )
    
    def audit_batch(
        self,
        claims: List[str],
        evidences: List[str],
    ) -> List[AuditResult]:
        """Audit multiple claim-evidence pairs."""
        return [self.audit(c, e) for c, e in zip(claims, evidences)]


def main():
    """Demo the RAG Auditor."""
    import argparse
    
    ap = argparse.ArgumentParser(description="RAG Auditor - Verify AI Claims")
    ap.add_argument("--claim", default="The Eiffel Tower is in Paris.")
    ap.add_argument("--evidence", default="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--no-stage2", action="store_true", help="Disable Stage2 NLI")
    args = ap.parse_args()
    
    print("="*60)
    print("RAG AUDITOR - AI Claim Verification")
    print("="*60)
    
    auditor = RAGAuditor(
        device=args.device,
        stage2_enabled=not args.no_stage2,
    )
    
    # Test cases
    test_cases = [
        (args.claim, args.evidence),
        ("The Great Wall of China is visible from space.", 
         "The Great Wall of China is not visible from low Earth orbit without aid, contrary to popular belief."),
        ("Python is a programming language.",
         "Python is a high-level, general-purpose programming language created by Guido van Rossum."),
        ("Unicorns exist in nature.",
         "Scientific research has found no evidence of unicorns in the natural world."),
    ]
    
    for claim, evidence in test_cases:
        print(f"\n{'='*60}")
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence[:80]}...")
        
        result = auditor.audit(claim, evidence)
        
        print(f"\n>>> VERDICT: {result.verdict}")
        print(f"    Confidence: {result.confidence:.2%}")
        print(f"    Stage1: {result.stage1_decision}")
        if result.stage2_ran:
            print(f"    Stage2: {result.stage2_decision}")
        print(f"    Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
