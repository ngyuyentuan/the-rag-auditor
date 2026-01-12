"""
Stage2 NLI Service - Real NLI Model Integration

This module provides real NLI inference using transformer models.
It processes claim-evidence pairs and returns verdict predictions.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class NLIResult:
    """Result from NLI inference."""
    verdict: str  # SUPPORTS, REFUTES, NEI
    decision: str  # ACCEPT, REJECT, UNCERTAIN
    confidence: float
    probs: Dict[str, float]  # entailment, contradiction, neutral
    reason: str


class Stage2NLIService:
    """
    Stage2 NLI Service for claim verification.
    
    Uses a transformer-based NLI model to determine if evidence
    supports, refutes, or is neutral to a claim.
    """
    
    # NLI label mapping
    LABEL_MAP = {
        'ENTAILMENT': 'SUPPORTS',
        'CONTRADICTION': 'REFUTES',
        'NEUTRAL': 'NEI',
        'entailment': 'SUPPORTS',
        'contradiction': 'REFUTES',
        'neutral': 'NEI',
    }
    
    def __init__(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        device: str = "auto",
        t_entail: float = 0.85,
        t_contra: float = 0.30,
        t_neutral: float = 0.50,
    ):
        """
        Initialize NLI service.
        
        Args:
            model_name: HuggingFace model for NLI
            device: 'cpu', 'cuda', or 'auto'
            t_entail: threshold for entailment → ACCEPT
            t_contra: threshold for contradiction → REJECT
            t_neutral: threshold for neutral → UNCERTAIN
        """
        self.model_name = model_name
        self.t_entail = t_entail
        self.t_contra = t_contra
        self.t_neutral = t_neutral
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the NLI model and tokenizer."""
        if self._loaded:
            return
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"[Stage2] Loading NLI model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping from model config
        self.id2label = self.model.config.id2label
        self._loaded = True
        print(f"[Stage2] Loaded on {self.device}")
    
    def _truncate_text(self, text: str, max_chars: int = 1500) -> str:
        """Truncate text to prevent token overflow."""
        if not isinstance(text, str):
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    def predict_single(
        self,
        claim: str,
        evidence: str,
    ) -> NLIResult:
        """
        Predict NLI for a single claim-evidence pair.
        
        Args:
            claim: The claim to verify
            evidence: The evidence text
        
        Returns:
            NLIResult with verdict and confidence
        """
        if not self._loaded:
            self.load()
        
        # Truncate if needed
        evidence = self._truncate_text(evidence)
        claim = self._truncate_text(claim, 500)
        
        # Tokenize (premise=evidence, hypothesis=claim for NLI)
        inputs = self.tokenizer(
            evidence,
            claim,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Map to labels
        prob_dict = {}
        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, str(idx)).lower()
            prob_dict[label] = float(prob)
        
        # Get probabilities
        p_entail = prob_dict.get('entailment', 0.0)
        p_contra = prob_dict.get('contradiction', 0.0)
        p_neutral = prob_dict.get('neutral', 0.0)
        
        # Make decision
        decision, reason = self._make_decision(p_entail, p_contra, p_neutral)
        
        # Map to verdict
        if decision == "ACCEPT":
            verdict = "SUPPORTS"
        elif decision == "REJECT":
            verdict = "REFUTES"
        else:
            verdict = "NEI"
        
        # Confidence is max probability
        confidence = max(p_entail, p_contra, p_neutral)
        
        return NLIResult(
            verdict=verdict,
            decision=decision,
            confidence=confidence,
            probs=prob_dict,
            reason=reason,
        )
    
    def _make_decision(
        self,
        p_entail: float,
        p_contra: float,
        p_neutral: float,
    ) -> Tuple[str, str]:
        """Make routing decision based on NLI probabilities."""
        # Check strongest signal above threshold
        if p_entail >= self.t_entail and p_entail > p_contra and p_entail > p_neutral:
            return "ACCEPT", "entailment_above_threshold"
        elif p_contra >= self.t_contra and p_contra > p_entail and p_contra > p_neutral:
            return "REJECT", "contradiction_above_threshold"
        elif p_neutral >= self.t_neutral:
            return "UNCERTAIN", "neutral_above_threshold"
        else:
            # Fallback: use argmax
            max_label = max(
                [('entailment', p_entail), ('contradiction', p_contra), ('neutral', p_neutral)],
                key=lambda x: x[1]
            )[0]
            if max_label == 'entailment':
                return "ACCEPT", "entailment_argmax"
            elif max_label == 'contradiction':
                return "REJECT", "contradiction_argmax"
            else:
                return "UNCERTAIN", "neutral_argmax"
    
    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],  # List of (claim, evidence)
        batch_size: int = 8,
    ) -> List[NLIResult]:
        """
        Predict NLI for multiple pairs.
        
        Args:
            pairs: List of (claim, evidence) tuples
            batch_size: Batch size for inference
        
        Returns:
            List of NLIResult
        """
        if not self._loaded:
            self.load()
        
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            for claim, evidence in batch:
                result = self.predict_single(claim, evidence)
                results.append(result)
        
        return results


def main():
    """Test the NLI service."""
    import argparse
    
    ap = argparse.ArgumentParser(description="Test Stage2 NLI Service")
    ap.add_argument("--claim", default="Climate change is caused by human activities.")
    ap.add_argument("--evidence", default="Scientific research shows that greenhouse gas emissions from burning fossil fuels are the primary driver of global warming.")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    
    print("="*60)
    print("Stage2 NLI Service Test")
    print("="*60)
    
    service = Stage2NLIService(device=args.device)
    
    # Test cases
    test_cases = [
        (args.claim, args.evidence),
        ("The Earth is flat.", "Scientific measurements and satellite imagery confirm Earth is a sphere."),
        ("Coffee is healthy.", "Studies show moderate coffee consumption may have some health benefits."),
    ]
    
    for claim, evidence in test_cases:
        print(f"\nClaim: {claim}")
        print(f"Evidence: {evidence[:100]}...")
        
        result = service.predict_single(claim, evidence)
        
        print(f"\n  Verdict: {result.verdict}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reason: {result.reason}")
        print(f"  Probs: E={result.probs.get('entailment', 0):.2%}, C={result.probs.get('contradiction', 0):.2%}, N={result.probs.get('neutral', 0):.2%}")


if __name__ == "__main__":
    main()
