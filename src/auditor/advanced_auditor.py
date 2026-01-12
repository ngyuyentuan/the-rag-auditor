import sys
import re
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("advanced_auditor")


@dataclass
class AuditResult:
    claim: str
    verdict: str
    confidence: float
    calibrated_confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class AdvancedAuditor:
    VERSION = "2.0.0"
    
    YEAR_PATTERN = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
    NUMBER_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    
    FACTUAL_VERBS = [
        'is', 'was', 'are', 'were', 'has', 'have', 'had',
        'born', 'died', 'released', 'founded', 'created', 'made',
        'starred', 'appeared', 'played', 'directed', 'wrote', 'won',
        'located', 'based', 'known', 'called', 'named',
    ]
    
    NEGATION_MARKERS = [
        'not', "n't", 'never', 'no', 'none', 'neither', 'nor',
        'without', 'lack', 'fail', 'denied', 'refused', 'rejected',
        'only', 'exclusively', 'solely', 'just', 'merely',
    ]
    
    NATIONALITY_MAP = {
        'american': 'usa', 'british': 'uk', 'english': 'uk', 'french': 'france',
        'german': 'germany', 'italian': 'italy', 'spanish': 'spain',
        'japanese': 'japan', 'chinese': 'china', 'korean': 'korea',
        'australian': 'australia', 'canadian': 'canada', 'indian': 'india',
        'russian': 'russia', 'brazilian': 'brazil', 'mexican': 'mexico',
        'swedish': 'sweden', 'danish': 'denmark', 'dutch': 'netherlands',
    }
    
    GENRE_MAP = {
        'comedy': 'comedy', 'drama': 'drama', 'horror': 'horror',
        'action': 'action', 'thriller': 'thriller', 'romance': 'romance',
        'documentary': 'documentary', 'musical': 'musical',
    }
    
    def __init__(
        self,
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device: str = "auto",
        enable_cache: bool = True,
    ):
        self.nli_model_name = nli_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nli_model = None
        self.nli_tokenizer = None
        self.embedder = None
        self._nli_loaded = False
        self._embedder_loaded = False
        self._cache = {}
    
    def _load_nli(self):
        if self._nli_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._nli_loaded = True
    
    def _load_embedder(self):
        if self._embedder_loaded:
            return
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self._embedder_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.sha256(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _extract_years(self, text: str) -> Set[str]:
        return set(self.YEAR_PATTERN.findall(text))
    
    def _extract_numbers(self, text: str) -> Set[str]:
        return set(self.NUMBER_PATTERN.findall(text))
    
    def _extract_entities(self, text: str) -> Set[str]:
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return set(words)
    
    def _detect_nationality(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for nat, code in self.NATIONALITY_MAP.items():
            if nat in text_lower:
                return code
        return None
    
    def _detect_genre(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for genre in self.GENRE_MAP:
            if genre in text_lower:
                return genre
        return None
    
    def _has_only_exclusively(self, text: str) -> bool:
        text_lower = text.lower()
        return any(w in text_lower for w in ['only', 'exclusively', 'solely', 'just'])
    
    def _count_negations(self, text: str) -> int:
        text_lower = text.lower()
        return sum(1 for w in self.NEGATION_MARKERS if w in text_lower)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        if not self._embedder_loaded:
            self._load_embedder()
        emb = self.embedder.encode([text1, text2])
        sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8)
        return float(sim)
    
    def _check_year_contradiction(self, claim: str, evidence: str) -> Tuple[bool, float]:
        claim_years = self._extract_years(claim)
        evidence_years = self._extract_years(evidence)
        if claim_years and evidence_years:
            if claim_years.isdisjoint(evidence_years):
                return True, 0.8
        return False, 0.0
    
    def _check_nationality_contradiction(self, claim: str, evidence: str) -> Tuple[bool, float]:
        claim_nat = self._detect_nationality(claim)
        evidence_nat = self._detect_nationality(evidence)
        if claim_nat and evidence_nat and claim_nat != evidence_nat:
            return True, 0.7
        return False, 0.0
    
    def _check_only_contradiction(self, claim: str, evidence: str) -> Tuple[bool, float]:
        if self._has_only_exclusively(claim):
            claim_nat = self._detect_nationality(claim)
            evidence_nat = self._detect_nationality(evidence)
            if claim_nat and evidence_nat and claim_nat != evidence_nat:
                return True, 0.9
            claim_genre = self._detect_genre(claim)
            evidence_genre = self._detect_genre(evidence)
            if claim_genre and evidence_genre and claim_genre != evidence_genre:
                return True, 0.85
            claim_lower = claim.lower().replace('only', '').replace('exclusively', '')
            evidence_lower = evidence.lower()
            claim_key_words = set(re.findall(r'\b\w{4,}\b', claim_lower))
            evidence_words = set(re.findall(r'\b\w{4,}\b', evidence_lower))
            if claim_key_words - evidence_words:
                return True, 0.6
        return False, 0.0
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._nli_loaded:
            self._load_nli()
        evidence = evidence[:1500]
        claim = claim[:500]
        inputs = self.nli_tokenizer(
            evidence, claim,
            padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        id2label = self.nli_model.config.id2label
        return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    def _decide(self, probs: Dict[str, float], claim: str, evidence: str) -> Tuple[str, str, float]:
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        year_contra, year_conf = self._check_year_contradiction(claim, evidence)
        nat_contra, nat_conf = self._check_nationality_contradiction(claim, evidence)
        only_contra, only_conf = self._check_only_contradiction(claim, evidence)
        
        negation_count = self._count_negations(evidence)
        
        sig = f"E={p_entail:.2f},C={p_contra:.2f},N={p_neutral:.2f}"
        
        if year_contra:
            return "REFUTES", f"year_mismatch ({sig})", year_conf
        
        if nat_contra:
            return "REFUTES", f"nationality_mismatch ({sig})", nat_conf
        
        if only_contra:
            return "REFUTES", f"only_exclusivity ({sig})", only_conf
        
        if p_contra > 0.70:
            return "REFUTES", f"high_contra ({sig})", min(p_contra, 1.0)
        
        if p_entail > 0.70:
            return "SUPPORTS", f"high_entail ({sig})", min(p_entail, 1.0)
        
        if negation_count >= 2 and p_contra > 0.25:
            return "REFUTES", f"negation_pattern ({sig})", min(p_contra + 0.2, 1.0)
        
        if p_contra > 0.50 and p_contra > p_entail:
            return "REFUTES", f"med_contra ({sig})", min(p_contra, 1.0)
        
        if p_entail > 0.50 and p_entail > p_contra:
            return "SUPPORTS", f"med_entail ({sig})", min(p_entail, 1.0)
        
        if p_neutral > 0.65:
            return "NEI", f"high_neutral ({sig})", min(p_neutral, 1.0)
        
        margin = 0.10
        if p_entail > p_contra + margin and p_entail > p_neutral:
            return "SUPPORTS", f"margin_entail ({sig})", min(p_entail, 1.0)
        if p_contra > p_entail + margin and p_contra > p_neutral:
            return "REFUTES", f"margin_contra ({sig})", min(p_contra, 1.0)
        if p_neutral > p_entail and p_neutral > p_contra:
            return "NEI", f"margin_neutral ({sig})", min(p_neutral, 1.0)
        
        if p_entail >= p_contra and p_entail >= p_neutral:
            return "SUPPORTS", f"argmax ({sig})", min(p_entail * 0.85, 1.0)
        elif p_contra > p_entail and p_contra >= p_neutral:
            return "REFUTES", f"argmax ({sig})", min(p_contra * 0.85, 1.0)
        else:
            return "NEI", f"argmax ({sig})", min(p_neutral * 0.85, 1.0)
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        start = time.time()
        
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return AuditResult(
                claim=cached.claim, verdict=cached.verdict,
                confidence=cached.confidence,
                calibrated_confidence=cached.calibrated_confidence,
                explanation=cached.explanation, signals=cached.signals,
                latency_ms=0.0
            )
        
        probs = self._nli_inference(claim, evidence)
        verdict, reason, confidence = self._decide(probs, claim, evidence)
        calibrated = min(confidence * 1.05, 1.0)
        latency = (time.time() - start) * 1000
        
        result = AuditResult(
            claim=claim, verdict=verdict, confidence=confidence,
            calibrated_confidence=calibrated, explanation=reason,
            signals={"probs": probs}, latency_ms=latency
        )
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED AUDITOR TEST")
    print("=" * 60)
    auditor = AdvancedAuditor(device="cpu")
    tests = [
        ("Telemundo is an English-language network.", "Telemundo is an American Spanish-language television network.", "REFUTES"),
        ("Damon Albarn's debut album was released in 2011.", "Damon Albarn released his debut solo album in 2014.", "REFUTES"),
        ("Brad Wilk helped co-found Rage in 1962.", "Brad Wilk co-founded Rage Against the Machine in 1991.", "REFUTES"),
        ("Alexandra Daddario is Canadian.", "Alexandra Daddario is an American actress.", "REFUTES"),
        ("Savages was exclusively a German film.", "Savages is an American crime thriller film.", "REFUTES"),
        ("Fox 2000 Pictures released Soul Food.", "Soul Food was produced by Fox 2000 Pictures.", "SUPPORTS"),
        ("Tim Roth is an English actor.", "Tim Roth is an English actor.", "SUPPORTS"),
    ]
    correct = 0
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        correct += 1 if match else 0
        print(f"{'Y' if match else 'N'} [{expected:8}->{result.verdict:8}] {result.explanation}")
    print(f"\nAccuracy: {correct}/{len(tests)}")
