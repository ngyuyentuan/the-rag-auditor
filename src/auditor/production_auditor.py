import sys
import re
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
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
    calibrated_confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class ProductionAuditor:
    
    NEGATION_WORDS = {
        'en': [
            'not', 'no', 'never', 'none', "n't", 'neither', 'nor', 'nothing',
            'nobody', 'nowhere', 'hardly', 'barely', 'scarcely', 'seldom',
            'rarely', 'without', 'lack', 'lacks', 'lacking', 'absent',
            'fail', 'fails', 'failed', 'failure', 'deny', 'denies', 'denied',
            'refuse', 'refuses', 'refused', 'reject', 'rejects', 'rejected',
            'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
            "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't",
        ],
        'vi': [
            'khong', 'chua', 'khong co', 'khong phai', 'khong duoc', 'khong the',
            'khong bao gio', 'chang', 'chang co', 'chang bao gio', 'khong he',
            'thieu', 'vang', 'khong con', 'het', 'mat',
        ],
    }
    
    CONTRADICTION_PATTERNS = [
        r'\b(false|incorrect|wrong|untrue|inaccurate|mistaken|erroneous)\b',
        r'\b(disprove[sd]?|refute[sd]?|contradict[s]?|debunk[s]?)\b',
        r'\b(myth|hoax|lie|fabricat|misinform|fake)\b',
        r'\b(no evidence|no proof|no link|no connection|no relation)\b',
        r'\b(contrary|opposite|inverse|reverse)\b',
        r'\b(sai|khong dung|khong chinh xac|bia dat|gia)\b',
    ]
    
    ANTONYM_PAIRS = [
        ('true', 'false'), ('correct', 'incorrect'), ('right', 'wrong'),
        ('safe', 'dangerous'), ('safe', 'unsafe'), ('healthy', 'unhealthy'),
        ('good', 'bad'), ('positive', 'negative'), ('increase', 'decrease'),
        ('more', 'less'), ('higher', 'lower'), ('larger', 'smaller'),
        ('before', 'after'), ('cause', 'prevent'), ('help', 'harm'),
        ('support', 'oppose'), ('agree', 'disagree'), ('accept', 'reject'),
        ('include', 'exclude'), ('allow', 'forbid'), ('enable', 'disable'),
        ('certain', 'uncertain'), ('probable', 'improbable'),
        ('possible', 'impossible'), ('exist', 'not exist'),
        ('proven', 'unproven'), ('confirmed', 'unconfirmed'),
        ('effective', 'ineffective'), ('successful', 'unsuccessful'),
        ('alive', 'dead'), ('present', 'absent'), ('real', 'fake'),
    ]
    
    SUPPORT_PATTERNS = [
        r'\b(confirm[s]?|prove[s]?|show[s]?|demonstrate[s]?|establish[es]?)\b',
        r'\b(evidence shows?|research shows?|studies show?|data shows?)\b',
        r'\b(consistent with|in line with|supports?)\b',
        r'\b(valid|accurate|correct|true|right)\b',
    ]
    
    NEI_PATTERNS = [
        r'\b(unclear|uncertain|unknown|debated?|controversial?)\b',
        r'\b(may|might|could|possibly|perhaps|maybe)\b',
        r'\b(mixed|inconclusive|insufficient|limited)\b',
        r'\b(some|partially|depends?|varies?)\b',
    ]
    
    CALIBRATION = {
        'SUPPORTS': {'a': 1.2, 'b': 0.1},
        'REFUTES': {'a': 1.5, 'b': 0.05},
        'NEI': {'a': 1.0, 'b': 0.15},
    }
    
    def __init__(
        self,
        nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto",
        enable_cache: bool = True,
    ):
        self.nli_model_name = nli_model
        self.embedding_model_name = embedding_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cache = enable_cache
        self.nli_model = None
        self.nli_tokenizer = None
        self.embedder = None
        self._nli_loaded = False
        self._embedder_loaded = False
        self._cache = {}
        self._contradiction_re = [re.compile(p, re.IGNORECASE) for p in self.CONTRADICTION_PATTERNS]
        self._support_re = [re.compile(p, re.IGNORECASE) for p in self.SUPPORT_PATTERNS]
        self._nei_re = [re.compile(p, re.IGNORECASE) for p in self.NEI_PATTERNS]
    
    def _load_nli(self):
        if self._nli_loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"[Production] Loading NLI: {self.nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self._nli_loaded = True
    
    def _load_embedder(self):
        if self._embedder_loaded:
            return
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        self._embedder_loaded = True
    
    def _cache_key(self, claim: str, evidence: str) -> str:
        return hashlib.md5(f"{claim}|||{evidence}".encode()).hexdigest()
    
    def _count_negations(self, text: str) -> int:
        text_lower = text.lower()
        count = 0
        for lang_words in self.NEGATION_WORDS.values():
            for word in lang_words:
                if word in text_lower:
                    count += 1
        return count
    
    def _check_antonyms(self, claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        antonym_score = 0.0
        for word1, word2 in self.ANTONYM_PAIRS:
            if (word1 in claim_lower and word2 in evidence_lower) or \
               (word2 in claim_lower and word1 in evidence_lower):
                antonym_score += 0.2
        return min(antonym_score, 0.5)
    
    def _check_patterns(self, text: str, patterns: List[re.Pattern]) -> int:
        count = 0
        for pattern in patterns:
            if pattern.search(text):
                count += 1
        return count
    
    def _nli_inference(self, claim: str, evidence: str) -> Dict[str, float]:
        if not self._nli_loaded:
            self._load_nli()
        evidence = evidence[:1500] if len(evidence) > 1500 else evidence
        claim = claim[:500] if len(claim) > 500 else claim
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
    
    def _calibrate_confidence(self, raw_conf: float, verdict: str) -> float:
        params = self.CALIBRATION.get(verdict, {'a': 1.0, 'b': 0.0})
        calibrated = 1 / (1 + np.exp(-(params['a'] * raw_conf + params['b'])))
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def _decide(self, probs: Dict[str, float], evidence: str, claim: str) -> Tuple[str, str, float]:
        p_entail = probs.get('entailment', 0)
        p_contra = probs.get('contradiction', 0)
        p_neutral = probs.get('neutral', 0)
        
        negation_count = self._count_negations(evidence)
        antonym_score = self._check_antonyms(claim, evidence)
        contra_patterns = self._check_patterns(evidence, self._contradiction_re)
        support_patterns = self._check_patterns(evidence, self._support_re)
        nei_patterns = self._check_patterns(evidence, self._nei_re)
        
        support_score = p_entail + (support_patterns * 0.12)
        refute_score = p_contra + (negation_count * 0.08) + (antonym_score * 0.5) + (contra_patterns * 0.15)
        nei_score = p_neutral + (nei_patterns * 0.10)
        
        signals = f"E={p_entail:.2f},C={p_contra:.2f},N={p_neutral:.2f},neg={negation_count},ant={antonym_score:.2f}"
        
        if p_contra > 0.45 or refute_score > 0.6:
            return "REFUTES", f"strong_contra ({signals})", min(refute_score, 1.0)
        
        if contra_patterns >= 2 or (negation_count >= 3 and p_contra > 0.25):
            return "REFUTES", f"multi_contra ({signals})", min(refute_score, 1.0)
        
        if antonym_score >= 0.3 and p_contra > 0.20:
            return "REFUTES", f"antonym_contra ({signals})", min(refute_score, 1.0)
        
        if p_entail > 0.55 or support_score > 0.65:
            return "SUPPORTS", f"strong_entail ({signals})", min(support_score, 1.0)
        
        if support_patterns >= 2 and p_entail > 0.30:
            return "SUPPORTS", f"pattern_entail ({signals})", min(support_score, 1.0)
        
        if (negation_count >= 2 or contra_patterns >= 1) and p_contra > 0.25:
            return "REFUTES", f"pattern_contra ({signals})", min(refute_score, 1.0)
        
        if nei_patterns >= 2 or p_neutral > 0.50:
            return "NEI", f"nei_signals ({signals})", min(nei_score, 1.0)
        
        margin = 0.10
        if p_entail > p_contra + margin and p_entail > p_neutral:
            return "SUPPORTS", f"margin_entail ({signals})", min(p_entail, 1.0)
        if p_contra > p_entail + margin and p_contra > p_neutral:
            return "REFUTES", f"margin_contra ({signals})", min(p_contra, 1.0)
        
        if support_score >= refute_score and support_score >= nei_score:
            return "SUPPORTS", f"score ({signals})", min(support_score, 1.0)
        elif refute_score > support_score and refute_score >= nei_score:
            return "REFUTES", f"score ({signals})", min(refute_score, 1.0)
        else:
            return "NEI", f"score ({signals})", min(nei_score, 1.0)
    
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        start = time.time()
        cache_key = self._cache_key(claim, evidence)
        if self.enable_cache and not skip_cache and cache_key in self._cache:
            return self._cache[cache_key]
        probs = self._nli_inference(claim, evidence)
        verdict, reason, raw_confidence = self._decide(probs, evidence, claim)
        calibrated = self._calibrate_confidence(raw_confidence, verdict)
        latency = (time.time() - start) * 1000
        result = AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=min(raw_confidence, 1.0),
            calibrated_confidence=calibrated,
            explanation=reason,
            signals={"probs": probs},
            latency_ms=latency,
        )
        if self.enable_cache:
            self._cache[cache_key] = result
        return result


def main():
    print("=" * 60)
    print("PRODUCTION AUDITOR - Enhanced REFUTES Detection")
    print("=" * 60)
    auditor = ProductionAuditor(device="cpu")
    tests = [
        ("Vaccines cause autism.", "There is no scientific evidence linking vaccines to autism.", "REFUTES"),
        ("The Earth is flat.", "Scientific evidence proves Earth is spherical.", "REFUTES"),
        ("5G causes COVID.", "There is no connection between 5G and COVID-19.", "REFUTES"),
        ("Coffee is dangerous.", "Studies show coffee is safe for healthy adults.", "REFUTES"),
        ("Exercise is harmful.", "Regular exercise is beneficial for health.", "REFUTES"),
        ("Vaccines are safe.", "Research confirms vaccines are safe and effective.", "SUPPORTS"),
        ("Exercise is healthy.", "Studies prove regular exercise improves health.", "SUPPORTS"),
        ("Water is essential.", "Scientific evidence shows water is essential for life.", "SUPPORTS"),
        ("AI will replace humans.", "The impact of AI on employment is debated.", "NEI"),
        ("Coffee has health effects.", "Studies show mixed results on coffee and health.", "NEI"),
    ]
    correct = 0
    for claim, evidence, expected in tests:
        result = auditor.audit(claim, evidence)
        match = result.verdict == expected
        correct += 1 if match else 0
        icon = "Y" if match else "N"
        print(f"{icon} [{expected:8}->{result.verdict:8}] conf={result.confidence:.2f} | {claim[:40]}")
    print(f"\nAccuracy: {correct}/{len(tests)} = {100*correct/len(tests):.1f}%")


if __name__ == "__main__":
    main()
