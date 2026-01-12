from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time


@dataclass
class AuditResult:
    claim: str
    verdict: str
    confidence: float
    calibrated_confidence: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "explanation": self.explanation,
            "signals": self.signals,
            "latency_ms": self.latency_ms,
        }


class BaseAuditor(ABC):
    VERSION: str = "0.0.0"
    
    @abstractmethod
    def audit(self, claim: str, evidence: str, skip_cache: bool = False) -> AuditResult:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
    
    def audit_batch(self, items: list, skip_cache: bool = False) -> list:
        results = []
        for item in items:
            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            result = self.audit(claim, evidence, skip_cache)
            results.append(result)
        return results


class TimingMixin:
    def _start_timer(self) -> float:
        return time.time()
    
    def _get_latency_ms(self, start: float) -> float:
        return (time.time() - start) * 1000


class CachingMixin:
    _cache: Dict[str, AuditResult]
    enable_cache: bool
    
    def _get_from_cache(self, key: str) -> Optional[AuditResult]:
        if not self.enable_cache:
            return None
        return self._cache.get(key)
    
    def _save_to_cache(self, key: str, result: AuditResult):
        if self.enable_cache:
            self._cache[key] = result
    
    def clear_cache(self):
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        return len(self._cache)
