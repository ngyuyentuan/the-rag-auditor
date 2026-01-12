import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.constants import Verdict, NLILabel, ThresholdConfig, PatternWeights
from src.auditor.base import BaseAuditor, AuditResult, TimingMixin, CachingMixin
from src.auditor.auditor import ProductionAuditor


class TestConstants:
    def test_verdict_enum(self):
        assert str(Verdict.SUPPORTS) == "SUPPORTS"
        assert str(Verdict.REFUTES) == "REFUTES"
        assert str(Verdict.NEI) == "NEI"
    
    def test_nli_label_enum(self):
        assert NLILabel.ENTAILMENT == "entailment"
        assert NLILabel.CONTRADICTION == "contradiction"
        assert NLILabel.NEUTRAL == "neutral"
    
    def test_threshold_config_values(self):
        assert 0 < ThresholdConfig.ENTAIL_HIGH < 1
        assert 0 < ThresholdConfig.CONTRA_HIGH < 1
        assert ThresholdConfig.ENTAIL_HIGH > ThresholdConfig.ENTAIL_MED
    
    def test_pattern_weights_range(self):
        assert 0 < PatternWeights.ONLY_NATIONALITY <= 1
        assert 0 < PatternWeights.NEVER <= 1


class TestAuditResult:
    def test_creation(self):
        result = AuditResult(
            claim="test claim",
            verdict="SUPPORTS",
            confidence=0.9,
            calibrated_confidence=0.95,
            explanation="test"
        )
        assert result.claim == "test claim"
        assert result.verdict == "SUPPORTS"
        assert result.confidence == 0.9
    
    def test_to_dict(self):
        result = AuditResult(
            claim="test",
            verdict="REFUTES",
            confidence=0.8,
            calibrated_confidence=0.85,
            explanation="reason"
        )
        d = result.to_dict()
        assert d["claim"] == "test"
        assert d["verdict"] == "REFUTES"
        assert "confidence" in d


class TestTimingMixin:
    def test_timing(self):
        class TestClass(TimingMixin):
            pass
        
        obj = TestClass()
        start = obj._start_timer()
        import time
        time.sleep(0.01)
        latency = obj._get_latency_ms(start)
        assert latency >= 10


class TestProductionAuditor:
    @pytest.fixture
    def auditor(self):
        return ProductionAuditor(device="cpu", nei_same_text_check=False)
    
    def test_version(self, auditor):
        assert auditor.VERSION == "8.0.0"
    
    def test_refutes_pattern_boost(self, auditor):
        boost = auditor._calculate_refutes_boost("He never played")
        assert boost > 0
        
        no_boost = auditor._calculate_refutes_boost("He played well")
        assert no_boost == 0
    
    def test_supports_boost(self, auditor):
        boost = auditor._calculate_supports_boost(
            "Born in 1990",
            "He was born in 1990"
        )
        assert boost > 0
    
    def test_is_same_text(self, auditor):
        assert auditor._is_same_text("Test claim", "Test claim")
        assert auditor._is_same_text("Test claim", "test claim")
        assert not auditor._is_same_text("Test claim", "Different text entirely")
    
    def test_cache_key(self, auditor):
        key1 = auditor._cache_key("claim1", "evidence1")
        key2 = auditor._cache_key("claim1", "evidence1")
        key3 = auditor._cache_key("claim2", "evidence1")
        assert key1 == key2
        assert key1 != key3
    
    def test_get_stats(self, auditor):
        stats = auditor.get_stats()
        assert "version" in stats
        assert "request_count" in stats
        assert "cache_size" in stats
    
    def test_audit_returns_result(self, auditor):
        result = auditor.audit("Test claim", "Test evidence")
        assert isinstance(result, AuditResult)
        assert result.verdict in ["SUPPORTS", "REFUTES", "NEI"]
        assert 0 <= result.confidence <= 1
    
    def test_cache_works(self, auditor):
        result1 = auditor.audit("cached claim", "cached evidence")
        result2 = auditor.audit("cached claim", "cached evidence")
        assert result2.latency_ms == 0.0
        assert result1.verdict == result2.verdict
    
    def test_skip_cache(self, auditor):
        result1 = auditor.audit("skip claim", "skip evidence")
        result2 = auditor.audit("skip claim", "skip evidence", skip_cache=True)
        assert result2.latency_ms > 0


class TestProductionAuditorDecisions:
    @pytest.fixture
    def auditor(self):
        return ProductionAuditor(device="cpu", nei_same_text_check=True)
    
    def test_same_text_nei(self, auditor):
        result = auditor.audit("Same text", "Same text")
        assert result.verdict == "NEI"
    
    def test_with_nei_disabled(self):
        auditor = ProductionAuditor(device="cpu", nei_same_text_check=False)
        result = auditor.audit("Same text", "Same text")
        assert result.verdict != "NEI" or result.explanation != "same_text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
