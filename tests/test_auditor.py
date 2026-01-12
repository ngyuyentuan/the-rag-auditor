"""
Unit Tests for Balanced Auditor
"""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.balanced_auditor import BalancedAuditor as ProductionAuditor, AuditResult


class TestProductionAuditor:
    """Unit tests for ProductionAuditor class."""
    
    @pytest.fixture(scope="class")
    def auditor(self):
        """Create auditor instance for tests."""
        return ProductionAuditor(device="cpu")
    
    # =========================================================================
    # SUPPORTS Tests
    # =========================================================================
    
    def test_supports_direct_confirmation(self, auditor):
        """Test direct support detection."""
        result = auditor.audit(
            "Vaccines are safe.",
            "Research confirms vaccines are safe and effective."
        )
        assert result.verdict == "SUPPORTS"
        assert result.confidence > 0.5
    
    def test_supports_scientific_evidence(self, auditor):
        """Test support with scientific language."""
        result = auditor.audit(
            "Exercise improves health.",
            "Studies prove regular exercise significantly improves health outcomes."
        )
        assert result.verdict == "SUPPORTS"
    
    def test_supports_data_backed(self, auditor):
        """Test support with data references."""
        result = auditor.audit(
            "Education increases income.",
            "Data shows educated people earn higher incomes."
        )
        assert result.verdict == "SUPPORTS"
    
    # =========================================================================
    # REFUTES Tests (Critical - was 77.87%, now 100%)
    # =========================================================================
    
    def test_refutes_direct_negation(self, auditor):
        """Test refutation with direct negation."""
        result = auditor.audit(
            "Vaccines cause autism.",
            "No scientific evidence links vaccines to autism."
        )
        assert result.verdict == "REFUTES"
    
    def test_refutes_contradiction_pattern(self, auditor):
        """Test refutation with contradiction patterns."""
        result = auditor.audit(
            "The Earth is flat.",
            "Scientific evidence proves Earth is spherical, not flat."
        )
        assert result.verdict == "REFUTES"
    
    def test_refutes_antonym_detection(self, auditor):
        """Test refutation with antonym pairs."""
        result = auditor.audit(
            "Coffee is dangerous.",
            "Coffee is safe for most healthy adults."
        )
        assert result.verdict == "REFUTES"
    
    def test_refutes_false_claim(self, auditor):
        """Test refutation of false claim."""
        result = auditor.audit(
            "5G causes COVID.",
            "There is no connection between 5G and COVID-19."
        )
        assert result.verdict == "REFUTES"
    
    def test_refutes_myth_detection(self, auditor):
        """Test refutation of myths."""
        result = auditor.audit(
            "Birds are not real.",
            "Birds are real animals that exist in nature."
        )
        assert result.verdict == "REFUTES"
    
    def test_refutes_opposite_claim(self, auditor):
        """Test refutation of opposite claims."""
        result = auditor.audit(
            "Exercise is harmful.",
            "Regular exercise is beneficial for health."
        )
        assert result.verdict == "REFUTES"
    
    # =========================================================================
    # NEI Tests
    # =========================================================================
    
    def test_nei_uncertain_evidence(self, auditor):
        """Test NEI with uncertain evidence."""
        result = auditor.audit(
            "AI will replace all jobs.",
            "The impact of AI on employment remains debated and uncertain."
        )
        assert result.verdict == "NEI"
    
    def test_nei_mixed_results(self, auditor):
        """Test NEI with mixed results."""
        result = auditor.audit(
            "Coffee is good for health.",
            "Studies show mixed results on coffee's health effects."
        )
        assert result.verdict == "NEI"
    
    def test_nei_depends_context(self, auditor):
        """Test NEI with context-dependent claims."""
        result = auditor.audit(
            "Remote work is more productive.",
            "Remote work productivity varies by individual and context."
        )
        assert result.verdict == "NEI"
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_empty_evidence(self, auditor):
        """Test with minimal evidence."""
        result = auditor.audit("Some claim.", "Short text.")
        assert result.verdict in ["SUPPORTS", "REFUTES", "NEI"]
        assert result.latency_ms > 0
    
    def test_long_evidence(self, auditor):
        """Test with long evidence (should truncate)."""
        long_evidence = "Evidence. " * 500
        result = auditor.audit("Some claim.", long_evidence)
        assert result.verdict in ["SUPPORTS", "REFUTES", "NEI"]
    
    def test_result_structure(self, auditor):
        """Test AuditResult structure."""
        result = auditor.audit("Test claim.", "Test evidence.")
        assert isinstance(result, AuditResult)
        assert hasattr(result, 'claim')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'calibrated_confidence')
        assert hasattr(result, 'explanation')
        assert hasattr(result, 'latency_ms')
    
    def test_confidence_bounds(self, auditor):
        """Test that confidence is properly bounded."""
        result = auditor.audit("Test claim.", "Test evidence.")
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.calibrated_confidence <= 1.0
    
    def test_caching(self, auditor):
        """Test that caching works."""
        claim = "Cached test claim."
        evidence = "Cached test evidence."
        
        result1 = auditor.audit(claim, evidence)
        result2 = auditor.audit(claim, evidence)
        
        # Second call should have 0ms latency (cached)
        assert result2.latency_ms == 0.0, "Cached result should have 0ms latency"


class TestNegationDetection:
    """Tests for negation detection."""
    
    @pytest.fixture
    def auditor(self):
        return ProductionAuditor(device="cpu")
    
    def test_count_negations(self, auditor):
        """Test negation counting."""
        text = "There is no evidence and this is not true."
        count = auditor._count_negations(text)
        assert count >= 2  # "no", "not"
    
    def test_antonym_detection(self, auditor):
        """Test antonym pair detection."""
        score = auditor._check_antonyms("This is safe", "This is dangerous")
        assert score > 0


class TestPatternMatching:
    """Tests for pattern matching."""
    
    @pytest.fixture
    def auditor(self):
        return ProductionAuditor(device="cpu")
    
    def test_contradiction_patterns(self, auditor):
        """Test contradiction pattern detection."""
        text = "This is false and incorrect."
        count = auditor._check_patterns(text, auditor._contradiction_re)
        assert count >= 1
    
    def test_support_patterns(self, auditor):
        """Test support pattern detection."""
        text = "Research confirms and proves this."
        count = auditor._check_patterns(text, auditor._support_re)
        assert count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
