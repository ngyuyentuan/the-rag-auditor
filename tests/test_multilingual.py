import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestVietnamesePatterns:
    def test_refutes_patterns_exist(self):
        from src.auditor.vietnamese import VIETNAMESE_REFUTES_PATTERNS
        assert len(VIETNAMESE_REFUTES_PATTERNS) > 0
    
    def test_supports_phrases_exist(self):
        from src.auditor.vietnamese import VIETNAMESE_SUPPORTS_PHRASES
        assert len(VIETNAMESE_SUPPORTS_PHRASES) > 0
    
    def test_nationality_mapping(self):
        from src.auditor.vietnamese import NATIONALITY_VI
        assert 'việt nam' in NATIONALITY_VI
        assert 'mỹ' in NATIONALITY_VI


class TestPersistentCache:
    def test_nli_cache(self, tmp_path):
        from src.utils.persistent_cache import NLICache
        cache = NLICache(tmp_path / "test_cache.db")
        
        cache.set_nli_result("claim1", "evidence1", {"entailment": 0.9})
        result = cache.get_nli_result("claim1", "evidence1")
        
        assert result is not None
        assert result["entailment"] == 0.9
    
    def test_cache_miss(self, tmp_path):
        from src.utils.persistent_cache import NLICache
        cache = NLICache(tmp_path / "test_cache2.db")
        
        result = cache.get_nli_result("not_exists", "not_exists")
        assert result is None
    
    def test_cache_stats(self, tmp_path):
        from src.utils.persistent_cache import NLICache
        cache = NLICache(tmp_path / "test_cache3.db")
        
        cache.set_nli_result("c1", "e1", {"entailment": 0.5})
        cache.set_nli_result("c2", "e2", {"contradiction": 0.5})
        
        stats = cache.get_stats()
        assert stats["total_entries"] == 2


class TestMultilingualAuditorV2:
    @pytest.fixture
    def auditor(self):
        from src.auditor.multilingual_v2 import MultilingualAuditorV2
        return MultilingualAuditorV2(device="cpu", use_persistent_cache=False)
    
    def test_version(self, auditor):
        assert auditor.VERSION == "9.0.0"
    
    def test_detect_vietnamese(self, auditor):
        lang = auditor._detect_language("Đây là tiếng Việt")
        assert lang == "vi"
    
    def test_detect_english(self, auditor):
        lang = auditor._detect_language("This is English")
        assert lang == "en"
    
    def test_audit_returns_language(self, auditor):
        result = auditor.audit("Test claim", "Test evidence")
        assert "language" in result.signals
    
    def test_nei_semantic_detection(self, auditor):
        is_nei, conf = auditor._detect_nei_semantic(
            "Different topic entirely",
            "Some other topic"
        )
        assert is_nei is False or conf > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
