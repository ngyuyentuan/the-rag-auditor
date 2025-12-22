from src.utils.evidence_match import (
    flatten_fever_evidence_sentences,
    check_sufficiency_substring,
)

def test_flatten_triple_format():
    ev = [["Shingles", "31", "The number of new cases per year ranges from 1.2 -- 3.4 per 1,000."]]
    out = flatten_fever_evidence_sentences(ev)
    assert out == ["The number of new cases per year ranges from 1.2 -- 3.4 per 1,000."]

def test_flatten_nested_groups():
    ev = [
        [["P1", "1", "S1"], ["P2", "2", "S2"]],
        [["P3", "3", "S3"]],
    ]
    out = flatten_fever_evidence_sentences(ev)
    assert out == ["S1", "S2", "S3"]

def test_flatten_malformed_safe():
    assert flatten_fever_evidence_sentences("not_a_list") == []
    assert flatten_fever_evidence_sentences(None) == []
    assert flatten_fever_evidence_sentences([]) == []
    assert flatten_fever_evidence_sentences([[]]) == []  # should not crash

def test_check_sufficiency_true_with_normalization():
    chunks = ["Evidence: The number of new cases per year ranges from 1.2--3.4 per 1,000 among healthy individuals!"]
    evs = ["The number of new cases per year ranges from 1.2 -- 3.4 per 1,000 among healthy individuals"]
    r = check_sufficiency_substring(chunks, evs, min_chars=10)
    assert r.sufficient is True
    assert r.matched_chunk is not None
    assert r.matched_evidence is not None

def test_check_sufficiency_false():
    chunks = ["Completely unrelated text."]
    evs = ["This is gold evidence sentence."]
    r = check_sufficiency_substring(chunks, evs, min_chars=10)
    assert r.sufficient is False

def test_check_filters_non_string_chunks():
    chunks = [None, 123, "Hello world this is a chunk containing evidence."]
    evs = ["chunk containing evidence"]
    r = check_sufficiency_substring(chunks, evs, min_chars=5)
    assert r.sufficient is True

def test_check_min_chars_blocks_too_short():
    chunks = ["abc def ghi jkl"]
    evs = ["abc"]
    r = check_sufficiency_substring(chunks, evs, min_chars=10)
    assert r.sufficient is False

def test_unicode_does_not_crash():
    chunks = ["Hello 😊 this is evidence."]
    evs = ["Hello 😊 this is evidence"]
    r = check_sufficiency_substring(chunks, evs, min_chars=5)
    assert isinstance(r.sufficient, bool)
