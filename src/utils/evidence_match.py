from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from src.utils.normalize import normalize_text


@dataclass(frozen=True)
class MatchResult:
    """Kết quả kiểm tra sufficiency."""
    sufficient: bool
    matched_chunk: Optional[str] = None
    matched_evidence: Optional[str] = None
    matched_evidence_norm: Optional[str] = None


def flatten_fever_evidence_sentences(evidence_field: Any) -> List[str]:
    """
    Extract evidence sentences from FEVER-like 'evidence' fields.

    Supports common shapes:
    1) Flattened triples: [["Page", "31", "Sentence text..."], ...]
    2) Nested groups: [[["Page","31","S1"], ...], [["Page","12","S2"], ...], ...]
    If malformed / unknown shape -> returns [] safely (no crash).
    """
    if not isinstance(evidence_field, list) or not evidence_field:
        return []

    out: List[str] = []

    # Case 1: flattened triples
    if all(isinstance(x, (list, tuple)) for x in evidence_field):
        # If items look like triples (>=3) with sentence at index 2
        if any((len(x) >= 3 and isinstance(x[2], str)) for x in evidence_field):
            for x in evidence_field:
                if isinstance(x, (list, tuple)) and len(x) >= 3 and isinstance(x[2], str):
                    out.append(x[2])
            return out

    # Case 2: nested groups
    # evidence_field: list of groups; each group is list of triples
    if all(isinstance(g, list) for g in evidence_field):
        for g in evidence_field:
            if not isinstance(g, list) or not g:
                continue
            for x in g:
                if isinstance(x, (list, tuple)) and len(x) >= 3 and isinstance(x[2], str):
                    out.append(x[2])

    return out


def check_sufficiency_substring(
    retrieved_chunks: Sequence[Any],
    gold_evidence_sentences: Sequence[Any],
    *,
    min_chars: int = 20,
) -> MatchResult:
    """
    Ground-truth rule for Stage-1 calibration labels.

    Returns sufficient=True iff exists evidence sentence e such that:
        normalize(e) is a substring of normalize(chunk)

    Notes:
    - Strict (no fuzzy) to keep labels deterministic & reproducible.
    - Filters out non-string / empty inputs defensively.
    - min_chars avoids accidental matches on very short evidence strings.
    """
    chunks: List[str] = [c for c in retrieved_chunks if isinstance(c, str) and c.strip()]
    evidences: List[str] = [e for e in gold_evidence_sentences if isinstance(e, str) and e.strip()]

    if not chunks or not evidences:
        return MatchResult(False)

    norm_chunks: List[Tuple[str, str]] = [(c, normalize_text(c)) for c in chunks]

    norm_evs: List[Tuple[str, str]] = []
    for e in evidences:
        ne = normalize_text(e)
        if ne and len(ne) >= min_chars:
            norm_evs.append((e, ne))

    if not norm_evs:
        return MatchResult(False)

    for raw_chunk, n_chunk in norm_chunks:
        if not n_chunk:
            continue
        for raw_ev, n_ev in norm_evs:
            if n_ev in n_chunk:
                return MatchResult(
                    True,
                    matched_chunk=raw_chunk,
                    matched_evidence=raw_ev,
                    matched_evidence_norm=n_ev,
                )

    return MatchResult(False)
