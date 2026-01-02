import pytest

from src.stage1.router_v2 import decide_router


def test_router_decision_rule():
    assert decide_router(0.9, 0.8, 0.2) == "ACCEPT"
    assert decide_router(0.1, 0.8, 0.2) == "REJECT"
    assert decide_router(0.5, 0.8, 0.2) == "UNCERTAIN"
    assert decide_router(0.95, 0.8, 0.2) != "REJECT"


def test_router_decision_requires_ordered_thresholds():
    with pytest.raises(ValueError):
        decide_router(0.5, 0.2, 0.2)
