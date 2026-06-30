"""Tests for downstream validity metrics."""
from __future__ import annotations

from exactkv.benchmarks.downstream_validity import (
    bfcl_tool_call_valid,
    mbpp_syntax_valid,
    summarise_downstream,
)


def test_bfcl_valid_json() -> None:
    assert bfcl_tool_call_valid('{"name": "get_weather", "arguments": {}}')
    assert not bfcl_tool_call_valid("not json at all")


def test_mbpp_syntax() -> None:
    assert mbpp_syntax_valid("```python\ndef add(a, b):\n    return a + b\n```")
    assert not mbpp_syntax_valid("def add(a b):\n    return a + b")


def test_preservation_summary() -> None:
    cells = [
        {
            "status": "ok",
            "full_kv_downstream_valid": True,
            "exactkv_downstream_valid": True,
            "downstream_metric": "bfcl_json_tool_call",
            "metrics": {"token_level_divergence": True, "acceptance_rate": 0.5},
        },
        {
            "status": "ok",
            "full_kv_downstream_valid": False,
            "exactkv_downstream_valid": False,
            "downstream_metric": "bfcl_json_tool_call",
            "metrics": {"token_level_divergence": False, "acceptance_rate": 1.0},
        },
    ]
    s = summarise_downstream(cells)
    assert s["valid_preserved_among_full_kv_valid"] == 1
    assert s["preservation_rate_given_full_valid"] == 1.0
