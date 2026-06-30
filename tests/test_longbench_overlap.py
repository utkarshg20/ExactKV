"""Tests for LongBench overlap scoring."""
from __future__ import annotations

from exactkv.benchmarks.longbench_overlap import (
    max_token_f1,
    normalize_references,
    token_f1,
)


def test_token_f1_identical() -> None:
    assert token_f1("the capital is Paris", "the capital is Paris") == 1.0


def test_token_f1_partial() -> None:
    assert 0.0 < token_f1("Paris France", "Paris") < 1.0


def test_max_token_f1_over_refs() -> None:
    assert max_token_f1("Paris", ["London", "Paris"]) == 1.0
    assert max_token_f1("answer: Paris", ["London", "Paris"]) > 0.5


def test_normalize_references_list() -> None:
    assert normalize_references(["a", "b"]) == ["a", "b"]
