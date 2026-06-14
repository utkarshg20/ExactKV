"""Tests for Phase 10I benchmark-gap analysis doc."""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "BENCHMARK_GAP_ANALYSIS.md"
_MATRIX = _ROOT / "docs" / "assets" / "benchmark_gap_matrix.md"


def _doc_text() -> str:
    assert _DOC.is_file()
    return _DOC.read_text(encoding="utf-8")


def _plain(text: str) -> str:
    """Lowercase doc text with markdown emphasis removed."""
    return re.sub(r"\*+", "", text).lower()


def _negated(text: str, phrase: str) -> bool:
    """True if phrase appears only inside a does-not / no-claim negation."""
    plain = _plain(text)
    if phrase not in plain:
        return False
    for line in plain.splitlines():
        if phrase in line and (
            "does not" in line
            or "do not" in line
            or line.strip().startswith("no ")
            or "not the focus" in line
        ):
            return True
    return False


def test_benchmark_gap_doc_exists() -> None:
    text = _doc_text()
    assert "Benchmark Scores Can Stay Green While KV Drift Happens" in text


def test_includes_core_examples() -> None:
    text = _doc_text().lower()
    assert "longbench" in text or "longbench-style" in text
    assert "exp 037" in text or "037" in text
    assert "shard" in text
    assert "exp 041" in text or "041" in text
    assert "spectralquant" in text
    assert "exp 045" in text or "045" in text
    assert "exactkv" in text
    assert "v13" in text or "exp 029" in text or "exp 012" in text


def test_complementary_wording() -> None:
    text = _doc_text().lower()
    assert "complementary to outcome benchmarks" in text
    assert "outcome benchmarks measure whether the final answer scored well" in text
    assert "kv compression changed" in text


def test_required_distinction_quote() -> None:
    text = _doc_text()
    assert "Outcome benchmarks can tell you whether the answer scored well" in text
    assert "KV compression changed the model's path before the answer looked fine" in text


def test_public_card_copy() -> None:
    text = _doc_text()
    assert "The answer can look right while the KV path drifted" in text
    assert "ExactKV catches the drift outcome scores hide" in text
    assert "crash-tested" in text.lower()


def test_no_forbidden_claims() -> None:
    text = _doc_text()
    plain = _plain(text)
    forbidden_positive = [
        "speedup over",
        "faster than full",
        "active gpu memory savings",
        "vram savings",
        "production serving ready",
        "production-ready serving",
        "accuracy improvement",
        "improves model accuracy",
    ]
    for phrase in forbidden_positive:
        if phrase in plain:
            assert _negated(text, phrase), f"forbidden positive claim: {phrase}"
    assert "does not prove speed" in plain or "no speed" in plain
    assert "does not prove active gpu memory" in plain or "no active memory" in plain


def test_does_not_call_benchmarks_bad() -> None:
    plain = _plain(_doc_text())
    assert "longbench" in plain
    assert "ruler" in plain
    bad_phrases = [
        "longbench is bad",
        "longbench is flawed",
        "ruler is bad",
        "ruler is flawed",
        "benchmarks are bad",
        "benchmarks are flawed",
        "outcome benchmarks are bad",
    ]
    for phrase in bad_phrases:
        if phrase in plain:
            assert _negated(_doc_text(), phrase), f"uncaveated bad claim: {phrase}"
    assert "does not mean longbench" in plain or "not mean longbench" in plain


def test_no_external_benchmark_numbers_as_exactkv() -> None:
    plain = _plain(_doc_text())
    assert "not cited here as exactkv results" in plain or "not exactkv results" in plain
    assert "external benchmark" in plain


def test_shard_and_spectralquant_caveats() -> None:
    plain = _plain(_doc_text())
    assert "external drafter" in plain or "external-drafter" in plain
    assert "not integrated" in plain or "not default registry" in plain
    assert "materializing" in plain
    assert "56.25" in plain or "18/32" in plain
    assert "0.481" in plain
    assert "11/12" in plain
    assert "exactkv_failures" in plain


def test_matrix_artifact_exists() -> None:
    assert _MATRIX.is_file()
    matrix = _MATRIX.read_text(encoding="utf-8").lower()
    assert "exp 037" in matrix or "037" in matrix
    assert "exp 041" in matrix or "041" in matrix
    assert "exp 045" in matrix or "045" in matrix
