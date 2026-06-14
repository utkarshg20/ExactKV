"""Tests for Phase 11A VeriCache parity audit and systems roadmap docs."""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_AUDIT = _ROOT / "docs" / "VERICACHE_PARITY_AUDIT.md"
_ROADMAP = _ROOT / "docs" / "VERICACHE_SYSTEMS_ROADMAP.md"
_README = _ROOT / "README.md"


def _plain(text: str) -> str:
    return re.sub(r"\*+", "", text).lower()


def _audit() -> str:
    assert _AUDIT.is_file()
    return _AUDIT.read_text(encoding="utf-8")


def _roadmap() -> str:
    assert _ROADMAP.is_file()
    return _ROADMAP.read_text(encoding="utf-8")


def test_audit_doc_exists() -> None:
    text = _audit()
    assert "VeriCache Parity Audit" in text


def test_roadmap_doc_exists() -> None:
    text = _roadmap()
    assert "VeriCache Systems Roadmap" in text


def test_audit_covers_required_capabilities() -> None:
    plain = _plain(_audit())
    required = [
        "vllm",
        "lmcache",
        "remote prefix",
        "cross-resource",
        "extended verification",
        "throughput",
        "batching",
        "kl",
        "distributional",
        "compressor",
        "benchmark",
    ]
    for term in required:
        assert term in plain, f"missing capability topic: {term}"


def test_audit_marks_serving_and_throughput_missing() -> None:
    plain = _plain(_audit())
    assert "vllm" in plain and "missing" in plain
    assert "lmcache" in plain and "missing" in plain
    assert "throughput" in plain
    assert "missing" in plain or "not reproduced" in plain or "forbidden" in plain
    assert "remote prefix" in plain and "missing" in plain


def test_audit_marks_core_draft_verify_done_or_partial() -> None:
    plain = _plain(_audit())
    assert "compressed-kv drafter" in plain or "compressed kv drafter" in plain
    assert "full-kv verifier" in plain or "full kv verifier" in plain
    assert "done" in plain or "mostly done" in plain
    assert "exact greedy" in plain


def test_forbidden_claim_firewall() -> None:
    combined = _plain(_audit() + _roadmap())
    assert "not a full vericache reproduction" in combined or "not a full vericache" in combined
    assert "algorithmic semantics" in combined
    assert "forbidden" in combined
    forbidden_positive = [
        "speedup over",
        "faster than vericache",
        "active gpu memory savings",
        "production serving ready",
        "vllm integration exists",
        "lmcache integration exists",
    ]
    for phrase in forbidden_positive:
        if phrase in combined:
            assert "forbidden" in combined or "not implemented" in combined or "no-go" in combined


def test_docs_do_not_claim_speed_memory_serving() -> None:
    combined = _plain(_audit() + _roadmap())
    risky = ["speedup", "memory savings", "production serving"]
    for term in risky:
        assert term in combined  # discussed
    # must be negated or forbidden, not promoted
    assert "forbidden" in combined
    assert "not" in combined or "missing" in combined


def test_readme_does_not_overclaim_vericache() -> None:
    readme = _plain(_README.read_text(encoding="utf-8"))
    overclaim = [
        "full vericache reproduction",
        "reproduces vericache completely",
        "vericache-equivalent functionality today",
        "vllm integration",
        "lmcache integration",
    ]
    for phrase in overclaim:
        assert phrase not in readme


def test_roadmap_has_stages() -> None:
    text = _roadmap()
    for i in range(11):
        assert f"Stage {i}" in text


def test_shard_spectralquant_not_system_parity() -> None:
    plain = _plain(_audit())
    assert "shard" in plain
    assert "spectralquant" in plain
    assert "restricted" in plain or "not integrated" in plain
