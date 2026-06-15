"""Tests for Experiment 059 vLLM feasibility probe docs and schema."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_probe import (
    EXPERIMENT_059_ID,
    EXP059_CLAIM_NOTE,
    FORBIDDEN_CLAIMS,
    build_vllm_blocked_report,
    validate_exp059_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_059_VLLM_FEASIBILITY_PROBE.md"


def test_report_schema_includes_blockers() -> None:
    report = build_vllm_blocked_report()
    assert "blockers" in report
    assert report["blockers"]
    assert validate_exp059_report(report) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "vllm feasibility probe",
        "not vllm integration",
        "no vllm runtime integration",
        "throughput",
        "vericache",
        "default exactkv generation behavior is unchanged",
        "environment blocker",
        "not an exactkv correctness failure",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("vllm integration exists", "achieves speedup", "production serving ready"):
        assert phrase not in text


def test_experiment_id_constant() -> None:
    report = build_vllm_blocked_report()
    assert report["experiment_id"] == EXPERIMENT_059_ID
    assert EXP059_CLAIM_NOTE in report["claim_note"]
    for term in FORBIDDEN_CLAIMS:
        assert term in report["forbidden_claims"]
