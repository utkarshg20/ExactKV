"""Tests for Experiment 086 claim-safe demo packaging (Phase 17A)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.phase16_closeout import ALLOWED_CLAIMS, FORBIDDEN_CLAIMS
from exactkv.demo.phase17_claim_safe_demo import (
    BENCHMARK_GAP_LINE,
    DEMO_HOOK,
    DEMO_PROBLEM_STATEMENT,
    EXPERIMENT_086_ID,
    SOURCE_DOCS,
    run_exp086_claim_safe_demo_packaging,
    validate_exp086_report,
)


def test_demo_report_schema_validates(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    assert report["experiment_id"] == EXPERIMENT_086_ID
    assert validate_exp086_report(report) == []


def test_required_hook_problem_benchmark_line(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    assert report["demo_hook"] == DEMO_HOOK
    assert report["demo_problem_statement"] == DEMO_PROBLEM_STATEMENT
    assert report["benchmark_gap_line"] == BENCHMARK_GAP_LINE
    assert report["demo_sections"]["hook"] == DEMO_HOOK


def test_all_demo_cards_have_evidence_and_limitations(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    for card in report["demo_cards"].values():
        assert card.get("evidence")
        assert card.get("limitations")
        assert card.get("title")
        assert card.get("source_docs")


def test_allowed_and_forbidden_claims_match_phase16_freeze(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    assert report["allowed_claims"] == list(ALLOWED_CLAIMS)
    assert report["forbidden_claims"] == list(FORBIDDEN_CLAIMS)


def test_q_and_a_rejects_overclaims(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    by_q = {q["question"]: q["answer"] for q in report["q_and_a"]}
    assert "no" in by_q["Are you claiming speedups?"].lower()
    assert "no" in by_q["Are you claiming memory savings?"].lower()
    assert "no" in by_q["Did you reproduce VeriCache serving?"].lower()
    assert "no" in by_q["Is streaming attention used for token commit?"].lower()
    assert "diagnostic" in by_q["What did Phase 16 prove?"].lower()


def test_missing_docs_handled_without_inventing(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    assert len(report["source_docs_missing"]) == len(SOURCE_DOCS)
    assert report["source_docs_found"] == []
    # Cards still build with fallback evidence strings, not fabricated metrics
    for card in report["demo_cards"].values():
        assert "per Phase 16" in card["evidence"] or "Exp " in card["evidence"]


def test_found_docs_used(tmp_path: Path) -> None:
    doc = tmp_path / SOURCE_DOCS[0]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Closeout\n\n- Phase 16 complete with 19 experiments.\n")
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    assert SOURCE_DOCS[0] in report["source_docs_found"]


def test_no_forbidden_positive_claim_phrases(tmp_path: Path) -> None:
    report = run_exp086_claim_safe_demo_packaging(root=tmp_path)
    for card in report["demo_cards"].values():
        for field in ("one_sentence_claim", "evidence", "limitations", "title"):
            dumped = json.dumps(card.get(field, "")).lower()
            for forbidden in (
                "speedup achieved",
                "throughput improved",
                "latency reduced",
                "tokens_per_second",
                "runtime_seconds",
                "active_gpu_memory_savings",
                "production_memory_savings",
                "production serving supported",
                "vericache throughput reproduced",
                "vericache serving reproduced",
                "streaming attention integrated into token commit",
            ):
                assert forbidden not in dumped
