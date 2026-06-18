"""Tests for Experiment 094 L3 shadow proposal provenance audit (Phase 18E)."""
from __future__ import annotations

import json

import pytest

from exactkv.safety.guarded_draft_shadow import (
    AUDIT_CATEGORIES,
    AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD,
    AUDIT_CATEGORY_SAFE_SHADOW_TOP1_AVAILABLE,
    AUDIT_CATEGORY_SHADOW_TOP1_MATCHES_COMMITTED,
    AUDIT_CATEGORY_SHADOW_TOP1_MISMATCHES_COMMITTED,
    AUDIT_CATEGORY_UNSAFE_SOURCE_REJECTED,
    COMMITTED_COMPARISON_ONLY_NOTE,
    DECISION_REPLACE_PROPOSAL_SOURCE,
    DECISION_STOP_L3_TOP1_PATH,
    EXPERIMENT_094_ID,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_SYNTHETIC,
    RECOMMENDED_NEXT_PHASE_18E,
    ShadowProposalAuditRecord,
    aggregate_provenance_audit,
    build_proposal_audit_record,
    classify_proposal_audit_categories,
    compute_decision_recommendation,
    extract_shadow_top1_candidate,
    run_exp094_shadow_proposal_provenance_audit,
    validate_exp094_report,
)
from exactkv.safety.integration_safety_spec import validate_integration_proposal

FORBIDDEN_CLAIM_PHRASES = (
    "speedup achieved",
    "throughput improved",
    "latency reduced",
    "tokens_per_second",
    "runtime_seconds",
    "active_gpu_memory_savings",
    "production_memory_savings",
    "production serving supported",
    "VeriCache throughput reproduced",
    "VeriCache serving reproduced",
    "streaming attention integrated into token commit",
    "draft shadow used for token commit",
    "verifier-mediated compressed draft implemented",
)

TOKENS = (100, 101, 102, 103)


def _baseline_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": list(TOKENS),
        "generated_text": "out",
        "exactkv_failures": 0,
        "token_exact_match": True,
        "blockers": [],
    }


def _draft_shadow_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": list(TOKENS),
        "generated_text": "out",
        "prompt_ids": [1, 2, 3],
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": [],
        "blockers": [],
    }


def _run_audit(**overrides: object) -> dict:
    defaults = {
        "prompts": [("p0", "a"), ("p1", "b")],
        "max_new_tokens_values": (4, 8),
        "compressors_requested": ["noop", "int8"],
        "proposal_source": PROPOSAL_SOURCE_SYNTHETIC,
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
    }
    defaults.update(overrides)
    return run_exp094_shadow_proposal_provenance_audit(**defaults)


def test_taxonomy_categories_exist() -> None:
    expected = {
        "safe_shadow_top1_available",
        "missing_shadow_top1_field",
        "shadow_top1_mismatches_committed",
        "shadow_top1_matches_committed",
        "round_alignment_unknown",
        "round_alignment_mismatch",
        "non_comparable_round",
        "blocked_no_safe_extraction",
        "unsafe_source_rejected",
    }
    assert expected == set(AUDIT_CATEGORIES)


def test_audit_record_separates_proposal_source_from_committed_comparison() -> None:
    rec = build_proposal_audit_record(
        prompt_id="p0",
        compressor="noop",
        max_new_tokens=4,
        proposal={
            "round_index": 0,
            "proposal_source": PROPOSAL_SOURCE_DECODE_TOP1,
            "extraction_status": "success",
            "extraction_source_field": "shadow_top1_token_id",
            "proposed_token_ids": [42],
            "proposal_status": "complete",
            "committed_token_id_for_comparison": 100,
            "matched_committed_token": False,
        },
        posthoc_shadow_cells=[{"round_index": 0, "shadow_top1_token_id": 42}],
        generated_token_ids=[100, 101],
    )
    assert rec.proposed_token_id == 42
    assert rec.committed_token_id_for_comparison == 100
    assert rec.proposed_token_id != rec.committed_token_id_for_comparison
    assert COMMITTED_COMPARISON_ONLY_NOTE in rec.interpretation_note


def test_committed_token_cannot_be_used_as_proposal_source() -> None:
    result = extract_shadow_top1_candidate({"committed_token_id": 99})
    assert result.extraction_status == "unsafe_rejected"


def test_missing_top1_field_categorized_correctly() -> None:
    cats = classify_proposal_audit_categories(
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
        extraction_status="blocked",
        proposal_status="blocked",
        block_reason="no explicit shadow top-1 diagnostic field available",
        proposed_token_id=None,
        committed_token_id=100,
        round_index=0,
        posthoc_shadow_cells=[{}],
        generated_token_ids=[100],
    )
    assert AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD in cats
    assert AUDIT_CATEGORY_SAFE_SHADOW_TOP1_AVAILABLE not in cats


def test_shadow_top1_mismatch_categorized_correctly() -> None:
    cats = classify_proposal_audit_categories(
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
        extraction_status="success",
        proposal_status="complete",
        block_reason=None,
        proposed_token_id=42,
        committed_token_id=100,
        round_index=0,
        posthoc_shadow_cells=[{"round_index": 0}],
        generated_token_ids=[100],
    )
    assert AUDIT_CATEGORY_SHADOW_TOP1_MISMATCHES_COMMITTED in cats


def test_shadow_top1_match_categorized_correctly() -> None:
    cats = classify_proposal_audit_categories(
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
        extraction_status="success",
        proposal_status="complete",
        block_reason=None,
        proposed_token_id=100,
        committed_token_id=100,
        round_index=0,
        posthoc_shadow_cells=[{"round_index": 0}],
        generated_token_ids=[100],
    )
    assert AUDIT_CATEGORY_SHADOW_TOP1_MATCHES_COMMITTED in cats


def test_category_aggregation_by_dimensions() -> None:
    records = (
        ShadowProposalAuditRecord(
            prompt_id="p0",
            compressor="noop",
            max_new_tokens=4,
            round_index=0,
            proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
            extraction_status="success",
            extraction_source_field="shadow_top1_token_id",
            proposed_token_id=1,
            proposed_token_text=None,
            committed_token_id_for_comparison=2,
            committed_token_text_for_comparison=None,
            matched_committed_token=False,
            block_reason=None,
            audit_categories=(AUDIT_CATEGORY_SHADOW_TOP1_MISMATCHES_COMMITTED,),
            interpretation_note="note",
        ),
        ShadowProposalAuditRecord(
            prompt_id="p1",
            compressor="int8",
            max_new_tokens=8,
            round_index=1,
            proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
            extraction_status="blocked",
            extraction_source_field=None,
            proposed_token_id=None,
            proposed_token_text=None,
            committed_token_id_for_comparison=3,
            committed_token_text_for_comparison=None,
            matched_committed_token=None,
            block_reason="no explicit shadow top-1 diagnostic field available",
            audit_categories=(AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD,),
            interpretation_note="note",
        ),
    )
    agg = aggregate_provenance_audit(records)
    assert agg["category_summary_by_compressor"]["noop"][
        AUDIT_CATEGORY_SHADOW_TOP1_MISMATCHES_COMMITTED
    ] == 1
    assert agg["category_summary_by_prompt"]["p1"][
        AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD
    ] == 1
    assert agg["category_summary_by_max_new_tokens"]["8"][
        AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD
    ] == 1
    assert agg["category_summary_by_round_index"]["1"][
        AUDIT_CATEGORY_MISSING_SHADOW_TOP1_FIELD
    ] == 1


def test_decision_gate_recommends_replace_for_low_coverage_zero_match() -> None:
    decision, reason = compute_decision_recommendation(
        total_audited_rounds=152,
        safe_extraction_count=53,
        unsafe_rejected_count=0,
        match_rate_successful_extractions=0.0,
    )
    assert decision == DECISION_REPLACE_PROPOSAL_SOURCE
    assert "low coverage" in reason


def test_decision_gate_recommends_stop_for_unsafe_source_dependency() -> None:
    decision, _reason = compute_decision_recommendation(
        total_audited_rounds=100,
        safe_extraction_count=5,
        unsafe_rejected_count=20,
        match_rate_successful_extractions=0.0,
    )
    assert decision == DECISION_STOP_L3_TOP1_PATH


def test_safety_spec_validation_passes_l3() -> None:
    result = validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)
    assert result["pass"] is True


def test_report_schema_validates() -> None:
    report = _run_audit()
    assert report["experiment_id"] == EXPERIMENT_094_ID
    assert validate_exp094_report(report) == []
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_18E
    assert report["decision_recommendation"] in (
        DECISION_REPLACE_PROPOSAL_SOURCE,
        DECISION_STOP_L3_TOP1_PATH,
        "continue_with_decode_time_shadow_top1",
        "needs_more_evidence",
    )


def test_no_forbidden_positive_claims_in_report() -> None:
    report = _run_audit()
    for cell in report.get("cells") or []:
        dumped = json.dumps(cell.get("blockers", "")).lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            assert phrase.lower() not in dumped


def test_synthetic_panel_produces_audit_records() -> None:
    report = _run_audit()
    assert report["total_audited_rounds"] > 0
    assert len(report["audit_records"]) == report["total_audited_rounds"]
