"""Tests for Experiment 093 L3 shadow top-1 extraction hardening (Phase 18D)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.safety.guarded_draft_shadow import (
    DEFAULT_EXP092_REPORT,
    EXPERIMENT_093_ID,
    EXTRACTION_SOURCES_ALLOWED,
    EXTRACTION_SOURCES_FORBIDDEN,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_SYNTHETIC,
    RECOMMENDED_NEXT_PHASE_18D,
    ShadowTop1ExtractionResult,
    aggregate_extraction_results,
    compute_coverage_delta,
    extract_shadow_top1_candidate,
    load_exp092_previous_coverage,
    run_exp093_shadow_top1_extraction_hardening,
    validate_exp093_report,
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


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "prompts": [("p0", "a"), ("p1", "b")],
        "max_new_tokens_values": (4, 8),
        "compressors_requested": ["noop", "int8"],
        "proposal_source": PROPOSAL_SOURCE_SYNTHETIC,
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
        "exp092_report_path": Path("/nonexistent/exp092.json"),
    }
    defaults.update(overrides)
    return run_exp093_shadow_top1_extraction_hardening(**defaults)


def test_explicit_shadow_top1_token_id_extraction_succeeds() -> None:
    result = extract_shadow_top1_candidate({"shadow_top1_token_id": 42})
    assert result.extraction_status == "success"
    assert result.proposed_token_id == 42
    assert result.extraction_source_field == "shadow_top1_token_id"
    assert result.is_shadow_derived is True
    assert result.uses_committed_token is False
    assert result.uses_baseline_token is False


def test_explicit_shadow_topk_token_ids_rank0_extraction_succeeds() -> None:
    result = extract_shadow_top1_candidate({"shadow_topk_token_ids": [77, 88]})
    assert result.extraction_status == "success"
    assert result.proposed_token_id == 77
    assert result.extraction_source_field == "shadow_topk_token_ids[0]"
    assert result.extraction_confidence == "topk_rank0"


def test_committed_token_source_is_rejected() -> None:
    result = extract_shadow_top1_candidate({"committed_token_id": 99})
    assert result.extraction_status == "unsafe_rejected"
    assert result.block_reason == "committed token source rejected"


def test_baseline_token_source_is_rejected() -> None:
    result = extract_shadow_top1_candidate({"baseline_token_id": 55})
    assert result.extraction_status == "unsafe_rejected"
    assert result.block_reason == "baseline token source rejected"


def test_missing_top1_source_blocks() -> None:
    result = extract_shadow_top1_candidate({"shadow_status": "shadow_blocked"})
    assert result.extraction_status == "blocked"
    assert result.block_reason == "no safe top1 extraction from shadow output"


def test_unsafe_retokenization_disabled_by_default() -> None:
    result = extract_shadow_top1_candidate(
        {"unsafe_retokenization_token_id": 12, "shadow_top1_token_id": 12},
    )
    assert result.extraction_status == "blocked"
    assert "unsafe retokenization" in (result.block_reason or "")


def test_source_provenance_is_recorded() -> None:
    result = extract_shadow_top1_candidate(
        {
            "topk_agreement_metrics": {"shadow_top1_token_id": 314},
            "shadow_top1_token_text": "pi",
        },
    )
    assert result.extraction_status == "success"
    assert result.extraction_source_field == "topk_agreement_metrics.shadow_top1_token_id"
    d = result.to_dict()
    assert d["is_shadow_derived"] is True
    assert d["uses_committed_token"] is False


def test_successful_extraction_requires_is_shadow_derived_true() -> None:
    result = extract_shadow_top1_candidate(
        {
            "_extraction_source_field": "committed_token_id",
            "shadow_top1_token_id": 1,
        },
    )
    assert result.extraction_status == "unsafe_rejected"


def test_successful_extraction_requires_uses_committed_token_false() -> None:
    result = extract_shadow_top1_candidate({"shadow_top1_token_id": 5, "committed_token_id": 5})
    assert result.extraction_status == "success"
    assert result.uses_committed_token is False


def test_coverage_comparison_handles_missing_exp092_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    prev = load_exp092_previous_coverage(missing)
    assert prev["previous_report_available"] is False
    assert prev["previous_coverage_rate"] is None
    cov = compute_coverage_delta(prev, {"current_coverage_rate": 0.5})
    assert cov["coverage_delta"] is None


def test_coverage_comparison_computes_delta_when_previous_report_exists(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "exp092.json"
    report_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp092_guarded_draft_shadow_panel_validation",
                "total_proposals": 100,
                "successful_proposals": 35,
                "blocked_proposals": 65,
                "proposal_coverage_rate": 0.35,
            },
        ),
    )
    prev = load_exp092_previous_coverage(report_path)
    cov = compute_coverage_delta(prev, {"current_coverage_rate": 0.5})
    assert cov["coverage_delta"] == pytest.approx(0.15)
    assert prev["previous_successful_proposals"] == 35


def test_safety_spec_validation_passes_l3() -> None:
    result = validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)
    assert result["pass"] is True
    assert result["proposed_level"] == "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT"


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_093_ID
    assert validate_exp093_report(report) == []
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_18D
    assert list(report["extraction_sources_allowed"]) == list(EXTRACTION_SOURCES_ALLOWED)
    assert list(report["extraction_sources_forbidden"]) == list(EXTRACTION_SOURCES_FORBIDDEN)


def test_no_forbidden_positive_claims_in_report() -> None:
    report = _run_panel()
    for cell in report["cells"]:
        dumped = json.dumps(cell.get("blockers", "")).lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            assert phrase.lower() not in dumped


def test_aggregate_extraction_results_from_cells() -> None:
    cells = [
        {
            "proposals": [
                {
                    "extraction_status": "success",
                    "extraction_source_field": "shadow_top1_token_id",
                    "extraction_confidence": "explicit_field",
                    "is_shadow_derived": True,
                    "uses_committed_token": False,
                    "uses_baseline_token": False,
                },
                {
                    "extraction_status": "blocked",
                    "block_reason": "no safe top1 extraction from shadow output",
                    "is_shadow_derived": False,
                    "uses_committed_token": False,
                    "uses_baseline_token": False,
                },
            ],
        },
    ]
    agg = aggregate_extraction_results(cells)
    assert agg["successful_extractions"] == 1
    assert agg["blocked_extractions"] == 1
    assert agg["extraction_source_summary"]["shadow_top1_token_id"] == 1


def test_exp093_does_not_require_exp092_report() -> None:
    report = _run_panel(exp092_report_path=DEFAULT_EXP092_REPORT)
    assert report["previous_coverage"]["previous_report_available"] is (
        DEFAULT_EXP092_REPORT.is_file()
    )
