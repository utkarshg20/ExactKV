"""Tests for Experiment 095 L3 round-log draft proposal source (Phase 19A)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.attention.live_round_observer import LiveRoundSnapshot, build_live_round_snapshot
from exactkv.safety.guarded_draft_shadow import (
    DEFAULT_EXP094_REPORT,
    EXPERIMENT_095_ID,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_ROUND_LOG,
    RECOMMENDED_NEXT_PHASE_19A,
    aggregate_round_log_proposal_coverage,
    build_round_log_draft_proposals,
    compute_source_comparison_delta,
    load_exp094_previous_source_comparison,
    run_exp095_round_log_draft_proposal_source,
    validate_exp095_report,
)
from exactkv.safety.integration_safety_spec import validate_integration_proposal
from exactkv.verification.acceptance import AcceptanceResult, VerificationTrace

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
PROMPT = (1, 2, 3)


def _acceptance(*, draft: list[int], accepted: list[int]) -> AcceptanceResult:
    return AcceptanceResult(
        draft_tokens=draft,
        verifier_tokens=accepted,
        accepted_tokens=accepted,
        correction_token=None,
        rejected_tokens=[],
        bonus_token=None,
        all_matched=draft[: len(accepted)] == accepted,
        num_accepted=len(accepted),
        num_rejected=max(0, len(draft) - len(accepted)),
    )


def _fake_snapshot(round_index: int, draft: tuple[int, ...] | None) -> LiveRoundSnapshot:
    gen_before = TOKENS[:round_index]
    gen_after = TOKENS[: round_index + 2]
    return build_live_round_snapshot(
        round_index=round_index,
        prompt_token_ids=PROMPT,
        generated_token_ids_before=gen_before,
        generated_token_ids_after=gen_after,
        draft_token_ids=draft,
        acceptance=_acceptance(draft=list(draft or ()), accepted=list(gen_after[len(gen_before):])),
        compressor_name="noop",
        max_new_tokens=8,
        full_seq_len_before=len(PROMPT) + len(gen_before),
        full_seq_len_after=len(PROMPT) + len(gen_after),
    )


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
        "prompt_ids": list(PROMPT),
        "exactkv_failures": 0,
        "token_exact_match": True,
        "live_snapshots": [_fake_snapshot(0, (200, 201)), _fake_snapshot(1, (202, 203))],
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "prompts": [("p0", "a"), ("p1", "b")],
        "max_new_tokens_values": (4, 8),
        "compressors_requested": ["noop", "int8"],
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
        "exp094_report_path": Path("/nonexistent/exp094.json"),
    }
    defaults.update(overrides)
    return run_exp095_round_log_draft_proposal_source(**defaults)


def test_proposal_source_exists() -> None:
    assert PROPOSAL_SOURCE_ROUND_LOG == "exactkv_round_log_draft_tokens"


def test_draft_token_extraction_from_fake_round_log_succeeds() -> None:
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={
            "live_snapshots": [_fake_snapshot(0, (200, 201))],
        },
    )
    assert len(proposals) == 1
    assert proposals[0].proposal_status == "complete"
    assert proposals[0].proposed_token_ids == (200, 201)
    meta = dict(proposals[0].metadata)
    assert meta.get("source_is_round_log_draft") == "True"
    assert meta.get("uses_committed_token") == "False"


def test_draft_token_extraction_from_trace_succeeds() -> None:
    trace = VerificationTrace(
        round_idx=0,
        draft_tokens=[300, 301],
        acceptance=_acceptance(draft=[300, 301], accepted=[300, 301]),
        full_seq_len_before=10,
        full_seq_len_after=12,
        compressed_seq_len_after=12,
    )
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"result_traces": [trace]},
    )
    assert proposals[0].proposed_token_ids == (300, 301)


def test_missing_draft_tokens_block() -> None:
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"live_snapshots": [_fake_snapshot(0, None)]},
    )
    assert proposals[0].proposal_status == "blocked"
    assert "missing draft token" in (proposals[0].exception or "")


def test_committed_tokens_cannot_be_used_as_proposal_source() -> None:
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"generated_token_ids": list(TOKENS)},
    )
    assert proposals[0].proposal_status == "blocked"
    assert proposals[0].proposed_token_ids == ()


def test_baseline_tokens_not_used_as_proposal_source() -> None:
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"baseline_token_ids": list(TOKENS)},
    )
    assert proposals[0].proposed_token_ids == ()


def test_verifier_tokens_not_used_as_proposal_source() -> None:
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"verifier_token_ids": list(TOKENS)},
    )
    assert proposals[0].proposed_token_ids == ()


def test_comparison_only_committed_fields_allowed() -> None:
    snap = _fake_snapshot(0, (200, 201))
    proposals = build_round_log_draft_proposals(
        prompt_id="p0",
        compressor="noop",
        draft_shadow_out={"live_snapshots": [snap]},
    )
    meta = dict(proposals[0].metadata)
    assert meta.get("committed_token_ids_for_comparison")
    assert proposals[0].proposed_token_ids != tuple(
        int(x) for x in meta["committed_token_ids_for_comparison"].split(",") if x
    )


def test_provenance_fields_recorded() -> None:
    report = _run_panel()
    rec = report["proposal_records"][0]
    assert rec["proposal_source"] == PROPOSAL_SOURCE_ROUND_LOG
    assert rec["source_is_round_log_draft"] is True
    assert rec["uses_committed_token"] is False
    assert rec["uses_baseline_token"] is False
    assert rec["uses_verifier_token"] is False
    assert "source_field_path" in rec


def test_coverage_aggregation_works() -> None:
    records = [
        {"proposed_token_ids": [1, 2], "matched_committed_prefix": True},
        {"proposed_token_ids": [], "block_reason": "missing", "matched_committed_prefix": None},
    ]
    agg = aggregate_round_log_proposal_coverage(records)
    assert agg["rounds_with_draft_tokens"] == 1
    assert agg["rounds_missing_draft_tokens"] == 1
    assert agg["proposal_coverage_rate"] == 0.5


def test_accepted_rejected_token_count_summaries() -> None:
    records = [
        {
            "proposed_token_ids": [1],
            "accepted_token_count_for_comparison": 2,
            "rejected_or_corrected_token_count_for_comparison": 0,
        },
    ]
    agg = aggregate_round_log_proposal_coverage(records)
    assert agg["accepted_token_count_summary"]["2"] == 1
    assert agg["rejected_or_corrected_token_count_summary"]["0"] == 1


def test_comparison_against_exp094_handles_missing_prior_report(tmp_path: Path) -> None:
    prev = load_exp094_previous_source_comparison(tmp_path / "missing.json")
    assert prev["previous_report_available"] is False
    delta = compute_source_comparison_delta(prev, {"current_coverage_rate": 1.0})
    assert delta["coverage_delta"] is None


def test_comparison_against_exp094_computes_deltas_when_prior_report_exists(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "exp094.json"
    report_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp094_shadow_proposal_provenance_audit",
                "proposal_source": PROPOSAL_SOURCE_DECODE_TOP1,
                "total_audited_rounds": 152,
                "safe_extraction_count": 53,
                "match_rate_total_rounds": 0.0,
            },
        ),
    )
    prev = load_exp094_previous_source_comparison(report_path)
    delta = compute_source_comparison_delta(
        prev,
        {"current_coverage_rate": 1.0, "current_match_rate_total_rounds": 0.5},
    )
    assert delta["coverage_delta"] == pytest.approx(1.0 - (53 / 152))
    assert delta["match_rate_delta"] == 0.5


def test_safety_spec_validation_passes_l3() -> None:
    result = validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)
    assert result["pass"] is True


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_095_ID
    assert validate_exp095_report(report) == []
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_19A


def test_no_forbidden_positive_claims_in_report() -> None:
    report = _run_panel()
    for cell in report.get("cells") or []:
        dumped = json.dumps(cell.get("blockers", "")).lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            assert phrase.lower() not in dumped


def test_exp095_does_not_require_exp094_report() -> None:
    report = _run_panel(exp094_report_path=DEFAULT_EXP094_REPORT)
    assert report["previous_source_comparison"]["previous_report_available"] is (
        DEFAULT_EXP094_REPORT.is_file()
    )
