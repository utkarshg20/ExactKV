"""Tests for Experiment 096 L3 proposal source comparison panel (Phase 19B)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from exactkv.attention.live_round_observer import LiveRoundSnapshot, build_live_round_snapshot
from exactkv.safety.guarded_draft_shadow import (
    DECISION_INSUFFICIENT_EVIDENCE,
    DECISION_KEEP_COMPARING,
    DECISION_PROMOTE_ROUND_LOG,
    DECISION_REPLACE_BOTH,
    EXPERIMENT_096_ID,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_ROUND_LOG,
    RECOMMENDED_NEXT_PHASE_19B,
    aggregate_side_by_side_summary,
    build_side_by_side_round_records,
    compute_comparison_decision,
    run_exp096_round_log_proposal_source_comparison_panel,
    summarize_proposal_source_rounds,
    validate_exp096_report,
)
from exactkv.safety.integration_safety_spec import validate_integration_proposal
from exactkv.verification.acceptance import AcceptanceResult

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
        "live_snapshots": [
            _fake_snapshot(0, (200, 201)),
            _fake_snapshot(1, (202, 203)),
        ],
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults: dict[str, Any] = {
        "model_ids": ["test/model"],
        "prompts": [("p0", "a"), ("p1", "b")],
        "max_new_tokens_values": (4, 8),
        "compressors_requested": ["noop", "int8"],
        "proposal_sources": (PROPOSAL_SOURCE_ROUND_LOG, PROPOSAL_SOURCE_DECODE_TOP1),
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
    }
    defaults.update(overrides)
    return run_exp096_round_log_proposal_source_comparison_panel(**defaults)


def _round_log_record(
    round_index: int,
    *,
    proposed: list[int],
    matched: bool | None = True,
    block_reason: str | None = None,
) -> dict:
    return {
        "round_index": round_index,
        "proposed_token_ids": proposed,
        "matched_committed_prefix": matched if proposed else None,
        "block_reason": block_reason,
        "source_field_path": f"live_snapshots[{round_index}].draft_token_ids",
        "committed_token_ids_for_comparison": [100],
    }


def _shadow_record(
    round_index: int,
    *,
    proposed: int | None,
    matched: bool | None = False,
    block_reason: str | None = None,
) -> dict:
    return {
        "round_index": round_index,
        "proposed_token_id": proposed,
        "proposed_token_ids": [proposed] if proposed is not None else [],
        "matched_committed_token": matched if proposed is not None else None,
        "block_reason": block_reason,
        "extraction_source_field": "shadow_top1_token_id" if proposed else None,
        "committed_token_id_for_comparison": 100,
    }


def test_safety_spec_validation_passes_l3() -> None:
    result = validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)
    assert result["pass"] is True


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_096_ID
    assert validate_exp096_report(report) == []
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_19B
    assert len(report["proposal_sources"]) == 2


def test_two_proposal_sources_compared() -> None:
    report = _run_panel()
    cell = report["cells"][0]
    psr = cell["proposal_source_records"]
    assert PROPOSAL_SOURCE_ROUND_LOG in psr
    assert PROPOSAL_SOURCE_DECODE_TOP1 in psr
    assert len(psr[PROPOSAL_SOURCE_ROUND_LOG]) > 0


def test_side_by_side_classifies_availability() -> None:
    records = build_side_by_side_round_records(
        round_log_records=[
            _round_log_record(0, proposed=[200]),
            _round_log_record(1, proposed=[], block_reason="missing"),
            _round_log_record(2, proposed=[300]),
            _round_log_record(4, proposed=[], block_reason="missing"),
        ],
        shadow_records=[
            _shadow_record(0, proposed=200),
            _shadow_record(1, proposed=111),
            _shadow_record(3, proposed=400),
            _shadow_record(4, proposed=None, block_reason="missing"),
        ],
    )
    agg = aggregate_side_by_side_summary(records, round_log_prefix_match_rate=0.5, shadow_prefix_match_rate=0.0)
    assert agg["rounds_where_both_sources_available"] == 1
    assert agg["rounds_where_only_round_log_available"] == 1
    assert agg["rounds_where_only_shadow_top1_available"] == 2
    assert agg["rounds_where_neither_available"] == 1


def test_source_agreement_aggregation() -> None:
    records = build_side_by_side_round_records(
        round_log_records=[_round_log_record(0, proposed=[200]), _round_log_record(1, proposed=[300])],
        shadow_records=[_shadow_record(0, proposed=200), _shadow_record(1, proposed=999)],
    )
    agg = aggregate_side_by_side_summary(records, round_log_prefix_match_rate=1.0, shadow_prefix_match_rate=0.0)
    assert agg["rounds_where_sources_agree"] == 1
    assert agg["rounds_where_sources_disagree"] == 1
    assert records[0]["sources_agree"] is True
    assert records[1]["sources_agree"] is False


def test_round_log_better_recommends_promote() -> None:
    rl = summarize_proposal_source_rounds(
        [_round_log_record(i, proposed=[200 + i]) for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_ROUND_LOG,
    )
    sh = summarize_proposal_source_rounds(
        [_shadow_record(i, proposed=None, block_reason="missing") for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    decision, reason = compute_comparison_decision(
        source_summaries=[rl, sh],
        total_generation_cells=8,
        successful_generation_cells=8,
        blocked_generation_cells=0,
        safety_gates_all_ok=True,
    )
    assert decision == DECISION_PROMOTE_ROUND_LOG
    assert "coverage" in reason.lower() or "match" in reason.lower()


def test_mixed_source_performance_recommends_keep_comparing() -> None:
    rl = summarize_proposal_source_rounds(
        [_round_log_record(i, proposed=[200], matched=True) for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_ROUND_LOG,
    )
    sh = summarize_proposal_source_rounds(
        [_shadow_record(i, proposed=200, matched=True) for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    decision, _ = compute_comparison_decision(
        source_summaries=[rl, sh],
        total_generation_cells=8,
        successful_generation_cells=8,
        blocked_generation_cells=0,
        safety_gates_all_ok=True,
    )
    assert decision == DECISION_KEEP_COMPARING


def test_both_low_coverage_recommends_replace_both() -> None:
    rl = summarize_proposal_source_rounds(
        [_round_log_record(i, proposed=[], block_reason="missing") for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_ROUND_LOG,
    )
    sh = summarize_proposal_source_rounds(
        [_shadow_record(i, proposed=None, block_reason="missing") for i in range(10)],
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    decision, _ = compute_comparison_decision(
        source_summaries=[rl, sh],
        total_generation_cells=8,
        successful_generation_cells=8,
        blocked_generation_cells=0,
        safety_gates_all_ok=True,
    )
    assert decision == DECISION_REPLACE_BOTH


def test_model_blocked_behavior() -> None:
    def _failing_loader(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("model unavailable")

    report = _run_panel(
        model_ids=["blocked/model"],
        runtime_loader=_failing_loader,
        baseline_generation_fn=None,
        draft_shadow_generation_fn=None,
    )
    assert report["models_blocked"]
    assert report["blocked_generation_cells"] > 0
    assert report["models_loaded"] == []


def test_provider_blocked_behavior() -> None:
    def _no_shadow_snapshots(**kwargs: object) -> dict:
        del kwargs
        return {
            "generation_completed": True,
            "generated_token_ids": list(TOKENS),
            "generated_text": "out",
            "prompt_ids": list(PROMPT),
            "exactkv_failures": 0,
            "live_snapshots": [],
            "blockers": [],
        }

    report = _run_panel(draft_shadow_generation_fn=_no_shadow_snapshots)
    sh_summary = next(
        s for s in report["source_summaries"] if s["proposal_source"] == PROPOSAL_SOURCE_DECODE_TOP1
    )
    assert sh_summary["blocked_proposals"] > 0
    assert sh_summary["proposal_coverage_rate"] == 0.0


def test_report_fails_if_safety_gate_fails() -> None:
    def _mismatch_draft(**kwargs: object) -> dict:
        out = _draft_shadow_fn(**kwargs)
        out["generated_token_ids"] = [999, 998]
        return out

    report = _run_panel(draft_shadow_generation_fn=_mismatch_draft)
    assert report["status"] == "failed"
    assert report["parity_summary"]["failed_cells"] > 0


def test_committed_tokens_are_comparison_only() -> None:
    report = _run_panel()
    for cell in report["cells"]:
        for recs in (cell.get("proposal_source_records") or {}).values():
            for rec in recs:
                if rec.get("proposed_token_ids"):
                    assert rec.get("uses_committed_token") is not True
                if rec.get("proposed_token_id") is not None:
                    uses = rec.get("uses_committed_token")
                    assert uses is not True


def test_no_forbidden_positive_claims_in_report() -> None:
    report = _run_panel()
    for cell in report.get("cells") or []:
        dumped = json.dumps(cell.get("blockers", "")).lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            assert phrase.lower() not in dumped
    narrative = json.dumps(
        {
            "claim_note": report.get("claim_note"),
            "limitations": report.get("limitations"),
            "decision_reason": report.get("decision_reason"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_insufficient_evidence_when_too_many_blocked_cells() -> None:
    decision, _ = compute_comparison_decision(
        source_summaries=[],
        total_generation_cells=10,
        successful_generation_cells=2,
        blocked_generation_cells=8,
        safety_gates_all_ok=True,
    )
    assert decision == DECISION_INSUFFICIENT_EVIDENCE
