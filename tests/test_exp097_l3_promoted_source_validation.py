"""Tests for Experiment 097 L3 promoted round-log source validation (Phase 19C)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from exactkv.attention.live_round_observer import LiveRoundSnapshot, build_live_round_snapshot
from exactkv.safety.guarded_draft_shadow import (
    DECISION_L3_SOURCE_NEEDS_MORE_VALIDATION,
    DECISION_L3_SOURCE_NOT_PROMOTED,
    DECISION_L3_SOURCE_PROMOTED,
    EXPERIMENT_097_ID,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_ROUND_LOG,
    PROMOTED_SOURCE_COVERAGE_GATE_THRESHOLD,
    RECOMMENDED_NEXT_PHASE_19C,
    aggregate_promoted_source_breakdowns,
    aggregate_source_viability_gate_summary,
    build_promoted_source_policy,
    compute_promoted_source_decision,
    evaluate_cell_source_viability_gates,
    run_exp097_l3_promoted_source_validation,
    validate_exp097_report,
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


def _proposal_record(
    round_index: int,
    *,
    proposed: list[int],
    matched: bool | None = True,
    block_reason: str | None = None,
    uses_committed: bool = False,
    source_is_round_log: bool = True,
    proposal_source: str = PROPOSAL_SOURCE_ROUND_LOG,
) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "proposal_source": proposal_source,
        "source_field_path": f"live_snapshots[{round_index}].draft_token_ids",
        "proposed_token_ids": proposed,
        "proposed_text": None,
        "source_is_round_log_draft": source_is_round_log,
        "uses_committed_token": uses_committed,
        "uses_baseline_token": False,
        "uses_verifier_token": False,
        "committed_token_ids_for_comparison": [100],
        "accepted_token_count_for_comparison": 1,
        "rejected_or_corrected_token_count_for_comparison": 0,
        "matched_committed_prefix": matched if proposed else None,
        "block_reason": block_reason,
        "interpretation_note": "diagnostic only",
    }


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
    defaults: dict[str, Any] = {
        "model_ids": ["test/model"],
        "prompts": [("p0", "a"), ("p1", "b")],
        "max_new_tokens_values": (4, 8),
        "compressors_requested": ["noop", "int8"],
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
    }
    defaults.update(overrides)
    return run_exp097_l3_promoted_source_validation(**defaults)


def _good_gates() -> dict[str, bool]:
    return {
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "proposal_return_value_ignored": True,
        "proposal_exception_affects_generation": False,
        "generated_output_modified_by_proposal": False,
        "default_runtime_changed": False,
        "baseline_generation_completed": True,
        "draft_shadow_generation_completed": True,
        "baseline_vs_draft_shadow_token_match": True,
        "baseline_vs_draft_shadow_text_match": True,
    }


def test_promoted_source_policy_schema_validates() -> None:
    policy = build_promoted_source_policy()
    assert policy["promoted_source"] == PROPOSAL_SOURCE_ROUND_LOG
    assert policy["authorized_for_l4_token_commit"] is False
    assert policy["exposed_to_generator_decisions"] is False
    assert policy["used_to_accept_reject_commit_tokens"] is False
    demoted = {d["source"]: d["demotion_reasons"] for d in policy["demoted_sources"]}
    assert PROPOSAL_SOURCE_DECODE_TOP1 in demoted
    assert "low coverage" in demoted[PROPOSAL_SOURCE_DECODE_TOP1][0]


def test_decode_time_shadow_top1_is_demoted_with_reason() -> None:
    policy = build_promoted_source_policy()
    entry = next(
        d for d in policy["demoted_sources"] if d["source"] == PROPOSAL_SOURCE_DECODE_TOP1
    )
    reasons = " ".join(entry["demotion_reasons"]).lower()
    assert "coverage" in reasons
    assert "prefix match" in reasons
    assert "disagreement" in reasons


def test_source_viability_gates_pass_for_good_synthetic_report() -> None:
    proposals = [_proposal_record(0, proposed=[200]), _proposal_record(1, proposed=[201])]
    gates = evaluate_cell_source_viability_gates(
        proposals=proposals,
        baseline_vs_promoted_source_token_match=True,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 0, "promoted_source": 0},
        safety_gates=_good_gates(),
    )
    assert gates["proposal_coverage_gate"] is True
    assert gates["proposal_provenance_gate"] is True
    assert gates["proposal_isolation_gate"] is True
    assert gates["generation_parity_gate"] is True
    assert gates["exactkv_failure_gate"] is True


def test_coverage_gate_fails_below_threshold() -> None:
    proposals = [
        _proposal_record(0, proposed=[200]),
        _proposal_record(1, proposed=[], block_reason="missing"),
    ]
    gates = evaluate_cell_source_viability_gates(
        proposals=proposals,
        baseline_vs_promoted_source_token_match=True,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 0, "promoted_source": 0},
        safety_gates=_good_gates(),
    )
    assert gates["proposal_coverage_gate"] is False
    assert 0.5 < PROMOTED_SOURCE_COVERAGE_GATE_THRESHOLD


def test_provenance_gate_fails_if_committed_token_source_used() -> None:
    proposals = [_proposal_record(0, proposed=[200], uses_committed=True)]
    gates = evaluate_cell_source_viability_gates(
        proposals=proposals,
        baseline_vs_promoted_source_token_match=True,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 0, "promoted_source": 0},
        safety_gates=_good_gates(),
    )
    assert gates["proposal_provenance_gate"] is False


def test_isolation_gate_fails_if_proposal_affects_commit() -> None:
    bad_gates = _good_gates()
    bad_gates["proposal_used_for_token_commit"] = True
    gates = evaluate_cell_source_viability_gates(
        proposals=[_proposal_record(0, proposed=[200])],
        baseline_vs_promoted_source_token_match=True,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 0, "promoted_source": 0},
        safety_gates=bad_gates,
    )
    assert gates["proposal_isolation_gate"] is False


def test_parity_gate_fails_on_token_mismatch() -> None:
    gates = evaluate_cell_source_viability_gates(
        proposals=[_proposal_record(0, proposed=[200])],
        baseline_vs_promoted_source_token_match=False,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 0, "promoted_source": 0},
        safety_gates=_good_gates(),
    )
    assert gates["generation_parity_gate"] is False


def test_exactkv_failure_gate_fails_when_exactkv_failures_positive() -> None:
    gates = evaluate_cell_source_viability_gates(
        proposals=[_proposal_record(0, proposed=[200])],
        baseline_vs_promoted_source_token_match=True,
        baseline_vs_promoted_source_text_match=True,
        exactkv_failures={"baseline": 1, "promoted_source": 0},
        safety_gates=_good_gates(),
    )
    assert gates["exactkv_failure_gate"] is False


def test_decision_recommends_l3_source_promoted_when_gates_pass() -> None:
    summary = {
        gate: {"pass": True, "cells_passing": 4, "cells_total": 4}
        for gate in (
            "proposal_coverage_gate",
            "proposal_block_gate",
            "proposal_provenance_gate",
            "proposal_isolation_gate",
            "generation_parity_gate",
            "exactkv_failure_gate",
            "claim_boundary_gate",
        )
    }
    summary["all_required_gates_pass"] = True
    decision, _ = compute_promoted_source_decision(
        source_viability_gate_summary=summary,
        successful_cells=4,
        total_cells=4,
        blocked_cells=0,
        failed_cells=0,
        models_blocked=[],
    )
    assert decision == DECISION_L3_SOURCE_PROMOTED


def test_decision_recommends_needs_more_validation_for_blocked_model() -> None:
    summary = {
        gate: {"pass": True, "cells_passing": 2, "cells_total": 2}
        for gate in (
            "proposal_coverage_gate",
            "proposal_provenance_gate",
            "proposal_isolation_gate",
            "generation_parity_gate",
            "exactkv_failure_gate",
            "claim_boundary_gate",
        )
    }
    summary["proposal_block_gate"] = {"pass": True, "cells_passing": 2, "cells_total": 2}
    decision, _ = compute_promoted_source_decision(
        source_viability_gate_summary=summary,
        successful_cells=2,
        total_cells=4,
        blocked_cells=2,
        failed_cells=0,
        models_blocked=[{"model_id": "blocked/model"}],
    )
    assert decision == DECISION_L3_SOURCE_NEEDS_MORE_VALIDATION


def test_decision_recommends_not_promoted_for_safety_failure() -> None:
    summary = {
        "proposal_coverage_gate": {"pass": True},
        "proposal_provenance_gate": {"pass": False},
        "proposal_isolation_gate": {"pass": True},
        "generation_parity_gate": {"pass": True},
        "exactkv_failure_gate": {"pass": True},
        "claim_boundary_gate": {"pass": True},
    }
    decision, _ = compute_promoted_source_decision(
        source_viability_gate_summary=summary,
        successful_cells=4,
        total_cells=4,
        blocked_cells=0,
        failed_cells=1,
        models_blocked=[],
    )
    assert decision == DECISION_L3_SOURCE_NOT_PROMOTED


def test_breakdown_aggregation_works() -> None:
    report = _run_panel()
    assert report["breakdowns_by_model"]
    assert report["breakdowns_by_compressor"]
    assert report["breakdowns_by_prompt"]
    assert report["breakdowns_by_max_new_tokens"]
    assert report["breakdowns_by_round_index"]
    bd = aggregate_promoted_source_breakdowns(report["cells"])
    assert "breakdowns_by_model" in bd
    assert bd["breakdowns_by_model"]["test/model"]["total_cells"] == 8


def test_report_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_097_ID
    assert validate_exp097_report(report) == []
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_19C
    assert report["promoted_source_policy"]["promoted_source"] == PROPOSAL_SOURCE_ROUND_LOG


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
    assert report["blocked_cells"] > 0
    assert report["decision_recommendation"] == DECISION_L3_SOURCE_NEEDS_MORE_VALIDATION


def test_report_fails_if_safety_gate_fails() -> None:
    def _mismatch_draft(**kwargs: object) -> dict:
        out = _draft_shadow_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_panel(draft_shadow_generation_fn=_mismatch_draft)
    assert report["status"] == "failed"
    assert report["parity_summary"]["failed_cells"] > 0


def test_safety_spec_validation_passes_l3() -> None:
    assert validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)["pass"] is True


def test_no_forbidden_positive_claims_in_report() -> None:
    report = _run_panel()
    narrative = json.dumps(
        {
            "claim_note": report.get("claim_note"),
            "limitations": report.get("limitations"),
            "decision_reason": report.get("decision_reason"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_aggregate_source_viability_gate_summary() -> None:
    cells = [
        {
            "generation_completed": True,
            "source_viability_gates": {
                "proposal_coverage_gate": True,
                "proposal_provenance_gate": True,
            },
        },
        {
            "generation_completed": True,
            "source_viability_gates": {
                "proposal_coverage_gate": False,
                "proposal_provenance_gate": True,
            },
        },
    ]
    summary = aggregate_source_viability_gate_summary(cells)
    assert summary["proposal_coverage_gate"]["cells_passing"] == 1
    assert summary["proposal_coverage_gate"]["pass"] is False
