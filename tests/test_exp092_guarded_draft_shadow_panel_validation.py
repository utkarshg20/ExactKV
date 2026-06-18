"""Tests for Experiment 092 L3 guarded draft-shadow panel validation (Phase 18C)."""
from __future__ import annotations

import json

from exactkv.safety.guarded_draft_shadow import (
    DEFAULT_MAX_NEW_TOKENS_VALUES,
    DEFAULT_PANEL_COMPRESSORS,
    EXPERIMENT_092_ID,
    L3_PANEL_SAFETY_SPEC_PROPOSAL,
    PROPOSAL_SOURCE_BLOCKED,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_SYNTHETIC,
    RECOMMENDED_NEXT_PHASE_18C,
    GuardedDraftShadowProposal,
    aggregate_proposal_block_reasons,
    build_blocked_proposals,
    default_no_commit_safety_result,
    run_exp092_guarded_draft_shadow_panel_validation,
    summarize_proposal_coverage,
    summarize_proposal_match,
    validate_exp092_report,
)
from exactkv.safety.integration_safety_spec import (
    IntegrationProposal,
    validate_integration_proposal,
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
    }
    defaults.update(overrides)
    return run_exp092_guarded_draft_shadow_panel_validation(**defaults)


def test_expanded_panel_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_092_ID
    assert validate_exp092_report(report) == []
    assert report["total_cells"] == 2 * 2 * 2


def test_real_panel_default_proposal_source_is_decode_top1() -> None:
    import inspect

    sig = inspect.signature(run_exp092_guarded_draft_shadow_panel_validation)
    assert sig.parameters["proposal_source"].default == PROPOSAL_SOURCE_DECODE_TOP1


def test_default_panel_dimensions() -> None:
    report = run_exp092_guarded_draft_shadow_panel_validation(
        prompts=[("p0", "a"), ("p1", "b"), ("p2", "c"), ("p3", "d")],
        max_new_tokens_values=DEFAULT_MAX_NEW_TOKENS_VALUES,
        compressors_requested=DEFAULT_PANEL_COMPRESSORS,
        proposal_source=PROPOSAL_SOURCE_SYNTHETIC,
        baseline_generation_fn=_baseline_fn,
        draft_shadow_generation_fn=_draft_shadow_fn,
    )
    assert report["total_cells"] == 32


def test_proposal_coverage_aggregation() -> None:
    report = _run_panel()
    assert report["total_proposals"] > 0
    assert "proposal_coverage_rate" in report
    assert report["proposal_coverage_rate"] > 0


def test_proposal_block_reason_aggregation() -> None:
    blocked = build_blocked_proposals(
        prompt_id="p0", compressor="noop", reason="other_top1_token_id unavailable",
    )
    summary = summarize_proposal_coverage(blocked)
    assert summary["blocked_proposals"] == 1
    cell = {"proposal_block_reasons": summary["proposal_block_reasons"]}
    agg = aggregate_proposal_block_reasons([cell])
    assert agg["other_top1_token_id unavailable"] == 1
    report = _run_panel(
        draft_shadow_generation_fn=lambda **kw: {
            **_draft_shadow_fn(**kw),
            "live_snapshots": [],
            "_hf_model": None,
        },
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    assert report["proposal_block_reason_summary"]


def test_blocked_provider() -> None:
    report = _run_panel(proposal_source=PROPOSAL_SOURCE_BLOCKED)
    assert report["blocked_proposals"] > 0


def test_synthetic_provider_test_only() -> None:
    report = _run_panel(proposal_source=PROPOSAL_SOURCE_SYNTHETIC)
    assert report["successful_proposals"] > 0


def test_proposal_match_summary_zero_matches() -> None:
    prop = GuardedDraftShadowProposal(
        round_index=0,
        prompt_id="p",
        compressor="noop",
        prefix_token_ids=(1,),
        proposed_token_ids=(999,),
        proposed_text=None,
        proposal_source=PROPOSAL_SOURCE_DECODE_TOP1,
        proposal_status="complete",
        exception=None,
        metadata=(("committed_token_id", "100"),),
    )
    summary = summarize_proposal_match((prop,), committed_token_ids=[100])
    assert summary["proposals_matching_committed_token"] == 0
    assert summary["proposals_not_matching_committed_token"] == 1


def test_safety_spec_validation_passes_l3() -> None:
    report = _run_panel()
    assert report["safety_spec_validation"]["pass"] is True
    assert validate_integration_proposal(L3_PANEL_SAFETY_SPEC_PROPOSAL)["pass"]


def test_report_fails_if_safety_spec_fails() -> None:
    bad = IntegrationProposal(
        proposal_id="bad",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=True,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=False,
        makes_memory_claim=False,
        makes_serving_claim=False,
        makes_vericache_claim=False,
    )
    report = _run_panel(safety_spec_proposal=bad)
    assert report["status"] == "failed"
    assert validate_exp092_report(report) != []


def test_baseline_parity_pass() -> None:
    report = _run_panel()
    assert report["baseline_vs_draft_shadow_token_match_cells"] == report["total_cells"]


def test_report_fails_on_token_mismatch() -> None:
    def _bad(**kwargs: object) -> dict:
        out = _draft_shadow_fn(**kwargs)
        out["generated_token_ids"] = [999]
        return out

    report = _run_panel(draft_shadow_generation_fn=_bad)
    assert report["status"] == "failed"


def test_safety_gates_no_commit() -> None:
    report = _run_panel()
    safety = default_no_commit_safety_result()
    for cell in report["cells"]:
        gates = cell["safety_gates"]
        assert gates["proposal_used_for_token_commit"] is False
        assert gates["proposal_exposed_to_generator"] is False
        assert gates["generated_output_modified_by_proposal"] is False
        assert gates["default_runtime_changed"] is False
        assert safety.proposal_return_value_ignored is True


def test_recommended_next_phase() -> None:
    report = _run_panel()
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE_18C


def test_no_forbidden_positive_claim_phrases() -> None:
    report = _run_panel()
    for cell in report["cells"]:
        dumped = json.dumps(cell.get("blockers", "")).lower()
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
            "draft shadow used for token commit",
            "verifier-mediated compressed draft implemented",
        ):
            assert forbidden not in dumped
