"""Tests for Experiment 091 L3 guarded draft-shadow no-commit scaffold (Phase 18B)."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from exactkv.safety.guarded_draft_shadow import (
    EXPERIMENT_091_ID,
    PROPOSAL_SOURCE_BLOCKED,
    PROPOSAL_SOURCE_SYNTHETIC,
    RECOMMENDED_NEXT_PHASE,
    GuardedDraftShadowProposal,
    GuardedDraftShadowSafetyResult,
    build_blocked_proposals,
    build_synthetic_proposals,
    default_no_commit_safety_result,
    extract_proposals,
    run_exp091_guarded_draft_shadow_no_commit_scaffold,
    validate_exp091_report,
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
        "live_snapshots": [],
        "blockers": [],
    }


def _run_panel(**overrides: object) -> dict:
    defaults = {
        "prompts": [("p0", "hello"), ("p1", "world")],
        "compressors_requested": ["noop", "int8"],
        "proposal_source": PROPOSAL_SOURCE_SYNTHETIC,
        "baseline_generation_fn": _baseline_fn,
        "draft_shadow_generation_fn": _draft_shadow_fn,
    }
    defaults.update(overrides)
    return run_exp091_guarded_draft_shadow_no_commit_scaffold(**defaults)


def test_proposal_dataclasses_immutable() -> None:
    prop = GuardedDraftShadowProposal(
        round_index=0,
        prompt_id="p",
        compressor="noop",
        prefix_token_ids=(1, 2),
        proposed_token_ids=(3,),
        proposed_text=None,
        proposal_source=PROPOSAL_SOURCE_SYNTHETIC,
        proposal_status="complete",
        exception=None,
    )
    with pytest.raises(FrozenInstanceError):
        prop.round_index = 1  # type: ignore[misc]


def test_synthetic_provider_creates_proposals() -> None:
    props = build_synthetic_proposals(
        prompt_id="p0",
        compressor="noop",
        generated_token_ids=TOKENS,
        prefix_token_ids=[1, 2],
    )
    assert len(props) == 4
    assert props[0].proposal_source == PROPOSAL_SOURCE_SYNTHETIC
    assert props[0].proposed_token_ids == (100,)


def test_blocked_provider() -> None:
    props = build_blocked_proposals(
        prompt_id="p0",
        compressor="noop",
        reason="no provider",
    )
    assert len(props) == 1
    assert props[0].proposal_status == "blocked"
    assert props[0].proposal_source == PROPOSAL_SOURCE_BLOCKED


def test_proposal_safety_gates_no_commit() -> None:
    safety = default_no_commit_safety_result()
    assert safety.proposal_used_for_token_commit is False
    assert safety.proposal_exposed_to_generator is False
    assert safety.generated_output_modified_by_proposal is False
    assert safety.all_gates_ok


def test_safety_spec_passes_l3_no_commit() -> None:
    report = _run_panel()
    assert report["safety_spec_validation"]["pass"] is True


def test_safety_spec_fails_direct_commit() -> None:
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
    result = validate_integration_proposal(bad)
    assert result["pass"] is False
    assert "no_direct_shadow_commit_gate" in result["failed_gates"]


def test_baseline_draft_shadow_parity_aggregation() -> None:
    report = _run_panel()
    assert report["total_cells"] == 4
    assert report["baseline_vs_draft_shadow_token_match_cells"] == 4
    assert report["baseline_vs_draft_shadow_text_match_cells"] == 4


def test_proposal_match_summary() -> None:
    report = _run_panel()
    assert report["total_proposals"] == 16
    assert report["proposal_match_summary"]["proposals_matching_committed_token"] == 16


def test_first_proposal_mismatch_round() -> None:
    props = build_synthetic_proposals(
        prompt_id="p0",
        compressor="noop",
        generated_token_ids=(100, 101),
        prefix_token_ids=[1],
    )
    # Override committed metadata mismatch for diagnostic summary test
    mismatched = GuardedDraftShadowProposal(
        round_index=props[0].round_index,
        prompt_id=props[0].prompt_id,
        compressor=props[0].compressor,
        prefix_token_ids=props[0].prefix_token_ids,
        proposed_token_ids=props[0].proposed_token_ids,
        proposed_text=props[0].proposed_text,
        proposal_source=props[0].proposal_source,
        proposal_status=props[0].proposal_status,
        exception=props[0].exception,
        metadata=(("committed_token_id", "999"),),
    )
    from exactkv.safety.guarded_draft_shadow import summarize_proposal_match

    summary = summarize_proposal_match(
        (mismatched,),
        committed_token_ids=[999],
    )
    assert summary["first_proposal_mismatch_round"] == 0
    assert summary["proposals_not_matching_committed_token"] == 1
    summary2 = summarize_proposal_match(
        (mismatched,),
        committed_token_ids=[100],
    )
    assert summary2["first_proposal_mismatch_round"] is None
    assert summary2["proposals_matching_committed_token"] == 1


def test_schema_validates() -> None:
    report = _run_panel()
    assert report["experiment_id"] == EXPERIMENT_091_ID
    assert validate_exp091_report(report) == []


def test_recommended_next_phase() -> None:
    report = _run_panel()
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE


def test_hidden_divergence_fails_safety_spec() -> None:
    bad = IntegrationProposal(
        proposal_id="hidden",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=True,
        makes_performance_claim=False,
        makes_memory_claim=False,
        makes_serving_claim=False,
        makes_vericache_claim=False,
    )
    assert validate_integration_proposal(bad)["pass"] is False


def test_overclaim_proposals_fail() -> None:
    for flag, gate in (
        ("makes_performance_claim", "claim_boundary_gate"),
        ("makes_memory_claim", "claim_boundary_gate"),
        ("makes_serving_claim", "claim_boundary_gate"),
        ("makes_vericache_claim", "claim_boundary_gate"),
    ):
        kwargs = {
            "proposal_id": flag,
            "proposed_level": "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "opt_in_only": True,
            "modifies_default_runtime": False,
            "verifier_source_of_truth": True,
            "shadow_can_commit_directly": False,
            "compressed_draft_can_commit_without_verifier": False,
            "fallback_to_baseline": True,
            "reports_exactkv_failures": True,
            "hides_token_divergence": False,
            "makes_performance_claim": False,
            "makes_memory_claim": False,
            "makes_serving_claim": False,
            "makes_vericache_claim": False,
        }
        kwargs[flag] = True
        result = validate_integration_proposal(IntegrationProposal(**kwargs))
        assert result["pass"] is False
        assert gate in result["failed_gates"]


def test_no_forbidden_positive_claim_phrases() -> None:
    report = _run_panel()
    for cell in report["cells"]:
        for field in ("blockers",):
            dumped = json.dumps(cell.get(field, "")).lower()
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


def test_extract_proposals_blocked_source() -> None:
    props = extract_proposals(
        proposal_source=PROPOSAL_SOURCE_BLOCKED,
        prompt_id="p",
        compressor="noop",
        draft_shadow_out=_draft_shadow_fn(),
    )
    assert props[0].proposal_status == "blocked"
