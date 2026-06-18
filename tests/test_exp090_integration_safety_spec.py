"""Tests for Experiment 090 integration safety spec (Phase 18A)."""
from __future__ import annotations

import json

from exactkv.attention.phase16_closeout import ALLOWED_CLAIMS, FORBIDDEN_CLAIMS
from exactkv.safety.integration_safety_spec import (
    EXPERIMENT_090_ID,
    GATES,
    MANDATORY_INVARIANTS,
    RECOMMENDED_NEXT_PHASE,
    SAFETY_LEVELS,
    IntegrationProposal,
    run_exp090_integration_safety_spec,
    validate_exp090_report,
    validate_integration_proposal,
)

LEVEL_IDS = (
    "L2_DIAGNOSTIC_OBSERVER",
    "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
    "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
    "L5_BACKEND_INTEGRATION",
)


def test_schema_validates() -> None:
    report = run_exp090_integration_safety_spec()
    assert report["experiment_id"] == EXPERIMENT_090_ID
    assert validate_exp090_report(report) == []


def test_safety_levels_l2_through_l5() -> None:
    report = run_exp090_integration_safety_spec()
    for level_id in LEVEL_IDS:
        assert level_id in report["safety_levels"]


def test_mandatory_invariants_verifier_source_of_truth() -> None:
    assert "full_verifier_remains_source_of_truth_for_token_commit" in MANDATORY_INVARIANTS


def test_mandatory_invariants_no_direct_shadow_commit() -> None:
    assert "compressed_draft_output_cannot_commit_directly" in MANDATORY_INVARIANTS
    assert "shadow_output_cannot_bypass_verification" in MANDATORY_INVARIANTS


def test_gates_have_pass_fail_conditions() -> None:
    for gate_id, gate in GATES.items():
        assert gate["pass_condition"], gate_id
        assert gate["fail_condition"], gate_id
        assert gate["applies_to_levels"], gate_id


def test_passing_l3_proposal() -> None:
    report = run_exp090_integration_safety_spec()
    l3 = next(
        p for p in report["passing_synthetic_proposals"]
        if p["proposal_id"] == "l3_diagnostic_draft_shadow_no_commit"
    )
    assert l3["pass"] is True
    assert l3["failed_gates"] == []


def test_passing_l4_proposal_requires_verifier() -> None:
    report = run_exp090_integration_safety_spec()
    l4 = next(
        p for p in report["passing_synthetic_proposals"]
        if p["proposal_id"] == "l4_verifier_mediated_compressed_draft_with_full_verifier"
    )
    assert l4["pass"] is True

    bad_l4 = validate_integration_proposal(
        IntegrationProposal(
            proposal_id="l4_no_verifier",
            proposed_level="L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            opt_in_only=True,
            modifies_default_runtime=False,
            verifier_source_of_truth=False,
            shadow_can_commit_directly=False,
            compressed_draft_can_commit_without_verifier=False,
            fallback_to_baseline=True,
            reports_exactkv_failures=True,
            hides_token_divergence=False,
            makes_performance_claim=False,
            makes_memory_claim=False,
            makes_serving_claim=False,
            makes_vericache_claim=False,
        ),
    )
    assert bad_l4["pass"] is False
    assert "verifier_source_of_truth_gate" in bad_l4["failed_gates"]


def test_failing_proposals_rejected() -> None:
    report = run_exp090_integration_safety_spec()
    failing = {p["proposal_id"]: p for p in report["failing_synthetic_proposals"]}
    assert failing["shadow_direct_commit"]["pass"] is False
    assert failing["verifier_bypass"]["pass"] is False
    assert failing["default_runtime_changed"]["pass"] is False
    assert failing["hidden_token_divergence"]["pass"] is False
    assert failing["performance_claim_without_measurement"]["pass"] is False
    assert failing["memory_claim_without_active_measurement"]["pass"] is False
    assert failing["serving_claim_without_backend"]["pass"] is False
    assert failing["vericache_reproduction_overclaim"]["pass"] is False


def test_shadow_direct_commit_fails_no_direct_shadow_gate() -> None:
    result = validate_integration_proposal(
        IntegrationProposal(
            proposal_id="shadow_direct_commit",
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
        ),
    )
    assert "no_direct_shadow_commit_gate" in result["failed_gates"]


def test_recommended_next_phase() -> None:
    report = run_exp090_integration_safety_spec()
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE
    assert report["recommended_next_phase"] == (
        "phase18b_guarded_draft_shadow_no_commit_spec_or_scaffold"
    )


def test_allowed_and_forbidden_claims_match_freeze() -> None:
    report = run_exp090_integration_safety_spec()
    assert report["allowed_claims"] == list(ALLOWED_CLAIMS)
    assert report["forbidden_claims"] == list(FORBIDDEN_CLAIMS)


def test_implemented_l2_future_l3_l4_deferred_l5() -> None:
    levels = SAFETY_LEVELS
    assert levels["L2_DIAGNOSTIC_OBSERVER"]["status"] == "implemented"
    assert levels["L3_GUARDED_DRAFT_SHADOW_NO_COMMIT"]["status"] == "future"
    assert levels["L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"]["status"] == "future"
    assert levels["L5_BACKEND_INTEGRATION"]["status"] == "deferred"


def test_no_forbidden_positive_claim_phrases() -> None:
    report = run_exp090_integration_safety_spec()
    for level in report["safety_levels"].values():
        for field in ("description", "claim_boundary"):
            dumped = json.dumps(level.get(field, "")).lower()
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
