"""Integration safety specification (Phase 18A / Exp 090).

Machine-readable safety contract governing future L3/L4 integration work.
Specification only — no runtime changes or integration implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from exactkv.attention.phase16_closeout import (
    ALLOWED_CLAIMS,
    FORBIDDEN_CLAIMS,
    FUTURE_DEFERRED_CLAIMS,
    TOPK_INTERPRETATION_NOTE,
)

EXPERIMENT_090_ID = "exp090_integration_safety_spec"
DEFAULT_EXP090_REPORT = Path("reports/experiment_090_integration_safety_spec.json")
PHASE_18A = "18A"

RECOMMENDED_NEXT_PHASE = "phase18b_guarded_draft_shadow_no_commit_spec_or_scaffold"

NO_PERFORMANCE_CLAIMS_NOTE = (
    "No speed, throughput, latency, serving, measured active GPU memory, "
    "or production-memory claim is made."
)

MANDATORY_INVARIANTS: tuple[str, ...] = (
    "default_runtime_unchanged_unless_explicit_opt_in",
    "fallback_path_restores_baseline_generation",
    "full_verifier_remains_source_of_truth_for_token_commit",
    "shadow_output_cannot_bypass_verification",
    "compressed_draft_output_cannot_commit_directly",
    "observer_shadow_return_values_cannot_affect_generation_unless_l4_verified",
    "exactkv_failures_must_be_zero_in_gate_tests",
    "baseline_vs_integrated_token_parity_on_fixed_greedy_tests",
    "all_token_divergences_must_be_surfaced_not_hidden",
    "all_safety_gate_failures_must_fail_the_report",
    "topk_agreement_supplementary_only_not_exactness",
    "no_performance_claims_without_measurement",
    "no_memory_claims_without_active_memory_measurement",
    "no_serving_claims_without_backend_validation",
    "no_vericache_reproduction_claims_without_serving_backend_parity",
)

DEFERRED_WORK_ITEMS: tuple[dict[str, str], ...] = (
    {"item": "L3 guarded draft shadow implementation", "status": "blocked_until_gates_pass", "phase": "18B+"},
    {"item": "L4 verifier-mediated compressed draft", "status": "blocked_until_gates_pass", "phase": "18B+"},
    {"item": "L5 CUDA/Triton/vLLM/LMCache/serving", "status": "deferred", "phase": "future"},
    {"item": "Direct vLLM integration", "status": "no-go", "phase": "V11+"},
    {"item": "Direct LMCache integration", "status": "no-go", "phase": "V11+"},
    {"item": "Measured active GPU memory savings", "status": "deferred", "phase": "future"},
    {"item": "Production serving", "status": "deferred", "phase": "future"},
)

SAFETY_LEVELS: dict[str, dict[str, Any]] = {
    "L2_DIAGNOSTIC_OBSERVER": {
        "level_id": "L2_DIAGNOSTIC_OBSERVER",
        "status": "implemented",
        "description": (
            "Live observer and guarded decode-time diagnostic shadow; "
            "observer/shadow diagnostics cannot affect token commits."
        ),
        "allowed_behavior": [
            "Post-commit shadow callbacks with return values ignored.",
            "Baseline-vs-guarded parity panels on fixed greedy settings.",
            "Decode-time vs post-hoc shadow comparison for diagnostics.",
        ],
        "forbidden_behavior": [
            "Using shadow logits or top-k to accept or reject tokens.",
            "Modifying default generation without explicit opt-in.",
            "Committing tokens from shadow or compressed-draft paths.",
        ],
        "required_gates": [
            "default_runtime_gate",
            "no_direct_shadow_commit_gate",
            "divergence_visibility_gate",
            "claim_boundary_gate",
        ],
        "required_tests": [
            "Guarded shadow panel with safety_gate checks.",
            "Baseline-vs-guarded token and text parity tests.",
        ],
        "claim_boundary": (
            "Panel-scoped diagnostic claims only; top-k supplementary; "
            "no exactness or production claims."
        ),
    },
    "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT": {
        "level_id": "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        "status": "future",
        "description": (
            "Draft/shadow compressed-attention diagnostics run during generation "
            "but cannot commit tokens."
        ),
        "allowed_behavior": [
            "Opt-in guarded draft shadow during decode.",
            "Recording draft-vs-baseline divergence in reports.",
            "Fallback to baseline generation on shadow failure.",
        ],
        "forbidden_behavior": [
            "Direct token commit from shadow or draft output.",
            "Default runtime changes without explicit flag.",
            "Hiding token divergence or safety gate failures.",
        ],
        "required_gates": [
            "default_runtime_gate",
            "fallback_gate",
            "no_direct_shadow_commit_gate",
            "no_verifier_bypass_gate",
            "exactkv_failure_gate",
            "divergence_visibility_gate",
            "baseline_token_parity_gate",
            "claim_boundary_gate",
            "audit_gate",
        ],
        "required_tests": [
            "Synthetic and panel parity tests with exactkv_failures == 0.",
            "Proposal validator pass for L3 plan.",
            "Safety gates fail report on any gate failure.",
        ],
        "claim_boundary": (
            "Diagnostic draft-shadow only; no token-commit integration claims."
        ),
    },
    "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT": {
        "level_id": "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
        "status": "future",
        "description": (
            "Compressed draft may propose tokens; full verifier must remain "
            "source of truth for any accepted commit."
        ),
        "allowed_behavior": [
            "Compressed draft proposes candidate tokens to full verifier.",
            "Full verifier accepts or rejects; baseline fallback on failure.",
            "Panel-scoped parity evidence on fixed greedy settings.",
        ],
        "forbidden_behavior": [
            "Compressed draft committing without full verifier approval.",
            "Shadow bypassing verification.",
            "Changing default runtime without opt-in.",
        ],
        "required_gates": [
            "default_runtime_gate",
            "fallback_gate",
            "verifier_source_of_truth_gate",
            "no_verifier_bypass_gate",
            "no_direct_shadow_commit_gate",
            "exactkv_failure_gate",
            "divergence_visibility_gate",
            "baseline_token_parity_gate",
            "claim_boundary_gate",
            "report_schema_gate",
            "audit_gate",
        ],
        "required_tests": [
            "All L3 tests plus verifier-source-of-truth enforcement tests.",
            "Proposal validator pass only with verifier_source_of_truth=true.",
        ],
        "claim_boundary": (
            "Panel-scoped parity only; full verifier source of truth; "
            "no general exact-generation preservation claim."
        ),
    },
    "L5_BACKEND_INTEGRATION": {
        "level_id": "L5_BACKEND_INTEGRATION",
        "status": "deferred",
        "description": (
            "CUDA/Triton/vLLM/LMCache/serving backend integration — "
            "deferred until L3/L4 safety contract satisfied."
        ),
        "allowed_behavior": [
            "Documented feasibility probes with no integration claims.",
            "Sidecar diagnostics per deferred work register.",
        ],
        "forbidden_behavior": [
            "Production serving claims without backend validation.",
            "VeriCache throughput or serving reproduction claims.",
            "Skipping L4 gate policy.",
        ],
        "required_gates": [
            gate for gate in (
                "default_runtime_gate",
                "fallback_gate",
                "verifier_source_of_truth_gate",
                "no_verifier_bypass_gate",
                "no_direct_shadow_commit_gate",
                "exactkv_failure_gate",
                "divergence_visibility_gate",
                "baseline_token_parity_gate",
                "claim_boundary_gate",
                "report_schema_gate",
                "audit_gate",
            )
        ],
        "required_tests": [
            "All L4 gate tests.",
            "Backend-specific validation panels before serving claims.",
        ],
        "claim_boundary": (
            "No serving, VeriCache reproduction, or memory savings claims "
            "without measured backend validation."
        ),
    },
}

GATES: dict[str, dict[str, Any]] = {
    "default_runtime_gate": {
        "gate_id": "default_runtime_gate",
        "name": "Default runtime unchanged",
        "purpose": "Ensure default generation behavior is unchanged unless explicit opt-in.",
        "required_evidence": [
            "opt_in_only=true and modifies_default_runtime=false in proposal.",
            "Safety gate default_runtime_changed=false in validation panels.",
        ],
        "pass_condition": "opt_in_only is true and modifies_default_runtime is false.",
        "fail_condition": "modifies_default_runtime is true without documented opt-in.",
        "applies_to_levels": [
            "L2_DIAGNOSTIC_OBSERVER",
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "fallback_gate": {
        "gate_id": "fallback_gate",
        "name": "Fallback restores baseline",
        "purpose": "Integration failure or disable must restore baseline generation.",
        "required_evidence": ["fallback_to_baseline=true in proposal.", "Documented fallback path."],
        "pass_condition": "fallback_to_baseline is true.",
        "fail_condition": "fallback_to_baseline is false for L3+ proposals.",
        "applies_to_levels": [
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "verifier_source_of_truth_gate": {
        "gate_id": "verifier_source_of_truth_gate",
        "name": "Full verifier source of truth",
        "purpose": "Committed tokens must come from full verifier for L4+.",
        "required_evidence": [
            "verifier_source_of_truth=true for L4 proposals.",
            "Documented verifier-mediated accept/reject path.",
        ],
        "pass_condition": (
            "For L4/L5: verifier_source_of_truth is true. "
            "For L2/L3: not applicable or true if specified."
        ),
        "fail_condition": "L4/L5 proposal with verifier_source_of_truth=false.",
        "applies_to_levels": [
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "no_verifier_bypass_gate": {
        "gate_id": "no_verifier_bypass_gate",
        "name": "No verifier bypass",
        "purpose": "Compressed draft cannot commit without full verifier.",
        "required_evidence": ["compressed_draft_can_commit_without_verifier=false."],
        "pass_condition": "compressed_draft_can_commit_without_verifier is false.",
        "fail_condition": "compressed_draft_can_commit_without_verifier is true.",
        "applies_to_levels": [
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "no_direct_shadow_commit_gate": {
        "gate_id": "no_direct_shadow_commit_gate",
        "name": "No direct shadow commit",
        "purpose": "Shadow output cannot directly commit tokens.",
        "required_evidence": ["shadow_can_commit_directly=false."],
        "pass_condition": "shadow_can_commit_directly is false.",
        "fail_condition": "shadow_can_commit_directly is true.",
        "applies_to_levels": [
            "L2_DIAGNOSTIC_OBSERVER",
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "baseline_token_parity_gate": {
        "gate_id": "baseline_token_parity_gate",
        "name": "Baseline token parity",
        "purpose": "Fixed greedy baseline-vs-integrated token parity required for L3+.",
        "required_evidence": [
            "Panel report with baseline_vs_guarded_token_match on all cells.",
            "hides_token_divergence=false.",
        ],
        "pass_condition": "hides_token_divergence is false for L3+ proposals.",
        "fail_condition": "hides_token_divergence is true.",
        "applies_to_levels": [
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "exactkv_failure_gate": {
        "gate_id": "exactkv_failure_gate",
        "name": "ExactKV failures reported",
        "purpose": "exactkv_failures must be zero in gate tests and reported.",
        "required_evidence": [
            "reports_exactkv_failures=true in proposal.",
            "Panel exactkv_failure_summary with zero failures.",
        ],
        "pass_condition": "reports_exactkv_failures is true for L3+ proposals.",
        "fail_condition": "reports_exactkv_failures is false for L3+ proposals.",
        "applies_to_levels": [
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "divergence_visibility_gate": {
        "gate_id": "divergence_visibility_gate",
        "name": "Divergence visibility",
        "purpose": "All token divergences must be surfaced, not hidden.",
        "required_evidence": ["hides_token_divergence=false.", "Failed cells mark report failed."],
        "pass_condition": "hides_token_divergence is false.",
        "fail_condition": "hides_token_divergence is true.",
        "applies_to_levels": [
            "L2_DIAGNOSTIC_OBSERVER",
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "claim_boundary_gate": {
        "gate_id": "claim_boundary_gate",
        "name": "Claim boundary",
        "purpose": "No forbidden performance, memory, serving, or VeriCache claims.",
        "required_evidence": [
            "makes_performance_claim=false.",
            "makes_memory_claim=false.",
            "makes_serving_claim=false.",
            "makes_vericache_claim=false.",
        ],
        "pass_condition": "All claim flags are false.",
        "fail_condition": "Any performance, memory, serving, or VeriCache claim flag is true.",
        "applies_to_levels": [
            "L2_DIAGNOSTIC_OBSERVER",
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "report_schema_gate": {
        "gate_id": "report_schema_gate",
        "name": "Report schema",
        "purpose": "Integration panels must emit schema-valid reports with safety gates.",
        "required_evidence": ["validate_*_report passes.", "safety_gates per cell."],
        "pass_condition": "Proposal includes schema-valid reporting plan for L4+.",
        "fail_condition": "L4+ proposal without reporting plan (implicit pass for L2/L3 spec).",
        "applies_to_levels": [
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
    "audit_gate": {
        "gate_id": "audit_gate",
        "name": "Public claims audit",
        "purpose": "audit_public_claims.py must pass after integration changes.",
        "required_evidence": ["scripts/audit_public_claims.py pass in CI."],
        "pass_condition": "claim_boundary_gate passes and audit planned for L3+.",
        "fail_condition": "Forbidden claims in proposal or planned public messaging.",
        "applies_to_levels": [
            "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
            "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
            "L5_BACKEND_INTEGRATION",
        ],
    },
}


@dataclass
class IntegrationProposal:
    """Proposed future integration plan for gate evaluation."""

    proposal_id: str
    proposed_level: str
    opt_in_only: bool
    modifies_default_runtime: bool
    verifier_source_of_truth: bool
    shadow_can_commit_directly: bool
    compressed_draft_can_commit_without_verifier: bool
    fallback_to_baseline: bool
    reports_exactkv_failures: bool
    hides_token_divergence: bool
    makes_performance_claim: bool
    makes_memory_claim: bool
    makes_serving_claim: bool
    makes_vericache_claim: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _level_applies(gate_id: str, level: str) -> bool:
    gate = GATES[gate_id]
    return level in gate["applies_to_levels"]


def validate_integration_proposal(proposal: IntegrationProposal) -> dict[str, Any]:
    """Evaluate a proposed integration plan against safety gates."""
    failed: list[str] = []
    warnings: list[str] = list(proposal.warnings)
    required_next: list[str] = []
    level = proposal.proposed_level

    # default_runtime_gate
    if _level_applies("default_runtime_gate", level):
        if proposal.modifies_default_runtime or not proposal.opt_in_only:
            failed.append("default_runtime_gate")

    # fallback_gate
    if _level_applies("fallback_gate", level):
        if not proposal.fallback_to_baseline:
            failed.append("fallback_gate")

    # verifier_source_of_truth_gate
    if _level_applies("verifier_source_of_truth_gate", level):
        if not proposal.verifier_source_of_truth:
            failed.append("verifier_source_of_truth_gate")

    # no_verifier_bypass_gate
    if _level_applies("no_verifier_bypass_gate", level):
        if proposal.compressed_draft_can_commit_without_verifier:
            failed.append("no_verifier_bypass_gate")

    # no_direct_shadow_commit_gate
    if _level_applies("no_direct_shadow_commit_gate", level):
        if proposal.shadow_can_commit_directly:
            failed.append("no_direct_shadow_commit_gate")

    # baseline_token_parity_gate
    if _level_applies("baseline_token_parity_gate", level):
        if proposal.hides_token_divergence:
            failed.append("baseline_token_parity_gate")

    # exactkv_failure_gate
    if _level_applies("exactkv_failure_gate", level):
        if not proposal.reports_exactkv_failures:
            failed.append("exactkv_failure_gate")

    # divergence_visibility_gate
    if _level_applies("divergence_visibility_gate", level):
        if proposal.hides_token_divergence:
            failed.append("divergence_visibility_gate")

    # claim_boundary_gate
    if _level_applies("claim_boundary_gate", level):
        if (
            proposal.makes_performance_claim
            or proposal.makes_memory_claim
            or proposal.makes_serving_claim
            or proposal.makes_vericache_claim
        ):
            failed.append("claim_boundary_gate")

    # report_schema_gate — L4+ requires reporting commitment
    if _level_applies("report_schema_gate", level):
        if not proposal.reports_exactkv_failures or proposal.hides_token_divergence:
            failed.append("report_schema_gate")

    # audit_gate
    if _level_applies("audit_gate", level):
        if "claim_boundary_gate" in failed:
            failed.append("audit_gate")

    if level in (
        "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
    ):
        required_next.append("Deterministic panel with baseline-vs-integrated token parity.")
        required_next.append("Safety gates per cell with report failure on mismatch.")
    if level == "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT":
        required_next.append("Documented full-verifier accept/reject path for draft proposals.")

    passed = len(failed) == 0
    return {
        "proposal_id": proposal.proposal_id,
        "proposed_level": proposal.proposed_level,
        "pass": passed,
        "failed_gates": failed,
        "warnings": warnings,
        "required_next_evidence": required_next,
    }


# --- Synthetic proposals ---

PASSING_SYNTHETIC_PROPOSALS: tuple[IntegrationProposal, ...] = (
    IntegrationProposal(
        proposal_id="l3_diagnostic_draft_shadow_no_commit",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
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
    IntegrationProposal(
        proposal_id="l4_verifier_mediated_compressed_draft_with_full_verifier",
        proposed_level="L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
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

FAILING_SYNTHETIC_PROPOSALS: tuple[IntegrationProposal, ...] = (
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
    IntegrationProposal(
        proposal_id="verifier_bypass",
        proposed_level="L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=False,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=True,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=False,
        makes_memory_claim=False,
        makes_serving_claim=False,
        makes_vericache_claim=False,
    ),
    IntegrationProposal(
        proposal_id="default_runtime_changed",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=False,
        modifies_default_runtime=True,
        verifier_source_of_truth=True,
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
    IntegrationProposal(
        proposal_id="hidden_token_divergence",
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
    ),
    IntegrationProposal(
        proposal_id="performance_claim_without_measurement",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=True,
        makes_memory_claim=False,
        makes_serving_claim=False,
        makes_vericache_claim=False,
    ),
    IntegrationProposal(
        proposal_id="memory_claim_without_active_measurement",
        proposed_level="L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=False,
        makes_memory_claim=True,
        makes_serving_claim=False,
        makes_vericache_claim=False,
    ),
    IntegrationProposal(
        proposal_id="serving_claim_without_backend",
        proposed_level="L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=False,
        makes_memory_claim=False,
        makes_serving_claim=True,
        makes_vericache_claim=False,
    ),
    IntegrationProposal(
        proposal_id="vericache_reproduction_overclaim",
        proposed_level="L5_BACKEND_INTEGRATION",
        opt_in_only=True,
        modifies_default_runtime=False,
        verifier_source_of_truth=True,
        shadow_can_commit_directly=False,
        compressed_draft_can_commit_without_verifier=False,
        fallback_to_baseline=True,
        reports_exactkv_failures=True,
        hides_token_divergence=False,
        makes_performance_claim=False,
        makes_memory_claim=False,
        makes_serving_claim=False,
        makes_vericache_claim=True,
    ),
)


def run_exp090_integration_safety_spec() -> dict[str, Any]:
    """Build Experiment 090 integration safety spec report."""
    passing_results = [validate_integration_proposal(p) for p in PASSING_SYNTHETIC_PROPOSALS]
    failing_results = [validate_integration_proposal(p) for p in FAILING_SYNTHETIC_PROPOSALS]

    all_passing_ok = all(r["pass"] for r in passing_results)
    all_failing_rejected = all(not r["pass"] for r in failing_results)

    if all_passing_ok and all_failing_rejected:
        status = "spec_complete"
    elif all_passing_ok:
        status = "spec_partial"
    else:
        status = "spec_failed"

    return {
        "experiment_id": EXPERIMENT_090_ID,
        "status": status,
        "phase": PHASE_18A,
        "safety_levels": SAFETY_LEVELS,
        "mandatory_invariants": list(MANDATORY_INVARIANTS),
        "gates": GATES,
        "topk_interpretation_note": TOPK_INTERPRETATION_NOTE,
        "proposal_validator_summary": {
            "passing_synthetic_count": len(passing_results),
            "failing_synthetic_count": len(failing_results),
            "all_passing_accepted": all_passing_ok,
            "all_failing_rejected": all_failing_rejected,
        },
        "passing_synthetic_proposals": passing_results,
        "failing_synthetic_proposals": failing_results,
        "allowed_claims": list(ALLOWED_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "future_deferred_claims": list(FUTURE_DEFERRED_CLAIMS),
        "deferred_work": [dict(item) for item in DEFERRED_WORK_ITEMS],
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "limitations": [
            "Integration safety specification only — no L3/L4/L5 implementation.",
            "ExactKV default generation and ExactKVGenerator unchanged.",
            "Streaming attention is not integrated into token commit.",
            "Full verification must remain source of truth before compressed draft acceptance.",
            "Shadow output cannot directly commit tokens.",
            "No new model experiments were run for this spec.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp090_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "safety_levels",
        "mandatory_invariants",
        "gates",
        "proposal_validator_summary",
        "passing_synthetic_proposals",
        "failing_synthetic_proposals",
        "allowed_claims",
        "forbidden_claims",
        "deferred_work",
        "recommended_next_phase",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_090_ID:
        errors.append("experiment_id mismatch")

    for level_id in (
        "L2_DIAGNOSTIC_OBSERVER",
        "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT",
        "L5_BACKEND_INTEGRATION",
    ):
        if level_id not in (report.get("safety_levels") or {}):
            errors.append(f"missing safety level: {level_id}")

    for gate_id in GATES:
        if gate_id not in (report.get("gates") or {}):
            errors.append(f"missing gate: {gate_id}")

    invariants = report.get("mandatory_invariants") or []
    if "full_verifier_remains_source_of_truth_for_token_commit" not in invariants:
        errors.append("missing verifier source of truth invariant")
    if "compressed_draft_output_cannot_commit_directly" not in invariants:
        errors.append("missing no direct shadow/draft commit invariant")

    if report.get("recommended_next_phase") != RECOMMENDED_NEXT_PHASE:
        errors.append("recommended_next_phase mismatch")

    if list(report.get("allowed_claims") or []) != list(ALLOWED_CLAIMS):
        errors.append("allowed_claims must match Phase 16 claim freeze")

    if list(report.get("forbidden_claims") or []) != list(FORBIDDEN_CLAIMS):
        errors.append("forbidden_claims must match Phase 16 claim freeze")

    return errors
