"""L4 verifier-mediated compressed draft design specification (Phase 20B / Exp 099).

Design and contract objects only — no runtime integration or generation changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_099_ID = "exp099_l4_verifier_mediated_design_spec"
DEFAULT_EXP099_REPORT = Path(
    "reports/experiment_099_l4_verifier_mediated_design_spec.json",
)
PHASE_20B = "20B"
RECOMMENDED_NEXT_PHASE_20B = "phase20c_l4_contract_tests_no_runtime"
FORBIDDEN_NEXT_PHASE_20B = "phase20c_l4_runtime_implementation"

L4_OPT_IN_FLAG = "--experimental-l4-verifier-mediated-draft"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

DESIGN_OUTCOME_COMPLETE = "l4_design_spec_complete"
DESIGN_OUTCOME_INCOMPLETE = "l4_design_spec_incomplete"
DESIGN_OUTCOME_BLOCKED = "l4_design_spec_blocked"

DESIGN_OUTCOMES: tuple[str, ...] = (
    DESIGN_OUTCOME_COMPLETE,
    DESIGN_OUTCOME_INCOMPLETE,
    DESIGN_OUTCOME_BLOCKED,
)

L4_READINESS_GATE_NAMES: tuple[str, ...] = (
    "default_runtime_unchanged_gate",
    "explicit_opt_in_gate",
    "verifier_source_of_truth_gate",
    "no_verifier_bypass_gate",
    "no_direct_proposal_commit_gate",
    "rollback_on_mismatch_gate",
    "fallback_restores_baseline_gate",
    "trace_completeness_gate",
    "baseline_token_parity_gate",
    "exactkv_failure_gate",
    "claim_boundary_gate",
)

INTENDED_L4_FLOW_STEPS: tuple[str, ...] = (
    "Existing default runtime remains unchanged unless explicit opt-in is enabled.",
    "Compressed draft path proposes one or more tokens from an explicit proposal source.",
    "Full-KV verifier evaluates proposed tokens against verifier output.",
    "Only verified matching prefix may be accepted for commit.",
    "Any mismatch triggers rollback/correction to full verifier output.",
    "All commit decisions are traced in round logs and acceptance metadata.",
    "Any safety failure falls back to existing baseline generation behavior.",
    "exactkv_failures > 0 fails the L4 readiness gate.",
    "No performance, memory, or serving claim is made by L4 design or validation.",
)

FORBIDDEN_DESIGN_CLAIM_PHRASES: tuple[str, ...] = (
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


@dataclass(frozen=True)
class L4DraftProposalContract:
    """Contract for future L4 draft token proposals."""

    proposal_source_must_be_explicit: bool = True
    proposal_source_must_not_commit_directly: bool = True
    proposal_source_must_have_provenance: bool = True
    forbidden_proposal_sources: tuple[str, ...] = (
        "committed_tokens",
        "baseline_tokens",
        "verifier_tokens",
        "retokenized_generated_text",
    )
    promoted_l3_source: str = PROPOSAL_SOURCE_ROUND_LOG
    notes: str = (
        "Draft proposals are inputs to verification only; never commit sources."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FullVerifierContract:
    """Contract requiring full verifier as source of truth."""

    full_verifier_is_source_of_truth: bool = True
    verifier_cannot_be_bypassed: bool = True
    verifier_result_controls_acceptance: bool = True
    verifier_mismatch_must_be_surfaced: bool = True
    notes: str = "Full verification remains required before any compressed draft acceptance."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4AcceptanceContract:
    """Contract for verifier-mediated prefix acceptance."""

    only_longest_verified_matching_prefix: bool = True
    accepted_tokens_must_be_traceable: bool = True
    rejected_or_corrected_tokens_must_be_traceable: bool = True
    no_silent_divergence: bool = True
    notes: str = "Acceptance follows verifier-matched prefix only; divergences are visible."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4RollbackContract:
    """Contract for rollback on L4 failure modes."""

    rollback_restores_baseline_safe_behavior: bool = True
    rollback_on_verifier_mismatch: bool = True
    rollback_on_proposal_exception: bool = True
    rollback_on_missing_verifier_evidence: bool = True
    rollback_on_safety_gate_failure: bool = True
    notes: str = "Rollback must restore baseline-safe generation; never hide failures."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FallbackContract:
    """Contract for default-runtime preservation and opt-out path."""

    default_runtime_must_be_unchanged: bool = True
    opt_out_path_equals_existing_behavior: bool = True
    fallback_must_not_depend_on_compressed_proposal_state: bool = True
    notes: str = "Fallback restores existing generation when L4 is off or fails."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4OptInContract:
    """Contract for future explicit L4 opt-in (design-only flag)."""

    l4_disabled_by_default: bool = True
    opt_in_flag_must_be_explicit_and_experimental: bool = True
    proposed_opt_in_flag: str = L4_OPT_IN_FLAG
    flag_implemented: bool = False
    notes: str = "Opt-in flag is design-only in Phase 20B; not implemented."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4IntegrationPoint:
    """Future integration touchpoint (documentation only)."""

    path: str
    why_future_changes_may_be_needed: str
    what_must_not_change_by_default: str
    safety_risk: str
    required_tests_before_modification: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TestMatrix:
    """Required future tests before L4 implementation."""

    unit_tests: tuple[str, ...]
    synthetic_integration_tests: tuple[str, ...]
    model_tests: tuple[str, ...]
    forbidden_tests_for_design_phase: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4ClaimBoundary:
    """Claim boundaries for L4 design and future implementation."""

    allowed_claim_categories: tuple[str, ...]
    forbidden_claim_categories: tuple[str, ...]
    no_performance_claims_note: str = NO_PERFORMANCE_CLAIMS_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4ImplementationBlocker:
    """Remaining blocker before L4 runtime implementation."""

    blocker_id: str
    description: str
    resolved_by_design_spec: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4ReadinessGate:
    """L4 readiness gate definition."""

    name: str
    purpose: str
    required_evidence: str
    pass_condition: str
    fail_condition: str
    applies_before_implementation: bool
    applies_during_implementation: bool
    applies_before_public_claim: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierMediatedDesignSpec:
    """Top-level L4 design specification aggregate."""

    spec_id: str
    safety_level: str
    intended_l4_flow: tuple[str, ...]
    draft_proposal_contract: L4DraftProposalContract
    full_verifier_contract: L4FullVerifierContract
    acceptance_contract: L4AcceptanceContract
    rollback_contract: L4RollbackContract
    fallback_contract: L4FallbackContract
    opt_in_contract: L4OptInContract
    integration_points: tuple[L4IntegrationPoint, ...]
    test_matrix: L4TestMatrix
    readiness_gates: tuple[L4ReadinessGate, ...]
    claim_boundaries: L4ClaimBoundary
    implementation_blockers: tuple[L4ImplementationBlocker, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "safety_level": self.safety_level,
            "intended_l4_flow": list(self.intended_l4_flow),
            "draft_proposal_contract": self.draft_proposal_contract.to_dict(),
            "full_verifier_contract": self.full_verifier_contract.to_dict(),
            "acceptance_contract": self.acceptance_contract.to_dict(),
            "rollback_contract": self.rollback_contract.to_dict(),
            "fallback_contract": self.fallback_contract.to_dict(),
            "opt_in_contract": self.opt_in_contract.to_dict(),
            "integration_points": [p.to_dict() for p in self.integration_points],
            "test_matrix": self.test_matrix.to_dict(),
            "readiness_gates": [g.to_dict() for g in self.readiness_gates],
            "claim_boundaries": self.claim_boundaries.to_dict(),
            "implementation_blockers": [b.to_dict() for b in self.implementation_blockers],
        }


@dataclass(frozen=True)
class L4DesignReviewResult:
    """Design review outcome for Phase 20B."""

    outcome: str
    l4_design_spec_complete: bool
    l4_implementation_authorized: bool
    next_phase_authorized: str
    forbidden_next_phase: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_l4_integration_points() -> tuple[L4IntegrationPoint, ...]:
    """List future L4 integration touchpoints without modifying runtime."""
    return (
        L4IntegrationPoint(
            path="exactkv/runtime/exactkv_generator.py",
            why_future_changes_may_be_needed=(
                "Future opt-in L4 path may wire compressed draft proposals into "
                "verifier-mediated acceptance behind an experimental flag."
            ),
            what_must_not_change_by_default=(
                "Default generation path, sampling, and commit behavior without "
                "explicit --experimental-l4-verifier-mediated-draft."
            ),
            safety_risk="Direct proposal commit or default-runtime change without opt-in.",
            required_tests_before_modification=(
                "baseline_token_parity_gate",
                "no_direct_proposal_commit_gate",
                "default_runtime_unchanged_gate",
            ),
        ),
        L4IntegrationPoint(
            path="existing verifier path",
            why_future_changes_may_be_needed=(
                "L4 must route draft proposals through existing full verifier "
                "acceptance without bypass."
            ),
            what_must_not_change_by_default=(
                "Verifier remains authoritative; no shortcut around full verification."
            ),
            safety_risk="Verifier bypass or silent acceptance of unverified tokens.",
            required_tests_before_modification=(
                "verifier_source_of_truth_gate",
                "no_verifier_bypass_gate",
            ),
        ),
        L4IntegrationPoint(
            path="ExactKVResult / round trace structure",
            why_future_changes_may_be_needed=(
                "Trace fields for proposal source, acceptance, rollback, and "
                "comparison-only committed tokens."
            ),
            what_must_not_change_by_default=(
                "Existing trace schema for default runtime; L4 fields opt-in only."
            ),
            safety_risk="Incomplete traces hiding divergence or proposal provenance.",
            required_tests_before_modification=("trace_completeness_gate",),
        ),
        L4IntegrationPoint(
            path="L3 proposal source policy",
            why_future_changes_may_be_needed=(
                "Promoted exactkv_round_log_draft_tokens may feed L4 proposals "
                "with provenance metadata."
            ),
            what_must_not_change_by_default=(
                "L3 diagnostic-only policy; proposals never commit in L3."
            ),
            safety_risk="Using committed or shadow tokens as proposal sources.",
            required_tests_before_modification=("no_direct_proposal_commit_gate",),
        ),
        L4IntegrationPoint(
            path="safety gate validation",
            why_future_changes_may_be_needed=(
                "L4 readiness gates must be evaluated in reports before merge."
            ),
            what_must_not_change_by_default="Phase 18A–20A safety spec invariants.",
            safety_risk="Merging L4 code without gate validation.",
            required_tests_before_modification=("claim_boundary_gate",),
        ),
        L4IntegrationPoint(
            path="report generation",
            why_future_changes_may_be_needed=(
                "L4 panels must record acceptance, rollback, and exactkv_failures."
            ),
            what_must_not_change_by_default="No performance/memory/serving metrics.",
            safety_risk="Performance or exactness claims without evidence.",
            required_tests_before_modification=("exactkv_failure_gate",),
        ),
        L4IntegrationPoint(
            path="CLI experimental flag path",
            why_future_changes_may_be_needed=(
                f"Future {L4_OPT_IN_FLAG} opt-in entry point."
            ),
            what_must_not_change_by_default="Flag absent from default CLI; disabled by default.",
            safety_risk="Implicit L4 enablement without explicit opt-in.",
            required_tests_before_modification=("explicit_opt_in_gate",),
        ),
    )


def build_l4_test_matrix() -> L4TestMatrix:
    """Define required future tests before L4 implementation."""
    return L4TestMatrix(
        unit_tests=(
            "verifier_source_of_truth_contract",
            "no_direct_proposal_commit",
            "rollback_on_mismatch",
            "fallback_on_exception",
            "opt_in_disabled_by_default",
            "trace_schema_completeness",
            "claim_boundary_audit",
        ),
        synthetic_integration_tests=(
            "proposal_all_match",
            "proposal_partial_match",
            "proposal_first_token_mismatch",
            "proposal_exception",
            "missing_verifier_evidence",
            "hidden_divergence_attempt",
            "fallback_restores_baseline_behavior",
        ),
        model_tests=(
            "baseline_vs_l4_token_parity",
            "baseline_vs_l4_text_parity",
            "exactkv_failures_zero",
            "accepted_rejected_token_trace_consistency",
            "compressor_grid",
            "prompt_grid",
            "max_new_tokens_grid",
        ),
        forbidden_tests_for_design_phase=(
            "performance_benchmark",
            "memory_benchmark",
            "vllm_lmcache_serving_benchmark",
            "cuda_triton_backend_benchmark",
        ),
    )


def build_l4_readiness_gates() -> tuple[L4ReadinessGate, ...]:
    """Build L4 readiness gate definitions."""
    definitions: list[tuple[str, str, str, str, str]] = [
        (
            "default_runtime_unchanged_gate",
            "Ensure default generation path is unchanged without explicit L4 opt-in.",
            "Default-runtime parity tests; opt-in flag absent in default CLI.",
            "Default path identical to baseline without experimental flag.",
            "Any default-runtime behavior change without opt-in.",
        ),
        (
            "explicit_opt_in_gate",
            "L4 must be disabled by default and enabled only via experimental flag.",
            f"CLI design for {L4_OPT_IN_FLAG}; flag default off.",
            "L4 path inactive unless flag explicitly set.",
            "L4 active by default or implicit enablement.",
        ),
        (
            "verifier_source_of_truth_gate",
            "Full verifier output controls acceptance decisions.",
            "Verifier-mediated acceptance tests; trace shows verifier tokens.",
            "All accepted tokens match verifier-approved prefix.",
            "Proposal or shadow tokens accepted without verifier agreement.",
        ),
        (
            "no_verifier_bypass_gate",
            "Compressed draft cannot skip full verification.",
            "Tests attempting bypass must fail gate.",
            "No code path commits draft without verifier evaluation.",
            "Verifier bypass detected in integration tests.",
        ),
        (
            "no_direct_proposal_commit_gate",
            "Draft proposals never commit directly.",
            "Proposal provenance tests; commit trace shows verifier step.",
            "Commits occur only after verifier-mediated acceptance.",
            "Proposal tokens appear in committed output without verifier step.",
        ),
        (
            "rollback_on_mismatch_gate",
            "Rollback on verifier mismatch and proposal failures.",
            "Synthetic mismatch tests; rollback trace recorded.",
            "Mismatch triggers rollback/correction to verifier output.",
            "Silent continuation after mismatch.",
        ),
        (
            "fallback_restores_baseline_gate",
            "Fallback restores baseline generation on L4 failure.",
            "Exception and safety-failure synthetic tests.",
            "Fallback output matches baseline path behavior.",
            "Fallback depends on compressed state or diverges from baseline.",
        ),
        (
            "trace_completeness_gate",
            "Round traces record proposal, verifier, acceptance, rollback.",
            "Trace schema validation on L4 panels.",
            "All L4 decisions have trace fields with provenance.",
            "Missing trace fields for proposal or acceptance.",
        ),
        (
            "baseline_token_parity_gate",
            "Baseline vs L4-opt-in-off token parity on fixed greedy panels.",
            "Model panel with flag disabled.",
            "Token and text parity when L4 not enabled.",
            "Parity failure with L4 disabled.",
        ),
        (
            "exactkv_failure_gate",
            "exactkv_failures must be zero on L4 gate panels.",
            "L4 panel reports exactkv_failures summary.",
            "baseline_failures == 0 and l4_failures == 0.",
            "Any exactkv_failures > 0 on gate panel.",
        ),
        (
            "claim_boundary_gate",
            "No performance/memory/serving/VeriCache claims without evidence.",
            "audit_public_claims.py; report forbidden_claims lists.",
            "No forbidden positive claims in L4 reports or docs.",
            "Performance, memory, serving, or VeriCache reproduction claims.",
        ),
    ]
    gates: list[L4ReadinessGate] = []
    for name, purpose, evidence, pass_cond, fail_cond in definitions:
        gates.append(
            L4ReadinessGate(
                name=name,
                purpose=purpose,
                required_evidence=evidence,
                pass_condition=pass_cond,
                fail_condition=fail_cond,
                applies_before_implementation=True,
                applies_during_implementation=True,
                applies_before_public_claim=True,
            ),
        )
    return tuple(gates)


def build_l4_implementation_blockers() -> tuple[L4ImplementationBlocker, ...]:
    """Remaining blockers after design spec; design-resolved items marked."""
    resolved_ids = {
        "explicit_l4_design_spec",
        "verifier_mediated_acceptance_contract",
        "rollback_behavior_defined",
        "l4_test_matrix_defined",
        "l4_opt_in_flag_designed",
    }
    mapping = {
        "explicit L4 design spec missing": "explicit_l4_design_spec",
        "ExactKVGenerator integration plan missing": "exactkv_generator_integration_plan",
        "fallback path not yet implemented for L4": "l4_fallback_path",
        "L4 opt-in flag not yet designed": "l4_opt_in_flag_designed",
        "verifier-mediated acceptance contract not yet defined": (
            "verifier_mediated_acceptance_contract"
        ),
        "rollback behavior not yet defined": "rollback_behavior_defined",
        "L4 test matrix not yet defined": "l4_test_matrix_defined",
        "no L4 baseline-vs-integrated parity panel": "l4_parity_panel",
        "no L4 exactkv_failures gate run": "l4_exactkv_failures_gate_run",
        "no active GPU memory measurement": "gpu_memory_measurement",
        "no performance benchmark": "performance_benchmark",
        "no serving integration": "serving_integration",
    }
    blockers: list[L4ImplementationBlocker] = []
    for text in L4_IMPLEMENTATION_BLOCKERS:
        bid = mapping.get(text, text.replace(" ", "_").lower()[:40])
        blockers.append(
            L4ImplementationBlocker(
                blocker_id=bid,
                description=text,
                resolved_by_design_spec=bid in resolved_ids,
            ),
        )
    return tuple(blockers)


def build_l4_claim_boundaries() -> L4ClaimBoundary:
    return L4ClaimBoundary(
        allowed_claim_categories=(
            "l4_design_spec_documentation",
            "verifier_mediated_acceptance_contract",
            "rollback_and_fallback_contract",
            "readiness_gate_definitions",
            "panel_scoped_diagnostic_claims",
        ),
        forbidden_claim_categories=(
            "speedup",
            "throughput",
            "latency",
            "tokens_per_second",
            "runtime_seconds",
            "active_gpu_memory",
            "production_memory",
            "serving",
            "vericache_reproduction",
            "l4_implementation_complete",
            "model_output_preservation_generally",
            "exact_generation_preservation_generally",
        ),
    )


def build_l4_verifier_mediated_design_spec() -> L4VerifierMediatedDesignSpec:
    """Build the complete L4 verifier-mediated design specification."""
    return L4VerifierMediatedDesignSpec(
        spec_id=EXPERIMENT_099_ID,
        safety_level=L4_SAFETY_LEVEL,
        intended_l4_flow=INTENDED_L4_FLOW_STEPS,
        draft_proposal_contract=L4DraftProposalContract(),
        full_verifier_contract=L4FullVerifierContract(),
        acceptance_contract=L4AcceptanceContract(),
        rollback_contract=L4RollbackContract(),
        fallback_contract=L4FallbackContract(),
        opt_in_contract=L4OptInContract(),
        integration_points=build_l4_integration_points(),
        test_matrix=build_l4_test_matrix(),
        readiness_gates=build_l4_readiness_gates(),
        claim_boundaries=build_l4_claim_boundaries(),
        implementation_blockers=build_l4_implementation_blockers(),
    )


def evaluate_l4_design_review(
    spec: L4VerifierMediatedDesignSpec,
) -> L4DesignReviewResult:
    """Evaluate whether the L4 design specification is complete."""
    required_gate_names = set(L4_READINESS_GATE_NAMES)
    present_gates = {g.name for g in spec.readiness_gates}
    gates_ok = required_gate_names <= present_gates

    contracts_ok = all(
        (
            spec.draft_proposal_contract.proposal_source_must_not_commit_directly,
            spec.full_verifier_contract.verifier_cannot_be_bypassed,
            spec.acceptance_contract.only_longest_verified_matching_prefix,
            spec.rollback_contract.rollback_on_verifier_mismatch,
            spec.fallback_contract.default_runtime_must_be_unchanged,
            spec.opt_in_contract.l4_disabled_by_default,
        ),
    )

    flow_text = " ".join(spec.intended_l4_flow).lower()
    flow_ok = (
        "verifier" in flow_text
        and "rollback" in flow_text
        and "opt-in" in flow_text
    )

    matrix_ok = (
        len(spec.test_matrix.unit_tests) > 0
        and len(spec.test_matrix.synthetic_integration_tests) > 0
        and len(spec.test_matrix.model_tests) > 0
    )

    complete = gates_ok and contracts_ok and flow_ok and matrix_ok

    if complete:
        return L4DesignReviewResult(
            outcome=DESIGN_OUTCOME_COMPLETE,
            l4_design_spec_complete=True,
            l4_implementation_authorized=False,
            next_phase_authorized=RECOMMENDED_NEXT_PHASE_20B,
            forbidden_next_phase=FORBIDDEN_NEXT_PHASE_20B,
            reason=(
                "L4 verifier-mediated design spec defines flow, contracts, integration "
                "points, test matrix, and readiness gates; implementation not authorized"
            ),
        )

    return L4DesignReviewResult(
        outcome=DESIGN_OUTCOME_INCOMPLETE,
        l4_design_spec_complete=False,
        l4_implementation_authorized=False,
        next_phase_authorized=RECOMMENDED_NEXT_PHASE_20B,
        forbidden_next_phase=FORBIDDEN_NEXT_PHASE_20B,
        reason="one or more required design spec sections incomplete",
    )


def run_exp099_l4_verifier_mediated_design_spec() -> dict[str, Any]:
    """Run Experiment 099 L4 design specification (no runtime changes)."""
    spec = build_l4_verifier_mediated_design_spec()
    review = evaluate_l4_design_review(spec)

    remaining_blockers = [
        b.to_dict()
        for b in spec.implementation_blockers
        if not b.resolved_by_design_spec
    ]

    status = "spec_complete" if review.l4_design_spec_complete else "spec_incomplete"

    return {
        "experiment_id": EXPERIMENT_099_ID,
        "status": status,
        "phase": PHASE_20B,
        "design_spec_objects": {
            "L4VerifierMediatedDesignSpec": spec.to_dict(),
            "L4DesignReviewResult": review.to_dict(),
        },
        "intended_l4_flow": list(spec.intended_l4_flow),
        "mandatory_contracts": {
            "draft_proposal_contract": spec.draft_proposal_contract.to_dict(),
            "full_verifier_contract": spec.full_verifier_contract.to_dict(),
            "acceptance_contract": spec.acceptance_contract.to_dict(),
            "rollback_contract": spec.rollback_contract.to_dict(),
            "fallback_contract": spec.fallback_contract.to_dict(),
            "opt_in_contract": spec.opt_in_contract.to_dict(),
        },
        "integration_points": [p.to_dict() for p in spec.integration_points],
        "l4_test_matrix": spec.test_matrix.to_dict(),
        "l4_readiness_gates": [g.to_dict() for g in spec.readiness_gates],
        "design_review_result": review.to_dict(),
        "l4_implementation_authorized": False,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_20B,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_20B,
        "implementation_blockers_remaining": remaining_blockers,
        "claim_boundaries": spec.claim_boundaries.to_dict(),
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE_20B,
        "limitations": [
            "L4 design specification only; not L4 implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "No new model experiments were run for this spec.",
            "Opt-in flag is design-only; not wired to CLI.",
            "Fallback and rollback behavior specified but not implemented.",
            "Performance, memory, and serving claims remain forbidden.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp099_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "design_spec_objects",
        "intended_l4_flow",
        "mandatory_contracts",
        "integration_points",
        "l4_test_matrix",
        "l4_readiness_gates",
        "design_review_result",
        "l4_implementation_authorized",
        "allowed_next_phase",
        "forbidden_next_phase",
        "implementation_blockers_remaining",
        "claim_boundaries",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_099_ID:
        errors.append("experiment_id mismatch")

    if report.get("l4_implementation_authorized") is not False:
        errors.append("l4_implementation_authorized must be false")

    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")

    if report.get("default_runtime_changed") is not False:
        errors.append("default_runtime_changed must be false")

    review = report.get("design_review_result") or {}
    if review.get("outcome") not in DESIGN_OUTCOMES:
        errors.append("invalid design_review_result.outcome")

    if review.get("outcome") == "l4_design_spec_complete":
        if review.get("l4_implementation_authorized") is not False:
            errors.append("design complete but implementation authorized")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_20B:
        errors.append("allowed_next_phase must be phase20c_l4_contract_tests_no_runtime")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_20B:
        errors.append("forbidden_next_phase must be phase20c_l4_runtime_implementation")

    gate_keys = (
        "name",
        "purpose",
        "required_evidence",
        "pass_condition",
        "fail_condition",
        "applies_before_implementation",
        "applies_during_implementation",
        "applies_before_public_claim",
    )
    for idx, gate in enumerate(report.get("l4_readiness_gates") or []):
        for gk in gate_keys:
            if gk not in gate:
                errors.append(f"l4_readiness_gates[{idx}] missing {gk}")

    contract_keys = (
        "draft_proposal_contract",
        "full_verifier_contract",
        "acceptance_contract",
        "rollback_contract",
        "fallback_contract",
        "opt_in_contract",
    )
    contracts = report.get("mandatory_contracts") or {}
    for ck in contract_keys:
        if ck not in contracts:
            errors.append(f"mandatory_contracts missing {ck}")

    return errors
