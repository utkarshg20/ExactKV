"""L4 integration plan review (Phase 20D / Exp 101).

Planning objects only — must not be wired to runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    L4_OPT_IN_FLAG,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_101_ID = "exp101_l4_integration_plan_review"
DEFAULT_EXP101_REPORT = Path(
    "reports/experiment_101_l4_integration_plan_review.json",
)
PHASE_20D = "20D"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_20D = "phase21a_l4_noop_opt_in_scaffold"
FORBIDDEN_NEXT_PHASES_20D: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "phase20d_l4_runtime_implementation",
    "phase21_l4_runtime_commit_implementation",
    "cuda_backend",
    "cuda_backend_implementation",
    "vllm_integration",
    "lmcache_integration",
    "performance_benchmark",
    "memory_benchmark",
)

DECISION_READY_STAGE_1 = "ready_for_stage_1_noop_opt_in_scaffold_design"
DECISION_NOT_READY_STAGE_1 = "not_ready_for_stage_1"
DECISION_BLOCKED_MISSING_PLAN = "blocked_missing_plan"
DECISION_BLOCKED_SAFETY_RISK = "blocked_safety_risk"

INTEGRATION_DECISIONS: tuple[str, ...] = (
    DECISION_READY_STAGE_1,
    DECISION_NOT_READY_STAGE_1,
    DECISION_BLOCKED_MISSING_PLAN,
    DECISION_BLOCKED_SAFETY_RISK,
)

STAGE_IDS: tuple[str, ...] = (
    "stage_0_current_no_runtime",
    "stage_1_noop_opt_in_scaffold",
    "stage_2_trace_only_l4_dry_run",
    "stage_3_verifier_mediated_dry_run",
    "stage_4_runtime_commit_candidate",
)

REQUIRED_RISK_IDS: tuple[str, ...] = (
    "default_runtime_change",
    "verifier_bypass",
    "direct_proposal_commit",
    "hidden_divergence",
    "fallback_not_equivalent_to_baseline",
    "rollback_corrupts_generation_state",
    "trace_incomplete",
    "cli_flag_accidental_enable",
    "test_matrix_too_small",
    "performance_memory_overclaim",
    "vericache_reproduction_overclaim",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

PLAN_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
    {
        "explicit_l4_design_spec",
        "verifier_mediated_acceptance_contract",
        "rollback_behavior_defined",
        "l4_test_matrix_defined",
        "l4_opt_in_flag_designed",
        "l4_synthetic_contract_tests_no_runtime",
        "exactkv_generator_integration_plan",
    },
)


@dataclass(frozen=True)
class L4FutureChangeTarget:
    """Future file or subsystem that may need L4 changes (planning only)."""

    path: str
    why_future_changes_may_be_needed: str
    required_safety_constraint: str
    default_runtime_risk: str
    verifier_bypass_risk: str
    rollback_fallback_requirement: str
    tests_required_before_touching: tuple[str, ...]
    current_status: str = "not_modified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureInterface:
    """Future L4 interface definition (not implemented at runtime)."""

    name: str
    purpose: str
    required_inputs: tuple[str, ...]
    required_outputs: tuple[str, ...]
    failure_behavior: str
    trace_fields: tuple[str, ...]
    safety_gates: tuple[str, ...]
    can_affect_token_commit_in_future: bool
    affects_token_commit_now: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureFlagPlan:
    """Future CLI opt-in flag plan (design only)."""

    flag_name: str
    default_enabled: bool
    required_warnings: tuple[str, ...]
    flag_implemented: bool = False
    notes: str = "Flag is planned only; not wired to CLI in Phase 20D."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureTracePlan:
    """Future trace fields for L4 dry-run and scaffold phases."""

    trace_fields: tuple[str, ...]
    schema_location: str
    opt_in_only: bool
    must_not_affect_default_trace: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureFallbackPlan:
    """Future runtime fallback behavior plan (not implemented)."""

    trigger_conditions: tuple[str, ...]
    required_behavior: str
    must_restore_baseline_path: bool
    must_not_depend_on_proposal_state: bool
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureRollbackPlan:
    """Future runtime rollback behavior plan (not implemented)."""

    trigger_conditions: tuple[str, ...]
    required_behavior: str
    must_preserve_generation_state: bool
    must_surface_mismatch: bool
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureTestGate:
    """Gate that must pass before or after a future implementation stage."""

    gate_id: str
    purpose: str
    pass_condition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4FutureImplementationStage:
    """Staged future L4 implementation plan."""

    stage_id: str
    description: str
    files_likely_touched: tuple[str, ...]
    allowed_behavior: tuple[str, ...]
    forbidden_behavior: tuple[str, ...]
    gates_required_before_starting: tuple[str, ...]
    gates_required_before_completing: tuple[str, ...]
    rollback_fallback_requirements: tuple[str, ...]
    claim_boundaries: tuple[str, ...]
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4IntegrationRisk:
    """Risk in future L4 integration."""

    risk_id: str
    description: str
    severity: str
    mitigation: str
    current_status: str
    must_pass_gate: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4IntegrationPlanDecision:
    """Integration plan review decision."""

    decision: str
    l4_runtime_commit_authorized: bool
    stage_1_noop_scaffold_authorized: bool
    allowed_next_phase: str
    forbidden_next_phases: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4IntegrationPlanReview:
    """Top-level L4 integration plan review aggregate."""

    review_id: str
    safety_level: str
    future_change_targets: tuple[L4FutureChangeTarget, ...]
    future_interfaces: tuple[L4FutureInterface, ...]
    future_flag_plan: L4FutureFlagPlan
    future_trace_plan: L4FutureTracePlan
    future_fallback_plan: L4FutureFallbackPlan
    future_rollback_plan: L4FutureRollbackPlan
    future_implementation_stages: tuple[L4FutureImplementationStage, ...]
    risk_register: tuple[L4IntegrationRisk, ...]
    decision: L4IntegrationPlanDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "safety_level": self.safety_level,
            "future_change_targets": [t.to_dict() for t in self.future_change_targets],
            "future_interfaces": [i.to_dict() for i in self.future_interfaces],
            "future_flag_plan": self.future_flag_plan.to_dict(),
            "future_trace_plan": self.future_trace_plan.to_dict(),
            "future_fallback_plan": self.future_fallback_plan.to_dict(),
            "future_rollback_plan": self.future_rollback_plan.to_dict(),
            "future_implementation_stages": [
                s.to_dict() for s in self.future_implementation_stages
            ],
            "risk_register": [r.to_dict() for r in self.risk_register],
            "decision": self.decision.to_dict(),
        }


def build_l4_future_change_targets() -> tuple[L4FutureChangeTarget, ...]:
    """List future L4 change targets without modifying runtime."""
    return (
        L4FutureChangeTarget(
            path="exactkv/runtime/exactkv_generator.py",
            why_future_changes_may_be_needed=(
                "Future opt-in L4 scaffold may add no-op flag plumbing and trace hooks "
                "behind experimental flag without changing default commit path."
            ),
            required_safety_constraint=(
                "Default generation path unchanged; no proposal commit without verifier."
            ),
            default_runtime_risk="Accidental default-path behavior change.",
            verifier_bypass_risk="Shortcut around full verifier for draft acceptance.",
            rollback_fallback_requirement=(
                "On mismatch or failure, rollback to verifier output; fallback to baseline."
            ),
            tests_required_before_touching=(
                "default_runtime_unchanged_gate",
                "baseline_token_parity_gate",
                "no_direct_proposal_commit_gate",
            ),
        ),
        L4FutureChangeTarget(
            path="verifier interface/path",
            why_future_changes_may_be_needed=(
                "L4 must route proposals through existing full verifier without bypass."
            ),
            required_safety_constraint="Verifier remains source of truth for acceptance.",
            default_runtime_risk="Verifier path change affecting default generation.",
            verifier_bypass_risk="Compressed draft skipping full verification.",
            rollback_fallback_requirement="Missing verifier evidence triggers fallback.",
            tests_required_before_touching=(
                "verifier_source_of_truth_gate",
                "no_verifier_bypass_gate",
            ),
        ),
        L4FutureChangeTarget(
            path="ExactKVResult / round trace structure",
            why_future_changes_may_be_needed=(
                "L4 dry-run and scaffold phases need proposal, acceptance, rollback trace fields."
            ),
            required_safety_constraint="L4 trace fields opt-in only; default schema unchanged.",
            default_runtime_risk="Default trace schema breakage.",
            verifier_bypass_risk="Incomplete traces hiding divergence.",
            rollback_fallback_requirement="Trace must record rollback and fallback events.",
            tests_required_before_touching=("trace_completeness_gate",),
        ),
        L4FutureChangeTarget(
            path="L3 proposal source policy",
            why_future_changes_may_be_needed=(
                "Promoted exactkv_round_log_draft_tokens may feed L4 proposals with provenance."
            ),
            required_safety_constraint="Proposals diagnostic until L4 commit stage; provenance required.",
            default_runtime_risk="L3 proposals affecting default commits.",
            verifier_bypass_risk="Committed or shadow tokens used as proposal source.",
            rollback_fallback_requirement="Proposal exception triggers fallback.",
            tests_required_before_touching=("no_direct_proposal_commit_gate",),
        ),
        L4FutureChangeTarget(
            path="report schema",
            why_future_changes_may_be_needed=(
                "L4 panels must record acceptance, rollback, exactkv_failures, gate outcomes."
            ),
            required_safety_constraint="No performance/memory/serving metrics in L4 reports.",
            default_runtime_risk="N/A for default runtime.",
            verifier_bypass_risk="Claims without gate evidence.",
            rollback_fallback_requirement="Reports must show fallback and rollback counts.",
            tests_required_before_touching=(
                "exactkv_failure_gate",
                "claim_boundary_gate",
            ),
        ),
        L4FutureChangeTarget(
            path="CLI experimental flag path",
            why_future_changes_may_be_needed=(
                f"Future {L4_OPT_IN_FLAG} opt-in entry for no-op scaffold."
            ),
            required_safety_constraint="Flag disabled by default; explicit experimental warnings.",
            default_runtime_risk="Implicit L4 enablement on default CLI.",
            verifier_bypass_risk="Flag enables commit without gates passing.",
            rollback_fallback_requirement="Opt-out must equal baseline behavior.",
            tests_required_before_touching=("explicit_opt_in_gate",),
        ),
        L4FutureChangeTarget(
            path="tests for L4 runtime scaffold",
            why_future_changes_may_be_needed=(
                "Stage 1–3 require unit, synthetic, and model parity tests before commit stage."
            ),
            required_safety_constraint="Test matrix from Phase 20B must pass before each stage.",
            default_runtime_risk="Insufficient regression coverage.",
            verifier_bypass_risk="Bypass paths untested.",
            rollback_fallback_requirement="Fallback and rollback synthetic tests required.",
            tests_required_before_touching=(
                "baseline_token_parity_gate",
                "fallback_restores_baseline_gate",
                "rollback_on_mismatch_gate",
            ),
        ),
        L4FutureChangeTarget(
            path="docs/claims audit",
            why_future_changes_may_be_needed=(
                "L4 phases must pass audit_public_claims before any public claim."
            ),
            required_safety_constraint="No forbidden positive claims in docs or reports.",
            default_runtime_risk="N/A.",
            verifier_bypass_risk="N/A.",
            rollback_fallback_requirement="N/A.",
            tests_required_before_touching=("claim_boundary_gate",),
        ),
    )


def build_l4_future_interfaces() -> tuple[L4FutureInterface, ...]:
    """Define future L4 interfaces (planning only)."""
    return (
        L4FutureInterface(
            name="L4DraftProposalProvider",
            purpose="Supply explicit draft token proposals with provenance metadata.",
            required_inputs=(
                "round_index",
                "proposal_source_id",
                "compressed_kv_state_ref",
            ),
            required_outputs=(
                "proposal_token_ids",
                "proposal_source",
                "proposal_provenance",
            ),
            failure_behavior="Raise or signal proposal_exception; trigger fallback.",
            trace_fields=(
                "l4_proposal_token_ids",
                "l4_proposal_source",
                "l4_proposal_provenance",
            ),
            safety_gates=("no_direct_proposal_commit_gate",),
            can_affect_token_commit_in_future=True,
        ),
        L4FutureInterface(
            name="L4FullVerifier",
            purpose="Full-KV verifier evaluation; source of truth for acceptance.",
            required_inputs=("proposal_token_ids", "full_kv_state", "round_context"),
            required_outputs=("verifier_token_ids", "verifier_evidence_present"),
            failure_behavior="Missing evidence triggers fallback; mismatch surfaced.",
            trace_fields=("l4_verifier_token_ids", "l4_verifier_evidence_present"),
            safety_gates=(
                "verifier_source_of_truth_gate",
                "no_verifier_bypass_gate",
            ),
            can_affect_token_commit_in_future=True,
        ),
        L4FutureInterface(
            name="L4AcceptanceDecision",
            purpose="Compute longest verified matching prefix for commit consideration.",
            required_inputs=("proposal_token_ids", "verifier_token_ids"),
            required_outputs=(
                "accepted_prefix",
                "rejected_suffix",
                "acceptance_decision_id",
            ),
            failure_behavior="Empty prefix on mismatch; no silent divergence.",
            trace_fields=(
                "l4_accepted_prefix",
                "l4_rejected_suffix",
                "l4_acceptance_decision_id",
            ),
            safety_gates=("no_direct_proposal_commit_gate",),
            can_affect_token_commit_in_future=True,
        ),
        L4FutureInterface(
            name="L4RollbackController",
            purpose="Restore verifier output on mismatch or safety failure.",
            required_inputs=("accepted_prefix", "verifier_token_ids", "mismatch_reason"),
            required_outputs=("rollback_applied", "corrected_tokens"),
            failure_behavior="Must not corrupt generation state; surface mismatch.",
            trace_fields=("l4_rollback_applied", "l4_rollback_reason"),
            safety_gates=("rollback_on_mismatch_gate",),
            can_affect_token_commit_in_future=True,
        ),
        L4FutureInterface(
            name="L4FallbackController",
            purpose="Restore baseline generation on exception or gate failure.",
            required_inputs=("failure_reason", "baseline_path_ref"),
            required_outputs=("fallback_triggered", "baseline_restored"),
            failure_behavior="Must not depend on compressed proposal state.",
            trace_fields=("l4_fallback_triggered", "l4_fallback_reason"),
            safety_gates=("fallback_restores_baseline_gate",),
            can_affect_token_commit_in_future=True,
        ),
        L4FutureInterface(
            name="L4TraceRecorder",
            purpose="Record L4 decisions in round traces without affecting commits in dry-run.",
            required_inputs=("round_trace", "l4_decision_bundle"),
            required_outputs=("trace_updated", "trace_complete"),
            failure_behavior="Incomplete trace fails trace_completeness_gate.",
            trace_fields=(
                "l4_trace_complete",
                "l4_decision_steps",
            ),
            safety_gates=("trace_completeness_gate",),
            can_affect_token_commit_in_future=False,
        ),
        L4FutureInterface(
            name="L4SafetyGateEvaluator",
            purpose="Evaluate L4 readiness gates before stage progression.",
            required_inputs=("gate_definitions", "panel_results"),
            required_outputs=("gates_passing", "gates_failing", "gate_evidence"),
            failure_behavior="Block stage progression on gate failure.",
            trace_fields=("l4_gate_results",),
            safety_gates=("claim_boundary_gate", "exactkv_failure_gate"),
            can_affect_token_commit_in_future=False,
        ),
    )


def build_l4_future_flag_plan() -> L4FutureFlagPlan:
    return L4FutureFlagPlan(
        flag_name=L4_OPT_IN_FLAG,
        default_enabled=False,
        required_warnings=(
            "experimental",
            "verifier-mediated only",
            "no performance claim",
            "no serving claim",
            "not VeriCache reproduction",
        ),
    )


def build_l4_future_trace_plan() -> L4FutureTracePlan:
    return L4FutureTracePlan(
        trace_fields=(
            "l4_proposal_token_ids",
            "l4_proposal_source",
            "l4_verifier_token_ids",
            "l4_accepted_prefix",
            "l4_rejected_suffix",
            "l4_rollback_applied",
            "l4_fallback_triggered",
            "l4_trace_complete",
            "l4_decision_steps",
        ),
        schema_location="ExactKVResult / round trace structure",
        opt_in_only=True,
        must_not_affect_default_trace=True,
        notes="Trace fields added only when L4 opt-in active; dry-run stages record without commit.",
    )


def build_l4_future_fallback_plan() -> L4FutureFallbackPlan:
    return L4FutureFallbackPlan(
        trigger_conditions=(
            "proposal_exception",
            "missing_verifier_evidence",
            "safety_gate_failure",
            "l4_internal_error",
        ),
        required_behavior=(
            "Restore baseline generation path; do not use compressed proposal state."
        ),
        must_restore_baseline_path=True,
        must_not_depend_on_proposal_state=True,
    )


def build_l4_future_rollback_plan() -> L4FutureRollbackPlan:
    return L4FutureRollbackPlan(
        trigger_conditions=(
            "verifier_mismatch",
            "partial_prefix_rejection",
            "hidden_divergence_detected",
        ),
        required_behavior=(
            "Correct to full verifier output; preserve generation state integrity."
        ),
        must_preserve_generation_state=True,
        must_surface_mismatch=True,
    )


def build_l4_future_implementation_stages() -> tuple[L4FutureImplementationStage, ...]:
    """Define staged future L4 implementation plan."""
    return (
        L4FutureImplementationStage(
            stage_id="stage_0_current_no_runtime",
            description="Current state: design spec, contract tests, integration plan only.",
            files_likely_touched=(
                "exactkv/safety/*",
                "docs/*",
                "tests/test_exp09*",
            ),
            allowed_behavior=(
                "Design specs",
                "Synthetic contract tests",
                "Integration plan reviews",
            ),
            forbidden_behavior=(
                "Runtime L4 wiring",
                "Token commit changes",
                "CLI flag implementation",
            ),
            gates_required_before_starting=(),
            gates_required_before_completing=("claim_boundary_gate",),
            rollback_fallback_requirements=(),
            claim_boundaries=("design_and_contract_claims_only",),
            blocked=False,
        ),
        L4FutureImplementationStage(
            stage_id="stage_1_noop_opt_in_scaffold",
            description=(
                "Future: opt-in flag and trace plumbing only; must not change token output."
            ),
            files_likely_touched=(
                "exactkv/runtime/exactkv_generator.py",
                "CLI flag path",
                "ExactKVResult trace schema",
            ),
            allowed_behavior=(
                "Parse experimental flag",
                "Record no-op L4 trace markers",
                "Baseline-identical token output",
            ),
            forbidden_behavior=(
                "Proposal commit",
                "Verifier bypass",
                "Default-runtime behavior change",
            ),
            gates_required_before_starting=(
                "default_runtime_unchanged_gate",
                "explicit_opt_in_gate",
            ),
            gates_required_before_completing=(
                "baseline_token_parity_gate",
                "trace_completeness_gate",
            ),
            rollback_fallback_requirements=("opt-out equals baseline",),
            claim_boundaries=("scaffold_only_no_commit_claims",),
            blocked=False,
        ),
        L4FutureImplementationStage(
            stage_id="stage_2_trace_only_l4_dry_run",
            description=(
                "Future: compute L4 contract decisions from existing traces; no commit effect."
            ),
            files_likely_touched=(
                "exactkv/safety/l4_contract_tests_no_runtime.py",
                "round trace readers",
                "report generators",
            ),
            allowed_behavior=(
                "Dry-run acceptance decisions in traces",
                "Compare proposal vs verifier in logs",
            ),
            forbidden_behavior=(
                "Commit effect",
                "Generator decision influence",
            ),
            gates_required_before_starting=(
                "no_direct_proposal_commit_gate",
                "verifier_source_of_truth_gate",
            ),
            gates_required_before_completing=("trace_completeness_gate",),
            rollback_fallback_requirements=("record rollback decisions in trace only",),
            claim_boundaries=("dry_run_no_commit_claims",),
            blocked=False,
        ),
        L4FutureImplementationStage(
            stage_id="stage_3_verifier_mediated_dry_run",
            description=(
                "Future: full verifier decision computed during generation; no commit effect."
            ),
            files_likely_touched=(
                "exactkv/runtime/exactkv_generator.py",
                "verifier path",
                "L4DraftProposalProvider",
            ),
            allowed_behavior=(
                "Live verifier comparison during generation",
                "Trace accepted/rejected prefixes",
            ),
            forbidden_behavior=(
                "Commit accepted prefix",
                "Change sampling or verifier decisions",
            ),
            gates_required_before_starting=(
                "no_verifier_bypass_gate",
                "rollback_on_mismatch_gate",
            ),
            gates_required_before_completing=(
                "baseline_token_parity_gate",
                "fallback_restores_baseline_gate",
            ),
            rollback_fallback_requirements=(
                "fallback on exception",
                "rollback trace on mismatch",
            ),
            claim_boundaries=("dry_run_verifier_mediated_claims",),
            blocked=False,
        ),
        L4FutureImplementationStage(
            stage_id="stage_4_runtime_commit_candidate",
            description=(
                "Future and blocked: runtime commit of verifier-matched prefix; "
                "only after stages 1–3 pass; full verifier remains source of truth."
            ),
            files_likely_touched=(
                "exactkv/runtime/exactkv_generator.py",
                "L4AcceptanceDecision",
                "L4RollbackController",
                "L4FallbackController",
            ),
            allowed_behavior=(),
            forbidden_behavior=(
                "Implementation without stages 1–3 gate passage",
                "Verifier bypass",
                "Direct proposal commit",
                "Default-runtime enablement",
            ),
            gates_required_before_starting=(
                "baseline_token_parity_gate",
                "exactkv_failure_gate",
                "no_direct_proposal_commit_gate",
                "rollback_on_mismatch_gate",
                "fallback_restores_baseline_gate",
            ),
            gates_required_before_completing=(
                "baseline_token_parity_gate",
                "exactkv_failure_gate",
            ),
            rollback_fallback_requirements=(
                "runtime rollback on mismatch",
                "runtime fallback on failure",
            ),
            claim_boundaries=("panel_scoped_parity_only",),
            blocked=True,
        ),
    )


def build_l4_risk_register() -> tuple[L4IntegrationRisk, ...]:
    """Build L4 integration risk register."""
    risks: list[tuple[str, str, str, str, str, str]] = [
        (
            "default_runtime_change",
            "L4 integration accidentally changes default generation path.",
            "critical",
            "Explicit opt-in only; default_runtime_unchanged_gate; parity tests.",
            "mitigated_by_design",
            "default_runtime_unchanged_gate",
        ),
        (
            "verifier_bypass",
            "Compressed draft accepted without full verifier evaluation.",
            "critical",
            "Verifier source-of-truth contract; no_verifier_bypass_gate.",
            "mitigated_by_design",
            "no_verifier_bypass_gate",
        ),
        (
            "direct_proposal_commit",
            "Draft proposals committed without verifier-mediated acceptance.",
            "critical",
            "no_direct_proposal_commit_gate; synthetic contract tests.",
            "mitigated_by_contract_tests",
            "no_direct_proposal_commit_gate",
        ),
        (
            "hidden_divergence",
            "Silent divergence between proposal and verifier output.",
            "critical",
            "Trace completeness; hidden divergence synthetic case.",
            "mitigated_by_contract_tests",
            "trace_completeness_gate",
        ),
        (
            "fallback_not_equivalent_to_baseline",
            "Fallback path diverges from baseline generation.",
            "high",
            "Fallback contract; synthetic fallback tests; parity panels.",
            "open",
            "fallback_restores_baseline_gate",
        ),
        (
            "rollback_corrupts_generation_state",
            "Rollback leaves KV or token state inconsistent.",
            "high",
            "Rollback contract; state preservation tests before stage 4.",
            "open",
            "rollback_on_mismatch_gate",
        ),
        (
            "trace_incomplete",
            "L4 decisions not fully traced.",
            "medium",
            "L4TraceRecorder; trace_completeness_gate.",
            "mitigated_by_contract_tests",
            "trace_completeness_gate",
        ),
        (
            "cli_flag_accidental_enable",
            "Experimental flag enabled by default or without warning.",
            "high",
            "explicit_opt_in_gate; required warnings in flag plan.",
            "mitigated_by_design",
            "explicit_opt_in_gate",
        ),
        (
            "test_matrix_too_small",
            "Insufficient test coverage before stage progression.",
            "medium",
            "L4 test matrix from Phase 20B; gate panels before each stage.",
            "open",
            "baseline_token_parity_gate",
        ),
        (
            "performance_memory_overclaim",
            "Performance or memory claims without measurement discipline.",
            "high",
            "claim_boundary_gate; audit_public_claims.py.",
            "mitigated_by_policy",
            "claim_boundary_gate",
        ),
        (
            "vericache_reproduction_overclaim",
            "Claims of VeriCache throughput or serving reproduction.",
            "high",
            "Explicit forbidden claims; claims audit.",
            "mitigated_by_policy",
            "claim_boundary_gate",
        ),
    ]
    return tuple(
        L4IntegrationRisk(
            risk_id=rid,
            description=desc,
            severity=sev,
            mitigation=mit,
            current_status=status,
            must_pass_gate=gate,
        )
        for rid, desc, sev, mit, status, gate in risks
    )


def evaluate_l4_integration_plan_decision(
    review: L4IntegrationPlanReview,
) -> L4IntegrationPlanDecision:
    """Evaluate integration plan readiness for stage 1 no-op scaffold design."""
    required_targets = {
        "exactkv/runtime/exactkv_generator.py",
        "verifier interface/path",
        "ExactKVResult / round trace structure",
    }
    present_targets = {t.path for t in review.future_change_targets}
    targets_ok = required_targets <= present_targets

    all_not_modified = all(
        t.current_status == "not_modified" for t in review.future_change_targets
    )

    interface_names = {i.name for i in review.future_interfaces}
    required_interfaces = {
        "L4DraftProposalProvider",
        "L4FullVerifier",
        "L4AcceptanceDecision",
        "L4RollbackController",
        "L4FallbackController",
    }
    interfaces_ok = required_interfaces <= interface_names

    none_affect_commit_now = all(
        not i.affects_token_commit_now for i in review.future_interfaces
    )

    flag_ok = (
        not review.future_flag_plan.default_enabled
        and not review.future_flag_plan.flag_implemented
    )

    stage_ids = {s.stage_id for s in review.future_implementation_stages}
    stages_ok = set(STAGE_IDS) <= stage_ids

    stage_1 = next(
        s for s in review.future_implementation_stages
        if s.stage_id == "stage_1_noop_opt_in_scaffold"
    )
    stage_1_noop = "no-op" in stage_1.description.lower() or any(
        "no-op" in b.lower() or "noop" in b.lower()
        for b in stage_1.allowed_behavior
    )

    stage_4 = next(
        s for s in review.future_implementation_stages
        if s.stage_id == "stage_4_runtime_commit_candidate"
    )
    stage_4_blocked = stage_4.blocked

    risk_ids = {r.risk_id for r in review.risk_register}
    risks_ok = set(REQUIRED_RISK_IDS) <= risk_ids

    ready = (
        targets_ok
        and all_not_modified
        and interfaces_ok
        and none_affect_commit_now
        and flag_ok
        and stages_ok
        and stage_1_noop
        and stage_4_blocked
        and risks_ok
    )

    if ready:
        return L4IntegrationPlanDecision(
            decision=DECISION_READY_STAGE_1,
            l4_runtime_commit_authorized=False,
            stage_1_noop_scaffold_authorized=True,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_20D,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_20D,
            reason=(
                "Integration plan defines change targets, interfaces, staged rollout, "
                "and risk register; stage 1 no-op scaffold design authorized; "
                "runtime commit remains blocked"
            ),
        )

    if not targets_ok or not interfaces_ok or not stages_ok:
        return L4IntegrationPlanDecision(
            decision=DECISION_BLOCKED_MISSING_PLAN,
            l4_runtime_commit_authorized=False,
            stage_1_noop_scaffold_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_20D,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_20D,
            reason="integration plan missing required targets, interfaces, or stages",
        )

    return L4IntegrationPlanDecision(
        decision=DECISION_NOT_READY_STAGE_1,
        l4_runtime_commit_authorized=False,
        stage_1_noop_scaffold_authorized=False,
        allowed_next_phase=RECOMMENDED_NEXT_PHASE_20D,
        forbidden_next_phases=FORBIDDEN_NEXT_PHASES_20D,
        reason="integration plan incomplete or safety preconditions not met",
    )


def build_l4_integration_plan_review() -> L4IntegrationPlanReview:
    """Build the complete L4 integration plan review."""
    targets = build_l4_future_change_targets()
    interfaces = build_l4_future_interfaces()
    flag_plan = build_l4_future_flag_plan()
    trace_plan = build_l4_future_trace_plan()
    fallback_plan = build_l4_future_fallback_plan()
    rollback_plan = build_l4_future_rollback_plan()
    stages = build_l4_future_implementation_stages()
    risks = build_l4_risk_register()

    partial = L4IntegrationPlanReview(
        review_id=EXPERIMENT_101_ID,
        safety_level=L4_SAFETY_LEVEL,
        future_change_targets=targets,
        future_interfaces=interfaces,
        future_flag_plan=flag_plan,
        future_trace_plan=trace_plan,
        future_fallback_plan=fallback_plan,
        future_rollback_plan=rollback_plan,
        future_implementation_stages=stages,
        risk_register=risks,
        decision=L4IntegrationPlanDecision(
            decision=DECISION_BLOCKED_MISSING_PLAN,
            l4_runtime_commit_authorized=False,
            stage_1_noop_scaffold_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_20D,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_20D,
            reason="pending evaluation",
        ),
    )
    decision = evaluate_l4_integration_plan_decision(partial)
    return L4IntegrationPlanReview(
        review_id=partial.review_id,
        safety_level=partial.safety_level,
        future_change_targets=partial.future_change_targets,
        future_interfaces=partial.future_interfaces,
        future_flag_plan=partial.future_flag_plan,
        future_trace_plan=partial.future_trace_plan,
        future_fallback_plan=partial.future_fallback_plan,
        future_rollback_plan=partial.future_rollback_plan,
        future_implementation_stages=partial.future_implementation_stages,
        risk_register=partial.risk_register,
        decision=decision,
    )


def _remaining_implementation_blockers() -> list[dict[str, Any]]:
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
    remaining: list[dict[str, Any]] = []
    for text in L4_IMPLEMENTATION_BLOCKERS:
        bid = mapping.get(text, text.replace(" ", "_").lower()[:40])
        if bid not in PLAN_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("stage_1_noop_scaffold", "stage 1 no-op opt-in scaffold not implemented"),
        ("stage_2_trace_dry_run", "stage 2 trace-only L4 dry-run not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp101_l4_integration_plan_review() -> dict[str, Any]:
    """Run Experiment 101 L4 integration plan review (no runtime changes)."""
    review = build_l4_integration_plan_review()
    decision = review.decision

    status = (
        "plan_review_complete"
        if decision.decision == DECISION_READY_STAGE_1
        else "plan_review_incomplete"
    )

    return {
        "experiment_id": EXPERIMENT_101_ID,
        "status": status,
        "phase": PHASE_20D,
        "safety_level": L4_SAFETY_LEVEL,
        "plan_objects": {
            "L4IntegrationPlanReview": review.to_dict(),
            "L4IntegrationPlanDecision": decision.to_dict(),
        },
        "future_change_targets": [t.to_dict() for t in review.future_change_targets],
        "future_interfaces": [i.to_dict() for i in review.future_interfaces],
        "future_flag_plan": review.future_flag_plan.to_dict(),
        "future_trace_plan": review.future_trace_plan.to_dict(),
        "future_fallback_plan": review.future_fallback_plan.to_dict(),
        "future_rollback_plan": review.future_rollback_plan.to_dict(),
        "future_implementation_stages": [
            s.to_dict() for s in review.future_implementation_stages
        ],
        "risk_register": [r.to_dict() for r in review.risk_register],
        "integration_plan_decision": decision.to_dict(),
        "allowed_next_phase": decision.allowed_next_phase,
        "forbidden_next_phases": list(decision.forbidden_next_phases),
        "l4_runtime_commit_authorized": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "runtime_generation_path_modified": False,
        "cli_flag_implemented": False,
        "model_experiments_run": False,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 integration plan review only; not runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "CLI flag planned but not implemented.",
            "Stage 4 runtime commit remains blocked.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }


def validate_exp101_report(report: dict[str, Any]) -> list[str]:
    """Validate Experiment 101 report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "plan_objects",
        "future_change_targets",
        "future_interfaces",
        "future_flag_plan",
        "future_trace_plan",
        "future_fallback_plan",
        "future_rollback_plan",
        "future_implementation_stages",
        "risk_register",
        "integration_plan_decision",
        "allowed_next_phase",
        "forbidden_next_phases",
        "l4_runtime_commit_authorized",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "runtime_generation_path_modified",
        "cli_flag_implemented",
        "model_experiments_run",
        "implementation_blockers_remaining",
        "claim_boundaries",
        "no_performance_claims_note",
        "limitations",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_101_ID:
        errors.append("experiment_id mismatch")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_20D:
        errors.append("allowed_next_phase must be phase21a_l4_noop_opt_in_scaffold")

    if report.get("l4_runtime_commit_authorized") is not False:
        errors.append("l4_runtime_commit_authorized must be false")

    for flag in (
        "exactkv_generator_modified",
        "default_runtime_changed",
        "runtime_generation_path_modified",
        "cli_flag_implemented",
        "model_experiments_run",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    decision = report.get("integration_plan_decision") or {}
    if decision.get("decision") not in INTEGRATION_DECISIONS:
        errors.append("invalid integration_plan_decision.decision")

    if decision.get("decision") != DECISION_READY_STAGE_1:
        errors.append("expected decision ready_for_stage_1_noop_opt_in_scaffold_design")

    target_paths = {t["path"] for t in report.get("future_change_targets") or []}
    if "exactkv/runtime/exactkv_generator.py" not in target_paths:
        errors.append("missing ExactKVGenerator change target")

    for target in report.get("future_change_targets") or []:
        if target.get("current_status") != "not_modified":
            errors.append(f"target {target.get('path')} must be not_modified")

    stage_ids = {s["stage_id"] for s in report.get("future_implementation_stages") or []}
    if not set(STAGE_IDS) <= stage_ids:
        errors.append("missing required implementation stages")

    stage_4 = next(
        (s for s in report.get("future_implementation_stages") or []
         if s.get("stage_id") == "stage_4_runtime_commit_candidate"),
        None,
    )
    if stage_4 is None or not stage_4.get("blocked"):
        errors.append("stage_4_runtime_commit_candidate must be blocked")

    flag_plan = report.get("future_flag_plan") or {}
    if flag_plan.get("default_enabled") is not False:
        errors.append("future_flag_plan.default_enabled must be false")

    risk_ids = {r["risk_id"] for r in report.get("risk_register") or []}
    if not set(REQUIRED_RISK_IDS) <= risk_ids:
        errors.append("missing required risks in risk_register")

    forbidden = set(FORBIDDEN_NEXT_PHASES_20D)
    for fp in report.get("forbidden_next_phases") or []:
        if fp not in forbidden:
            errors.append(f"unknown forbidden_next_phase: {fp}")

    required_forbidden = {
        "l4_runtime_commit_implementation",
        "cuda_backend",
        "vllm_integration",
        "lmcache_integration",
        "performance_benchmark",
        "memory_benchmark",
    }
    if not required_forbidden <= set(report.get("forbidden_next_phases") or []):
        errors.append("forbidden_next_phases missing required entries")

    return errors
