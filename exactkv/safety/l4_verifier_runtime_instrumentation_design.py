"""L4 verifier evidence runtime instrumentation design (Phase 21J / Exp 111).

Architecture design specification only — must not implement runtime hooks or generation changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_evidence_trace_schema_design import TRACE_SCHEMA_VERSION
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    L4_OPT_IN_FLAG,
    build_l4_claim_boundaries,
    build_l4_integration_points,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_111_ID = "exp111_l4_verifier_runtime_instrumentation_design"
DEFAULT_EXP111_REPORT = Path(
    "reports/experiment_111_l4_verifier_runtime_instrumentation_design.json",
)
PHASE_21J = "21J"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "verifier_runtime_instrumentation_design"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21J = "phase21k_l4_stage3_verifier_mediated_dry_run_design"
FORBIDDEN_NEXT_PHASES_21J: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_instrumentation_implementation",
    "l4_runtime_verifier_in_loop_execution",
    "l4_default_runtime_modification",
)

DESIGN_OUTCOME_COMPLETE = "runtime_instrumentation_design_complete"
DESIGN_OUTCOME_INCOMPLETE = "runtime_instrumentation_design_incomplete"
DESIGN_OUTCOME_BLOCKED = "runtime_instrumentation_design_blocked"

DESIGN_OUTCOMES: tuple[str, ...] = (
    DESIGN_OUTCOME_COMPLETE,
    DESIGN_OUTCOME_INCOMPLETE,
    DESIGN_OUTCOME_BLOCKED,
)

RUNTIME_HOOK_IDS: tuple[str, ...] = (
    "hook_pre_generation_session",
    "hook_round_proposal_intercept",
    "hook_per_token_generation",
    "hook_verifier_comparison",
    "hook_trace_record_emit",
    "hook_post_generation_finalize",
    "hook_rollback_decision",
)

INSTRUMENTATION_POINT_IDS: tuple[str, ...] = (
    "pre_generation",
    "per_token",
    "post_generation",
    "verifier_comparison",
)

DATA_FLOW_STEP_IDS: tuple[str, ...] = (
    "proposal_capture",
    "trace_record_write",
    "verifier_evidence_capture",
    "comparison_decision",
    "rollback_concept",
)

INTEGRATION_POINT_IDS: tuple[str, ...] = (
    "exactkv_generator",
    "round_log_system",
    "verifier_evidence_schema",
    "trace_only_dry_run",
)

FAILURE_MODE_IDS: tuple[str, ...] = (
    "missing_verifier_evidence",
    "proposal_verifier_alias",
    "verifier_bypass_attempt",
    "direct_proposal_commit",
    "instrumentation_enabled_without_opt_in",
    "trace_schema_version_mismatch",
    "verifier_exception",
    "rollback_not_triggered_on_mismatch",
)

INCORRECT_ENABLEMENT_SCENARIO_IDS: tuple[str, ...] = (
    "enabled_on_default_runtime",
    "verifier_treated_as_non_authoritative",
    "proposal_committed_without_verifier",
    "trace_only_decision_wired_to_commit",
    "instrumentation_without_fallback",
    "per_token_hook_mutates_generator_state",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

ARCHITECTURE_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ExactKVGenerator (UNCHANGED default path)               │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│  │ pre-gen hook │───▶│ per-token hook  │───▶│ post-generation hook     │  │
│  │ (conceptual) │    │ (NOT IMPLEMENTED)│   │ (conceptual)             │  │
│  └──────┬───────┘    └────────┬────────┘    └────────────┬─────────────┘  │
│         │                     │                          │                │
│         │         ┌───────────▼───────────┐              │                │
│         │         │ proposal intercept  │              │                │
│         │         │ (round-log source)  │              │                │
│         │         └───────────┬─────────┘              │                │
│         │                     │                          │                │
│         │         ┌───────────▼───────────┐              │                │
│         │         │ verifier comparison   │◀─────────────┘                │
│         │         │ hook (conceptual)   │                                 │
│         │         └───────────┬───────────┘                                 │
│         │                     │                                             │
│         │    proposal ──▶ trace record ──▶ verifier evidence ──▶ decision   │
│         │                     │              (schema v1)         │          │
│         │                     │                                  │          │
│         │                     └──────────▶ rollback (concept) ◀──┘          │
│         │                              (no execution in Phase 21J)          │
└─────────┴───────────────────────────────────────────────────────────────────┘
         ▲
         │ opt-in gate only: {opt_in_flag}
         │ default runtime bypasses all hooks
""".format(opt_in_flag=L4_OPT_IN_FLAG).strip()

DESIGN_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
    {
        "explicit_l4_design_spec",
        "verifier_mediated_acceptance_contract",
        "rollback_behavior_defined",
        "l4_test_matrix_defined",
        "l4_opt_in_flag_designed",
        "l4_synthetic_contract_tests_no_runtime",
        "exactkv_generator_integration_plan",
        "stage_1_noop_scaffold_design",
        "stage_1_noop_scaffold_panel_validation",
        "stage_2_trace_only_dry_run_design",
        "stage_2_trace_dry_run_scaffold",
        "stage_2_trace_panel_validation",
        "verifier_evidence_trace_schema_design",
        "verifier_evidence_trace_schema_scaffold",
        "schema_example_dry_run_validation",
        "trace_schema_stress_adversarial_panel",
        "runtime_instrumentation_design",
    },
)


@dataclass(frozen=True)
class L4RuntimeHookDefinition:
    """Conceptual runtime hook — design only, not implemented."""

    hook_id: str
    attach_location: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    must_not_modify_baseline: bool
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4InstrumentationPoint:
    """Instrumentation point in the future generation loop (design only)."""

    point_id: str
    phase: str
    description: str
    captures: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4DataFlowStep:
    """One step in the conceptual proposal→trace→verifier→decision→rollback flow."""

    step_id: str
    description: str
    input_artifacts: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    executed_at_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SafetyBoundary:
    """Safety boundary for future runtime instrumentation."""

    boundary_id: str
    protected_component: str
    must_not_change_by_default: str
    allowed_when_opt_in: str
    fallback_behavior: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4InstrumentationIntegrationPoint:
    """Conceptual integration with an existing ExactKV subsystem."""

    integration_id: str
    subsystem: str
    touchpoint: str
    design_only_contract: str
    changes_allowed_in_phase_21j: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4InstrumentationFailureMode:
    """Failure mode for future runtime instrumentation."""

    failure_id: str
    description: str
    detection_signal: str
    required_response: str
    blocks_commit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4IncorrectEnablementScenario:
    """What happens if instrumentation is enabled or wired incorrectly."""

    scenario_id: str
    misconfiguration: str
    expected_harm: str
    required_mitigation: str
    prevented_by_design_gate: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4RuntimeInstrumentationDesignDecision:
    """Design review outcome for Phase 21J."""

    outcome: str
    runtime_instrumentation_implementation_authorized: bool
    runtime_commit_authorized: bool
    stage_3_dry_run_design_authorized: bool
    allowed_next_phase: str
    forbidden_next_phases: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4RuntimeInstrumentationDesign:
    """Top-level L4 runtime instrumentation design aggregate."""

    design_id: str
    safety_level: str
    stage: str
    mode: str
    schema_version: str
    architecture_diagram: str
    runtime_hooks: tuple[L4RuntimeHookDefinition, ...]
    instrumentation_points: tuple[L4InstrumentationPoint, ...]
    data_flow_steps: tuple[L4DataFlowStep, ...]
    safety_boundaries: tuple[L4SafetyBoundary, ...]
    integration_points: tuple[L4InstrumentationIntegrationPoint, ...]
    failure_modes: tuple[L4InstrumentationFailureMode, ...]
    incorrect_enablement_scenarios: tuple[L4IncorrectEnablementScenario, ...]
    decision: L4RuntimeInstrumentationDesignDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "safety_level": self.safety_level,
            "stage": self.stage,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "architecture_diagram": self.architecture_diagram,
            "runtime_hooks": [h.to_dict() for h in self.runtime_hooks],
            "instrumentation_points": [p.to_dict() for p in self.instrumentation_points],
            "data_flow_steps": [s.to_dict() for s in self.data_flow_steps],
            "safety_boundaries": [b.to_dict() for b in self.safety_boundaries],
            "integration_points": [i.to_dict() for i in self.integration_points],
            "failure_modes": [f.to_dict() for f in self.failure_modes],
            "incorrect_enablement_scenarios": [
                s.to_dict() for s in self.incorrect_enablement_scenarios
            ],
            "decision": self.decision.to_dict(),
        }


def build_l4_runtime_hook_definitions() -> tuple[L4RuntimeHookDefinition, ...]:
    """Conceptual runtime hooks for future L4 instrumentation."""
    return (
        L4RuntimeHookDefinition(
            hook_id="hook_pre_generation_session",
            attach_location="ExactKVGenerator.generate() entry (after opt-in gate)",
            purpose="Initialize instrumentation session; verify opt-in and safety gates.",
            inputs=("opt_in_flag", "session_config"),
            outputs=("instrumentation_session_id", "baseline_snapshot_ref"),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_round_proposal_intercept",
            attach_location="Compressed draft / round-log proposal emission point",
            purpose="Intercept draft proposal token IDs before any commit decision.",
            inputs=("round_index", "proposal_source", "proposal_token_ids"),
            outputs=("proposal_trace_ref",),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_per_token_generation",
            attach_location="Inner generation loop per accepted token step",
            purpose="Observe per-token generation state for trace completeness.",
            inputs=("round_index", "token_index", "generator_state_ref"),
            outputs=("per_token_observation_ref",),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_verifier_comparison",
            attach_location="After full-KV verifier output is available for the round",
            purpose="Compare proposal tokens against verifier evidence; emit comparison metadata.",
            inputs=(
                "proposal_token_ids",
                "verifier_evidence_token_ids",
                "verifier_evidence_is_full_kv",
            ),
            outputs=(
                "verifier_matching_prefix_token_ids",
                "verifier_rejected_suffix_token_ids",
                "verifier_first_mismatch_index",
                "verifier_decision_status",
            ),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_trace_record_emit",
            attach_location="Round trace write path (diagnostic channel)",
            purpose="Emit verifier evidence trace record conforming to schema v1.",
            inputs=("round_metadata", "verifier_evidence_fields"),
            outputs=("trace_record_ref",),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_post_generation_finalize",
            attach_location="ExactKVGenerator.generate() exit",
            purpose="Finalize instrumentation session; attach summary diagnostics.",
            inputs=("instrumentation_session_id", "round_trace_refs"),
            outputs=("instrumentation_summary_ref",),
            must_not_modify_baseline=True,
        ),
        L4RuntimeHookDefinition(
            hook_id="hook_rollback_decision",
            attach_location="After verifier comparison when mismatch or block detected",
            purpose="Conceptual rollback to baseline-safe state; not executed in Phase 21J.",
            inputs=("verifier_decision_status", "baseline_snapshot_ref"),
            outputs=("rollback_status", "rollback_reason"),
            must_not_modify_baseline=True,
        ),
    )


def build_l4_instrumentation_points() -> tuple[L4InstrumentationPoint, ...]:
    """Instrumentation points in the future generation loop."""
    return (
        L4InstrumentationPoint(
            point_id="pre_generation",
            phase="before first token",
            description=(
                "Session setup, opt-in verification, baseline snapshot reference. "
                "No token generation side effects."
            ),
            captures=("opt_in_state", "session_id", "baseline_config_hash"),
            forbidden_actions=(
                "modify_generator_weights",
                "enable_commit_path",
                "skip_baseline_warmup",
            ),
        ),
        L4InstrumentationPoint(
            point_id="per_token",
            phase="each generation step",
            description=(
                "Per-token observation hook — design only; MUST NOT be implemented in Phase 21J. "
                "Would observe generator state without influencing token selection."
            ),
            captures=("round_index", "token_index", "observation_timestamp"),
            forbidden_actions=(
                "override_token_selection",
                "commit_proposal_without_verifier",
                "mutate_kv_cache_for_commit",
            ),
        ),
        L4InstrumentationPoint(
            point_id="verifier_comparison",
            phase="after verifier output available",
            description=(
                "Compare intercepted proposal against explicit verifier evidence fields. "
                "Verifier is source of truth when L4 is enabled."
            ),
            captures=(
                "proposal_token_ids",
                "verifier_evidence_token_ids",
                "verifier_first_mismatch_index",
                "verifier_decision_status",
            ),
            forbidden_actions=(
                "treat_proposal_as_verifier",
                "fabricate_verifier_on_missing_evidence",
                "bypass_full_kv_verifier",
            ),
        ),
        L4InstrumentationPoint(
            point_id="post_generation",
            phase="after generation completes",
            description=(
                "Finalize trace records, emit instrumentation summary, release session refs."
            ),
            captures=("round_count", "trace_complete_flags", "block_reasons"),
            forbidden_actions=(
                "retroactively_modify_committed_tokens",
                "hide_block_reasons",
                "expose_trace_to_generator_feedback_loop",
            ),
        ),
    )


def build_l4_data_flow_steps() -> tuple[L4DataFlowStep, ...]:
    """Conceptual data flow: proposal → trace → verifier → decision → rollback."""
    return (
        L4DataFlowStep(
            step_id="proposal_capture",
            description=(
                "Draft proposal tokens intercepted from explicit round-log proposal source "
                f"({PROPOSAL_SOURCE_ROUND_LOG}). Never from committed or verifier tokens."
            ),
            input_artifacts=("round_log_draft_proposal",),
            output_artifacts=("proposal_token_ids", "proposal_source"),
        ),
        L4DataFlowStep(
            step_id="trace_record_write",
            description=(
                "Write per-round trace record with proposal fields and metadata "
                f"(schema {TRACE_SCHEMA_VERSION}). Diagnostic channel only."
            ),
            input_artifacts=("proposal_token_ids", "round_index", "round_metadata"),
            output_artifacts=("trace_record",),
        ),
        L4DataFlowStep(
            step_id="verifier_evidence_capture",
            description=(
                "Capture explicit full-KV verifier output as verifier evidence fields. "
                "Missing evidence blocks; never fabricated."
            ),
            input_artifacts=("full_kv_verifier_output",),
            output_artifacts=(
                "verifier_evidence_token_ids",
                "verifier_evidence_is_full_kv",
                "verifier_evidence_available",
            ),
        ),
        L4DataFlowStep(
            step_id="comparison_decision",
            description=(
                "Compute longest matching prefix and decision status from proposal vs verifier. "
                "Non-authoritative until L4 commit stage; diagnostic in trace-only phases."
            ),
            input_artifacts=("proposal_token_ids", "verifier_evidence_token_ids"),
            output_artifacts=(
                "verifier_matching_prefix_token_ids",
                "verifier_decision_status",
                "verifier_first_mismatch_index",
            ),
        ),
        L4DataFlowStep(
            step_id="rollback_concept",
            description=(
                "On mismatch, missing evidence, or safety gate failure: conceptual rollback "
                "restores baseline-safe generation path. Not executed in Phase 21J."
            ),
            input_artifacts=("verifier_decision_status", "baseline_snapshot_ref"),
            output_artifacts=("rollback_status", "rollback_reason"),
        ),
    )


def build_l4_safety_boundaries() -> tuple[L4SafetyBoundary, ...]:
    """Safety boundary matrix for future runtime instrumentation."""
    return (
        L4SafetyBoundary(
            boundary_id="default_runtime_unchanged",
            protected_component="ExactKVGenerator default code path",
            must_not_change_by_default="Token selection, KV cache commits, round-log semantics",
            allowed_when_opt_in="Read-only observation hooks only after explicit opt-in",
            fallback_behavior="Opt-out or gate failure restores identical baseline behavior",
        ),
        L4SafetyBoundary(
            boundary_id="verifier_non_authoritative_until_l4",
            protected_component="Verifier evidence fields",
            must_not_change_by_default=(
                "Verifier evidence is diagnostic-only; cannot authorize commits"
            ),
            allowed_when_opt_in=(
                "Verifier becomes acceptance authority only after future L4 commit stage"
            ),
            fallback_behavior=(
                "Missing verifier evidence blocks decision; never fabricates tokens"
            ),
        ),
        L4SafetyBoundary(
            boundary_id="proposal_verifier_separation",
            protected_component="Proposal vs verifier trace fields",
            must_not_change_by_default="Fields must remain distinct; no aliasing",
            allowed_when_opt_in="Separate capture hooks for proposal and verifier sources",
            fallback_behavior="Alias or forgery detected → invalid trace / block",
        ),
        L4SafetyBoundary(
            boundary_id="no_direct_proposal_commit",
            protected_component="Token commit path",
            must_not_change_by_default="Proposal tokens never commit without verifier check",
            allowed_when_opt_in="Future L4 stage may commit verified prefix only",
            fallback_behavior="Direct proposal commit attempt → rollback + block",
        ),
        L4SafetyBoundary(
            boundary_id="trace_only_no_commit_wiring",
            protected_component="Trace-only dry-run evaluator",
            must_not_change_by_default="Dry-run decisions never wired to commit",
            allowed_when_opt_in="N/A in trace-only stages",
            fallback_behavior="Any commit wiring attempt fails safety gate",
        ),
        L4SafetyBoundary(
            boundary_id="opt_in_gate_required",
            protected_component=f"CLI flag {L4_OPT_IN_FLAG}",
            must_not_change_by_default="Flag absent → zero instrumentation hooks active",
            allowed_when_opt_in="Hooks may attach in read-only/observe mode",
            fallback_behavior="Flag parse error or missing gate → baseline only",
        ),
    )


def build_l4_instrumentation_integration_points() -> tuple[L4InstrumentationIntegrationPoint, ...]:
    """Conceptual integration points with existing ExactKV subsystems."""
    existing = build_l4_integration_points()
    generator_touch = next(
        (p for p in existing if "exactkv_generator" in p.path.lower()),
        None,
    )
    generator_contract = (
        generator_touch.why_future_changes_may_be_needed
        if generator_touch
        else "Future hook attachment at generate() boundary"
    )
    return (
        L4InstrumentationIntegrationPoint(
            integration_id="exactkv_generator",
            subsystem="exactkv/runtime/exactkv_generator.py",
            touchpoint="ExactKVGenerator.generate() entry/exit and inner loop observe points",
            design_only_contract=(
                f"{generator_contract} Phase 21J defines hooks only; "
                "ExactKVGenerator remains unmodified."
            ),
        ),
        L4InstrumentationIntegrationPoint(
            integration_id="round_log_system",
            subsystem="exactkv_round_log_draft_tokens / guarded draft shadow",
            touchpoint="Round-log proposal emission and draft token fields",
            design_only_contract=(
                f"Proposal source must be explicit ({PROPOSAL_SOURCE_ROUND_LOG}). "
                "Instrumentation reads proposal fields; does not rewrite round logs."
            ),
        ),
        L4InstrumentationIntegrationPoint(
            integration_id="verifier_evidence_schema",
            subsystem="l4_verifier_evidence_trace_schema_design / scaffold",
            touchpoint=f"Trace records conforming to {TRACE_SCHEMA_VERSION}",
            design_only_contract=(
                "All emitted trace records must pass schema validation and adversarial panel rules."
            ),
        ),
        L4InstrumentationIntegrationPoint(
            integration_id="trace_only_dry_run",
            subsystem="l4_trace_only_dry_run_scaffold",
            touchpoint="evaluate_l4_trace_only_input() diagnostic evaluator",
            design_only_contract=(
                "Runtime instrumentation feeds trace records into dry-run evaluator offline; "
                "evaluator decisions remain non-authoritative."
            ),
        ),
    )


def build_l4_instrumentation_failure_modes() -> tuple[L4InstrumentationFailureMode, ...]:
    """Failure mode analysis for future runtime instrumentation."""
    return (
        L4InstrumentationFailureMode(
            failure_id="missing_verifier_evidence",
            description="Verifier evidence fields absent or verifier_evidence_available=false",
            detection_signal="blocked_missing_verifier_evidence in trace validation",
            required_response="Block decision; emit block_reason; no fabricated tokens",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="proposal_verifier_alias",
            description="Proposal and verifier fields share object identity or mislabeled source",
            detection_signal="detected_poisoning in adversarial panel / schema alias rule",
            required_response="Reject trace as invalid; block commit path",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="verifier_bypass_attempt",
            description="Proposal accepted without full-KV verifier comparison",
            detection_signal="verifier_comparison hook skipped or verifier_evidence_is_full_kv=false",
            required_response="Rollback to baseline; mark safety gate failure",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="direct_proposal_commit",
            description="Draft proposal tokens committed without verifier mediation",
            detection_signal="failed_direct_commit_attempt in dry-run status",
            required_response="Rollback; surface failed_direct_commit_attempt",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="instrumentation_enabled_without_opt_in",
            description="Hooks active on default runtime without explicit experimental flag",
            detection_signal="opt_in_gate_required boundary violation",
            required_response="Disable hooks immediately; restore baseline path",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="trace_schema_version_mismatch",
            description=f"Trace record schema_version != {TRACE_SCHEMA_VERSION}",
            detection_signal="invalid_trace in schema validation",
            required_response="Reject record; do not feed dry-run evaluator",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="verifier_exception",
            description="Verifier computation raises or returns incomplete evidence",
            detection_signal="verifier_exception field populated",
            required_response="Block decision; rollback per L4RollbackContract",
            blocks_commit=True,
        ),
        L4InstrumentationFailureMode(
            failure_id="rollback_not_triggered_on_mismatch",
            description="Verifier mismatch detected but rollback hook not invoked",
            detection_signal="verifier_decision_status mismatch without rollback_status",
            required_response="Treat as safety gate failure; force baseline fallback",
            blocks_commit=True,
        ),
    )


def build_l4_incorrect_enablement_scenarios() -> tuple[L4IncorrectEnablementScenario, ...]:
    """What happens if instrumentation is enabled or wired incorrectly."""
    return (
        L4IncorrectEnablementScenario(
            scenario_id="enabled_on_default_runtime",
            misconfiguration="Instrumentation hooks active without opt-in flag",
            expected_harm="Silent behavior change on production default path",
            required_mitigation="Hard gate: hooks inert unless opt-in; parity panel required",
            prevented_by_design_gate="opt_in_gate_required",
        ),
        L4IncorrectEnablementScenario(
            scenario_id="verifier_treated_as_non_authoritative",
            misconfiguration="Proposal tokens commit when verifier disagrees",
            expected_harm="Unverified draft tokens enter KV cache",
            required_mitigation="Verifier comparison hook must block commit on mismatch",
            prevented_by_design_gate="verifier_non_authoritative_until_l4",
        ),
        L4IncorrectEnablementScenario(
            scenario_id="proposal_committed_without_verifier",
            misconfiguration="Proposal intercept wired directly to commit path",
            expected_harm="Bypasses full-KV verification entirely",
            required_mitigation="no_direct_proposal_commit boundary; contract tests",
            prevented_by_design_gate="no_direct_proposal_commit",
        ),
        L4IncorrectEnablementScenario(
            scenario_id="trace_only_decision_wired_to_commit",
            misconfiguration="Dry-run evaluator output feeds token commit logic",
            expected_harm="Diagnostic trace influences generation incorrectly",
            required_mitigation="Separate diagnostic channel; no commit wiring",
            prevented_by_design_gate="trace_only_no_commit_wiring",
        ),
        L4IncorrectEnablementScenario(
            scenario_id="instrumentation_without_fallback",
            misconfiguration="Hook failure crashes generation instead of falling back",
            expected_harm="Availability regression on experimental path",
            required_mitigation="Try/fallback wrapper restores baseline on any hook error",
            prevented_by_design_gate="default_runtime_unchanged",
        ),
        L4IncorrectEnablementScenario(
            scenario_id="per_token_hook_mutates_generator_state",
            misconfiguration="Per-token hook overrides token selection or KV state",
            expected_harm="Non-deterministic divergence from baseline",
            required_mitigation="Per-token hook observe-only; forbidden_actions enforced",
            prevented_by_design_gate="default_runtime_unchanged",
        ),
    )


def build_safety_boundary_matrix(
    boundaries: Sequence[L4SafetyBoundary],
) -> list[dict[str, str]]:
    """Tabular safety boundary matrix for reports."""
    return [
        {
            "boundary_id": b.boundary_id,
            "protected_component": b.protected_component,
            "must_not_change_by_default": b.must_not_change_by_default,
            "allowed_when_opt_in": b.allowed_when_opt_in,
            "fallback_behavior": b.fallback_behavior,
        }
        for b in boundaries
    ]


def evaluate_l4_runtime_instrumentation_design_decision(
    design: L4RuntimeInstrumentationDesign,
) -> L4RuntimeInstrumentationDesignDecision:
    """Evaluate whether Phase 21J design is complete."""
    hook_ids = {h.hook_id for h in design.runtime_hooks}
    point_ids = {p.point_id for p in design.instrumentation_points}
    flow_ids = {s.step_id for s in design.data_flow_steps}
    integration_ids = {i.integration_id for i in design.integration_points}
    failure_ids = {f.failure_id for f in design.failure_modes}
    scenario_ids = {s.scenario_id for s in design.incorrect_enablement_scenarios}

    missing: list[str] = []
    if not set(RUNTIME_HOOK_IDS) <= hook_ids:
        missing.append("incomplete runtime_hooks")
    if not set(INSTRUMENTATION_POINT_IDS) <= point_ids:
        missing.append("incomplete instrumentation_points")
    if not set(DATA_FLOW_STEP_IDS) <= flow_ids:
        missing.append("incomplete data_flow_steps")
    if not set(INTEGRATION_POINT_IDS) <= integration_ids:
        missing.append("incomplete integration_points")
    if not set(FAILURE_MODE_IDS) <= failure_ids:
        missing.append("incomplete failure_modes")
    if not set(INCORRECT_ENABLEMENT_SCENARIO_IDS) <= scenario_ids:
        missing.append("incomplete incorrect_enablement_scenarios")
    if not design.architecture_diagram.strip():
        missing.append("missing architecture_diagram")
    if any(h.implemented for h in design.runtime_hooks):
        missing.append("runtime hooks must not be marked implemented")
    if any(p.implemented for p in design.instrumentation_points):
        missing.append("instrumentation points must not be marked implemented")
    if any(s.executed_at_runtime for s in design.data_flow_steps):
        missing.append("data flow steps must not be marked executed_at_runtime")

    if missing:
        return L4RuntimeInstrumentationDesignDecision(
            outcome=DESIGN_OUTCOME_INCOMPLETE,
            runtime_instrumentation_implementation_authorized=False,
            runtime_commit_authorized=False,
            stage_3_dry_run_design_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21J,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21J,
            reason="; ".join(missing),
        )

    return L4RuntimeInstrumentationDesignDecision(
        outcome=DESIGN_OUTCOME_COMPLETE,
        runtime_instrumentation_implementation_authorized=False,
        runtime_commit_authorized=False,
        stage_3_dry_run_design_authorized=True,
        allowed_next_phase=RECOMMENDED_NEXT_PHASE_21J,
        forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21J,
        reason="Runtime instrumentation architecture design complete; implementation blocked.",
    )


def build_l4_runtime_instrumentation_design() -> L4RuntimeInstrumentationDesign:
    """Build the Phase 21J runtime instrumentation design aggregate."""
    hooks = build_l4_runtime_hook_definitions()
    points = build_l4_instrumentation_points()
    flow = build_l4_data_flow_steps()
    boundaries = build_l4_safety_boundaries()
    integrations = build_l4_instrumentation_integration_points()
    failures = build_l4_instrumentation_failure_modes()
    scenarios = build_l4_incorrect_enablement_scenarios()

    partial = L4RuntimeInstrumentationDesign(
        design_id=EXPERIMENT_111_ID,
        safety_level=L4_SAFETY_LEVEL,
        stage=STAGE,
        mode=MODE,
        schema_version=TRACE_SCHEMA_VERSION,
        architecture_diagram=ARCHITECTURE_DIAGRAM,
        runtime_hooks=hooks,
        instrumentation_points=points,
        data_flow_steps=flow,
        safety_boundaries=boundaries,
        integration_points=integrations,
        failure_modes=failures,
        incorrect_enablement_scenarios=scenarios,
        decision=L4RuntimeInstrumentationDesignDecision(
            outcome=DESIGN_OUTCOME_BLOCKED,
            runtime_instrumentation_implementation_authorized=False,
            runtime_commit_authorized=False,
            stage_3_dry_run_design_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21J,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21J,
            reason="pending evaluation",
        ),
    )
    decision = evaluate_l4_runtime_instrumentation_design_decision(partial)
    return L4RuntimeInstrumentationDesign(
        design_id=partial.design_id,
        safety_level=partial.safety_level,
        stage=partial.stage,
        mode=partial.mode,
        schema_version=partial.schema_version,
        architecture_diagram=partial.architecture_diagram,
        runtime_hooks=partial.runtime_hooks,
        instrumentation_points=partial.instrumentation_points,
        data_flow_steps=partial.data_flow_steps,
        safety_boundaries=partial.safety_boundaries,
        integration_points=partial.integration_points,
        failure_modes=partial.failure_modes,
        incorrect_enablement_scenarios=partial.incorrect_enablement_scenarios,
        decision=decision,
    )


def _remaining_implementation_blockers() -> list[dict[str, str]]:
    mapping = {
        "explicit L4 design spec missing": "explicit_l4_design_spec",
        "ExactKVGenerator integration plan missing": "exactkv_generator_integration_plan",
        "fallback path not yet implemented for L4": "l4_fallback_path",
        "no L4 baseline-vs-integrated parity panel": "l4_parity_panel",
        "no L4 exactkv_failures gate run": "l4_exactkv_failures_gate_run",
        "no active GPU memory measurement": "gpu_memory_measurement",
        "no performance benchmark": "performance_benchmark",
        "no serving integration": "serving_integration",
    }
    remaining: list[dict[str, str]] = []
    for text in L4_IMPLEMENTATION_BLOCKERS:
        bid = mapping.get(text, text.replace(" ", "_").lower()[:40])
        if bid not in DESIGN_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        (
            "runtime_verifier_instrumentation",
            "runtime verifier evidence instrumentation not implemented",
        ),
        (
            "runtime_instrumentation_implementation",
            "runtime hook implementation blocked until stage 3+",
        ),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp111_l4_verifier_runtime_instrumentation_design() -> dict[str, Any]:
    """Run Experiment 111 L4 runtime instrumentation design (no runtime changes)."""
    design = build_l4_runtime_instrumentation_design()
    decision = design.decision

    status = (
        "design_complete"
        if decision.outcome == DESIGN_OUTCOME_COMPLETE
        else "design_incomplete"
    )

    return {
        "experiment_id": EXPERIMENT_111_ID,
        "status": status,
        "phase": PHASE_21J,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": design.schema_version,
        "design_outcome": decision.outcome,
        "architecture_diagram": design.architecture_diagram,
        "design_objects": {
            "L4RuntimeInstrumentationDesign": design.to_dict(),
            "L4RuntimeInstrumentationDesignDecision": decision.to_dict(),
        },
        "runtime_hooks": [h.to_dict() for h in design.runtime_hooks],
        "instrumentation_points": [p.to_dict() for p in design.instrumentation_points],
        "data_flow_steps": [s.to_dict() for s in design.data_flow_steps],
        "data_flow_description": (
            "Conceptual flow: proposal_capture → trace_record_write → "
            "verifier_evidence_capture → comparison_decision → rollback_concept. "
            "None of these steps execute at runtime in Phase 21J."
        ),
        "safety_boundaries": [b.to_dict() for b in design.safety_boundaries],
        "safety_boundary_matrix": build_safety_boundary_matrix(design.safety_boundaries),
        "integration_points": [i.to_dict() for i in design.integration_points],
        "failure_modes": [f.to_dict() for f in design.failure_modes],
        "failure_mode_breakdown": {f.failure_id: f.description for f in design.failure_modes},
        "incorrect_enablement_scenarios": [
            s.to_dict() for s in design.incorrect_enablement_scenarios
        ],
        "what_happens_if_instrumentation_enabled_incorrectly": [
            {
                "scenario_id": s.scenario_id,
                "misconfiguration": s.misconfiguration,
                "expected_harm": s.expected_harm,
                "required_mitigation": s.required_mitigation,
            }
            for s in design.incorrect_enablement_scenarios
        ],
        "design_decision": decision.to_dict(),
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21J,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21J),
        "runtime_instrumentation_authorized": False,
        "runtime_instrumentation_implementation_authorized": False,
        "runtime_commit_authorized": False,
        "stage_3_dry_run_design_authorized": decision.stage_3_dry_run_design_authorized,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "model_experiments_run": False,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "Runtime instrumentation architecture design only; no hooks implemented.",
            "ExactKVGenerator and default runtime unchanged.",
            "Per-token hook defined conceptually but MUST NOT be implemented in this phase.",
            "Proposal → trace → verifier → decision → rollback flow is conceptual only.",
            "Verifier remains non-authoritative until future L4 commit stage.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }


def validate_exp111_report(report: Mapping[str, Any]) -> list[str]:
    """Validate Experiment 111 report schema."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "design_outcome",
        "architecture_diagram",
        "runtime_hooks",
        "instrumentation_points",
        "data_flow_steps",
        "data_flow_description",
        "safety_boundaries",
        "safety_boundary_matrix",
        "integration_points",
        "failure_modes",
        "incorrect_enablement_scenarios",
        "what_happens_if_instrumentation_enabled_incorrectly",
        "design_decision",
        "allowed_next_phase",
        "forbidden_next_phases",
        "runtime_instrumentation_authorized",
        "runtime_instrumentation_implementation_authorized",
        "runtime_commit_authorized",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "model_experiments_run",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_111_ID:
        errors.append("experiment_id mismatch")

    if report.get("design_outcome") != DESIGN_OUTCOME_COMPLETE:
        errors.append("design_outcome must be runtime_instrumentation_design_complete")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21J:
        errors.append(
            "allowed_next_phase must be phase21k_l4_stage3_verifier_mediated_dry_run_design",
        )

    forbidden = set(report.get("forbidden_next_phases") or [])
    if not set(FORBIDDEN_NEXT_PHASES_21J) <= forbidden:
        errors.append("missing required forbidden_next_phases")

    for flag in (
        "runtime_instrumentation_authorized",
        "runtime_instrumentation_implementation_authorized",
        "runtime_commit_authorized",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "model_experiments_run",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    decision = report.get("design_decision") or {}
    if decision.get("outcome") != DESIGN_OUTCOME_COMPLETE:
        errors.append("design_decision.outcome must be runtime_instrumentation_design_complete")

    if decision.get("runtime_instrumentation_implementation_authorized") is not False:
        errors.append(
            "design_decision.runtime_instrumentation_implementation_authorized must be false",
        )

    if decision.get("runtime_commit_authorized") is not False:
        errors.append("design_decision.runtime_commit_authorized must be false")

    if decision.get("stage_3_dry_run_design_authorized") is not True:
        errors.append("design_decision.stage_3_dry_run_design_authorized must be true")

    hook_ids = {h["hook_id"] for h in report.get("runtime_hooks") or []}
    if not set(RUNTIME_HOOK_IDS) <= hook_ids:
        errors.append("runtime_hooks missing required hook definitions")

    for hook in report.get("runtime_hooks") or []:
        if hook.get("implemented"):
            errors.append(f"runtime hook {hook.get('hook_id')} must not be implemented")

    point_ids = {p["point_id"] for p in report.get("instrumentation_points") or []}
    if not set(INSTRUMENTATION_POINT_IDS) <= point_ids:
        errors.append("instrumentation_points missing required points")

    for point in report.get("instrumentation_points") or []:
        if point.get("implemented"):
            errors.append(f"instrumentation point {point.get('point_id')} must not be implemented")

    flow_ids = {s["step_id"] for s in report.get("data_flow_steps") or []}
    if not set(DATA_FLOW_STEP_IDS) <= flow_ids:
        errors.append("data_flow_steps missing required steps")

    for step in report.get("data_flow_steps") or []:
        if step.get("executed_at_runtime"):
            errors.append(f"data flow step {step.get('step_id')} must not execute at runtime")

    integration_ids = {i["integration_id"] for i in report.get("integration_points") or []}
    if not set(INTEGRATION_POINT_IDS) <= integration_ids:
        errors.append("integration_points missing required integrations")

    failure_ids = {f["failure_id"] for f in report.get("failure_modes") or []}
    if not set(FAILURE_MODE_IDS) <= failure_ids:
        errors.append("failure_modes missing required failure modes")

    scenario_ids = {
        s["scenario_id"] for s in report.get("incorrect_enablement_scenarios") or []
    }
    if not set(INCORRECT_ENABLEMENT_SCENARIO_IDS) <= scenario_ids:
        errors.append("incorrect_enablement_scenarios missing required scenarios")

    diagram = str(report.get("architecture_diagram") or "")
    if "ExactKVGenerator" not in diagram or "proposal" not in diagram.lower():
        errors.append("architecture_diagram must reference generator and proposal flow")

    return errors
