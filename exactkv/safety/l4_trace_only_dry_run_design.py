"""L4 trace-only dry-run design (Phase 21C / Exp 104).

Stage 2 design specification only — must not be wired to runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_104_ID = "exp104_l4_trace_only_dry_run_design"
DEFAULT_EXP104_REPORT = Path(
    "reports/experiment_104_l4_trace_only_dry_run_design.json",
)
PHASE_21C = "21C"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "trace_only_dry_run_design"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21C = "phase21d_l4_trace_only_dry_run_scaffold"
FORBIDDEN_NEXT_PHASE_21C = "l4_runtime_commit_implementation"

DESIGN_OUTCOME_COMPLETE = "trace_only_dry_run_design_complete"
DESIGN_OUTCOME_INCOMPLETE = "trace_only_dry_run_design_incomplete"
DESIGN_OUTCOME_BLOCKED = "trace_only_dry_run_design_blocked"

DESIGN_OUTCOMES: tuple[str, ...] = (
    DESIGN_OUTCOME_COMPLETE,
    DESIGN_OUTCOME_INCOMPLETE,
    DESIGN_OUTCOME_BLOCKED,
)

DECISION_STATUSES: tuple[str, ...] = (
    "all_match",
    "partial_match",
    "first_token_mismatch",
    "blocked_missing_proposal",
    "blocked_missing_verifier_evidence",
    "failed_hidden_divergence",
    "failed_direct_commit_attempt",
    "invalid_trace",
)

STAGE_2_GATE_NAMES: tuple[str, ...] = (
    "default_runtime_unchanged_gate",
    "trace_only_gate",
    "proposal_source_gate",
    "verifier_evidence_gate",
    "verifier_source_of_truth_gate",
    "no_commit_effect_gate",
    "no_generator_exposure_gate",
    "missing_evidence_blocks_gate",
    "trace_completeness_gate",
    "claim_boundary_gate",
)

REQUIRED_RISK_IDS: tuple[str, ...] = (
    "trace_only_decision_influences_commits",
    "committed_tokens_as_proposal_source",
    "missing_verifier_evidence_silent_match",
    "retokenized_text_used_unsafely",
    "trace_schema_incomplete",
    "block_reasons_hidden",
    "dry_run_overclaimed_as_runtime",
    "prefix_match_overclaimed_as_exactness",
    "performance_memory_claims_inferred",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

INTENDED_TRACE_ONLY_BEHAVIOR: tuple[str, ...] = (
    "Generation runs exactly as today; default runtime unchanged.",
    "After or alongside generation, existing round traces are read.",
    "Draft proposal tokens come from explicit round-log draft proposal fields.",
    "Verifier evidence comes from explicit verifier/full-KV evidence fields where available.",
    "A trace-only accept/reject decision is computed from proposal vs verifier evidence.",
    "The computed decision is written to diagnostics only.",
    "The computed decision is never used to commit tokens.",
    "The computed decision is never exposed to generator decisions.",
    "Any missing evidence creates a blocked trace-only decision, not a fabricated one.",
    "Any safety failure marks the dry-run invalid.",
)

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
    },
)


@dataclass(frozen=True)
class L4TraceOnlyEvidencePlan:
    """Allowed and forbidden evidence sources for trace-only dry-run."""

    allowed_draft_proposal_sources: tuple[str, ...]
    allowed_verifier_evidence_sources: tuple[str, ...]
    forbidden_evidence_sources: tuple[str, ...]
    missing_proposal_blocks_decision: bool = True
    missing_verifier_evidence_blocks_decision: bool = True
    notes: str = "Missing evidence blocks; never fabricate tokens."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyInputSchema:
    """Input fields read from existing round traces for dry-run."""

    required_round_fields: tuple[str, ...]
    optional_round_fields: tuple[str, ...]
    proposal_field: str
    verifier_evidence_fields: tuple[str, ...]
    comparison_only_fields: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyDecisionSchema:
    """Future trace-only dry-run decision record schema."""

    field_names: tuple[str, ...]
    decision_statuses: tuple[str, ...]
    proposal_used_for_token_commit: bool = False
    dry_run_decision_used_for_token_commit: bool = False
    exposed_to_generator: bool = False
    interpretation_note: str = (
        "Trace-only dry-run decisions are diagnostic only; not commit authority."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyRecordSchema:
    """Per-round trace-only dry-run record schema."""

    record_fields: tuple[str, ...]
    must_include_block_reason_when_blocked: bool = True
    trace_complete_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlySafetyGate:
    """Stage 2 trace-only dry-run safety gate definition."""

    name: str
    purpose: str
    required_evidence: str
    pass_condition: str
    fail_condition: str
    applies_before_scaffold: bool
    applies_during_scaffold_validation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyStagePlan:
    """Stage 2 implementation plan (design only)."""

    stage_id: str
    description: str
    allowed_behavior: tuple[str, ...]
    forbidden_behavior: tuple[str, ...]
    prerequisite_stages: tuple[str, ...]
    claim_boundaries: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyRisk:
    """Risk in trace-only dry-run design or future scaffold."""

    risk_id: str
    description: str
    severity: str
    mitigation: str
    current_status: str
    must_pass_gate: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyDesignDecision:
    """Design review outcome for Phase 21C."""

    outcome: str
    stage_2_scaffold_design_authorized: bool
    runtime_commit_authorized: bool
    allowed_next_phase: str
    forbidden_next_phase: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyDryRunDesign:
    """Top-level L4 trace-only dry-run design aggregate."""

    design_id: str
    safety_level: str
    stage: str
    mode: str
    intended_trace_only_behavior: tuple[str, ...]
    evidence_plan: L4TraceOnlyEvidencePlan
    input_schema: L4TraceOnlyInputSchema
    decision_schema: L4TraceOnlyDecisionSchema
    record_schema: L4TraceOnlyRecordSchema
    stage_plan: L4TraceOnlyStagePlan
    safety_gates: tuple[L4TraceOnlySafetyGate, ...]
    risk_register: tuple[L4TraceOnlyRisk, ...]
    decision: L4TraceOnlyDesignDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "safety_level": self.safety_level,
            "stage": self.stage,
            "mode": self.mode,
            "intended_trace_only_behavior": list(self.intended_trace_only_behavior),
            "evidence_plan": self.evidence_plan.to_dict(),
            "input_schema": self.input_schema.to_dict(),
            "decision_schema": self.decision_schema.to_dict(),
            "record_schema": self.record_schema.to_dict(),
            "stage_plan": self.stage_plan.to_dict(),
            "safety_gates": [g.to_dict() for g in self.safety_gates],
            "risk_register": [r.to_dict() for r in self.risk_register],
            "decision": self.decision.to_dict(),
        }


def build_l4_trace_only_evidence_plan() -> L4TraceOnlyEvidencePlan:
    return L4TraceOnlyEvidencePlan(
        allowed_draft_proposal_sources=(
            PROPOSAL_SOURCE_ROUND_LOG,
            "exactkv_round_log_draft_tokens",
            "round_log_draft_token_ids",
            "explicit round-log draft token fields",
        ),
        allowed_verifier_evidence_sources=(
            "verifier_token_ids",
            "full_kv_verifier_token_ids",
            "verifier_evidence_token_ids",
            "explicit full-KV verifier output fields",
            "existing accepted/rejected trace fields for comparison only",
        ),
        forbidden_evidence_sources=(
            "retokenized_generated_text",
            "guessed_token_ids",
            "baseline_generated_tokens_as_proposal",
            "committed_tokens_as_proposal",
            "hidden_implicit_verifier_assumptions",
        ),
    )


def build_l4_trace_only_input_schema() -> L4TraceOnlyInputSchema:
    return L4TraceOnlyInputSchema(
        required_round_fields=(
            "round_index",
            "proposal_token_ids",
        ),
        optional_round_fields=(
            "verifier_evidence_token_ids",
            "accepted_prefix_token_ids",
            "rejected_suffix_token_ids",
            "block_reason",
        ),
        proposal_field="exactkv_round_log_draft_tokens",
        verifier_evidence_fields=(
            "verifier_token_ids",
            "full_kv_verifier_token_ids",
            "verifier_evidence_token_ids",
        ),
        comparison_only_fields=(
            "committed_token_ids",
            "baseline_token_ids",
            "accepted_prefix_token_ids",
            "rejected_suffix_token_ids",
        ),
        notes="Read-only from existing traces; no runtime generation mutation.",
    )


def build_l4_trace_only_decision_schema() -> L4TraceOnlyDecisionSchema:
    return L4TraceOnlyDecisionSchema(
        field_names=(
            "round_index",
            "proposal_token_ids",
            "verifier_evidence_token_ids",
            "accepted_prefix_token_ids",
            "rejected_suffix_token_ids",
            "decision_status",
            "block_reason",
            "verifier_source_of_truth",
            "proposal_used_for_token_commit",
            "dry_run_decision_used_for_token_commit",
            "exposed_to_generator",
            "fallback_required",
            "rollback_required",
            "trace_complete",
            "interpretation_note",
        ),
        decision_statuses=DECISION_STATUSES,
    )


def build_l4_trace_only_record_schema() -> L4TraceOnlyRecordSchema:
    return L4TraceOnlyRecordSchema(
        record_fields=(
            "round_index",
            "proposal_token_ids",
            "verifier_evidence_token_ids",
            "accepted_prefix_token_ids",
            "rejected_suffix_token_ids",
            "decision_status",
            "block_reason",
            "verifier_source_of_truth",
            "trace_complete",
            "interpretation_note",
        ),
    )


def build_l4_trace_only_stage_plan() -> L4TraceOnlyStagePlan:
    return L4TraceOnlyStagePlan(
        stage_id=STAGE,
        description=(
            "Future Stage 2: compute verifier-mediated accept/reject decisions "
            "from existing round traces; diagnostics only; no commit effect."
        ),
        allowed_behavior=(
            "Read round traces after generation",
            "Compute longest verified matching prefix",
            "Write dry-run decisions to diagnostic reports",
            "Block on missing proposal or verifier evidence",
        ),
        forbidden_behavior=(
            "Modify token commits",
            "Expose dry-run decisions to generator",
            "Fabricate missing verifier evidence",
            "Use committed tokens as proposal source",
            "Change default runtime generation",
        ),
        prerequisite_stages=("stage_1_noop_opt_in_scaffold",),
        claim_boundaries=(
            "trace_only_diagnostic_claims",
            "panel_scoped_dry_run_claims",
        ),
    )


def build_l4_trace_only_safety_gates() -> tuple[L4TraceOnlySafetyGate, ...]:
    definitions: list[tuple[str, str, str, str, str]] = [
        (
            "default_runtime_unchanged_gate",
            "Default generation path unchanged during trace-only dry-run.",
            "Baseline parity tests; no generator wiring.",
            "Generation output identical with dry-run enabled vs disabled.",
            "Dry-run changes default token output.",
        ),
        (
            "trace_only_gate",
            "Dry-run operates on traces only; no live commit path.",
            "Architecture review; no commit hooks in dry-run module.",
            "Dry-run reads traces post-generation only.",
            "Dry-run invoked inside commit path.",
        ),
        (
            "proposal_source_gate",
            "Proposal tokens from explicit round-log fields only.",
            "Provenance audit on proposal sources.",
            "All proposals trace to exactkv_round_log_draft_tokens or explicit fields.",
            "Committed or baseline tokens used as proposals.",
        ),
        (
            "verifier_evidence_gate",
            "Verifier evidence from explicit trace fields only.",
            "Trace schema validation for verifier fields.",
            "Verifier evidence present in trace or decision blocked.",
            "Implicit or guessed verifier tokens used.",
        ),
        (
            "verifier_source_of_truth_gate",
            "Verifier evidence controls accept/reject prefix computation.",
            "Dry-run decision tests vs verifier tokens.",
            "Accepted prefix matches verifier-approved prefix only.",
            "Proposal tokens accepted without verifier agreement.",
        ),
        (
            "no_commit_effect_gate",
            "Dry-run decisions never affect token commits.",
            "Commit isolation tests; no generator exposure.",
            "Committed tokens unchanged with dry-run enabled.",
            "Dry-run decision alters committed output.",
        ),
        (
            "no_generator_exposure_gate",
            "Dry-run decisions not exposed to generator decisions.",
            "Generator import audit; no dry-run callbacks.",
            "ExactKVGenerator unchanged; no dry-run hooks in commit loop.",
            "Generator reads dry-run decision output.",
        ),
        (
            "missing_evidence_blocks_gate",
            "Missing proposal or verifier evidence blocks decision.",
            "Synthetic blocked-evidence test cases.",
            "blocked_missing_proposal or blocked_missing_verifier_evidence status.",
            "Missing evidence treated as match or fabricated tokens used.",
        ),
        (
            "trace_completeness_gate",
            "Every dry-run record includes required fields and block reasons.",
            "Record schema validation on dry-run panels.",
            "trace_complete true with block_reason when blocked.",
            "Incomplete records or hidden block reasons.",
        ),
        (
            "claim_boundary_gate",
            "No performance/memory/serving/runtime-commit claims from dry-run.",
            "audit_public_claims.py on dry-run docs and reports.",
            "No forbidden positive claims in dry-run artifacts.",
            "Dry-run results claimed as runtime behavior or speedup.",
        ),
    ]
    gates: list[L4TraceOnlySafetyGate] = []
    for name, purpose, evidence, pass_cond, fail_cond in definitions:
        gates.append(
            L4TraceOnlySafetyGate(
                name=name,
                purpose=purpose,
                required_evidence=evidence,
                pass_condition=pass_cond,
                fail_condition=fail_cond,
                applies_before_scaffold=True,
                applies_during_scaffold_validation=True,
            ),
        )
    return tuple(gates)


def build_l4_trace_only_risk_register() -> tuple[L4TraceOnlyRisk, ...]:
    risks: list[tuple[str, str, str, str, str, str]] = [
        (
            "trace_only_decision_influences_commits",
            "Trace-only dry-run decision accidentally influences token commits.",
            "critical",
            "no_commit_effect_gate; generator import audit.",
            "mitigated_by_design",
            "no_commit_effect_gate",
        ),
        (
            "committed_tokens_as_proposal_source",
            "Committed tokens used as draft proposal source in dry-run.",
            "critical",
            "proposal_source_gate; forbidden source list.",
            "mitigated_by_design",
            "proposal_source_gate",
        ),
        (
            "missing_verifier_evidence_silent_match",
            "Missing verifier evidence silently treated as full match.",
            "critical",
            "missing_evidence_blocks_gate; blocked_missing_verifier_evidence status.",
            "mitigated_by_design",
            "missing_evidence_blocks_gate",
        ),
        (
            "retokenized_text_used_unsafely",
            "Retokenized generated text used as proposal or verifier evidence.",
            "high",
            "forbidden evidence sources; provenance audit.",
            "mitigated_by_design",
            "proposal_source_gate",
        ),
        (
            "trace_schema_incomplete",
            "Dry-run record missing required fields or block reasons.",
            "medium",
            "trace_completeness_gate; record schema validation.",
            "open",
            "trace_completeness_gate",
        ),
        (
            "block_reasons_hidden",
            "Blocked decisions omit block_reason in diagnostics.",
            "medium",
            "record schema requires block_reason when blocked.",
            "mitigated_by_design",
            "trace_completeness_gate",
        ),
        (
            "dry_run_overclaimed_as_runtime",
            "Dry-run diagnostic results overclaimed as runtime L4 behavior.",
            "high",
            "claim_boundary_gate; explicit interpretation_note.",
            "mitigated_by_policy",
            "claim_boundary_gate",
        ),
        (
            "prefix_match_overclaimed_as_exactness",
            "Prefix match rate overclaimed as exact generation preservation.",
            "high",
            "interpretation_note; panel-scoped claims only.",
            "mitigated_by_policy",
            "claim_boundary_gate",
        ),
        (
            "performance_memory_claims_inferred",
            "Performance or memory claims inferred from dry-run diagnostics.",
            "high",
            "claim_boundary_gate; no timing/memory fields in schema.",
            "mitigated_by_policy",
            "claim_boundary_gate",
        ),
    ]
    return tuple(
        L4TraceOnlyRisk(
            risk_id=rid,
            description=desc,
            severity=sev,
            mitigation=mit,
            current_status=status,
            must_pass_gate=gate,
        )
        for rid, desc, sev, mit, status, gate in risks
    )


def evaluate_l4_trace_only_design_decision(
    design: L4TraceOnlyDryRunDesign,
) -> L4TraceOnlyDesignDecision:
    """Evaluate whether trace-only dry-run design is complete."""
    gate_names = {g.name for g in design.safety_gates}
    gates_ok = set(STAGE_2_GATE_NAMES) <= gate_names

    evidence_ok = (
        design.evidence_plan.missing_proposal_blocks_decision
        and design.evidence_plan.missing_verifier_evidence_blocks_decision
        and PROPOSAL_SOURCE_ROUND_LOG in design.evidence_plan.allowed_draft_proposal_sources
    )

    decision_fields = set(design.decision_schema.field_names)
    required_fields = {
        "round_index",
        "proposal_token_ids",
        "verifier_evidence_token_ids",
        "accepted_prefix_token_ids",
        "rejected_suffix_token_ids",
        "decision_status",
        "block_reason",
        "dry_run_decision_used_for_token_commit",
        "exposed_to_generator",
        "trace_complete",
    }
    schema_ok = required_fields <= decision_fields

    statuses_ok = set(DECISION_STATUSES) <= set(design.decision_schema.decision_statuses)

    behavior_text = " ".join(design.intended_trace_only_behavior).lower()
    behavior_ok = (
        "unchanged" in behavior_text or "exactly as today" in behavior_text
    ) and "diagnostic" in behavior_text and "never used to commit" in behavior_text

    risk_ids = {r.risk_id for r in design.risk_register}
    risks_ok = set(REQUIRED_RISK_IDS) <= risk_ids

    complete = gates_ok and evidence_ok and schema_ok and statuses_ok and behavior_ok and risks_ok

    if complete:
        return L4TraceOnlyDesignDecision(
            outcome=DESIGN_OUTCOME_COMPLETE,
            stage_2_scaffold_design_authorized=True,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21C,
            forbidden_next_phase=FORBIDDEN_NEXT_PHASE_21C,
            reason=(
                "Trace-only dry-run design defines behavior, evidence plan, decision schema, "
                "safety gates, and risks; Stage 2 scaffold may begin; runtime commit blocked"
            ),
        )

    if not gates_ok or not schema_ok:
        return L4TraceOnlyDesignDecision(
            outcome=DESIGN_OUTCOME_INCOMPLETE,
            stage_2_scaffold_design_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21C,
            forbidden_next_phase=FORBIDDEN_NEXT_PHASE_21C,
            reason="trace-only dry-run design missing required gates or schema sections",
        )

    return L4TraceOnlyDesignDecision(
        outcome=DESIGN_OUTCOME_BLOCKED,
        stage_2_scaffold_design_authorized=False,
        runtime_commit_authorized=False,
        allowed_next_phase=RECOMMENDED_NEXT_PHASE_21C,
        forbidden_next_phase=FORBIDDEN_NEXT_PHASE_21C,
        reason="trace-only dry-run design blocked by safety preconditions",
    )


def build_l4_trace_only_dry_run_design() -> L4TraceOnlyDryRunDesign:
    """Build the complete L4 trace-only dry-run design."""
    evidence = build_l4_trace_only_evidence_plan()
    input_schema = build_l4_trace_only_input_schema()
    decision_schema = build_l4_trace_only_decision_schema()
    record_schema = build_l4_trace_only_record_schema()
    stage_plan = build_l4_trace_only_stage_plan()
    gates = build_l4_trace_only_safety_gates()
    risks = build_l4_trace_only_risk_register()

    partial = L4TraceOnlyDryRunDesign(
        design_id=EXPERIMENT_104_ID,
        safety_level=L4_SAFETY_LEVEL,
        stage=STAGE,
        mode=MODE,
        intended_trace_only_behavior=INTENDED_TRACE_ONLY_BEHAVIOR,
        evidence_plan=evidence,
        input_schema=input_schema,
        decision_schema=decision_schema,
        record_schema=record_schema,
        stage_plan=stage_plan,
        safety_gates=gates,
        risk_register=risks,
        decision=L4TraceOnlyDesignDecision(
            outcome=DESIGN_OUTCOME_BLOCKED,
            stage_2_scaffold_design_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21C,
            forbidden_next_phase=FORBIDDEN_NEXT_PHASE_21C,
            reason="pending evaluation",
        ),
    )
    decision = evaluate_l4_trace_only_design_decision(partial)
    return L4TraceOnlyDryRunDesign(
        design_id=partial.design_id,
        safety_level=partial.safety_level,
        stage=partial.stage,
        mode=partial.mode,
        intended_trace_only_behavior=partial.intended_trace_only_behavior,
        evidence_plan=partial.evidence_plan,
        input_schema=partial.input_schema,
        decision_schema=partial.decision_schema,
        record_schema=partial.record_schema,
        stage_plan=partial.stage_plan,
        safety_gates=partial.safety_gates,
        risk_register=partial.risk_register,
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
        ("stage_2_trace_dry_run_scaffold", "stage 2 trace-only dry-run scaffold not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp104_l4_trace_only_dry_run_design() -> dict[str, Any]:
    """Run Experiment 104 L4 trace-only dry-run design (no runtime changes)."""
    design = build_l4_trace_only_dry_run_design()
    decision = design.decision

    status = (
        "design_complete"
        if decision.outcome == DESIGN_OUTCOME_COMPLETE
        else "design_incomplete"
    )

    return {
        "experiment_id": EXPERIMENT_104_ID,
        "status": status,
        "phase": PHASE_21C,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "design_objects": {
            "L4TraceOnlyDryRunDesign": design.to_dict(),
            "L4TraceOnlyDesignDecision": decision.to_dict(),
        },
        "intended_trace_only_behavior": list(design.intended_trace_only_behavior),
        "evidence_source_plan": design.evidence_plan.to_dict(),
        "dry_run_decision_schema": design.decision_schema.to_dict(),
        "input_schema": design.input_schema.to_dict(),
        "record_schema": design.record_schema.to_dict(),
        "stage_plan": design.stage_plan.to_dict(),
        "stage_2_safety_gates": [g.to_dict() for g in design.safety_gates],
        "risk_register": [r.to_dict() for r in design.risk_register],
        "design_decision": decision.to_dict(),
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21C,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21C,
        "runtime_commit_authorized": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "model_experiments_run": False,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 trace-only dry-run design only; not runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Trace-only decisions are diagnostic only; no commit effect.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }


def validate_exp104_report(report: dict[str, Any]) -> list[str]:
    """Validate Experiment 104 report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "design_objects",
        "intended_trace_only_behavior",
        "evidence_source_plan",
        "dry_run_decision_schema",
        "stage_2_safety_gates",
        "risk_register",
        "design_decision",
        "allowed_next_phase",
        "forbidden_next_phase",
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

    if report.get("experiment_id") != EXPERIMENT_104_ID:
        errors.append("experiment_id mismatch")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21C:
        errors.append("allowed_next_phase must be phase21d_l4_trace_only_dry_run_scaffold")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_21C:
        errors.append("forbidden_next_phase must be l4_runtime_commit_implementation")

    if report.get("runtime_commit_authorized") is not False:
        errors.append("runtime_commit_authorized must be false")

    for flag in (
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
        errors.append("design_decision.outcome must be trace_only_dry_run_design_complete")

    if decision.get("runtime_commit_authorized") is not False:
        errors.append("design_decision.runtime_commit_authorized must be false")

    gate_names = {g["name"] for g in report.get("stage_2_safety_gates") or []}
    if not set(STAGE_2_GATE_NAMES) <= gate_names:
        errors.append("missing required stage_2_safety_gates")

    if "no_commit_effect_gate" not in gate_names:
        errors.append("missing no_commit_effect_gate")
    if "no_generator_exposure_gate" not in gate_names:
        errors.append("missing no_generator_exposure_gate")

    statuses = set((report.get("dry_run_decision_schema") or {}).get("decision_statuses") or [])
    if not set(DECISION_STATUSES) <= statuses:
        errors.append("dry_run_decision_schema missing required decision_statuses")

    risk_ids = {r["risk_id"] for r in report.get("risk_register") or []}
    if not set(REQUIRED_RISK_IDS) <= risk_ids:
        errors.append("missing required risks in risk_register")

    evidence = report.get("evidence_source_plan") or {}
    if not evidence.get("missing_verifier_evidence_blocks_decision"):
        errors.append("evidence plan must block missing verifier evidence")

    return errors
