"""L4 Stage 3 verifier-mediated dry-run design (Phase 21K / Exp 112).

Stage 3 execution model design only — must not implement runtime or model execution.
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
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_112_ID = "exp112_l4_stage3_verifier_mediated_dry_run_design"
DEFAULT_EXP112_REPORT = Path(
    "reports/experiment_112_l4_stage3_verifier_mediated_dry_run_design.json",
)
PHASE_21K = "21K"
STAGE = "stage_3_verifier_mediated_dry_run"
MODE = "verifier_mediated_dry_run_design"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21K = "phase21l_l4_stage3_verifier_mediated_dry_run_scaffold"
FORBIDDEN_NEXT_PHASES_21K: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_instrumentation_implementation",
    "l4_stage3_runtime_execution",
    "l4_default_runtime_modification",
    "l4_verifier_in_loop_execution",
)

DESIGN_OUTCOME_COMPLETE = "stage3_verifier_mediated_dry_run_design_complete"
DESIGN_OUTCOME_INCOMPLETE = "stage3_verifier_mediated_dry_run_design_incomplete"
DESIGN_OUTCOME_BLOCKED = "stage3_verifier_mediated_dry_run_design_blocked"

TERMINAL_STATES: tuple[str, ...] = (
    "ACCEPT_PREFIX",
    "REJECT",
    "BLOCK_MISSING_EVIDENCE",
    "INVALID_TRACE",
)

DECISION_GRAPH_NODE_TYPES: tuple[str, ...] = (
    "proposal_token",
    "verifier_alignment_point",
    "match_transition",
    "mismatch_transition",
    "terminal",
)

DECISION_GRAPH_EDGE_TYPES: tuple[str, ...] = (
    "match",
    "mismatch",
    "block",
    "invalid",
)

FAILURE_MODE_IDS: tuple[str, ...] = (
    "missing_verifier_evidence",
    "proposal_verifier_mismatch",
    "invalid_trace_schema",
    "corrupted_proposal_source",
    "aliasing_attack_detected",
    "truncated_verifier_stream",
    "partial_prefix_disagreement",
    "conflicting_evidence_sources",
)

FAILURE_RESPONSE = "BLOCK_DRY_RUN_DECISION"

SAFETY_INVARIANT_IDS: tuple[str, ...] = (
    "default_runtime_unchanged",
    "no_token_commit",
    "no_generator_exposure",
    "verifier_is_not_executed",
    "trace_only",
    "deterministic_only",
    "no_external_effects",
)

SYNTHETIC_TEST_CASE_IDS: tuple[str, ...] = (
    "synthetic_full_match_trace",
    "synthetic_partial_prefix_mismatch",
    "synthetic_missing_verifier_evidence",
    "synthetic_corrupted_proposal",
    "synthetic_adversarial_aliasing",
    "synthetic_conflicting_verifier_sources",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

ARCHITECTURE_DIAGRAM = """
┌──────────────────────────────────────────────────────────────────────────────┐
│              Stage 3 Verifier-Mediated Dry-Run (DESIGN ONLY)                 │
│                                                                              │
│  ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐  │
│  │ L3 proposal     │     │ Evidence             │     │ Decision graph  │  │
│  │ ingestion       │────▶│ reconciliation       │────▶│ (prefix sim)    │  │
│  │ (round-log)     │     │ layer                │     │                 │  │
│  └─────────────────┘     └──────────┬───────────┘     └────────┬────────┘  │
│                                     │                          │           │
│  ┌─────────────────┐              │                          │           │
│  │ L4 verifier     │──────────────┘                          │           │
│  │ evidence schema │  (v1 trace fields — NOT executed)       │           │
│  │ (v1)            │                                         │           │
│  └─────────────────┘                                         ▼           │
│                                                    ┌─────────────────────┐ │
│                                                    │ L4Stage3DryRunResult│ │
│                                                    │ (trace-only output) │ │
│                                                    └─────────────────────┘ │
│                                                                              │
│  ExactKVGenerator ──▶ UNCHANGED (no exposure, no commit, no execution)      │
└──────────────────────────────────────────────────────────────────────────────┘
""".strip()

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
        "stage_3_verifier_dry_run_design",
    },
)


@dataclass(frozen=True)
class L4ProposalIngestionModel:
    """How L3 proposal sources enter Stage 3 dry-run (design only)."""

    allowed_proposal_sources: tuple[str, ...]
    required_proposal_fields: tuple[str, ...]
    forbidden_proposal_sources: tuple[str, ...]
    ingestion_is_read_only: bool = True
    notes: str = "Proposals are read from trace; never influence generation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceMappingModel:
    """How verifier evidence trace fields map into Stage 3 evaluation."""

    schema_version: str
    required_verifier_fields: tuple[str, ...]
    mapping_rules: tuple[str, ...]
    missing_field_policy: str
    notes: str = "Missing verifier fields block; never fabricated."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4DecisionGraphNode:
    """One node in the Stage 3 dry-run decision graph."""

    node_id: str
    node_type: str
    description: str
    token_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4DecisionGraphEdge:
    """Directed edge in the Stage 3 decision graph."""

    edge_id: str
    from_node_id: str
    to_node_id: str
    edge_type: str
    condition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4DecisionGraphModel:
    """Decision graph structure for prefix accept/reject simulation."""

    node_types: tuple[str, ...]
    edge_types: tuple[str, ...]
    terminal_states: tuple[str, ...]
    nodes: tuple[L4DecisionGraphNode, ...]
    edges: tuple[L4DecisionGraphEdge, ...]
    traversal_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_types": list(self.node_types),
            "edge_types": list(self.edge_types),
            "terminal_states": list(self.terminal_states),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "traversal_rule": self.traversal_rule,
        }


@dataclass(frozen=True)
class L4PrefixAcceptanceSimulationLogic:
    """Conceptual prefix acceptance simulation — no runtime execution."""

    algorithm_description: str
    match_rule: str
    prefix_length_computation: str
    accept_terminal_state: str
    reject_terminal_state: str
    executed_at_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4MismatchHandlingPolicy:
    """Policy for proposal/verifier mismatch in Stage 3 dry-run."""

    first_mismatch_action: str
    partial_match_allowed: bool
    mismatch_terminal_state: str
    block_reason_required: bool = True
    notes: str = "Mismatch surfaces in dry-run result; no token commit."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4RollbackSimulationModel:
    """Conceptual rollback simulation — no execution."""

    trigger_conditions: tuple[str, ...]
    simulated_rollback_state: str
    baseline_reference: str
    executed_at_runtime: bool = False
    notes: str = "Rollback is simulated in decision graph only."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3TraceOnlyOutputSchema:
    """Trace-only output schema for Stage 3 dry-run results."""

    result_type_name: str
    required_fields: tuple[str, ...]
    decision_status_values: tuple[str, ...]
    diagnostic_only: bool = True
    exposed_to_generator: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3DryRunResult:
    """Stage 3 dry-run result record schema (design specification)."""

    decision_status: str
    prefix_match_length: int
    proposal_source_id: str
    verifier_source_id: str
    block_reason: str | None
    trace_complete_flag: bool
    safety_gate_results: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4EvidenceReconciliationLayer:
    """How L3 proposals reconcile against L4 verifier traces."""

    reconciliation_steps: tuple[str, ...]
    proposal_field: str
    verifier_fields: tuple[str, ...]
    alias_detection_rule: str
    missing_evidence_blocks: bool = True
    no_generation_influence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3SafetyInvariant:
    """Safety invariant for Stage 3 dry-run design."""

    invariant_id: str
    description: str
    required_value: bool
    enforcement_gate: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3FailureMode:
    """Failure mode for Stage 3 dry-run evaluation."""

    failure_id: str
    description: str
    detection_signal: str
    required_response: str
    terminal_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3SyntheticTestCase:
    """Synthetic test case specification — expected outcomes only, no execution."""

    test_id: str
    description: str
    input_sketch: str
    expected_terminal_state: str
    expected_decision_status: str
    executes_at_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3DryRunDesignDecision:
    """Design review outcome for Phase 21K."""

    outcome: str
    stage_3_scaffold_authorized: bool
    runtime_execution_authorized: bool
    runtime_commit_authorized: bool
    allowed_next_phase: str
    forbidden_next_phases: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3VerifierMediatedDryRunDesign:
    """Top-level Stage 3 verifier-mediated dry-run design aggregate."""

    design_id: str
    safety_level: str
    stage: str
    mode: str
    schema_version: str
    architecture_diagram: str
    proposal_ingestion: L4ProposalIngestionModel
    verifier_evidence_mapping: L4VerifierEvidenceMappingModel
    evidence_reconciliation: L4EvidenceReconciliationLayer
    decision_graph: L4DecisionGraphModel
    prefix_acceptance_simulation: L4PrefixAcceptanceSimulationLogic
    mismatch_handling: L4MismatchHandlingPolicy
    rollback_simulation: L4RollbackSimulationModel
    output_schema: L4Stage3TraceOnlyOutputSchema
    safety_invariants: tuple[L4Stage3SafetyInvariant, ...]
    failure_modes: tuple[L4Stage3FailureMode, ...]
    synthetic_test_matrix: tuple[L4Stage3SyntheticTestCase, ...]
    decision: L4Stage3DryRunDesignDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "safety_level": self.safety_level,
            "stage": self.stage,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "architecture_diagram": self.architecture_diagram,
            "proposal_ingestion": self.proposal_ingestion.to_dict(),
            "verifier_evidence_mapping": self.verifier_evidence_mapping.to_dict(),
            "evidence_reconciliation": self.evidence_reconciliation.to_dict(),
            "decision_graph": self.decision_graph.to_dict(),
            "prefix_acceptance_simulation": self.prefix_acceptance_simulation.to_dict(),
            "mismatch_handling": self.mismatch_handling.to_dict(),
            "rollback_simulation": self.rollback_simulation.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "safety_invariants": [i.to_dict() for i in self.safety_invariants],
            "failure_modes": [f.to_dict() for f in self.failure_modes],
            "synthetic_test_matrix": [t.to_dict() for t in self.synthetic_test_matrix],
            "decision": self.decision.to_dict(),
        }


def build_l4_proposal_ingestion_model() -> L4ProposalIngestionModel:
    """L3 proposal ingestion model for Stage 3 dry-run."""
    return L4ProposalIngestionModel(
        allowed_proposal_sources=(PROPOSAL_SOURCE_ROUND_LOG,),
        required_proposal_fields=(
            "proposal_source",
            "proposal_token_ids",
            "round_index",
        ),
        forbidden_proposal_sources=(
            "committed_tokens",
            "baseline_tokens",
            "verifier_tokens",
            "retokenized_generated_text",
        ),
    )


def build_l4_verifier_evidence_mapping_model() -> L4VerifierEvidenceMappingModel:
    """Verifier evidence mapping from schema v1 into Stage 3."""
    return L4VerifierEvidenceMappingModel(
        schema_version=TRACE_SCHEMA_VERSION,
        required_verifier_fields=(
            "verifier_evidence_available",
            "verifier_evidence_source",
            "verifier_evidence_token_ids",
            "verifier_evidence_is_full_kv",
            "verifier_checked_proposal_token_ids",
        ),
        mapping_rules=(
            "verifier_evidence_token_ids maps to comparison right-hand side",
            "proposal_token_ids maps to comparison left-hand side",
            "verifier_first_mismatch_index derived from pairwise token walk",
            "verifier_decision_status derived from terminal graph state",
        ),
        missing_field_policy="BLOCK_MISSING_EVIDENCE; emit block_reason; no fabrication",
    )


def build_l4_evidence_reconciliation_layer() -> L4EvidenceReconciliationLayer:
    """Evidence reconciliation between L3 proposals and L4 verifier traces."""
    return L4EvidenceReconciliationLayer(
        reconciliation_steps=(
            "Validate trace schema version and diagnostic_only=true",
            "Load proposal_token_ids from explicit proposal_source",
            "Load verifier_evidence_token_ids from explicit verifier fields",
            "Reject if proposal and verifier share object identity (alias attack)",
            "Reject if verifier_evidence_available is false or fields missing",
            "Walk tokens left-to-right building decision graph nodes",
            "Emit terminal state without influencing generation",
        ),
        proposal_field="proposal_token_ids",
        verifier_fields=(
            "verifier_evidence_token_ids",
            "verifier_checked_proposal_token_ids",
            "verifier_matching_prefix_token_ids",
        ),
        alias_detection_rule=(
            "proposal_token_ids is verifier_evidence_token_ids → INVALID_TRACE"
        ),
    )


def build_l4_decision_graph_model() -> L4DecisionGraphModel:
    """Decision graph for prefix-level accept/reject simulation."""
    nodes = (
        L4DecisionGraphNode(
            node_id="start",
            node_type="verifier_alignment_point",
            description="Graph entry; validate inputs before token walk",
            token_index=None,
        ),
        L4DecisionGraphNode(
            node_id="token_0",
            node_type="proposal_token",
            description="First proposal token alignment check",
            token_index=0,
        ),
        L4DecisionGraphNode(
            node_id="token_n",
            node_type="proposal_token",
            description="Generic proposal token alignment check at index n",
            token_index=None,
        ),
        L4DecisionGraphNode(
            node_id="terminal_accept_prefix",
            node_type="terminal",
            description="Longest verified matching prefix accepted (simulated)",
        ),
        L4DecisionGraphNode(
            node_id="terminal_reject",
            node_type="terminal",
            description="Mismatch or policy rejection",
        ),
        L4DecisionGraphNode(
            node_id="terminal_block_missing",
            node_type="terminal",
            description="Missing verifier evidence blocks evaluation",
        ),
        L4DecisionGraphNode(
            node_id="terminal_invalid_trace",
            node_type="terminal",
            description="Schema or alias violation",
        ),
    )
    edges = (
        L4DecisionGraphEdge(
            edge_id="start_to_token_0",
            from_node_id="start",
            to_node_id="token_0",
            edge_type="match",
            condition="inputs valid and verifier evidence present",
        ),
        L4DecisionGraphEdge(
            edge_id="start_to_block",
            from_node_id="start",
            to_node_id="terminal_block_missing",
            edge_type="block",
            condition="verifier evidence missing or unavailable",
        ),
        L4DecisionGraphEdge(
            edge_id="start_to_invalid",
            from_node_id="start",
            to_node_id="terminal_invalid_trace",
            edge_type="invalid",
            condition="schema invalid or alias detected",
        ),
        L4DecisionGraphEdge(
            edge_id="token_match_next",
            from_node_id="token_n",
            to_node_id="token_n",
            edge_type="match",
            condition="proposal[i] == verifier[i]",
        ),
        L4DecisionGraphEdge(
            edge_id="token_match_accept",
            from_node_id="token_n",
            to_node_id="terminal_accept_prefix",
            edge_type="match",
            condition="all proposal tokens matched; end of proposal",
        ),
        L4DecisionGraphEdge(
            edge_id="token_mismatch_reject",
            from_node_id="token_n",
            to_node_id="terminal_reject",
            edge_type="mismatch",
            condition="proposal[i] != verifier[i]",
        ),
    )
    return L4DecisionGraphModel(
        node_types=DECISION_GRAPH_NODE_TYPES,
        edge_types=DECISION_GRAPH_EDGE_TYPES,
        terminal_states=TERMINAL_STATES,
        nodes=nodes,
        edges=edges,
        traversal_rule=(
            "Deterministic left-to-right walk: at each index emit match or mismatch "
            "edge; first mismatch → REJECT; full match → ACCEPT_PREFIX; "
            "missing evidence → BLOCK_MISSING_EVIDENCE; invalid input → INVALID_TRACE."
        ),
    )


def build_l4_prefix_acceptance_simulation_logic() -> L4PrefixAcceptanceSimulationLogic:
    """Conceptual prefix acceptance simulation."""
    return L4PrefixAcceptanceSimulationLogic(
        algorithm_description=(
            "Simulate longest common prefix between proposal_token_ids and "
            "verifier_evidence_token_ids without executing verifier or committing tokens."
        ),
        match_rule="proposal[i] == verifier[i] for all i in 0..prefix_length-1",
        prefix_length_computation=(
            "prefix_match_length = count of consecutive equal tokens from index 0 "
            "until first mismatch or end of shorter sequence"
        ),
        accept_terminal_state="ACCEPT_PREFIX",
        reject_terminal_state="REJECT",
    )


def build_l4_mismatch_handling_policy() -> L4MismatchHandlingPolicy:
    """Mismatch handling for Stage 3 dry-run."""
    return L4MismatchHandlingPolicy(
        first_mismatch_action="terminate graph at mismatch edge → REJECT",
        partial_match_allowed=True,
        mismatch_terminal_state="REJECT",
    )


def build_l4_rollback_simulation_model() -> L4RollbackSimulationModel:
    """Conceptual rollback simulation — no execution."""
    return L4RollbackSimulationModel(
        trigger_conditions=(
            "terminal_reject reached",
            "BLOCK_MISSING_EVIDENCE",
            "INVALID_TRACE",
            "BLOCK_DRY_RUN_DECISION from failure mode",
        ),
        simulated_rollback_state="baseline_safe_path_restored (conceptual)",
        baseline_reference="pre-proposal baseline generation snapshot (design reference only)",
    )


def build_l4_stage3_trace_only_output_schema() -> L4Stage3TraceOnlyOutputSchema:
    """Output schema for Stage 3 dry-run results."""
    return L4Stage3TraceOnlyOutputSchema(
        result_type_name="L4Stage3DryRunResult",
        required_fields=(
            "decision_status",
            "prefix_match_length",
            "proposal_source_id",
            "verifier_source_id",
            "block_reason",
            "trace_complete_flag",
            "safety_gate_results",
        ),
        decision_status_values=TERMINAL_STATES + (FAILURE_RESPONSE,),
    )


def build_l4_stage3_safety_invariants() -> tuple[L4Stage3SafetyInvariant, ...]:
    """Safety invariants — all required_value=True."""
    specs = (
        ("default_runtime_unchanged", "Default ExactKV runtime path is never modified", "default_runtime_unchanged_gate"),
        ("no_token_commit", "Dry-run never commits tokens to KV cache", "no_commit_effect_gate"),
        ("no_generator_exposure", "Dry-run results never exposed to ExactKVGenerator", "no_generator_exposure_gate"),
        ("verifier_is_not_executed", "Verifier is not executed; only trace fields consumed", "trace_only_gate"),
        ("trace_only", "All outputs are diagnostic trace records only", "trace_only_gate"),
        ("deterministic_only", "Same inputs always yield same decision graph", "trace_completeness_gate"),
        ("no_external_effects", "No model, GPU, serving, or inference side effects", "claim_boundary_gate"),
    )
    return tuple(
        L4Stage3SafetyInvariant(
            invariant_id=iid,
            description=desc,
            required_value=True,
            enforcement_gate=gate,
        )
        for iid, desc, gate in specs
    )


def build_l4_stage3_failure_modes() -> tuple[L4Stage3FailureMode, ...]:
    """Failure modes — all map to BLOCK_DRY_RUN_DECISION."""
    specs: list[tuple[str, str, str, str]] = [
        (
            "missing_verifier_evidence",
            "Verifier evidence fields absent or verifier_evidence_available=false",
            "verifier_evidence_available is false or required fields missing",
            "BLOCK_MISSING_EVIDENCE",
        ),
        (
            "proposal_verifier_mismatch",
            "Proposal and verifier token sequences disagree",
            "first mismatch index < len(proposal)",
            "REJECT",
        ),
        (
            "invalid_trace_schema",
            f"Trace record fails schema validation for {TRACE_SCHEMA_VERSION}",
            "schema validation errors non-empty",
            "INVALID_TRACE",
        ),
        (
            "corrupted_proposal_source",
            "Proposal source forbidden or proposal_token_ids malformed",
            "proposal_source in forbidden set or proposal_token_ids not a list",
            "INVALID_TRACE",
        ),
        (
            "aliasing_attack_detected",
            "Proposal and verifier fields alias same object or mislabeled source",
            "alias detection rule triggered",
            "INVALID_TRACE",
        ),
        (
            "truncated_verifier_stream",
            "Verifier evidence shorter than proposal without explicit partial policy",
            "len(verifier) < len(proposal) and no partial_match policy",
            "REJECT",
        ),
        (
            "partial_prefix_disagreement",
            "Matching prefix followed by suffix disagreement",
            "prefix_match_length > 0 and terminal REJECT",
            "REJECT",
        ),
        (
            "conflicting_evidence_sources",
            "Multiple verifier source fields disagree",
            "verifier_evidence_source conflicts with token content",
            "INVALID_TRACE",
        ),
    ]
    return tuple(
        L4Stage3FailureMode(
            failure_id=fid,
            description=desc,
            detection_signal=signal,
            required_response=FAILURE_RESPONSE,
            terminal_state=terminal,
        )
        for fid, desc, signal, terminal in specs
    )


def build_l4_stage3_synthetic_test_matrix() -> tuple[L4Stage3SyntheticTestCase, ...]:
    """Synthetic test specifications — expected outcomes only."""
    return (
        L4Stage3SyntheticTestCase(
            test_id="synthetic_full_match_trace",
            description="Proposal and verifier tokens fully agree",
            input_sketch="proposal=[1,2,3], verifier=[1,2,3], schema valid",
            expected_terminal_state="ACCEPT_PREFIX",
            expected_decision_status="ACCEPT_PREFIX",
        ),
        L4Stage3SyntheticTestCase(
            test_id="synthetic_partial_prefix_mismatch",
            description="Proposal matches verifier for first k tokens then diverges",
            input_sketch="proposal=[1,2,9], verifier=[1,2,3], schema valid",
            expected_terminal_state="REJECT",
            expected_decision_status="REJECT",
        ),
        L4Stage3SyntheticTestCase(
            test_id="synthetic_missing_verifier_evidence",
            description="Verifier evidence block missing entirely",
            input_sketch="proposal=[1,2], verifier_evidence_available=false",
            expected_terminal_state="BLOCK_MISSING_EVIDENCE",
            expected_decision_status="BLOCK_MISSING_EVIDENCE",
        ),
        L4Stage3SyntheticTestCase(
            test_id="synthetic_corrupted_proposal",
            description="Proposal from forbidden committed_tokens source",
            input_sketch="proposal_source=committed_tokens",
            expected_terminal_state="INVALID_TRACE",
            expected_decision_status="INVALID_TRACE",
        ),
        L4Stage3SyntheticTestCase(
            test_id="synthetic_adversarial_aliasing",
            description="Proposal and verifier fields share identity",
            input_sketch="proposal_token_ids is verifier_evidence_token_ids",
            expected_terminal_state="INVALID_TRACE",
            expected_decision_status="INVALID_TRACE",
        ),
        L4Stage3SyntheticTestCase(
            test_id="synthetic_conflicting_verifier_sources",
            description="Two verifier source fields disagree on token content",
            input_sketch="verifier_evidence_token_ids != verifier_checked_proposal_token_ids",
            expected_terminal_state="INVALID_TRACE",
            expected_decision_status="BLOCK_DRY_RUN_DECISION",
        ),
    )


def build_l4_stage3_dry_run_result_example() -> L4Stage3DryRunResult:
    """Example L4Stage3DryRunResult for schema documentation."""
    invariants = build_l4_stage3_safety_invariants()
    return L4Stage3DryRunResult(
        decision_status="ACCEPT_PREFIX",
        prefix_match_length=3,
        proposal_source_id=PROPOSAL_SOURCE_ROUND_LOG,
        verifier_source_id="full_kv_verifier_output_tokens",
        block_reason=None,
        trace_complete_flag=True,
        safety_gate_results={i.invariant_id: True for i in invariants},
    )


def evaluate_l4_stage3_dry_run_design_decision(
    design: L4Stage3VerifierMediatedDryRunDesign,
) -> L4Stage3DryRunDesignDecision:
    """Evaluate whether Phase 21K design is complete."""
    missing: list[str] = []

    failure_ids = {f.failure_id for f in design.failure_modes}
    if not set(FAILURE_MODE_IDS) <= failure_ids:
        missing.append("incomplete failure_modes")

    invariant_ids = {i.invariant_id for i in design.safety_invariants}
    if not set(SAFETY_INVARIANT_IDS) <= invariant_ids:
        missing.append("incomplete safety_invariants")

    test_ids = {t.test_id for t in design.synthetic_test_matrix}
    if not set(SYNTHETIC_TEST_CASE_IDS) <= test_ids:
        missing.append("incomplete synthetic_test_matrix")

    if not set(TERMINAL_STATES) <= set(design.decision_graph.terminal_states):
        missing.append("decision_graph missing terminal states")

    for mode in design.failure_modes:
        if mode.required_response != FAILURE_RESPONSE:
            missing.append(f"failure {mode.failure_id} must map to BLOCK_DRY_RUN_DECISION")

    for inv in design.safety_invariants:
        if not inv.required_value:
            missing.append(f"invariant {inv.invariant_id} must be required_value=true")

    if design.prefix_acceptance_simulation.executed_at_runtime:
        missing.append("prefix acceptance must not execute at runtime")
    if design.rollback_simulation.executed_at_runtime:
        missing.append("rollback simulation must not execute at runtime")
    if any(t.executes_at_runtime for t in design.synthetic_test_matrix):
        missing.append("synthetic tests must not execute at runtime")

    if missing:
        return L4Stage3DryRunDesignDecision(
            outcome=DESIGN_OUTCOME_INCOMPLETE,
            stage_3_scaffold_authorized=False,
            runtime_execution_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21K,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21K,
            reason="; ".join(missing),
        )

    return L4Stage3DryRunDesignDecision(
        outcome=DESIGN_OUTCOME_COMPLETE,
        stage_3_scaffold_authorized=True,
        runtime_execution_authorized=False,
        runtime_commit_authorized=False,
        allowed_next_phase=RECOMMENDED_NEXT_PHASE_21K,
        forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21K,
        reason="Stage 3 verifier-mediated dry-run design complete; scaffold authorized only.",
    )


def build_l4_stage3_verifier_mediated_dry_run_design() -> L4Stage3VerifierMediatedDryRunDesign:
    """Build the Phase 21K Stage 3 dry-run design aggregate."""
    partial = L4Stage3VerifierMediatedDryRunDesign(
        design_id=EXPERIMENT_112_ID,
        safety_level=L4_SAFETY_LEVEL,
        stage=STAGE,
        mode=MODE,
        schema_version=TRACE_SCHEMA_VERSION,
        architecture_diagram=ARCHITECTURE_DIAGRAM,
        proposal_ingestion=build_l4_proposal_ingestion_model(),
        verifier_evidence_mapping=build_l4_verifier_evidence_mapping_model(),
        evidence_reconciliation=build_l4_evidence_reconciliation_layer(),
        decision_graph=build_l4_decision_graph_model(),
        prefix_acceptance_simulation=build_l4_prefix_acceptance_simulation_logic(),
        mismatch_handling=build_l4_mismatch_handling_policy(),
        rollback_simulation=build_l4_rollback_simulation_model(),
        output_schema=build_l4_stage3_trace_only_output_schema(),
        safety_invariants=build_l4_stage3_safety_invariants(),
        failure_modes=build_l4_stage3_failure_modes(),
        synthetic_test_matrix=build_l4_stage3_synthetic_test_matrix(),
        decision=L4Stage3DryRunDesignDecision(
            outcome=DESIGN_OUTCOME_BLOCKED,
            stage_3_scaffold_authorized=False,
            runtime_execution_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21K,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21K,
            reason="pending evaluation",
        ),
    )
    decision = evaluate_l4_stage3_dry_run_design_decision(partial)
    return L4Stage3VerifierMediatedDryRunDesign(
        design_id=partial.design_id,
        safety_level=partial.safety_level,
        stage=partial.stage,
        mode=partial.mode,
        schema_version=partial.schema_version,
        architecture_diagram=partial.architecture_diagram,
        proposal_ingestion=partial.proposal_ingestion,
        verifier_evidence_mapping=partial.verifier_evidence_mapping,
        evidence_reconciliation=partial.evidence_reconciliation,
        decision_graph=partial.decision_graph,
        prefix_acceptance_simulation=partial.prefix_acceptance_simulation,
        mismatch_handling=partial.mismatch_handling,
        rollback_simulation=partial.rollback_simulation,
        output_schema=partial.output_schema,
        safety_invariants=partial.safety_invariants,
        failure_modes=partial.failure_modes,
        synthetic_test_matrix=partial.synthetic_test_matrix,
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
        ("stage_3_verifier_dry_run_scaffold", "stage 3 dry-run scaffold not implemented"),
        (
            "runtime_verifier_instrumentation",
            "runtime verifier evidence instrumentation not implemented",
        ),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp112_l4_stage3_verifier_mediated_dry_run_design() -> dict[str, Any]:
    """Run Experiment 112 Stage 3 verifier-mediated dry-run design (no runtime)."""
    design = build_l4_stage3_verifier_mediated_dry_run_design()
    decision = design.decision
    example_result = build_l4_stage3_dry_run_result_example()

    status = (
        "design_complete"
        if decision.outcome == DESIGN_OUTCOME_COMPLETE
        else "design_incomplete"
    )

    safety_invariant_flags = {
        i.invariant_id: i.required_value for i in design.safety_invariants
    }

    return {
        "experiment_id": EXPERIMENT_112_ID,
        "status": status,
        "phase": PHASE_21K,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": design.schema_version,
        "design_outcome": decision.outcome,
        "architecture_diagram": design.architecture_diagram,
        "design_objects": {
            "L4Stage3VerifierMediatedDryRunDesign": design.to_dict(),
            "L4Stage3DryRunDesignDecision": decision.to_dict(),
            "L4Stage3DryRunResult": example_result.to_dict(),
        },
        "proposal_ingestion_model": design.proposal_ingestion.to_dict(),
        "verifier_evidence_mapping_model": design.verifier_evidence_mapping.to_dict(),
        "evidence_reconciliation_layer": design.evidence_reconciliation.to_dict(),
        "decision_graph_model": design.decision_graph.to_dict(),
        "prefix_acceptance_simulation": design.prefix_acceptance_simulation.to_dict(),
        "mismatch_handling_policy": design.mismatch_handling.to_dict(),
        "rollback_simulation_model": design.rollback_simulation.to_dict(),
        "output_schema": design.output_schema.to_dict(),
        "safety_invariants": [i.to_dict() for i in design.safety_invariants],
        "safety_invariant_flags": safety_invariant_flags,
        "failure_modes": [f.to_dict() for f in design.failure_modes],
        "failure_mode_taxonomy": {
            f.failure_id: {
                "description": f.description,
                "required_response": f.required_response,
                "terminal_state": f.terminal_state,
            }
            for f in design.failure_modes
        },
        "synthetic_test_matrix": [t.to_dict() for t in design.synthetic_test_matrix],
        "design_decision": decision.to_dict(),
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21K,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21K),
        "runtime_execution_authorized": False,
        "runtime_instrumentation_authorized": False,
        "runtime_commit_authorized": False,
        "stage_3_scaffold_authorized": decision.stage_3_scaffold_authorized,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "model_experiments_run": False,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "Stage 3 verifier-mediated dry-run design only; no runtime execution.",
            "ExactKVGenerator and default runtime unchanged.",
            "Verifier is not executed; only trace field reconciliation is designed.",
            "Prefix acceptance and rollback are simulation models only.",
            "Synthetic test matrix defines expected outcomes; tests do not execute.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }


def validate_exp112_report(report: Mapping[str, Any]) -> list[str]:
    """Validate Experiment 112 report schema."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "design_outcome",
        "architecture_diagram",
        "proposal_ingestion_model",
        "verifier_evidence_mapping_model",
        "evidence_reconciliation_layer",
        "decision_graph_model",
        "prefix_acceptance_simulation",
        "mismatch_handling_policy",
        "rollback_simulation_model",
        "output_schema",
        "safety_invariants",
        "safety_invariant_flags",
        "failure_modes",
        "synthetic_test_matrix",
        "design_decision",
        "allowed_next_phase",
        "forbidden_next_phases",
        "runtime_execution_authorized",
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

    if report.get("experiment_id") != EXPERIMENT_112_ID:
        errors.append("experiment_id mismatch")

    if report.get("design_outcome") != DESIGN_OUTCOME_COMPLETE:
        errors.append("design_outcome must be stage3_verifier_mediated_dry_run_design_complete")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21K:
        errors.append(
            "allowed_next_phase must be phase21l_l4_stage3_verifier_mediated_dry_run_scaffold",
        )

    forbidden = set(report.get("forbidden_next_phases") or [])
    if not set(FORBIDDEN_NEXT_PHASES_21K) <= forbidden:
        errors.append("missing required forbidden_next_phases")

    for flag in (
        "runtime_execution_authorized",
        "runtime_instrumentation_authorized",
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
        errors.append(
            "design_decision.outcome must be stage3_verifier_mediated_dry_run_design_complete",
        )
    if decision.get("runtime_execution_authorized") is not False:
        errors.append("design_decision.runtime_execution_authorized must be false")
    if decision.get("runtime_commit_authorized") is not False:
        errors.append("design_decision.runtime_commit_authorized must be false")
    if decision.get("stage_3_scaffold_authorized") is not True:
        errors.append("design_decision.stage_3_scaffold_authorized must be true")

    invariant_flags = report.get("safety_invariant_flags") or {}
    for iid in SAFETY_INVARIANT_IDS:
        if invariant_flags.get(iid) is not True:
            errors.append(f"safety_invariant_flags.{iid} must be true")

    failure_ids = {f["failure_id"] for f in report.get("failure_modes") or []}
    if not set(FAILURE_MODE_IDS) <= failure_ids:
        errors.append("failure_modes missing required failure modes")

    for mode in report.get("failure_modes") or []:
        if mode.get("required_response") != FAILURE_RESPONSE:
            errors.append(
                f"failure {mode.get('failure_id')} must map to BLOCK_DRY_RUN_DECISION",
            )

    graph = report.get("decision_graph_model") or {}
    terminals = set(graph.get("terminal_states") or [])
    if not set(TERMINAL_STATES) <= terminals:
        errors.append("decision_graph_model missing required terminal states")

    test_ids = {t["test_id"] for t in report.get("synthetic_test_matrix") or []}
    if not set(SYNTHETIC_TEST_CASE_IDS) <= test_ids:
        errors.append("synthetic_test_matrix missing required test cases")

    for test in report.get("synthetic_test_matrix") or []:
        if test.get("executes_at_runtime"):
            errors.append(f"synthetic test {test.get('test_id')} must not execute at runtime")

    output = report.get("output_schema") or {}
    required_fields = set(output.get("required_fields") or [])
    result_fields = {
        "decision_status",
        "prefix_match_length",
        "proposal_source_id",
        "verifier_source_id",
        "block_reason",
        "trace_complete_flag",
        "safety_gate_results",
    }
    if not result_fields <= required_fields:
        errors.append("output_schema missing L4Stage3DryRunResult required fields")

    sim = report.get("prefix_acceptance_simulation") or {}
    if sim.get("executed_at_runtime"):
        errors.append("prefix_acceptance_simulation must not execute at runtime")

    rollback = report.get("rollback_simulation_model") or {}
    if rollback.get("executed_at_runtime"):
        errors.append("rollback_simulation_model must not execute at runtime")

    return errors
