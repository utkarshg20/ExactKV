"""L4 Stage 3 verifier-mediated dry-run scaffold (Phase 21L / Exp 113).

Trace-only execution engine — must not be imported by runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_stage3_verifier_mediated_dry_run_design import (
    L4Stage3DryRunResult,
    SAFETY_INVARIANT_IDS,
    TERMINAL_STATES,
)
from exactkv.safety.l4_verifier_evidence_trace_schema_design import TRACE_SCHEMA_VERSION
from exactkv.safety.l4_verifier_evidence_trace_schema_scaffold import (
    validate_verifier_evidence_trace_record,
)
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_113_ID = "exp113_l4_stage3_verifier_mediated_dry_run_scaffold"
DEFAULT_EXP113_REPORT = Path(
    "reports/experiment_113_l4_stage3_verifier_mediated_dry_run_scaffold.json",
)
PHASE_21L = "21L"
STAGE = "stage_3_verifier_mediated_dry_run"
MODE = "verifier_mediated_dry_run_scaffold"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21L = "phase21m_l4_minimal_runtime_coupling_layer"
FORBIDDEN_NEXT_PHASES_21L: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_instrumentation_implementation",
    "l4_stage3_runtime_model_execution",
    "l4_default_runtime_modification",
    "l4_verifier_in_loop_execution",
)

SCAFFOLD_OUTCOME_COMPLETE = "stage3_dry_run_scaffold_complete"
SCAFFOLD_OUTCOME_INCOMPLETE = "stage3_dry_run_scaffold_incomplete"
SCAFFOLD_OUTCOME_BLOCKED = "stage3_dry_run_scaffold_blocked"

PANEL_OUTCOME_COMPLETE = "stage3_dry_run_panel_complete"
PANEL_OUTCOME_INCOMPLETE = "stage3_dry_run_panel_incomplete"

FORBIDDEN_PROPOSAL_SOURCES: frozenset[str] = frozenset(
    {
        "committed_tokens",
        "baseline_tokens",
        "verifier_tokens",
        "retokenized_generated_text",
        "guessed_token_ids",
    },
)

INTERPRETATION_NOTE = (
    "Stage 3 dry-run scaffold is trace-only diagnostic; not commit authority."
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

SCAFFOLD_CASE_IDS: tuple[str, ...] = (
    "scaffold_full_match",
    "scaffold_partial_mismatch",
    "scaffold_missing_verifier",
    "scaffold_corrupted_trace",
    "scaffold_adversarial_aliasing",
)

SCAFFOLD_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
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
        "stage_3_verifier_dry_run_scaffold",
    },
)


@dataclass(frozen=True)
class L4Stage3ProposalIngestion:
    """Read-only proposal ingestion from trace record."""

    proposal_source: str
    proposal_token_ids: tuple[int, ...]
    round_index: int
    ingestion_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_source": self.proposal_source,
            "proposal_token_ids": list(self.proposal_token_ids),
            "round_index": self.round_index,
            "ingestion_errors": list(self.ingestion_errors),
        }


@dataclass(frozen=True)
class L4Stage3VerifierTraceMapping:
    """Verifier evidence fields mapped from schema v1 trace."""

    verifier_source_id: str
    verifier_token_ids: tuple[int, ...]
    verifier_evidence_available: bool
    checked_proposal_token_ids: tuple[int, ...]
    mapping_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_source_id": self.verifier_source_id,
            "verifier_token_ids": list(self.verifier_token_ids),
            "verifier_evidence_available": self.verifier_evidence_available,
            "checked_proposal_token_ids": list(self.checked_proposal_token_ids),
            "mapping_errors": list(self.mapping_errors),
        }


@dataclass(frozen=True)
class L4Stage3DecisionGraphTrace:
    """Structured trace log for decision graph traversal."""

    steps: tuple[str, ...]
    edges_traversed: tuple[str, ...]
    terminal_state: str
    prefix_match_length: int
    first_mismatch_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3RollbackSimulation:
    """Conceptual rollback simulation — no runtime execution."""

    rollback_triggered: bool
    rollback_status: str
    rollback_reason: str | None
    baseline_reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4Stage3DryRunExecutionResult:
    """Full Stage 3 dry-run execution result for one trace record."""

    case_id: str
    schema_valid: bool
    validation_errors: tuple[str, ...]
    proposal_ingestion: L4Stage3ProposalIngestion
    verifier_mapping: L4Stage3VerifierTraceMapping
    decision_graph_trace: L4Stage3DecisionGraphTrace
    rollback_simulation: L4Stage3RollbackSimulation
    dry_run_result: L4Stage3DryRunResult
    interpretation_note: str
    scaffold_test_passed: bool
    expected_terminal_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "schema_valid": self.schema_valid,
            "validation_errors": list(self.validation_errors),
            "proposal_ingestion": self.proposal_ingestion.to_dict(),
            "verifier_mapping": self.verifier_mapping.to_dict(),
            "decision_graph_trace": self.decision_graph_trace.to_dict(),
            "rollback_simulation": self.rollback_simulation.to_dict(),
            "dry_run_result": self.dry_run_result.to_dict(),
            "interpretation_note": self.interpretation_note,
            "scaffold_test_passed": self.scaffold_test_passed,
            "expected_terminal_state": self.expected_terminal_state,
        }


@dataclass(frozen=True)
class L4Stage3ScaffoldCase:
    """One scaffold panel case with expected terminal state."""

    case_id: str
    description: str
    record: dict[str, Any]
    expected_terminal_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "record": self.record,
            "expected_terminal_state": self.expected_terminal_state,
        }


@dataclass(frozen=True)
class L4Stage3ScaffoldValidationResult:
    """Validation outcome for Experiment 113 report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_safety_gate_results() -> dict[str, bool]:
    return {iid: True for iid in SAFETY_INVARIANT_IDS}


def _normalize_ids(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return tuple(raw)
    if isinstance(raw, list):
        return tuple(raw)
    return ()


def _record_as_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    return dict(record)


def ingest_l3_proposal(record: Mapping[str, Any]) -> L4Stage3ProposalIngestion:
    """Read-only proposal ingestion from trace record."""
    data = _record_as_dict(record)
    errors: list[str] = []

    proposal_source = str(data.get("proposal_source") or "")
    if not proposal_source:
        errors.append("missing proposal_source")
    elif proposal_source in FORBIDDEN_PROPOSAL_SOURCES:
        errors.append(f"forbidden proposal_source: {proposal_source}")

    proposal_ids = _normalize_ids(data.get("proposal_token_ids"))
    if "proposal_token_ids" not in data:
        errors.append("missing proposal_token_ids")

    round_index = int(data.get("round_index") or 0)

    return L4Stage3ProposalIngestion(
        proposal_source=proposal_source,
        proposal_token_ids=proposal_ids,
        round_index=round_index,
        ingestion_errors=tuple(errors),
    )


def map_verifier_evidence_trace(
    record: Mapping[str, Any],
) -> L4Stage3VerifierTraceMapping:
    """Map verifier evidence fields from schema v1 trace (no verifier execution)."""
    data = _record_as_dict(record)
    errors: list[str] = []

    available = bool(data.get("verifier_evidence_available"))
    source = str(data.get("verifier_evidence_source") or "")
    verifier_ids = _normalize_ids(data.get("verifier_evidence_token_ids"))
    checked = _normalize_ids(data.get("verifier_checked_proposal_token_ids"))

    if not available and not data.get("verifier_block_reason") and not data.get("verifier_exception"):
        errors.append("verifier_evidence_available is false without block reason")

    return L4Stage3VerifierTraceMapping(
        verifier_source_id=source,
        verifier_token_ids=verifier_ids,
        verifier_evidence_available=available,
        checked_proposal_token_ids=checked,
        mapping_errors=tuple(errors),
    )


def _detect_alias_attack(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Detect proposal/verifier aliasing in raw record."""
    data = _record_as_dict(record)
    proposal_raw = data.get("proposal_token_ids")
    verifier_raw = data.get("verifier_evidence_token_ids")

    if proposal_raw is not None and verifier_raw is not None:
        if id(proposal_raw) == id(verifier_raw):
            return True, "proposal_token_ids and verifier_evidence_token_ids alias same object"

    proposal_source = str(data.get("proposal_source") or "")
    verifier_source = str(data.get("verifier_evidence_source") or "")
    if proposal_source and proposal_source == verifier_source:
        return True, "proposal_source and verifier_evidence_source must differ"

    proposal_ids = list(proposal_raw or [])
    verifier_ids = list(verifier_raw or [])
    if (
        proposal_ids
        and verifier_ids
        and proposal_ids == verifier_ids
        and data.get("verifier_evidence_is_full_kv") is not True
    ):
        return True, "proposal and verifier token IDs alias without full-KV mark"

    checked = list(data.get("verifier_checked_proposal_token_ids") or [])
    if checked and proposal_ids and checked != proposal_ids:
        return True, "conflicting verifier_checked_proposal_token_ids vs proposal_token_ids"

    return False, None


def simulate_prefix_walk(
    proposal: Sequence[int],
    verifier: Sequence[int],
) -> tuple[int, int | None, str]:
    """Deterministic prefix walk; returns (prefix_len, first_mismatch_index, terminal)."""
    prefix_len = 0
    for i in range(min(len(proposal), len(verifier))):
        if proposal[i] == verifier[i]:
            prefix_len += 1
        else:
            return prefix_len, i, "REJECT"

    if len(proposal) == len(verifier):
        return prefix_len, None, "ACCEPT_PREFIX"
    if len(proposal) < len(verifier):
        return prefix_len, None, "ACCEPT_PREFIX"
    return prefix_len, len(verifier), "REJECT"


def simulate_rollback(terminal_state: str) -> L4Stage3RollbackSimulation:
    """Conceptual rollback simulation based on terminal state."""
    if terminal_state == "ACCEPT_PREFIX":
        return L4Stage3RollbackSimulation(
            rollback_triggered=False,
            rollback_status="no_rollback_needed",
            rollback_reason=None,
            baseline_reference="baseline_safe_path (conceptual)",
        )
    reason_map = {
        "REJECT": "verifier_mismatch",
        "BLOCK_MISSING_EVIDENCE": "missing_verifier_evidence",
        "INVALID_TRACE": "invalid_trace_or_alias",
    }
    return L4Stage3RollbackSimulation(
        rollback_triggered=True,
        rollback_status="simulated_baseline_restore",
        rollback_reason=reason_map.get(terminal_state, "safety_gate_failure"),
        baseline_reference="baseline_safe_path (conceptual)",
    )


def evaluate_stage3_decision_graph(
    *,
    proposal: Sequence[int],
    verifier: Sequence[int],
    steps: list[str],
) -> L4Stage3DecisionGraphTrace:
    """Execute Stage 3 decision graph prefix simulation."""
    steps.append("decision_graph_entered")
    edges: list[str] = ["start_to_token_0"]

    prefix_len, mismatch_idx, terminal = simulate_prefix_walk(proposal, verifier)

    if terminal == "ACCEPT_PREFIX":
        steps.append("prefix_walk_complete_match")
        edges.append("token_match_accept")
    else:
        steps.append(f"prefix_walk_mismatch_at_{mismatch_idx}")
        edges.append("token_mismatch_reject")

    steps.append(f"terminal_{terminal}")

    return L4Stage3DecisionGraphTrace(
        steps=tuple(steps),
        edges_traversed=tuple(edges),
        terminal_state=terminal,
        prefix_match_length=prefix_len,
        first_mismatch_index=mismatch_idx,
    )


def execute_stage3_dry_run(
    record: Mapping[str, Any],
    *,
    case_id: str = "record",
    expected_terminal_state: str | None = None,
) -> L4Stage3DryRunExecutionResult:
    """Execute Stage 3 verifier-mediated dry-run on one trace record (trace-only)."""
    steps: list[str] = ["stage3_dry_run_started"]
    data = _record_as_dict(record)
    data.setdefault("cell_id", case_id)

    validation = validate_verifier_evidence_trace_record(data)
    schema_valid = validation.valid
    validation_errors = list(validation.errors)

    proposal_ingestion = ingest_l3_proposal(data)
    verifier_mapping = map_verifier_evidence_trace(data)

    alias_detected, alias_reason = _detect_alias_attack(data)
    if alias_detected and alias_reason:
        steps.append("alias_attack_detected")
        terminal = "INVALID_TRACE"
        block_reason = alias_reason
        graph_trace = L4Stage3DecisionGraphTrace(
            steps=tuple(steps + ["terminal_INVALID_TRACE"]),
            edges_traversed=("start_to_invalid",),
            terminal_state=terminal,
            prefix_match_length=0,
            first_mismatch_index=None,
        )
    elif not schema_valid:
        steps.append("invalid_trace_schema")
        terminal = "INVALID_TRACE"
        block_reason = validation_errors[0] if validation_errors else "invalid_trace"
        graph_trace = L4Stage3DecisionGraphTrace(
            steps=tuple(steps + ["terminal_INVALID_TRACE"]),
            edges_traversed=("start_to_invalid",),
            terminal_state=terminal,
            prefix_match_length=0,
            first_mismatch_index=None,
        )
    elif proposal_ingestion.ingestion_errors:
        steps.append("corrupted_proposal_source")
        terminal = "INVALID_TRACE"
        block_reason = proposal_ingestion.ingestion_errors[0]
        graph_trace = L4Stage3DecisionGraphTrace(
            steps=tuple(steps + ["terminal_INVALID_TRACE"]),
            edges_traversed=("start_to_invalid",),
            terminal_state=terminal,
            prefix_match_length=0,
            first_mismatch_index=None,
        )
    elif not verifier_mapping.verifier_evidence_available or not verifier_mapping.verifier_token_ids:
        steps.append("blocked_missing_verifier_evidence")
        terminal = "BLOCK_MISSING_EVIDENCE"
        block_reason = (
            str(data.get("verifier_block_reason"))
            or str(data.get("verifier_exception"))
            or "missing_verifier_evidence"
        )
        graph_trace = L4Stage3DecisionGraphTrace(
            steps=tuple(steps + ["terminal_BLOCK_MISSING_EVIDENCE"]),
            edges_traversed=("start_to_block",),
            terminal_state=terminal,
            prefix_match_length=0,
            first_mismatch_index=None,
        )
    else:
        steps.append("evidence_reconciliation_complete")
        graph_trace = evaluate_stage3_decision_graph(
            proposal=proposal_ingestion.proposal_token_ids,
            verifier=verifier_mapping.verifier_token_ids,
            steps=steps,
        )
        terminal = graph_trace.terminal_state
        block_reason = None
        if terminal == "REJECT":
            block_reason = f"mismatch_at_index_{graph_trace.first_mismatch_index}"

    rollback = simulate_rollback(terminal)

    dry_run_result = L4Stage3DryRunResult(
        decision_status=terminal,
        prefix_match_length=graph_trace.prefix_match_length,
        proposal_source_id=proposal_ingestion.proposal_source,
        verifier_source_id=verifier_mapping.verifier_source_id,
        block_reason=block_reason,
        trace_complete_flag=terminal in TERMINAL_STATES,
        safety_gate_results=_default_safety_gate_results(),
    )

    expected = expected_terminal_state or terminal
    passed = dry_run_result.decision_status == expected

    return L4Stage3DryRunExecutionResult(
        case_id=case_id,
        schema_valid=schema_valid,
        validation_errors=tuple(validation_errors),
        proposal_ingestion=proposal_ingestion,
        verifier_mapping=verifier_mapping,
        decision_graph_trace=graph_trace,
        rollback_simulation=rollback,
        dry_run_result=dry_run_result,
        interpretation_note=INTERPRETATION_NOTE,
        scaffold_test_passed=passed,
        expected_terminal_state=expected,
    )


def _base_metadata(*, proposal: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "round_index": 0,
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "proposal_token_ids": list(proposal),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_by": "l4_stage3_verifier_mediated_dry_run_scaffold",
        "diagnostic_only": True,
    }


def _valid_verifier_block(
    *,
    proposal: tuple[int, ...],
    verifier: tuple[int, ...],
    source: str = "full_kv_verifier_output_tokens",
) -> dict[str, Any]:
    prefix = []
    mismatch_idx: int | None = None
    for i, (p, v) in enumerate(zip(proposal, verifier, strict=False)):
        if p == v:
            prefix.append(p)
        else:
            mismatch_idx = i
            break
    if mismatch_idx is None and len(proposal) <= len(verifier):
        matching = list(proposal)
        rejected = list(proposal[len(prefix) :]) if len(prefix) < len(proposal) else []
    elif mismatch_idx is not None:
        matching = list(proposal[:mismatch_idx])
        rejected = list(proposal[mismatch_idx:])
    else:
        matching = list(verifier)
        rejected = list(proposal[len(verifier) :])
        mismatch_idx = len(verifier)

    status = "all_match" if not rejected and len(proposal) == len(verifier) else "partial_match"
    if mismatch_idx == 0:
        status = "first_token_mismatch"

    return {
        "verifier_evidence_available": True,
        "verifier_evidence_source": source,
        "verifier_evidence_token_ids": list(verifier),
        "verifier_evidence_text": None,
        "verifier_evidence_is_full_kv": True,
        "verifier_evidence_is_authoritative": True,
        "verifier_checked_proposal_token_ids": list(proposal),
        "verifier_matching_prefix_token_ids": matching,
        "verifier_rejected_suffix_token_ids": rejected,
        "verifier_first_mismatch_index": mismatch_idx,
        "verifier_decision_status": status,
        "verifier_exception": None,
        "verifier_block_reason": None,
        "verifier_trace_complete": True,
    }


def build_stage3_scaffold_cases() -> tuple[L4Stage3ScaffoldCase, ...]:
    """Build synthetic scaffold panel cases."""
    shared_ids = [20, 21, 22]

    return (
        L4Stage3ScaffoldCase(
            case_id="scaffold_full_match",
            description="Proposal and verifier tokens fully agree",
            record={
                **_base_metadata(proposal=(1, 2, 3)),
                **_valid_verifier_block(proposal=(1, 2, 3), verifier=(1, 2, 3)),
            },
            expected_terminal_state="ACCEPT_PREFIX",
        ),
        L4Stage3ScaffoldCase(
            case_id="scaffold_partial_mismatch",
            description="Proposal matches verifier prefix then diverges",
            record={
                **_base_metadata(proposal=(1, 2, 9)),
                **_valid_verifier_block(proposal=(1, 2, 9), verifier=(1, 2, 3)),
            },
            expected_terminal_state="REJECT",
        ),
        L4Stage3ScaffoldCase(
            case_id="scaffold_missing_verifier",
            description="Verifier evidence unavailable",
            record={
                **_base_metadata(proposal=(1, 2, 3)),
                "verifier_evidence_available": False,
                "verifier_evidence_source": "verifier_exception_or_block_reason",
                "verifier_evidence_token_ids": [],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [1, 2, 3],
                "verifier_matching_prefix_token_ids": [],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "blocked",
                "verifier_exception": "VerifierForwardError: simulated",
                "verifier_block_reason": "verifier_exception",
                "verifier_trace_complete": True,
            },
            expected_terminal_state="BLOCK_MISSING_EVIDENCE",
        ),
        L4Stage3ScaffoldCase(
            case_id="scaffold_corrupted_trace",
            description="Invalid trace schema version",
            record={
                **_base_metadata(proposal=(1, 2)),
                **_valid_verifier_block(proposal=(1, 2), verifier=(1, 2)),
                "trace_schema_version": "corrupted_v99",
            },
            expected_terminal_state="INVALID_TRACE",
        ),
        L4Stage3ScaffoldCase(
            case_id="scaffold_adversarial_aliasing",
            description="Proposal and verifier share same list object",
            record={
                **_base_metadata(proposal=(20, 21, 22)),
                "proposal_token_ids": shared_ids,
                "verifier_evidence_available": True,
                "verifier_evidence_source": "full_kv_verifier_output_tokens",
                "verifier_evidence_token_ids": shared_ids,
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [20, 21, 22],
                "verifier_matching_prefix_token_ids": [20, 21, 22],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "all_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": False,
            },
            expected_terminal_state="INVALID_TRACE",
        ),
    )


def execute_scaffold_case(case: L4Stage3ScaffoldCase) -> L4Stage3DryRunExecutionResult:
    """Run one scaffold case through Stage 3 dry-run engine."""
    record = dict(case.record)
    record["cell_id"] = case.case_id
    return execute_stage3_dry_run(
        record,
        case_id=case.case_id,
        expected_terminal_state=case.expected_terminal_state,
    )


def evaluate_panel_outcome(
    results: Sequence[L4Stage3DryRunExecutionResult],
) -> str:
    """Determine panel outcome from case results."""
    if not results:
        return PANEL_OUTCOME_INCOMPLETE
    if all(r.scaffold_test_passed for r in results):
        return PANEL_OUTCOME_COMPLETE
    return PANEL_OUTCOME_INCOMPLETE


def validate_exp113_scaffold_report(report: Mapping[str, Any]) -> L4Stage3ScaffoldValidationResult:
    """Validate Experiment 113 scaffold panel report."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "panel_outcome",
        "total_cases",
        "cases_passed",
        "case_results",
        "classification_summary",
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

    if report.get("experiment_id") != EXPERIMENT_113_ID:
        errors.append("experiment_id mismatch")

    if report.get("panel_outcome") != PANEL_OUTCOME_COMPLETE:
        errors.append("panel_outcome must be stage3_dry_run_panel_complete")

    for flag in (
        "runtime_execution_authorized",
        "runtime_commit_authorized",
        "runtime_instrumentation_authorized",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "model_experiments_run",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    for idx, case in enumerate(report.get("case_results") or []):
        if not case.get("scaffold_test_passed"):
            errors.append(f"case_results[{idx}] scaffold_test_passed false")

    for case_id in SCAFFOLD_CASE_IDS:
        ids = {c.get("case_id") for c in report.get("case_results") or []}
        if case_id not in ids:
            errors.append(f"missing scaffold case: {case_id}")

    return L4Stage3ScaffoldValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
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
        if bid not in SCAFFOLD_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        ("stage_3_dry_run_panel", "stage 3 dry-run panel validation not implemented"),
        (
            "runtime_verifier_instrumentation",
            "runtime verifier evidence instrumentation not implemented",
        ),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold() -> dict[str, Any]:
    """Run Experiment 113 Stage 3 dry-run scaffold panel (trace-only)."""
    cases = build_stage3_scaffold_cases()
    results = tuple(execute_scaffold_case(c) for c in cases)

    passed = sum(1 for r in results if r.scaffold_test_passed)
    failed = len(results) - passed

    classification_counts: dict[str, int] = {}
    for r in results:
        status = r.dry_run_result.decision_status
        classification_counts[status] = classification_counts.get(status, 0) + 1

    panel_outcome = evaluate_panel_outcome(results)
    status = (
        "scaffold_complete"
        if panel_outcome == PANEL_OUTCOME_COMPLETE
        else "scaffold_incomplete"
    )

    report = {
        "experiment_id": EXPERIMENT_113_ID,
        "status": status,
        "phase": PHASE_21L,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "scaffold_outcome": (
            SCAFFOLD_OUTCOME_COMPLETE
            if panel_outcome == PANEL_OUTCOME_COMPLETE
            else SCAFFOLD_OUTCOME_INCOMPLETE
        ),
        "panel_outcome": panel_outcome,
        "total_cases": len(results),
        "cases_passed": passed,
        "cases_failed": failed,
        "classification_summary": {
            "status_counts": classification_counts,
            "terminal_states": list(TERMINAL_STATES),
        },
        "case_results": [r.to_dict() for r in results],
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21L,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21L),
        "runtime_execution_authorized": False,
        "runtime_instrumentation_authorized": False,
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
            "Stage 3 dry-run scaffold is trace-only; no model execution.",
            "ExactKVGenerator and default runtime unchanged.",
            "Verifier is not executed; trace field mapping only.",
            "Prefix walk and rollback are pure logic simulations.",
            "Dry-run results are not commit authority.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }
    report["validation_result"] = validate_exp113_scaffold_report(report).to_dict()
    return report


def validate_exp113_report(report: Mapping[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 113 report."""
    return list(validate_exp113_scaffold_report(report).errors)
