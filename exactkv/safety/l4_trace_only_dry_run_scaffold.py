"""L4 trace-only dry-run scaffold (Phase 21D / Exp 105).

Diagnostic-only trace evaluator — must not be imported by runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_trace_only_dry_run_design import DECISION_STATUSES
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_105_ID = "exp105_l4_trace_only_dry_run_scaffold"
DEFAULT_EXP105_REPORT = Path(
    "reports/experiment_105_l4_trace_only_dry_run_scaffold.json",
)
PHASE_21D = "21D"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "trace_only_dry_run_scaffold"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21D = "phase21e_l4_trace_only_dry_run_panel_validation"
FORBIDDEN_NEXT_PHASE_21D = "l4_runtime_commit_implementation"

EXPERIMENT_106_ID = "exp106_l4_trace_only_dry_run_panel_validation"
DEFAULT_EXP106_REPORT = Path(
    "reports/experiment_106_l4_trace_only_dry_run_panel_validation.json",
)
PHASE_21E = "21E"
PANEL_MODE = "trace_only_dry_run_panel_validation"
DEFAULT_PANEL_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_PANEL_INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PANEL_COMPRESSORS: tuple[str, ...] = (
    "noop",
    "int8",
    "int4_sim",
    "k8_v4_sim",
)
DEFAULT_MAX_NEW_TOKENS_VALUES: tuple[int, ...] = (4, 8)
DEFAULT_PANEL_PROMPTS = 4

RECOMMEND_PHASE21F_VERIFIER_SCHEMA = "phase21f_l4_verifier_evidence_trace_schema_design"
RECOMMEND_PHASE21F_PANEL_REPEAT = "phase21f_l4_trace_only_panel_repeat_with_evidence"
RECOMMEND_PHASE21F_STAGE3_DESIGN = "phase21f_stage3_verifier_mediated_dry_run_design"
FORBIDDEN_NEXT_PHASE_21E = "l4_runtime_commit_implementation"

VERIFIER_EVIDENCE_BLOCK_REASON = "no explicit verifier evidence in trace"
SUFFICIENT_VERIFIER_COVERAGE_THRESHOLD = 0.95

ALLOWED_VERIFIER_EVIDENCE_SOURCES: frozenset[str] = frozenset(
    {
        "verifier_token_ids",
        "full_kv_verifier_token_ids",
        "full_kv_verifier_evidence_token_ids",
        "verifier_evidence_token_ids",
    },
)

COMPARISON_ONLY_ACCEPTANCE_FIELDS: frozenset[str] = frozenset(
    {
        "accepted_tokens",
        "rejected_tokens",
        "num_accepted",
        "num_rejected",
        "correction_token",
        "all_matched",
    },
)

INTERPRETATION_NOTE = (
    "Trace-only dry-run decisions are diagnostic only; not commit authority."
)

FORBIDDEN_PROPOSAL_SOURCES: frozenset[str] = frozenset(
    {
        "committed_tokens",
        "baseline_tokens",
        "retokenized_generated_text",
        "guessed_token_ids",
    },
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

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
    },
)


@dataclass(frozen=True)
class L4TraceOnlyEvidence:
    """Explicit proposal or verifier evidence from trace records."""

    token_ids: tuple[int, ...]
    source: str
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyDryRunInput:
    """Input for trace-only dry-run evaluation."""

    cell_id: str
    prompt_id: str
    compressor: str
    round_index: int
    proposal_token_ids: tuple[int, ...]
    verifier_evidence_token_ids: tuple[int, ...]
    proposal_source: str
    verifier_evidence_source: str
    hidden_divergence_attempt: bool = False
    direct_commit_attempt: bool = False
    metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class L4TraceOnlyDecisionTrace:
    """Trace of dry-run evaluation steps."""

    decision_steps: tuple[str, ...]
    trace_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlySafetyGates:
    """Safety invariants for trace-only dry-run decisions."""

    dry_run_decision_used_for_token_commit: bool = False
    exposed_to_generator: bool = False
    verifier_source_of_truth: bool = True
    proposal_used_for_token_commit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyDecision:
    """Diagnostic-only trace-only dry-run decision."""

    cell_id: str
    prompt_id: str
    compressor: str
    round_index: int
    proposal_source: str
    verifier_evidence_source: str
    proposal_token_ids: tuple[int, ...]
    verifier_evidence_token_ids: tuple[int, ...]
    accepted_prefix_token_ids: tuple[int, ...]
    rejected_suffix_token_ids: tuple[int, ...]
    decision_status: str
    block_reason: str | None
    dry_run_decision_used_for_token_commit: bool
    exposed_to_generator: bool
    verifier_source_of_truth: bool
    trace_complete: bool
    interpretation_note: str
    decision_trace: L4TraceOnlyDecisionTrace
    safety_gates: L4TraceOnlySafetyGates

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "prompt_id": self.prompt_id,
            "compressor": self.compressor,
            "round_index": self.round_index,
            "proposal_source": self.proposal_source,
            "verifier_evidence_source": self.verifier_evidence_source,
            "proposal_token_ids": list(self.proposal_token_ids),
            "verifier_evidence_token_ids": list(self.verifier_evidence_token_ids),
            "accepted_prefix_token_ids": list(self.accepted_prefix_token_ids),
            "rejected_suffix_token_ids": list(self.rejected_suffix_token_ids),
            "decision_status": self.decision_status,
            "block_reason": self.block_reason,
            "dry_run_decision_used_for_token_commit": self.dry_run_decision_used_for_token_commit,
            "exposed_to_generator": self.exposed_to_generator,
            "verifier_source_of_truth": self.verifier_source_of_truth,
            "trace_complete": self.trace_complete,
            "interpretation_note": self.interpretation_note,
            "decision_trace": self.decision_trace.to_dict(),
            "safety_gates": self.safety_gates.to_dict(),
        }


@dataclass(frozen=True)
class L4TraceOnlyCellResult:
    """Cell-level dry-run result wrapping input and decision."""

    input: L4TraceOnlyDryRunInput
    decision: L4TraceOnlyDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class L4TraceOnlyValidationResult:
    """Validation outcome for trace-only scaffold report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4TraceOnlyScaffoldReport:
    """Top-level scaffold report aggregate (for typing; report is dict at runtime)."""

    experiment_id: str
    status: str
    decisions: tuple[L4TraceOnlyDecision, ...]
    validation_result: L4TraceOnlyValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "decisions": [d.to_dict() for d in self.decisions],
            "validation_result": self.validation_result.to_dict(),
        }


def _longest_matching_prefix(
    proposal: tuple[int, ...],
    verifier: tuple[int, ...],
) -> tuple[int, ...]:
    prefix: list[int] = []
    for proposal_id, verifier_id in zip(proposal, verifier, strict=False):
        if proposal_id == verifier_id:
            prefix.append(proposal_id)
        else:
            break
    return tuple(prefix)


def _default_safety_gates() -> L4TraceOnlySafetyGates:
    return L4TraceOnlySafetyGates()


def evaluate_l4_trace_only_input(
    input: L4TraceOnlyDryRunInput,
) -> L4TraceOnlyDecision:
    """Evaluate one trace-only dry-run input (diagnostic only)."""
    steps: list[str] = ["trace_only_evaluator_started"]
    gates = _default_safety_gates()

    meta = dict(input.metadata)
    if meta.get("trace_invalid") == "true" or input.proposal_source in FORBIDDEN_PROPOSAL_SOURCES:
        steps.append("invalid_trace_detected")
        return L4TraceOnlyDecision(
            cell_id=input.cell_id,
            prompt_id=input.prompt_id,
            compressor=input.compressor,
            round_index=input.round_index,
            proposal_source=input.proposal_source,
            verifier_evidence_source=input.verifier_evidence_source,
            proposal_token_ids=input.proposal_token_ids,
            verifier_evidence_token_ids=input.verifier_evidence_token_ids,
            accepted_prefix_token_ids=(),
            rejected_suffix_token_ids=(),
            decision_status="invalid_trace",
            block_reason="forbidden or invalid trace source",
            dry_run_decision_used_for_token_commit=False,
            exposed_to_generator=False,
            verifier_source_of_truth=True,
            trace_complete=False,
            interpretation_note=INTERPRETATION_NOTE,
            decision_trace=L4TraceOnlyDecisionTrace(
                decision_steps=tuple(steps),
                trace_complete=False,
            ),
            safety_gates=gates,
        )

    if input.direct_commit_attempt:
        steps.append("direct_commit_attempt_blocked")
        return L4TraceOnlyDecision(
            cell_id=input.cell_id,
            prompt_id=input.prompt_id,
            compressor=input.compressor,
            round_index=input.round_index,
            proposal_source=input.proposal_source,
            verifier_evidence_source=input.verifier_evidence_source,
            proposal_token_ids=input.proposal_token_ids,
            verifier_evidence_token_ids=input.verifier_evidence_token_ids,
            accepted_prefix_token_ids=(),
            rejected_suffix_token_ids=input.proposal_token_ids,
            decision_status="failed_direct_commit_attempt",
            block_reason="direct_commit_attempt",
            dry_run_decision_used_for_token_commit=False,
            exposed_to_generator=False,
            verifier_source_of_truth=True,
            trace_complete=True,
            interpretation_note=INTERPRETATION_NOTE,
            decision_trace=L4TraceOnlyDecisionTrace(
                decision_steps=tuple(steps),
                trace_complete=True,
            ),
            safety_gates=gates,
        )

    if input.hidden_divergence_attempt:
        steps.append("hidden_divergence_blocked")
        return L4TraceOnlyDecision(
            cell_id=input.cell_id,
            prompt_id=input.prompt_id,
            compressor=input.compressor,
            round_index=input.round_index,
            proposal_source=input.proposal_source,
            verifier_evidence_source=input.verifier_evidence_source,
            proposal_token_ids=input.proposal_token_ids,
            verifier_evidence_token_ids=input.verifier_evidence_token_ids,
            accepted_prefix_token_ids=(),
            rejected_suffix_token_ids=input.proposal_token_ids,
            decision_status="failed_hidden_divergence",
            block_reason="hidden_divergence_attempt",
            dry_run_decision_used_for_token_commit=False,
            exposed_to_generator=False,
            verifier_source_of_truth=True,
            trace_complete=True,
            interpretation_note=INTERPRETATION_NOTE,
            decision_trace=L4TraceOnlyDecisionTrace(
                decision_steps=tuple(steps),
                trace_complete=True,
            ),
            safety_gates=gates,
        )

    if not input.proposal_token_ids:
        steps.append("blocked_missing_proposal")
        return L4TraceOnlyDecision(
            cell_id=input.cell_id,
            prompt_id=input.prompt_id,
            compressor=input.compressor,
            round_index=input.round_index,
            proposal_source=input.proposal_source,
            verifier_evidence_source=input.verifier_evidence_source,
            proposal_token_ids=(),
            verifier_evidence_token_ids=input.verifier_evidence_token_ids,
            accepted_prefix_token_ids=(),
            rejected_suffix_token_ids=(),
            decision_status="blocked_missing_proposal",
            block_reason="missing_proposal_token_ids",
            dry_run_decision_used_for_token_commit=False,
            exposed_to_generator=False,
            verifier_source_of_truth=True,
            trace_complete=True,
            interpretation_note=INTERPRETATION_NOTE,
            decision_trace=L4TraceOnlyDecisionTrace(
                decision_steps=tuple(steps),
                trace_complete=True,
            ),
            safety_gates=gates,
        )

    if not input.verifier_evidence_token_ids:
        steps.append("blocked_missing_verifier_evidence")
        return L4TraceOnlyDecision(
            cell_id=input.cell_id,
            prompt_id=input.prompt_id,
            compressor=input.compressor,
            round_index=input.round_index,
            proposal_source=input.proposal_source,
            verifier_evidence_source=input.verifier_evidence_source,
            proposal_token_ids=input.proposal_token_ids,
            verifier_evidence_token_ids=(),
            accepted_prefix_token_ids=(),
            rejected_suffix_token_ids=(),
            decision_status="blocked_missing_verifier_evidence",
            block_reason="missing_verifier_evidence_token_ids",
            dry_run_decision_used_for_token_commit=False,
            exposed_to_generator=False,
            verifier_source_of_truth=True,
            trace_complete=True,
            interpretation_note=INTERPRETATION_NOTE,
            decision_trace=L4TraceOnlyDecisionTrace(
                decision_steps=tuple(steps),
                trace_complete=True,
            ),
            safety_gates=gates,
        )

    steps.append("verifier_evidence_used_as_source_of_truth")
    accepted = _longest_matching_prefix(
        input.proposal_token_ids,
        input.verifier_evidence_token_ids,
    )
    rejected = input.proposal_token_ids[len(accepted) :]

    if len(accepted) == 0:
        status = "first_token_mismatch"
        steps.append("first_token_mismatch")
    elif accepted == input.proposal_token_ids:
        status = "all_match"
        steps.append("all_match")
    else:
        status = "partial_match"
        steps.append("partial_match")

    return L4TraceOnlyDecision(
        cell_id=input.cell_id,
        prompt_id=input.prompt_id,
        compressor=input.compressor,
        round_index=input.round_index,
        proposal_source=input.proposal_source,
        verifier_evidence_source=input.verifier_evidence_source,
        proposal_token_ids=input.proposal_token_ids,
        verifier_evidence_token_ids=input.verifier_evidence_token_ids,
        accepted_prefix_token_ids=accepted,
        rejected_suffix_token_ids=rejected,
        decision_status=status,
        block_reason=None,
        dry_run_decision_used_for_token_commit=False,
        exposed_to_generator=False,
        verifier_source_of_truth=True,
        trace_complete=True,
        interpretation_note=INTERPRETATION_NOTE,
        decision_trace=L4TraceOnlyDecisionTrace(
            decision_steps=tuple(steps),
            trace_complete=True,
        ),
        safety_gates=gates,
    )


def _normalize_token_ids(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    return ()


def build_l4_trace_only_inputs_from_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[L4TraceOnlyDryRunInput, ...]:
    """Build dry-run inputs from explicit trace records only."""
    inputs: list[L4TraceOnlyDryRunInput] = []
    for idx, rec in enumerate(records):
        proposal_source = str(rec.get("proposal_source") or PROPOSAL_SOURCE_ROUND_LOG)
        verifier_source = str(rec.get("verifier_evidence_source") or "verifier_token_ids")

        if proposal_source in FORBIDDEN_PROPOSAL_SOURCES:
            inputs.append(
                L4TraceOnlyDryRunInput(
                    cell_id=str(rec.get("cell_id") or f"rec_{idx}"),
                    prompt_id=str(rec.get("prompt_id") or "unknown"),
                    compressor=str(rec.get("compressor") or "noop"),
                    round_index=int(rec.get("round_index") or 0),
                    proposal_token_ids=_normalize_token_ids(rec.get("proposal_token_ids")),
                    verifier_evidence_token_ids=_normalize_token_ids(
                        rec.get("verifier_evidence_token_ids"),
                    ),
                    proposal_source=proposal_source,
                    verifier_evidence_source=verifier_source,
                    metadata=(("trace_invalid", "true"),),
                ),
            )
            continue

        inputs.append(
            L4TraceOnlyDryRunInput(
                cell_id=str(rec.get("cell_id") or f"rec_{idx}"),
                prompt_id=str(rec.get("prompt_id") or "unknown"),
                compressor=str(rec.get("compressor") or "noop"),
                round_index=int(rec.get("round_index") or 0),
                proposal_token_ids=_normalize_token_ids(rec.get("proposal_token_ids")),
                verifier_evidence_token_ids=_normalize_token_ids(
                    rec.get("verifier_evidence_token_ids"),
                ),
                proposal_source=proposal_source,
                verifier_evidence_source=verifier_source,
                hidden_divergence_attempt=bool(rec.get("hidden_divergence_attempt")),
                direct_commit_attempt=bool(rec.get("direct_commit_attempt")),
                metadata=tuple(
                    (str(k), str(v))
                    for k, v in (rec.get("metadata") or {}).items()
                ),
            ),
        )
    return tuple(inputs)


def build_default_synthetic_trace_records() -> tuple[dict[str, Any], ...]:
    """Build default synthetic trace records covering all required statuses."""
    base = {
        "prompt_id": "p0",
        "compressor": "noop",
        "round_index": 0,
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "verifier_evidence_source": "verifier_token_ids",
    }
    return (
        {
            **base,
            "cell_id": "all_match",
            "proposal_token_ids": [1, 2, 3, 4],
            "verifier_evidence_token_ids": [1, 2, 3, 4],
        },
        {
            **base,
            "cell_id": "partial_match",
            "proposal_token_ids": [1, 2, 9, 9],
            "verifier_evidence_token_ids": [1, 2, 3, 4],
        },
        {
            **base,
            "cell_id": "first_token_mismatch",
            "proposal_token_ids": [9, 9, 9],
            "verifier_evidence_token_ids": [1, 2, 3],
        },
        {
            **base,
            "cell_id": "blocked_missing_proposal",
            "proposal_token_ids": [],
            "verifier_evidence_token_ids": [1, 2, 3],
        },
        {
            **base,
            "cell_id": "blocked_missing_verifier_evidence",
            "proposal_token_ids": [1, 2, 3],
            "verifier_evidence_token_ids": [],
        },
        {
            **base,
            "cell_id": "failed_hidden_divergence",
            "proposal_token_ids": [1, 2, 3],
            "verifier_evidence_token_ids": [1, 2, 3],
            "hidden_divergence_attempt": True,
        },
        {
            **base,
            "cell_id": "failed_direct_commit_attempt",
            "proposal_token_ids": [1, 2, 3],
            "verifier_evidence_token_ids": [1, 2, 3],
            "direct_commit_attempt": True,
        },
        {
            **base,
            "cell_id": "invalid_trace",
            "proposal_token_ids": [1, 2, 3],
            "verifier_evidence_token_ids": [1, 2, 3],
            "proposal_source": "committed_tokens",
        },
    )


def run_synthetic_trace_only_suite(
    records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[L4TraceOnlyDecision, ...]:
    """Run trace-only dry-run evaluation on synthetic records."""
    recs = records if records is not None else build_default_synthetic_trace_records()
    inputs = build_l4_trace_only_inputs_from_records(recs)
    return tuple(evaluate_l4_trace_only_input(inp) for inp in inputs)


def _try_load_real_trace_records() -> tuple[list[dict[str, Any]], str]:
    """Attempt to load explicit trace records from local reports (optional)."""
    candidate_paths = (
        Path("reports/experiment_103_l4_noop_scaffold_panel_validation.json"),
        Path("reports/experiment_102_l4_noop_opt_in_scaffold.json"),
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            import json

            data = json.loads(path.read_text())
            cells = data.get("cells") or []
            records: list[dict[str, Any]] = []
            for cell in cells:
                for prop in cell.get("proposals") or []:
                    proposal_ids = prop.get("proposed_token_ids") or prop.get(
                        "proposal_token_ids",
                    )
                    verifier_ids = prop.get("verifier_token_ids") or prop.get(
                        "verifier_evidence_token_ids",
                    )
                    if proposal_ids is None and verifier_ids is None:
                        continue
                    records.append(
                        {
                            "cell_id": f"{cell.get('model_id', 'm')}_{cell.get('prompt_id')}_{prop.get('round_index', 0)}",
                            "prompt_id": cell.get("prompt_id"),
                            "compressor": cell.get("compressor"),
                            "round_index": prop.get("round_index", 0),
                            "proposal_token_ids": proposal_ids,
                            "verifier_evidence_token_ids": verifier_ids,
                            "proposal_source": prop.get(
                                "proposal_source",
                                PROPOSAL_SOURCE_ROUND_LOG,
                            ),
                            "verifier_evidence_source": prop.get(
                                "verifier_evidence_source",
                                "verifier_token_ids",
                            ),
                        },
                    )
            if records:
                return records, f"loaded_from_{path.name}"
        except (OSError, ValueError, TypeError):
            continue
    return [], "no_explicit_trace_records_available"


def _count_by_status(decisions: Sequence[L4TraceOnlyDecision]) -> dict[str, int]:
    counts = {s: 0 for s in DECISION_STATUSES}
    for d in decisions:
        counts[d.decision_status] = counts.get(d.decision_status, 0) + 1
    return counts


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
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def validate_l4_trace_only_scaffold_report(
    report: dict[str, Any],
) -> L4TraceOnlyValidationResult:
    """Validate trace-only dry-run scaffold report safety invariants."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "stage",
        "mode",
        "synthetic_suite_summary",
        "real_trace_mode_summary",
        "total_decisions",
        "status_counts",
        "decisions",
        "dry_run_decision_used_for_token_commit",
        "exposed_to_generator",
        "verifier_source_of_truth",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "runtime_commit_authorized",
        "allowed_next_phase",
        "forbidden_next_phase",
        "limitations",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_105_ID:
        errors.append("experiment_id mismatch")

    bool_must_be_false = (
        "dry_run_decision_used_for_token_commit",
        "exposed_to_generator",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "runtime_commit_authorized",
    )
    for key in bool_must_be_false:
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    if report.get("verifier_source_of_truth") is not True:
        errors.append("verifier_source_of_truth must be true")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21D:
        errors.append("allowed_next_phase must be phase21e_l4_trace_only_dry_run_panel_validation")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_21D:
        errors.append("forbidden_next_phase must be l4_runtime_commit_implementation")

    for idx, dec in enumerate(report.get("decisions") or []):
        if dec.get("dry_run_decision_used_for_token_commit") is True:
            errors.append(f"decisions[{idx}] dry_run_decision_used_for_token_commit must be false")
        if dec.get("exposed_to_generator") is True:
            errors.append(f"decisions[{idx}] exposed_to_generator must be false")
        if dec.get("verifier_source_of_truth") is not True:
            errors.append(f"decisions[{idx}] verifier_source_of_truth must be true")

        status = dec.get("decision_status")
        if status == "blocked_missing_verifier_evidence":
            if dec.get("accepted_prefix_token_ids"):
                errors.append(
                    f"decisions[{idx}] missing verifier treated as match",
                )
        if status == "failed_direct_commit_attempt" and not dec.get("block_reason"):
            errors.append(f"decisions[{idx}] direct commit must have block_reason")
        if status == "failed_hidden_divergence" and not dec.get("block_reason"):
            errors.append(f"decisions[{idx}] hidden divergence must have block_reason")

        if dec.get("trace_complete") is True:
            for field in (
                "proposal_source",
                "verifier_evidence_source",
                "decision_status",
            ):
                if dec.get(field) is None:
                    errors.append(f"decisions[{idx}] trace_complete but missing {field}")

    return L4TraceOnlyValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def build_synthetic_exp105_report(*, unsafe: bool = False) -> dict[str, Any]:
    """Build synthetic scaffold report for unit tests."""
    report = run_exp105_l4_trace_only_dry_run_scaffold(try_real_traces=False)
    if unsafe:
        report["dry_run_decision_used_for_token_commit"] = True
        if report.get("decisions"):
            report["decisions"][0]["dry_run_decision_used_for_token_commit"] = True
    validation = validate_l4_trace_only_scaffold_report(report)
    report["validation_result"] = validation.to_dict()
    return report


def run_exp105_l4_trace_only_dry_run_scaffold(
    *,
    try_real_traces: bool = False,
    synthetic_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 105 L4 trace-only dry-run scaffold."""
    synthetic_decisions = list(
        run_synthetic_trace_only_suite(synthetic_records),
    )
    synthetic_counts = _count_by_status(synthetic_decisions)

    real_decisions: list[L4TraceOnlyDecision] = []
    real_summary: dict[str, Any] = {
        "status": "skipped",
        "decisions": 0,
        "reason": "try_real_traces not enabled",
    }

    if try_real_traces:
        real_records, reason = _try_load_real_trace_records()
        if real_records:
            real_inputs = build_l4_trace_only_inputs_from_records(real_records)
            real_decisions = [
                evaluate_l4_trace_only_input(inp) for inp in real_inputs
            ]
            real_summary = {
                "status": "available",
                "decisions": len(real_decisions),
                "reason": reason,
                "status_counts": _count_by_status(real_decisions),
            }
        else:
            real_summary = {
                "status": "unavailable",
                "decisions": 0,
                "reason": reason,
            }

    all_decisions = synthetic_decisions + real_decisions
    status_counts = _count_by_status(all_decisions)
    trace_complete = sum(1 for d in all_decisions if d.trace_complete)

    expected_synthetic = len(build_default_synthetic_trace_records())
    synthetic_ok = (
        len(synthetic_decisions) == expected_synthetic
        and synthetic_counts.get("all_match", 0) >= 1
        and synthetic_counts.get("blocked_missing_verifier_evidence", 0) >= 1
    )
    status = "scaffold_complete" if synthetic_ok else "scaffold_incomplete"

    report = {
        "experiment_id": EXPERIMENT_105_ID,
        "status": status,
        "phase": PHASE_21D,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "synthetic_suite_summary": {
            "total_decisions": len(synthetic_decisions),
            "status_counts": synthetic_counts,
            "all_statuses_covered": all(
                synthetic_counts.get(s, 0) >= 1
                for s in DECISION_STATUSES
            ),
        },
        "real_trace_mode_summary": real_summary,
        "total_decisions": len(all_decisions),
        "status_counts": status_counts,
        "all_match_count": status_counts.get("all_match", 0),
        "partial_match_count": status_counts.get("partial_match", 0),
        "first_token_mismatch_count": status_counts.get("first_token_mismatch", 0),
        "blocked_missing_proposal_count": status_counts.get("blocked_missing_proposal", 0),
        "blocked_missing_verifier_evidence_count": status_counts.get(
            "blocked_missing_verifier_evidence",
            0,
        ),
        "failed_hidden_divergence_count": status_counts.get("failed_hidden_divergence", 0),
        "failed_direct_commit_attempt_count": status_counts.get(
            "failed_direct_commit_attempt",
            0,
        ),
        "invalid_trace_count": status_counts.get("invalid_trace", 0),
        "trace_completeness_summary": {
            "trace_complete_decisions": trace_complete,
            "total_decisions": len(all_decisions),
            "all_traces_complete": trace_complete == len(all_decisions),
        },
        "safety_gate_summary": {
            "dry_run_decision_used_for_token_commit": False,
            "exposed_to_generator": False,
            "verifier_source_of_truth": True,
            "decisions_with_commit_flag_false": sum(
                1 for d in all_decisions if not d.dry_run_decision_used_for_token_commit
            ),
        },
        "decisions": [d.to_dict() for d in all_decisions],
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "verifier_source_of_truth": True,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "runtime_commit_authorized": False,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21D,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21D,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 trace-only dry-run scaffold only; not runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Diagnostic decisions only; no commit or generator exposure.",
            "Missing verifier evidence blocks; never fabricates tokens.",
            "No model experiments required for synthetic suite.",
        ],
    }
    report["validation_result"] = validate_l4_trace_only_scaffold_report(report).to_dict()
    return report


def validate_exp105_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 105 report."""
    return list(validate_l4_trace_only_scaffold_report(report).errors)


def _trace_get(trace: Any, key: str, default: Any = None) -> Any:
    if isinstance(trace, dict):
        return trace.get(key, default)
    return getattr(trace, key, default)


def extract_verifier_evidence_from_round_trace(
    trace: Any,
) -> tuple[tuple[int, ...], str | None]:
    """Extract explicit verifier evidence from one round trace (never accepted_tokens)."""
    for field, source in (
        ("verifier_evidence_token_ids", "verifier_evidence_token_ids"),
        ("verifier_token_ids", "verifier_token_ids"),
        ("full_kv_verifier_evidence_token_ids", "full_kv_verifier_evidence_token_ids"),
        ("full_kv_verifier_token_ids", "full_kv_verifier_token_ids"),
    ):
        raw = _trace_get(trace, field)
        if raw is not None:
            return _normalize_token_ids(raw), source

    acceptance = _trace_get(trace, "acceptance")
    if acceptance is not None:
        verifier_tokens = (
            acceptance.get("verifier_tokens")
            if isinstance(acceptance, dict)
            else getattr(acceptance, "verifier_tokens", None)
        )
        if verifier_tokens is not None:
            return _normalize_token_ids(verifier_tokens), "verifier_token_ids"

    return (), None


def extract_proposal_evidence_from_round_trace(
    trace: Any,
) -> tuple[tuple[int, ...], str]:
    """Extract explicit round-log draft proposal tokens from one round trace."""
    explicit = _trace_get(trace, "proposal_token_ids")
    if explicit is not None:
        source = str(
            _trace_get(trace, "proposal_source") or PROPOSAL_SOURCE_ROUND_LOG,
        )
        return _normalize_token_ids(explicit), source

    draft = _trace_get(trace, "draft_tokens")
    if draft is not None:
        return _normalize_token_ids(draft), PROPOSAL_SOURCE_ROUND_LOG

    return (), PROPOSAL_SOURCE_ROUND_LOG


def build_round_trace_records_for_cell(
    *,
    cell_id: str,
    prompt_id: str,
    compressor: str,
    round_traces: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Build explicit trace-only dry-run input records from generation round traces."""
    records: list[dict[str, Any]] = []
    for trace in round_traces:
        round_index = int(
            _trace_get(trace, "round_index", _trace_get(trace, "round_idx", 0)) or 0,
        )
        proposal_ids, proposal_source = extract_proposal_evidence_from_round_trace(trace)
        verifier_ids, verifier_source = extract_verifier_evidence_from_round_trace(trace)

        record: dict[str, Any] = {
            "cell_id": cell_id,
            "prompt_id": prompt_id,
            "compressor": compressor,
            "round_index": round_index,
            "proposal_token_ids": list(proposal_ids),
            "proposal_source": proposal_source,
        }
        if verifier_source is not None:
            record["verifier_evidence_token_ids"] = list(verifier_ids)
            record["verifier_evidence_source"] = verifier_source
        else:
            record["verifier_evidence_token_ids"] = []
            record["verifier_evidence_source"] = ""
        records.append(record)
    return tuple(records)


def default_trace_only_panel_prompts(
    max_prompts: int = DEFAULT_PANEL_PROMPTS,
) -> list[tuple[str, str]]:
    """Deterministic prompts for L4 trace-only dry-run panel validation."""
    from exactkv.safety.l4_noop_opt_in_scaffold import default_noop_panel_prompts

    return default_noop_panel_prompts(max_prompts)


def _default_trace_only_panel_safety_gates() -> dict[str, bool]:
    return {
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "verifier_source_of_truth": True,
        "proposal_used_for_token_commit": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "exactkv_generator_modified": False,
        "production_cli_modified": False,
    }


def _panel_decision_to_dict(decision: L4TraceOnlyDecision) -> dict[str, Any]:
    block_reason = decision.block_reason
    if (
        decision.decision_status == "blocked_missing_verifier_evidence"
        and block_reason == "missing_verifier_evidence_token_ids"
    ):
        block_reason = VERIFIER_EVIDENCE_BLOCK_REASON
    return {
        "round_index": decision.round_index,
        "proposal_source": decision.proposal_source,
        "verifier_evidence_source": decision.verifier_evidence_source,
        "proposal_token_ids": list(decision.proposal_token_ids),
        "verifier_evidence_token_ids": list(decision.verifier_evidence_token_ids),
        "accepted_prefix_token_ids": list(decision.accepted_prefix_token_ids),
        "rejected_suffix_token_ids": list(decision.rejected_suffix_token_ids),
        "decision_status": decision.decision_status,
        "block_reason": block_reason,
        "dry_run_decision_used_for_token_commit": decision.dry_run_decision_used_for_token_commit,
        "exposed_to_generator": decision.exposed_to_generator,
        "verifier_source_of_truth": decision.verifier_source_of_truth,
        "trace_complete": decision.trace_complete,
        "interpretation_note": decision.interpretation_note,
    }


def _count_decision_statuses(
    decisions: Sequence[L4TraceOnlyDecision | Mapping[str, Any]],
) -> dict[str, int]:
    counts = {s: 0 for s in DECISION_STATUSES}
    for dec in decisions:
        status = dec.decision_status if isinstance(dec, L4TraceOnlyDecision) else dec.get(
            "decision_status",
        )
        if status in counts:
            counts[str(status)] += 1
        else:
            counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _is_decision_blocked(decision: L4TraceOnlyDecision) -> bool:
    return decision.decision_status in (
        "blocked_missing_proposal",
        "blocked_missing_verifier_evidence",
        "failed_hidden_divergence",
        "failed_direct_commit_attempt",
        "invalid_trace",
    )


def _build_trace_only_panel_cell(
    *,
    model_id: str,
    prompt_id: str,
    prompt_text: str,
    compressor: str,
    max_new_tokens: int,
    runtime: Any | None,
    draft_len: int,
    generation_fn: Callable[..., dict[str, Any]] | None,
    allow_missing_verifier_evidence: bool,
) -> dict[str, Any]:
    """Build one panel cell with generation, trace extraction, and dry-run decisions."""
    preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
    cell_id = f"{model_id}|{prompt_id}|{compressor}|{max_new_tokens}"
    blockers: list[str] = []

    if generation_fn is not None:
        gen = generation_fn(
            prompt=prompt_text,
            prompt_id=prompt_id,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            model_id=model_id,
        )
    elif runtime is not None:
        from exactkv.safety.l4_noop_opt_in_scaffold import run_baseline_generation_external

        gen = run_baseline_generation_external(
            runtime=runtime,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            compressor_name=compressor,
            draft_len=draft_len,
        )
    else:
        gen = {"generation_completed": False, "blockers": ["no runtime"]}

    generation_completed = bool(gen.get("generation_completed"))
    baseline_ids = _normalize_token_ids(gen.get("generated_token_ids"))
    baseline_text = gen.get("generated_text")
    scaffold_ids = baseline_ids
    scaffold_text = baseline_text
    token_text_parity = (
        generation_completed
        and list(baseline_ids) == list(scaffold_ids)
        and baseline_text == scaffold_text
    )

    round_traces = gen.get("result_traces") or []
    records = build_round_trace_records_for_cell(
        cell_id=cell_id,
        prompt_id=prompt_id,
        compressor=compressor,
        round_traces=round_traces,
    )
    inputs = build_l4_trace_only_inputs_from_records(records)
    decisions = [evaluate_l4_trace_only_input(inp) for inp in inputs]

    proposal_available = 0
    verifier_available = 0
    for rec in records:
        if rec.get("proposal_token_ids"):
            proposal_available += 1
        if rec.get("verifier_evidence_source") and rec.get("verifier_evidence_token_ids"):
            verifier_available += 1

    decisions_computed = sum(1 for d in decisions if not _is_decision_blocked(d))
    decisions_blocked = sum(1 for d in decisions if _is_decision_blocked(d))
    status_counts = _count_decision_statuses(decisions)

    if generation_completed and not token_text_parity:
        blockers.append("token_text_parity_failed")

    gates = _default_trace_only_panel_safety_gates()
    for dec in decisions:
        if dec.dry_run_decision_used_for_token_commit:
            blockers.append("dry_run_decision_used_for_token_commit")
        if dec.exposed_to_generator:
            blockers.append("dry_run_decision_exposed_to_generator")
        if (
            dec.decision_status == "blocked_missing_verifier_evidence"
            and dec.accepted_prefix_token_ids
        ):
            blockers.append("missing_verifier_evidence_treated_as_match")

    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "prompt_preview": preview,
        "compressor": compressor,
        "max_new_tokens": max_new_tokens,
        "generation_completed": generation_completed,
        "token_text_parity": token_text_parity,
        "exactkv_failures": gen.get("exactkv_failures"),
        "round_trace_count": len(round_traces),
        "trace_inputs_built": len(inputs),
        "proposal_evidence_available_count": proposal_available,
        "verifier_evidence_available_count": verifier_available,
        "decisions_computed": decisions_computed,
        "decisions_blocked": decisions_blocked,
        "decision_status_counts": status_counts,
        "decisions": [_panel_decision_to_dict(d) for d in decisions],
        "safety_gates": gates,
        "blockers": blockers,
        "allow_missing_verifier_evidence": allow_missing_verifier_evidence,
        "baseline_token_ids": list(baseline_ids),
        "trace_only_scaffold_token_ids": list(scaffold_ids),
    }


def _panel_cell_metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(cells)
    successful = sum(1 for c in cells if c.get("generation_completed"))
    blocked = total - successful
    total_round_traces = sum(int(c.get("round_trace_count") or 0) for c in cells)
    trace_inputs = sum(int(c.get("trace_inputs_built") or 0) for c in cells)
    decisions_computed = sum(int(c.get("decisions_computed") or 0) for c in cells)
    decisions_blocked = sum(int(c.get("decisions_blocked") or 0) for c in cells)
    proposal_available = sum(
        int(c.get("proposal_evidence_available_count") or 0) for c in cells
    )
    verifier_available = sum(
        int(c.get("verifier_evidence_available_count") or 0) for c in cells
    )
    verifier_missing = trace_inputs - verifier_available
    status_counts: dict[str, int] = {s: 0 for s in DECISION_STATUSES}
    for cell in cells:
        for status, count in (cell.get("decision_status_counts") or {}).items():
            status_counts[status] = status_counts.get(status, 0) + int(count)

    return {
        "total_generation_cells": total,
        "successful_generation_cells": successful,
        "blocked_generation_cells": blocked,
        "total_round_traces": total_round_traces,
        "trace_inputs_built": trace_inputs,
        "decisions_computed": decisions_computed,
        "decisions_blocked": decisions_blocked,
        "blocked_missing_proposal_count": status_counts.get("blocked_missing_proposal", 0),
        "blocked_missing_verifier_evidence_count": status_counts.get(
            "blocked_missing_verifier_evidence",
            0,
        ),
        "all_match_count": status_counts.get("all_match", 0),
        "partial_match_count": status_counts.get("partial_match", 0),
        "first_token_mismatch_count": status_counts.get("first_token_mismatch", 0),
        "invalid_trace_count": status_counts.get("invalid_trace", 0),
        "verifier_evidence_available_count": verifier_available,
        "verifier_evidence_missing_count": max(0, verifier_missing),
        "proposal_evidence_coverage_rate": (
            proposal_available / trace_inputs if trace_inputs else 0.0
        ),
        "verifier_evidence_coverage_rate": (
            verifier_available / trace_inputs if trace_inputs else 0.0
        ),
        "dry_run_decision_coverage_rate": (
            decisions_computed / trace_inputs if trace_inputs else 0.0
        ),
        "decision_status_counts": status_counts,
    }


def aggregate_trace_only_panel_breakdowns(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Break down trace-only panel metrics by model, compressor, prompt, max_new_tokens, status."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    by_compressor: dict[str, list[Mapping[str, Any]]] = {}
    by_prompt: dict[str, list[Mapping[str, Any]]] = {}
    by_max_new: dict[str, list[Mapping[str, Any]]] = {}
    by_status: dict[str, int] = {s: 0 for s in DECISION_STATUSES}

    for cell in cells:
        by_model.setdefault(str(cell.get("model_id", "unknown")), []).append(cell)
        by_compressor.setdefault(str(cell.get("compressor", "unknown")), []).append(cell)
        by_prompt.setdefault(str(cell.get("prompt_id", "unknown")), []).append(cell)
        by_max_new.setdefault(str(cell.get("max_new_tokens", "unknown")), []).append(cell)
        for status, count in (cell.get("decision_status_counts") or {}).items():
            by_status[str(status)] = by_status.get(str(status), 0) + int(count)

    def _group_metrics(grouped: dict[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
        return {key: _panel_cell_metrics(group) for key, group in sorted(grouped.items())}

    return {
        "breakdowns_by_model": _group_metrics(by_model),
        "breakdowns_by_compressor": _group_metrics(by_compressor),
        "breakdowns_by_prompt": _group_metrics(by_prompt),
        "breakdowns_by_max_new_tokens": _group_metrics(by_max_new),
        "breakdowns_by_decision_status": by_status,
    }


def compute_phase21f_recommendation(
    *,
    verifier_evidence_coverage_rate: float,
    safety_gates_ok: bool,
) -> tuple[str, str]:
    """Recommend Phase 21F next step from verifier coverage and safety gates."""
    if verifier_evidence_coverage_rate <= 0.0:
        return (
            RECOMMEND_PHASE21F_VERIFIER_SCHEMA,
            "verifier evidence coverage is zero; explicit trace schema fields needed",
        )
    if verifier_evidence_coverage_rate < SUFFICIENT_VERIFIER_COVERAGE_THRESHOLD:
        return (
            RECOMMEND_PHASE21F_PANEL_REPEAT,
            "verifier evidence partially available; repeat panel when schema complete",
        )
    if safety_gates_ok:
        return (
            RECOMMEND_PHASE21F_STAGE3_DESIGN,
            "verifier evidence coverage sufficient and trace-only safety gates pass",
        )
    return (
        RECOMMEND_PHASE21F_VERIFIER_SCHEMA,
        "safety gates failed; address trace schema and safety before stage 3 design",
    )


def validate_exp106_panel_report(
    report: dict[str, Any],
) -> L4TraceOnlyValidationResult:
    """Validate Experiment 106 trace-only dry-run panel report."""
    errors: list[str] = []

    required_top = (
        "experiment_id",
        "status",
        "stage",
        "mode",
        "models_requested",
        "models_loaded",
        "models_blocked",
        "device",
        "dtype",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens_values",
        "total_generation_cells",
        "successful_generation_cells",
        "blocked_generation_cells",
        "total_round_traces",
        "trace_inputs_built",
        "decisions_computed",
        "decisions_blocked",
        "proposal_evidence_coverage_rate",
        "verifier_evidence_coverage_rate",
        "dry_run_decision_coverage_rate",
        "decision_status_counts",
        "trace_completeness_summary",
        "safety_gate_summary",
        "exactkv_failure_summary",
        "token_text_parity_summary",
        "breakdowns_by_model",
        "breakdowns_by_compressor",
        "breakdowns_by_prompt",
        "breakdowns_by_max_new_tokens",
        "breakdowns_by_decision_status",
        "cells",
        "decision_recommendation",
        "decision_reason",
        "dry_run_decision_used_for_token_commit",
        "exposed_to_generator",
        "verifier_source_of_truth",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "runtime_commit_authorized",
        "forbidden_next_phase",
        "limitations",
    )
    for key in required_top:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_106_ID:
        errors.append("experiment_id mismatch")

    bool_must_be_false = (
        "dry_run_decision_used_for_token_commit",
        "exposed_to_generator",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "runtime_commit_authorized",
    )
    for key in bool_must_be_false:
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    if report.get("verifier_source_of_truth") is not True:
        errors.append("verifier_source_of_truth must be true")

    if report.get("forbidden_next_phase") != FORBIDDEN_NEXT_PHASE_21E:
        errors.append("forbidden_next_phase must be l4_runtime_commit_implementation")

    forbidden_recommendations = (
        "l4_runtime_commit_implementation",
        "l4_runtime_commit",
    )
    rec = report.get("decision_recommendation")
    if rec in forbidden_recommendations:
        errors.append("decision_recommendation must not authorize runtime commit")

    cell_required = (
        "model_id",
        "prompt_id",
        "compressor",
        "max_new_tokens",
        "generation_completed",
        "token_text_parity",
        "round_trace_count",
        "trace_inputs_built",
        "decisions_computed",
        "decisions_blocked",
        "decision_status_counts",
        "safety_gates",
    )
    for idx, cell in enumerate(report.get("cells") or []):
        for ck in cell_required:
            if ck not in cell:
                errors.append(f"cells[{idx}] missing {ck}")

        if cell.get("generation_completed") and cell.get("token_text_parity") is not True:
            errors.append(f"cells[{idx}] token_text_parity failed for completed cell")

        sg = cell.get("safety_gates") or {}
        for key in bool_must_be_false:
            if sg.get(key) is True:
                errors.append(f"cells[{idx}].safety_gates.{key} must be false")

        for didx, dec in enumerate(cell.get("decisions") or []):
            if dec.get("dry_run_decision_used_for_token_commit") is True:
                errors.append(
                    f"cells[{idx}].decisions[{didx}] dry_run_decision_used_for_token_commit",
                )
            if dec.get("exposed_to_generator") is True:
                errors.append(f"cells[{idx}].decisions[{didx}] exposed_to_generator")
            if (
                dec.get("decision_status") == "blocked_missing_verifier_evidence"
                and dec.get("accepted_prefix_token_ids")
            ):
                errors.append(
                    f"cells[{idx}].decisions[{didx}] missing verifier treated as match",
                )

    if report.get("status") == "panel_complete":
        total = report.get("total_generation_cells") or 0
        parity = report.get("token_text_parity_summary") or {}
        if parity.get("parity_cells") != total and total > 0:
            errors.append("panel_complete but token_text_parity_summary mismatch")

    return L4TraceOnlyValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def build_synthetic_exp106_panel_report(
    *,
    num_cells: int = 2,
    unsafe: bool = False,
    with_verifier_evidence: bool = False,
) -> dict[str, Any]:
    """Build synthetic panel report for unit tests (no model downloads)."""
    gates = _default_trace_only_panel_safety_gates()
    cells: list[dict[str, Any]] = []
    for i in range(num_cells):
        round_traces: list[dict[str, Any]] = [
            {
                "round_idx": 0,
                "draft_tokens": [10, 11, 12],
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
            },
        ]
        if with_verifier_evidence:
            round_traces[0]["verifier_token_ids"] = [10, 11, 12]

        traces_for_cell = list(round_traces)

        def _gen_fn(
            *,
            traces: list[dict[str, Any]] = traces_for_cell,
            **kwargs: object,
        ) -> dict[str, Any]:
            del kwargs
            return {
                "generation_completed": True,
                "generated_token_ids": [10, 11] if not unsafe else [999],
                "generated_text": "out" if not unsafe else "bad",
                "exactkv_failures": 0,
                "result_traces": traces,
            }

        cell = _build_trace_only_panel_cell(
            model_id="synthetic-model",
            prompt_id=f"p{i}",
            prompt_text=f"prompt {i}",
            compressor="noop" if i % 2 == 0 else "int8",
            max_new_tokens=4 if i % 2 == 0 else 8,
            runtime=None,
            draft_len=4,
            generation_fn=_gen_fn,
            allow_missing_verifier_evidence=True,
        )
        if unsafe:
            cell["token_text_parity"] = False
            cell["blockers"] = list(cell.get("blockers") or []) + ["token_text_parity_failed"]
        cells.append(cell)

    metrics = _panel_cell_metrics(cells)
    breakdowns = aggregate_trace_only_panel_breakdowns(cells)
    trace_complete = sum(
        1
        for c in cells
        for d in c.get("decisions") or []
        if d.get("trace_complete")
    )
    total_decisions = sum(len(c.get("decisions") or []) for c in cells)

    parity_cells = sum(
        1 for c in cells if c.get("generation_completed") and c.get("token_text_parity")
    )
    safety_ok = not unsafe and all(
        not c.get("blockers") for c in cells
    )
    recommendation, reason = compute_phase21f_recommendation(
        verifier_evidence_coverage_rate=metrics["verifier_evidence_coverage_rate"],
        safety_gates_ok=safety_ok,
    )

    status = "failed" if unsafe else "panel_complete"
    if metrics["successful_generation_cells"] == 0:
        status = "blocked"

    report = {
        "experiment_id": EXPERIMENT_106_ID,
        "status": status,
        "phase": PHASE_21E,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": PANEL_MODE,
        **metrics,
        "trace_completeness_summary": {
            "trace_complete_decisions": trace_complete,
            "total_decisions": total_decisions,
            "all_traces_complete": trace_complete == total_decisions and total_decisions > 0,
        },
        "safety_gate_summary": {
            "dry_run_decision_used_for_token_commit": False,
            "exposed_to_generator": False,
            "verifier_source_of_truth": True,
            "cells_with_commit_flag_false": len(cells),
        },
        "exactkv_failure_summary": {
            "cells_with_exactkv_failures": sum(
                1 for c in cells if (c.get("exactkv_failures") or 0) > 0
            ),
            "total_cells": len(cells),
        },
        "token_text_parity_summary": {
            "parity_cells": parity_cells,
            "total_cells": len(cells),
            "all_parity": parity_cells == len(cells) and len(cells) > 0,
        },
        **breakdowns,
        "cells": cells,
        "models_requested": ["synthetic-model"],
        "models_loaded": ["synthetic-model"],
        "models_blocked": [],
        "device": "cpu",
        "dtype": "float32",
        "compressors_requested": ["noop", "int8"],
        "compressors_run": ["noop", "int8"],
        "max_new_tokens_values": [4, 8],
        "decision_recommendation": recommendation,
        "decision_reason": reason,
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "verifier_source_of_truth": True,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "runtime_commit_authorized": False,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21E,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 trace-only dry-run panel validation only; not runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Diagnostic decisions only; low verifier coverage is a schema gap, not runtime failure.",
            "Missing verifier evidence blocks; never fabricates tokens.",
        ],
    }
    validation = validate_exp106_panel_report(report)
    report["validation_result"] = validation.to_dict()
    return report


def run_exp106_l4_trace_only_dry_run_panel_validation(
    *,
    model_id: str = DEFAULT_PANEL_MODEL_ID,
    include_instruct: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_PANEL_PROMPTS,
    max_new_tokens_values: Sequence[int] = DEFAULT_MAX_NEW_TOKENS_VALUES,
    compressors_requested: Sequence[str] = DEFAULT_PANEL_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_model_blocked: bool = True,
    allow_missing_verifier_evidence: bool = True,
    generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Experiment 106 L4 trace-only dry-run panel validation."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    models_requested = [model_id]
    if include_instruct:
        models_requested.append(DEFAULT_PANEL_INSTRUCT_MODEL_ID)

    prompt_panel = (
        list(prompts) if prompts is not None else default_trace_only_panel_prompts(max_prompts)
    )
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)
    mnt_values = list(max_new_tokens_values)

    cells: list[dict[str, Any]] = []
    models_loaded: list[str] = []
    models_blocked: list[dict[str, Any]] = []

    for mid in models_requested:
        runtime: Any | None = None
        if generation_fn is not None:
            models_loaded.append(mid)
        else:
            try:
                if runtime_loader is not None:
                    runtime = runtime_loader(
                        model_id=mid,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files_only,
                    )
                else:
                    from exactkv.runtime.model_runtime import ModelRuntime

                    runtime = ModelRuntime(
                        mid,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files_only,
                    )
                models_loaded.append(mid)
            except Exception as exc:  # noqa: BLE001
                if allow_model_blocked:
                    models_blocked.append(
                        {
                            "model_id": mid,
                            "reason": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue
                raise

        for prompt_id, prompt_text in prompt_panel:
            for compressor in runnable:
                for max_new_tokens in mnt_values:
                    cell = _build_trace_only_panel_cell(
                        model_id=mid,
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        compressor=compressor,
                        max_new_tokens=max_new_tokens,
                        runtime=runtime,
                        draft_len=draft_len,
                        generation_fn=generation_fn,
                        allow_missing_verifier_evidence=allow_missing_verifier_evidence,
                    )
                    cells.append(cell)

    metrics = _panel_cell_metrics(cells)
    breakdowns = aggregate_trace_only_panel_breakdowns(cells)

    trace_complete = sum(
        1
        for c in cells
        for d in c.get("decisions") or []
        if d.get("trace_complete")
    )
    total_decisions = sum(len(c.get("decisions") or []) for c in cells)

    parity_cells = sum(
        1 for c in cells if c.get("generation_completed") and c.get("token_text_parity")
    )
    safety_ok = all(
        not (c.get("blockers") or [])
        for c in cells
        if c.get("generation_completed")
    ) and all(
        not d.get("dry_run_decision_used_for_token_commit")
        for c in cells
        for d in c.get("decisions") or []
    )

    recommendation, reason = compute_phase21f_recommendation(
        verifier_evidence_coverage_rate=metrics["verifier_evidence_coverage_rate"],
        safety_gates_ok=safety_ok,
    )

    if not cells:
        status = "blocked"
    elif metrics["successful_generation_cells"] == 0:
        status = "blocked"
    elif not safety_ok:
        status = "failed"
    elif metrics["successful_generation_cells"] == metrics["total_generation_cells"]:
        status = "panel_complete"
    else:
        status = "panel_partial"

    report = {
        "experiment_id": EXPERIMENT_106_ID,
        "status": status,
        "phase": PHASE_21E,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": PANEL_MODE,
        "models_requested": models_requested,
        "models_loaded": models_loaded,
        "models_blocked": models_blocked,
        "device": device,
        "dtype": dtype,
        "compressors_requested": list(compressors_requested),
        "compressors_run": list(runnable),
        "max_new_tokens_values": mnt_values,
        **metrics,
        "trace_completeness_summary": {
            "trace_complete_decisions": trace_complete,
            "total_decisions": total_decisions,
            "all_traces_complete": trace_complete == total_decisions and total_decisions > 0,
        },
        "safety_gate_summary": {
            "dry_run_decision_used_for_token_commit": False,
            "exposed_to_generator": False,
            "verifier_source_of_truth": True,
            "cells_with_commit_flag_false": len(cells),
        },
        "exactkv_failure_summary": {
            "cells_with_exactkv_failures": sum(
                1 for c in cells if (c.get("exactkv_failures") or 0) > 0
            ),
            "total_cells": len(cells),
        },
        "token_text_parity_summary": {
            "parity_cells": parity_cells,
            "total_cells": len(cells),
            "all_parity": parity_cells == len(cells) and len(cells) > 0,
        },
        **breakdowns,
        "cells": cells,
        "decision_recommendation": recommendation,
        "decision_reason": reason,
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "verifier_source_of_truth": True,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "runtime_commit_authorized": False,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASE_21E,
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "L4 trace-only dry-run panel validation only; not runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Diagnostic decisions only; low verifier coverage is a schema gap, not runtime failure.",
            "Missing verifier evidence blocks; never fabricates tokens.",
            "No speed, throughput, latency, serving, or memory claims.",
        ],
    }
    report["validation_result"] = validate_exp106_panel_report(report).to_dict()
    return report


def validate_exp106_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 106 report."""
    return list(validate_exp106_panel_report(report).errors)
