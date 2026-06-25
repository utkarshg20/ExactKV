"""L4 verifier evidence trace schema scaffold (Phase 21G / Exp 108).

Schema validation and dry-run conversion only — must not be wired to runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_trace_only_dry_run_scaffold import (
    L4TraceOnlyDryRunInput,
    evaluate_l4_trace_only_input,
)
from exactkv.safety.l4_verifier_evidence_trace_schema_design import (
    ALL_SCHEMA_FIELDS,
    FORBIDDEN_VERIFIER_SOURCE_NAMES,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_VERIFIER_EVIDENCE_FIELDS,
    TRACE_SCHEMA_VERSION,
    build_l4_verifier_evidence_allowed_sources,
    build_l4_verifier_evidence_forbidden_sources,
)
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_108_ID = "exp108_l4_verifier_evidence_trace_schema_scaffold"
DEFAULT_EXP108_REPORT = Path(
    "reports/experiment_108_l4_verifier_evidence_trace_schema_scaffold.json",
)
PHASE_21G = "21G"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "verifier_evidence_trace_schema_scaffold"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21G = "phase21h_l4_trace_only_dry_run_with_schema_examples"
FORBIDDEN_NEXT_PHASES_21G: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_verifier_instrumentation",
    "l4_stage3_verifier_mediated_dry_run",
)

SCAFFOLD_OUTCOME_COMPLETE = "schema_scaffold_complete"
SCAFFOLD_OUTCOME_INCOMPLETE = "schema_scaffold_incomplete"
SCAFFOLD_OUTCOME_BLOCKED = "schema_scaffold_blocked"

SCAFFOLD_OUTCOMES: tuple[str, ...] = (
    SCAFFOLD_OUTCOME_COMPLETE,
    SCAFFOLD_OUTCOME_INCOMPLETE,
    SCAFFOLD_OUTCOME_BLOCKED,
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

INTERPRETATION_NOTE = (
    "Verifier evidence schema scaffold is diagnostic only; not commit authority."
)

ALLOWED_VERIFIER_SOURCE_NAMES: frozenset[str] = frozenset(
    s.source_name for s in build_l4_verifier_evidence_allowed_sources()
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
    },
)


@dataclass(frozen=True)
class L4VerifierEvidenceTraceMetadata:
    """Metadata fields for a verifier evidence trace record."""

    round_index: int
    proposal_source: str
    proposal_token_ids: tuple[int, ...]
    trace_schema_version: str
    created_by: str
    diagnostic_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "proposal_source": self.proposal_source,
            "proposal_token_ids": list(self.proposal_token_ids),
            "trace_schema_version": self.trace_schema_version,
            "created_by": self.created_by,
            "diagnostic_only": self.diagnostic_only,
        }


@dataclass(frozen=True)
class L4VerifierEvidenceTraceRecord:
    """Immutable verifier evidence trace record (schema scaffold only)."""

    cell_id: str
    prompt_id: str
    compressor: str
    metadata: L4VerifierEvidenceTraceMetadata
    verifier_evidence_available: bool
    verifier_evidence_source: str
    verifier_evidence_token_ids: tuple[int, ...]
    verifier_evidence_text: str | None
    verifier_evidence_is_full_kv: bool
    verifier_evidence_is_authoritative: bool
    verifier_checked_proposal_token_ids: tuple[int, ...]
    verifier_matching_prefix_token_ids: tuple[int, ...]
    verifier_rejected_suffix_token_ids: tuple[int, ...]
    verifier_first_mismatch_index: int | None
    verifier_decision_status: str
    verifier_exception: str | None
    verifier_block_reason: str | None
    verifier_trace_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "prompt_id": self.prompt_id,
            "compressor": self.compressor,
            **self.metadata.to_dict(),
            "verifier_evidence_available": self.verifier_evidence_available,
            "verifier_evidence_source": self.verifier_evidence_source,
            "verifier_evidence_token_ids": list(self.verifier_evidence_token_ids),
            "verifier_evidence_text": self.verifier_evidence_text,
            "verifier_evidence_is_full_kv": self.verifier_evidence_is_full_kv,
            "verifier_evidence_is_authoritative": self.verifier_evidence_is_authoritative,
            "verifier_checked_proposal_token_ids": list(self.verifier_checked_proposal_token_ids),
            "verifier_matching_prefix_token_ids": list(self.verifier_matching_prefix_token_ids),
            "verifier_rejected_suffix_token_ids": list(self.verifier_rejected_suffix_token_ids),
            "verifier_first_mismatch_index": self.verifier_first_mismatch_index,
            "verifier_decision_status": self.verifier_decision_status,
            "verifier_exception": self.verifier_exception,
            "verifier_block_reason": self.verifier_block_reason,
            "verifier_trace_complete": self.verifier_trace_complete,
        }

    @staticmethod
    def from_mapping(
        record: Mapping[str, Any],
        *,
        cell_id: str = "record",
        prompt_id: str = "p0",
        compressor: str = "noop",
    ) -> L4VerifierEvidenceTraceRecord:
        """Build record from a mapping (e.g. synthetic example dict)."""
        meta = L4VerifierEvidenceTraceMetadata(
            round_index=int(record.get("round_index") or 0),
            proposal_source=str(record.get("proposal_source") or PROPOSAL_SOURCE_ROUND_LOG),
            proposal_token_ids=_normalize_ids(record.get("proposal_token_ids")),
            trace_schema_version=str(record.get("trace_schema_version") or TRACE_SCHEMA_VERSION),
            created_by=str(record.get("created_by") or "l4_verifier_evidence_trace_schema_scaffold"),
            diagnostic_only=bool(record.get("diagnostic_only")),
        )
        return L4VerifierEvidenceTraceRecord(
            cell_id=str(record.get("cell_id") or cell_id),
            prompt_id=str(record.get("prompt_id") or prompt_id),
            compressor=str(record.get("compressor") or compressor),
            metadata=meta,
            verifier_evidence_available=bool(record.get("verifier_evidence_available")),
            verifier_evidence_source=str(record.get("verifier_evidence_source") or ""),
            verifier_evidence_token_ids=_normalize_ids(record.get("verifier_evidence_token_ids")),
            verifier_evidence_text=record.get("verifier_evidence_text"),
            verifier_evidence_is_full_kv=bool(record.get("verifier_evidence_is_full_kv")),
            verifier_evidence_is_authoritative=bool(record.get("verifier_evidence_is_authoritative")),
            verifier_checked_proposal_token_ids=_normalize_ids(
                record.get("verifier_checked_proposal_token_ids"),
            ),
            verifier_matching_prefix_token_ids=_normalize_ids(
                record.get("verifier_matching_prefix_token_ids"),
            ),
            verifier_rejected_suffix_token_ids=_normalize_ids(
                record.get("verifier_rejected_suffix_token_ids"),
            ),
            verifier_first_mismatch_index=record.get("verifier_first_mismatch_index"),
            verifier_decision_status=str(record.get("verifier_decision_status") or ""),
            verifier_exception=record.get("verifier_exception"),
            verifier_block_reason=record.get("verifier_block_reason"),
            verifier_trace_complete=bool(record.get("verifier_trace_complete")),
        )


@dataclass(frozen=True)
class L4VerifierEvidenceValidationResult:
    """Outcome of validating one verifier evidence trace record."""

    valid: bool
    errors: tuple[str, ...]
    proposal_verifier_separated: bool
    diagnostic_only_ok: bool
    forbidden_source_rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceDryRunConversionResult:
    """Outcome of converting a trace record to trace-only dry-run input."""

    converted: bool
    dry_run_input: L4TraceOnlyDryRunInput | None
    conversion_errors: tuple[str, ...]
    decision_status: str | None
    dry_run_evaluated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "converted": self.converted,
            "dry_run_input": self.dry_run_input.to_dict() if self.dry_run_input else None,
            "conversion_errors": list(self.conversion_errors),
            "decision_status": self.decision_status,
            "dry_run_evaluated": self.dry_run_evaluated,
        }


@dataclass(frozen=True)
class L4VerifierEvidenceSchemaScaffoldValidationResult:
    """Validation outcome for scaffold report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceSchemaScaffoldReport:
    """Top-level scaffold report aggregate (typing; report is dict at runtime)."""

    experiment_id: str
    status: str
    schema_version: str
    example_results: tuple[dict[str, Any], ...]
    validation_result: L4VerifierEvidenceSchemaScaffoldValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "schema_version": self.schema_version,
            "example_results": list(self.example_results),
            "validation_result": self.validation_result.to_dict(),
        }


def _normalize_ids(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    return ()


def _record_as_mapping(
    record: Mapping[str, Any] | L4VerifierEvidenceTraceRecord,
) -> dict[str, Any]:
    if isinstance(record, L4VerifierEvidenceTraceRecord):
        return record.to_dict()
    return dict(record)


def validate_verifier_evidence_trace_record(
    record: Mapping[str, Any] | L4VerifierEvidenceTraceRecord,
) -> L4VerifierEvidenceValidationResult:
    """Validate one verifier evidence trace record (scaffold; no runtime effect)."""
    data = _record_as_mapping(record)
    errors: list[str] = []

    for field in ALL_SCHEMA_FIELDS:
        if field not in data:
            errors.append(f"missing field: {field}")

    diagnostic_only_ok = data.get("diagnostic_only") is True
    if not diagnostic_only_ok:
        errors.append("diagnostic_only must be true")

    if data.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        errors.append("trace_schema_version must match scaffold version")

    proposal_source = str(data.get("proposal_source") or "")
    verifier_source = str(data.get("verifier_evidence_source") or "")

    proposal_verifier_separated = True
    if proposal_source == verifier_source and verifier_source:
        errors.append("proposal_source and verifier_evidence_source must differ")
        proposal_verifier_separated = False

    if verifier_source in FORBIDDEN_VERIFIER_SOURCE_NAMES:
        errors.append(f"forbidden verifier_evidence_source: {verifier_source}")

    forbidden_source_rejected = verifier_source not in FORBIDDEN_VERIFIER_SOURCE_NAMES

    if verifier_source == PROPOSAL_SOURCE_ROUND_LOG:
        errors.append("round-log draft tokens are proposal evidence, not verifier evidence")
        forbidden_source_rejected = False

    if (
        verifier_source
        and verifier_source not in FORBIDDEN_VERIFIER_SOURCE_NAMES
        and verifier_source not in ALLOWED_VERIFIER_SOURCE_NAMES
        and data.get("verifier_evidence_available") is True
    ):
        errors.append(f"verifier_evidence_source not in allowed sources: {verifier_source}")

    proposal_ids = list(data.get("proposal_token_ids") or [])
    verifier_ids = list(data.get("verifier_evidence_token_ids") or [])

    if id(data.get("proposal_token_ids")) == id(data.get("verifier_evidence_token_ids")):
        errors.append("proposal_token_ids and verifier_evidence_token_ids must not alias same object")
        proposal_verifier_separated = False

    if (
        proposal_ids
        and verifier_ids
        and proposal_ids == verifier_ids
        and data.get("verifier_evidence_is_full_kv") is not True
    ):
        errors.append("proposal and verifier token IDs must not alias without full-KV mark")
        proposal_verifier_separated = False

    if data.get("verifier_evidence_available") is True:
        if not verifier_source:
            errors.append("verifier_evidence_source required when evidence available")
        if (
            not verifier_ids
            and not data.get("verifier_block_reason")
            and verifier_source != "verifier_exception_or_block_reason"
        ):
            errors.append("verifier_evidence_available but no token IDs or block reason")
        if data.get("verifier_evidence_is_full_kv") is not True:
            if verifier_source not in ("verifier_exception_or_block_reason",):
                errors.append("available verifier evidence must be full_kv authoritative")
        if data.get("verifier_evidence_is_authoritative") is not True:
            if data.get("verifier_evidence_available") is True and verifier_ids:
                errors.append("verifier evidence must be authoritative when token IDs present")

    if data.get("num_accepted") is not None and not verifier_ids:
        errors.append("cannot infer verifier evidence from num_accepted alone")
        forbidden_source_rejected = False

    if data.get("accepted_tokens") is not None and not verifier_ids:
        errors.append("cannot infer verifier evidence from accepted_tokens alone")
        forbidden_source_rejected = False

    if data.get("num_rejected") is not None and not verifier_ids:
        errors.append("cannot infer verifier evidence from num_rejected alone")

    valid = len(errors) == 0
    return L4VerifierEvidenceValidationResult(
        valid=valid,
        errors=tuple(errors),
        proposal_verifier_separated=proposal_verifier_separated and valid,
        diagnostic_only_ok=diagnostic_only_ok,
        forbidden_source_rejected=forbidden_source_rejected and valid,
    )


def convert_verifier_trace_to_l4_trace_only_input(
    record: Mapping[str, Any] | L4VerifierEvidenceTraceRecord,
    *,
    allow_invalid_for_diagnostics: bool = False,
) -> L4VerifierEvidenceDryRunConversionResult:
    """Convert a validated trace record to L4TraceOnlyDryRunInput (no fabrication)."""
    validation = validate_verifier_evidence_trace_record(record)
    if not validation.valid and not allow_invalid_for_diagnostics:
        return L4VerifierEvidenceDryRunConversionResult(
            converted=False,
            dry_run_input=None,
            conversion_errors=validation.errors,
            decision_status=None,
            dry_run_evaluated=False,
        )

    if isinstance(record, L4VerifierEvidenceTraceRecord):
        rec = record
        data = record.to_dict()
    else:
        data = dict(record)
        rec = L4VerifierEvidenceTraceRecord.from_mapping(data)

    if not validation.valid:
        return L4VerifierEvidenceDryRunConversionResult(
            converted=False,
            dry_run_input=None,
            conversion_errors=validation.errors,
            decision_status="invalid_trace",
            dry_run_evaluated=False,
        )

    proposal_ids = rec.metadata.proposal_token_ids
    if rec.verifier_evidence_available and rec.verifier_evidence_token_ids:
        verifier_ids = rec.verifier_evidence_token_ids
        verifier_source = rec.verifier_evidence_source
    else:
        verifier_ids = ()
        verifier_source = rec.verifier_evidence_source or "verifier_evidence_token_ids"

    dry_input = L4TraceOnlyDryRunInput(
        cell_id=rec.cell_id,
        prompt_id=rec.prompt_id,
        compressor=rec.compressor,
        round_index=rec.metadata.round_index,
        proposal_token_ids=proposal_ids,
        verifier_evidence_token_ids=verifier_ids,
        proposal_source=rec.metadata.proposal_source,
        verifier_evidence_source=verifier_source,
        metadata=(
            ("trace_schema_version", rec.metadata.trace_schema_version),
            ("diagnostic_only", str(rec.metadata.diagnostic_only)),
            ("created_by", rec.metadata.created_by),
        ),
    )

    decision = evaluate_l4_trace_only_input(dry_input)
    return L4VerifierEvidenceDryRunConversionResult(
        converted=True,
        dry_run_input=dry_input,
        conversion_errors=(),
        decision_status=decision.decision_status,
        dry_run_evaluated=True,
    )


def _base_metadata(*, round_index: int = 0, proposal: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "proposal_token_ids": list(proposal),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_by": "l4_verifier_evidence_trace_schema_scaffold",
        "diagnostic_only": True,
    }


def build_synthetic_schema_examples() -> tuple[dict[str, Any], ...]:
    """Build synthetic schema examples for scaffold validation."""
    shared_ids = [1, 2, 3]
    return (
        {
            "example_id": "complete_all_match_trace",
            "expected_validation": True,
            "expected_dry_run_status": "all_match",
            "record": {
                **_base_metadata(proposal=(1, 2, 3, 4)),
                "verifier_evidence_available": True,
                "verifier_evidence_source": "full_kv_verifier_output_tokens",
                "verifier_evidence_token_ids": [1, 2, 3, 4],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": True,
                "verifier_evidence_is_authoritative": True,
                "verifier_checked_proposal_token_ids": [1, 2, 3, 4],
                "verifier_matching_prefix_token_ids": [1, 2, 3, 4],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "all_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": True,
            },
        },
        {
            "example_id": "complete_partial_match_trace",
            "expected_validation": True,
            "expected_dry_run_status": "partial_match",
            "record": {
                **_base_metadata(proposal=(1, 2, 9, 9)),
                "verifier_evidence_available": True,
                "verifier_evidence_source": "verifier_matching_prefix_evidence",
                "verifier_evidence_token_ids": [1, 2, 3, 4],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": True,
                "verifier_evidence_is_authoritative": True,
                "verifier_checked_proposal_token_ids": [1, 2, 9, 9],
                "verifier_matching_prefix_token_ids": [1, 2],
                "verifier_rejected_suffix_token_ids": [9, 9],
                "verifier_first_mismatch_index": 2,
                "verifier_decision_status": "partial_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": True,
            },
        },
        {
            "example_id": "complete_first_mismatch_trace",
            "expected_validation": True,
            "expected_dry_run_status": "first_token_mismatch",
            "record": {
                **_base_metadata(proposal=(9, 9, 9)),
                "verifier_evidence_available": True,
                "verifier_evidence_source": "verifier_mismatch_evidence",
                "verifier_evidence_token_ids": [1, 2, 3],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": True,
                "verifier_evidence_is_authoritative": True,
                "verifier_checked_proposal_token_ids": [9, 9, 9],
                "verifier_matching_prefix_token_ids": [],
                "verifier_rejected_suffix_token_ids": [9, 9, 9],
                "verifier_first_mismatch_index": 0,
                "verifier_decision_status": "first_token_mismatch",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": True,
            },
        },
        {
            "example_id": "missing_verifier_evidence_trace",
            "expected_validation": True,
            "expected_dry_run_status": "blocked_missing_verifier_evidence",
            "record": {
                **_base_metadata(proposal=(1, 2, 3)),
                "verifier_evidence_available": False,
                "verifier_evidence_source": "",
                "verifier_evidence_token_ids": [],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [],
                "verifier_matching_prefix_token_ids": [],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "blocked",
                "verifier_exception": None,
                "verifier_block_reason": "no explicit verifier evidence in trace",
                "verifier_trace_complete": True,
            },
        },
        {
            "example_id": "verifier_exception_trace",
            "expected_validation": True,
            "expected_dry_run_status": "blocked_missing_verifier_evidence",
            "record": {
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
        },
        {
            "example_id": "invalid_committed_tokens_as_verifier_trace",
            "expected_validation": False,
            "expected_dry_run_status": "invalid_trace",
            "record": {
                **_base_metadata(proposal=(1, 2, 3)),
                "verifier_evidence_available": True,
                "verifier_evidence_source": "committed_token_ids_unmarked",
                "verifier_evidence_token_ids": [1, 2],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [1, 2, 3],
                "verifier_matching_prefix_token_ids": [1, 2],
                "verifier_rejected_suffix_token_ids": [3],
                "verifier_first_mismatch_index": 2,
                "verifier_decision_status": "partial_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": False,
            },
        },
        {
            "example_id": "invalid_counts_only_trace",
            "expected_validation": False,
            "expected_dry_run_status": "invalid_trace",
            "record": {
                **_base_metadata(proposal=(1, 2, 3, 4)),
                "verifier_evidence_available": True,
                "verifier_evidence_source": "accepted_token_counts_only",
                "verifier_evidence_token_ids": [],
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [1, 2, 3, 4],
                "verifier_matching_prefix_token_ids": [],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "all_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": False,
                "num_accepted": 4,
            },
        },
        {
            "example_id": "invalid_proposal_verifier_alias_trace",
            "expected_validation": False,
            "expected_dry_run_status": "invalid_trace",
            "record": {
                **_base_metadata(proposal=(1, 2, 3)),
                "proposal_source": "exactkv_round_log_draft_tokens",
                "verifier_evidence_available": True,
                "verifier_evidence_source": "exactkv_round_log_draft_tokens",
                "verifier_evidence_token_ids": shared_ids,
                "proposal_token_ids": shared_ids,
                "verifier_evidence_text": None,
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
                "verifier_checked_proposal_token_ids": [1, 2, 3],
                "verifier_matching_prefix_token_ids": [1, 2, 3],
                "verifier_rejected_suffix_token_ids": [],
                "verifier_first_mismatch_index": None,
                "verifier_decision_status": "all_match",
                "verifier_exception": None,
                "verifier_block_reason": None,
                "verifier_trace_complete": False,
            },
        },
    )


def process_schema_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, convert, and optionally dry-run evaluate one schema example."""
    example_id = str(example["example_id"])
    record = dict(example["record"])
    record["cell_id"] = example_id

    expected_validation = bool(example["expected_validation"])
    expected_dry_run = str(example["expected_dry_run_status"])

    validation = validate_verifier_evidence_trace_record(record)
    actual_validation = validation.valid

    conversion_errors: list[str] = []
    converted = False
    actual_dry_run: str | None = None

    if actual_validation:
        conv = convert_verifier_trace_to_l4_trace_only_input(record)
        converted = conv.converted
        actual_dry_run = conv.decision_status
        conversion_errors = list(conv.conversion_errors)
    elif example.get("allow_invalid_conversion"):
        conv = convert_verifier_trace_to_l4_trace_only_input(
            record,
            allow_invalid_for_diagnostics=True,
        )
        actual_dry_run = conv.decision_status

    proposal_source = str(record.get("proposal_source") or "")
    verifier_source = str(record.get("verifier_evidence_source") or "")

    return {
        "example_id": example_id,
        "expected_validation": expected_validation,
        "actual_validation": actual_validation,
        "expected_dry_run_status": expected_dry_run,
        "actual_dry_run_status": actual_dry_run,
        "converted_to_dry_run_input": converted,
        "validation_errors": list(validation.errors),
        "conversion_errors": conversion_errors,
        "source_summary": {
            "proposal_source": proposal_source,
            "verifier_evidence_source": verifier_source,
            "proposal_verifier_separated": validation.proposal_verifier_separated,
            "diagnostic_only": record.get("diagnostic_only"),
        },
        "interpretation_note": INTERPRETATION_NOTE,
    }


def evaluate_scaffold_decision(
    example_results: Sequence[Mapping[str, Any]],
) -> str:
    """Return scaffold outcome from example results."""
    if not example_results:
        return SCAFFOLD_OUTCOME_BLOCKED

    all_match_expectations = all(
        r.get("expected_validation") == r.get("actual_validation")
        and (
            not r.get("actual_validation")
            or r.get("expected_dry_run_status") == r.get("actual_dry_run_status")
        )
        for r in example_results
    )

    valid_complete = sum(
        1
        for r in example_results
        if r.get("actual_validation")
        and r.get("converted_to_dry_run_input")
    )
    blocked_ok = sum(
        1
        for r in example_results
        if r.get("actual_validation")
        and r.get("actual_dry_run_status") == "blocked_missing_verifier_evidence"
    )
    invalid_rejected = sum(1 for r in example_results if not r.get("actual_validation"))

    if (
        all_match_expectations
        and valid_complete >= 3
        and blocked_ok >= 2
        and invalid_rejected >= 3
    ):
        return SCAFFOLD_OUTCOME_COMPLETE

    if not all_match_expectations:
        return SCAFFOLD_OUTCOME_BLOCKED

    return SCAFFOLD_OUTCOME_INCOMPLETE


def validate_exp108_scaffold_report(report: dict[str, Any]) -> L4VerifierEvidenceSchemaScaffoldValidationResult:
    """Validate Experiment 108 scaffold report safety invariants."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "schema_version",
        "total_examples",
        "valid_examples",
        "invalid_examples",
        "converted_examples",
        "example_results",
        "validation_summary",
        "conversion_summary",
        "dry_run_status_summary",
        "forbidden_source_rejection_summary",
        "proposal_verifier_separation_summary",
        "diagnostic_only_summary",
        "scaffold_decision",
        "allowed_next_phase",
        "forbidden_next_phases",
        "runtime_instrumentation_authorized",
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

    if report.get("experiment_id") != EXPERIMENT_108_ID:
        errors.append("experiment_id mismatch")

    if report.get("scaffold_decision") != SCAFFOLD_OUTCOME_COMPLETE:
        errors.append("scaffold_decision must be schema_scaffold_complete")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21G:
        errors.append(
            "allowed_next_phase must be phase21h_l4_trace_only_dry_run_with_schema_examples",
        )

    forbidden = set(report.get("forbidden_next_phases") or [])
    if not set(FORBIDDEN_NEXT_PHASES_21G) <= forbidden:
        errors.append("missing required forbidden_next_phases")

    bool_must_be_false = (
        "runtime_instrumentation_authorized",
        "runtime_commit_authorized",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "generation_logic_changed",
        "production_cli_modified",
        "model_experiments_run",
    )
    for key in bool_must_be_false:
        if report.get(key) is not False:
            errors.append(f"{key} must be false")

    for idx, ex in enumerate(report.get("example_results") or []):
        if ex.get("expected_validation") != ex.get("actual_validation"):
            errors.append(f"example_results[{idx}] validation expectation mismatch")

    return L4VerifierEvidenceSchemaScaffoldValidationResult(
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
        ("runtime_verifier_instrumentation", "runtime verifier evidence instrumentation not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
        ("schema_example_dry_run_validation", "schema-example dry-run validation not run"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp108_l4_verifier_evidence_trace_schema_scaffold() -> dict[str, Any]:
    """Run Experiment 108 L4 verifier evidence trace schema scaffold."""
    examples = build_synthetic_schema_examples()
    example_results = [process_schema_example(ex) for ex in examples]

    valid_count = sum(1 for r in example_results if r["actual_validation"])
    invalid_count = len(example_results) - valid_count
    converted_count = sum(1 for r in example_results if r["converted_to_dry_run_input"])
    blocked_missing = sum(
        1
        for r in example_results
        if r.get("actual_dry_run_status") == "blocked_missing_verifier_evidence"
    )
    dry_run_evaluated = sum(
        1 for r in example_results if r.get("actual_dry_run_status") is not None
    )

    dry_run_status_counts: dict[str, int] = {}
    for r in example_results:
        status = r.get("actual_dry_run_status")
        if status:
            dry_run_status_counts[str(status)] = dry_run_status_counts.get(str(status), 0) + 1

    forbidden_rejected = sum(
        1
        for r in example_results
        if not r["actual_validation"]
        and any(
            "forbidden" in e for e in (r.get("validation_errors") or [])
        )
    )

    separation_ok = sum(
        1
        for r in example_results
        if (r.get("source_summary") or {}).get("proposal_verifier_separated") is True
        or not r["actual_validation"]
    )

    diagnostic_only_ok = sum(
        1
        for r in example_results
        if (r.get("source_summary") or {}).get("diagnostic_only") is True
    )

    scaffold_decision = evaluate_scaffold_decision(example_results)
    status = (
        "scaffold_complete"
        if scaffold_decision == SCAFFOLD_OUTCOME_COMPLETE
        else "scaffold_incomplete"
    )

    report = {
        "experiment_id": EXPERIMENT_108_ID,
        "status": status,
        "phase": PHASE_21G,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "total_examples": len(example_results),
        "valid_examples": valid_count,
        "invalid_examples": invalid_count,
        "converted_examples": converted_count,
        "blocked_missing_verifier_examples": blocked_missing,
        "dry_run_evaluated_examples": dry_run_evaluated,
        "validation_summary": {
            "valid_examples": valid_count,
            "invalid_examples": invalid_count,
            "all_expectations_met": all(
                r["expected_validation"] == r["actual_validation"] for r in example_results
            ),
        },
        "conversion_summary": {
            "converted_examples": converted_count,
            "conversion_rate": converted_count / len(example_results) if example_results else 0.0,
        },
        "dry_run_status_summary": {
            "status_counts": dry_run_status_counts,
            "evaluated_examples": dry_run_evaluated,
        },
        "forbidden_source_rejection_summary": {
            "forbidden_source_rejections": forbidden_rejected,
            "forbidden_sources_defined": len(build_l4_verifier_evidence_forbidden_sources()),
        },
        "proposal_verifier_separation_summary": {
            "examples_with_separation_ok": separation_ok,
            "total_examples": len(example_results),
        },
        "diagnostic_only_summary": {
            "examples_diagnostic_only": diagnostic_only_ok,
            "total_examples": len(example_results),
        },
        "example_results": example_results,
        "scaffold_decision": scaffold_decision,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21G,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21G),
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
            "Verifier evidence trace schema scaffold only; not runtime instrumentation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Validates and converts explicit schema records; never fabricates evidence.",
            "Missing verifier evidence blocks dry-run; proposal and verifier are separate.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }
    report["validation_result"] = validate_exp108_scaffold_report(report).to_dict()
    return report


def validate_exp108_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 108 report."""
    return list(validate_exp108_scaffold_report(report).errors)
