"""L4 verifier evidence trace schema design (Phase 21F / Exp 107).

Schema design specification only — must not be wired to runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_107_ID = "exp107_l4_verifier_evidence_trace_schema_design"
DEFAULT_EXP107_REPORT = Path(
    "reports/experiment_107_l4_verifier_evidence_trace_schema_design.json",
)
PHASE_21F = "21F"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "verifier_evidence_trace_schema_design"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"
TRACE_SCHEMA_VERSION = "l4_verifier_evidence_v1"

RECOMMENDED_NEXT_PHASE_21F = "phase21g_l4_verifier_evidence_trace_schema_scaffold"
FORBIDDEN_NEXT_PHASES_21F: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_verifier_instrumentation",
    "l4_stage3_verifier_mediated_dry_run",
)

DESIGN_OUTCOME_COMPLETE = "verifier_evidence_trace_schema_design_complete"
DESIGN_OUTCOME_INCOMPLETE = "verifier_evidence_trace_schema_design_incomplete"
DESIGN_OUTCOME_BLOCKED = "verifier_evidence_trace_schema_design_blocked"

DESIGN_OUTCOMES: tuple[str, ...] = (
    DESIGN_OUTCOME_COMPLETE,
    DESIGN_OUTCOME_INCOMPLETE,
    DESIGN_OUTCOME_BLOCKED,
)

REQUIRED_VERIFIER_EVIDENCE_FIELDS: tuple[str, ...] = (
    "verifier_evidence_available",
    "verifier_evidence_source",
    "verifier_evidence_token_ids",
    "verifier_evidence_text",
    "verifier_evidence_is_full_kv",
    "verifier_evidence_is_authoritative",
    "verifier_checked_proposal_token_ids",
    "verifier_matching_prefix_token_ids",
    "verifier_rejected_suffix_token_ids",
    "verifier_first_mismatch_index",
    "verifier_decision_status",
    "verifier_exception",
    "verifier_block_reason",
    "verifier_trace_complete",
)

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "round_index",
    "proposal_source",
    "proposal_token_ids",
    "trace_schema_version",
    "created_by",
    "diagnostic_only",
)

ALL_SCHEMA_FIELDS: tuple[str, ...] = REQUIRED_VERIFIER_EVIDENCE_FIELDS + REQUIRED_METADATA_FIELDS

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

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
    },
)


@dataclass(frozen=True)
class L4VerifierEvidenceField:
    """One field in the future verifier evidence trace schema."""

    field_name: str
    field_type: str
    required: bool
    description: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceSourceRule:
    """Allowed verifier evidence source with provenance and validation rules."""

    source_name: str
    required_fields: tuple[str, ...]
    provenance_rule: str
    validation_rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceForbiddenSource:
    """Forbidden verifier evidence source."""

    source_name: str
    reason: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceValidationRule:
    """Validation rule for verifier evidence trace records."""

    rule_id: str
    description: str
    pass_condition: str
    fail_condition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4VerifierEvidenceTraceExample:
    """Synthetic trace example for schema validation (design only)."""

    example_id: str
    description: str
    trace_record: dict[str, Any]
    expected_validation_passes: bool
    expected_dry_run_decision_status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "description": self.description,
            "trace_record": self.trace_record,
            "expected_validation_passes": self.expected_validation_passes,
            "expected_dry_run_decision_status": self.expected_dry_run_decision_status,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class L4VerifierEvidenceTraceSchemaDecision:
    """Design review outcome for Phase 21F."""

    outcome: str
    schema_scaffold_authorized: bool
    runtime_instrumentation_authorized: bool
    runtime_commit_authorized: bool
    allowed_next_phase: str
    forbidden_next_phases: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "forbidden_next_phases": list(self.forbidden_next_phases),
        }


@dataclass(frozen=True)
class L4VerifierEvidenceTraceSchemaDesign:
    """Top-level L4 verifier evidence trace schema design aggregate."""

    design_id: str
    safety_level: str
    stage: str
    mode: str
    schema_version: str
    schema_fields: tuple[L4VerifierEvidenceField, ...]
    allowed_sources: tuple[L4VerifierEvidenceSourceRule, ...]
    forbidden_sources: tuple[L4VerifierEvidenceForbiddenSource, ...]
    validation_rules: tuple[L4VerifierEvidenceValidationRule, ...]
    trace_examples: tuple[L4VerifierEvidenceTraceExample, ...]
    decision: L4VerifierEvidenceTraceSchemaDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "safety_level": self.safety_level,
            "stage": self.stage,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "schema_fields": [f.to_dict() for f in self.schema_fields],
            "allowed_sources": [s.to_dict() for s in self.allowed_sources],
            "forbidden_sources": [s.to_dict() for s in self.forbidden_sources],
            "validation_rules": [r.to_dict() for r in self.validation_rules],
            "trace_examples": [e.to_dict() for e in self.trace_examples],
            "decision": self.decision.to_dict(),
        }


def build_l4_verifier_evidence_schema_fields() -> tuple[L4VerifierEvidenceField, ...]:
    """Build required verifier evidence and metadata field definitions."""
    field_specs: list[tuple[str, str, bool, str, str]] = [
        ("verifier_evidence_available", "bool", True, "Whether explicit verifier evidence is present.", "verifier"),
        ("verifier_evidence_source", "str", True, "Named source of verifier evidence (not proposal).", "verifier"),
        ("verifier_evidence_token_ids", "tuple[int,...]", True, "Explicit verifier token IDs; immutable.", "verifier"),
        ("verifier_evidence_text", "str | None", False, "Optional decoded verifier text; not retokenized.", "verifier"),
        ("verifier_evidence_is_full_kv", "bool", True, "True when evidence is full-KV verifier output.", "verifier"),
        ("verifier_evidence_is_authoritative", "bool", True, "True when evidence is authoritative for dry-run.", "verifier"),
        ("verifier_checked_proposal_token_ids", "tuple[int,...]", True, "Proposal tokens verifier compared.", "verifier"),
        ("verifier_matching_prefix_token_ids", "tuple[int,...]", True, "Longest matching prefix from verifier.", "verifier"),
        ("verifier_rejected_suffix_token_ids", "tuple[int,...]", True, "Proposal suffix rejected by verifier.", "verifier"),
        ("verifier_first_mismatch_index", "int | None", False, "Index of first mismatch; None if all match.", "verifier"),
        ("verifier_decision_status", "str", True, "Verifier-side decision status for diagnostics.", "verifier"),
        ("verifier_exception", "str | None", False, "Verifier exception message if any.", "verifier"),
        ("verifier_block_reason", "str | None", False, "Block reason when evidence unavailable.", "verifier"),
        ("verifier_trace_complete", "bool", True, "Whether verifier trace fields are complete.", "verifier"),
        ("round_index", "int", True, "Round index in generation trace.", "metadata"),
        ("proposal_source", "str", True, "Proposal provenance (e.g. round-log draft).", "metadata"),
        ("proposal_token_ids", "tuple[int,...]", True, "Explicit proposal token IDs.", "metadata"),
        ("trace_schema_version", "str", True, "Schema version string.", "metadata"),
        ("created_by", "str", True, "Producer of trace record (diagnostic pipeline).", "metadata"),
        ("diagnostic_only", "bool", True, "Must be true; not commit authority.", "metadata"),
    ]
    return tuple(
        L4VerifierEvidenceField(
            field_name=name,
            field_type=ftype,
            required=req,
            description=desc,
            category=cat,
        )
        for name, ftype, req, desc, cat in field_specs
    )


def build_l4_verifier_evidence_allowed_sources() -> tuple[L4VerifierEvidenceSourceRule, ...]:
    return (
        L4VerifierEvidenceSourceRule(
            source_name="full_kv_verifier_output_tokens",
            required_fields=(
                "verifier_evidence_token_ids",
                "verifier_evidence_is_full_kv",
                "verifier_evidence_is_authoritative",
                "verifier_evidence_source",
            ),
            provenance_rule="Tokens from explicit full-KV verifier forward pass; not committed output.",
            validation_rule="verifier_evidence_is_full_kv=true and token IDs non-empty or explicitly blocked.",
        ),
        L4VerifierEvidenceSourceRule(
            source_name="verifier_comparison_output_for_proposal",
            required_fields=(
                "verifier_checked_proposal_token_ids",
                "verifier_evidence_token_ids",
                "verifier_evidence_source",
            ),
            provenance_rule="Verifier compared proposal tokens against full-KV predictions.",
            validation_rule="proposal and verifier token fields distinct; checked proposal matches proposal_token_ids.",
        ),
        L4VerifierEvidenceSourceRule(
            source_name="verifier_matching_prefix_evidence",
            required_fields=(
                "verifier_matching_prefix_token_ids",
                "verifier_rejected_suffix_token_ids",
                "verifier_first_mismatch_index",
            ),
            provenance_rule="Prefix/suffix split computed by verifier comparison; not from accepted counts.",
            validation_rule="matching_prefix + rejected_suffix reconstruct checked proposal when complete.",
        ),
        L4VerifierEvidenceSourceRule(
            source_name="verifier_mismatch_evidence",
            required_fields=(
                "verifier_first_mismatch_index",
                "verifier_rejected_suffix_token_ids",
                "verifier_decision_status",
            ),
            provenance_rule="Explicit mismatch index and rejected suffix from verifier.",
            validation_rule="first_mismatch_index set when decision_status indicates mismatch.",
        ),
        L4VerifierEvidenceSourceRule(
            source_name="verifier_exception_or_block_reason",
            required_fields=(
                "verifier_evidence_available",
                "verifier_block_reason",
            ),
            provenance_rule="Verifier failed or evidence withheld; dry-run must block.",
            validation_rule="verifier_evidence_available=false with non-empty verifier_block_reason.",
        ),
    )


def build_l4_verifier_evidence_forbidden_sources() -> tuple[L4VerifierEvidenceForbiddenSource, ...]:
    specs: list[tuple[str, str, str]] = [
        (
            "committed_token_ids_unmarked",
            "Committed token IDs used as verifier evidence without full-KV marking.",
            "critical",
        ),
        (
            "accepted_token_counts_only",
            "Accepted token counts alone without explicit verifier token IDs.",
            "critical",
        ),
        (
            "rejected_corrected_counts_only",
            "Rejected/corrected counts alone without verifier token evidence.",
            "critical",
        ),
        (
            "baseline_generated_tokens",
            "Baseline full-greedy tokens used as verifier evidence.",
            "critical",
        ),
        (
            "retokenized_generated_text",
            "Retokenized generated text used as verifier evidence.",
            "critical",
        ),
        (
            "guessed_token_ids",
            "Guessed or inferred token IDs without verifier provenance.",
            "critical",
        ),
        (
            "compressed_draft_tokens",
            "Compressed draft tokens used as verifier evidence.",
            "high",
        ),
        (
            "shadow_top1_proposal_tokens",
            "Shadow top-1 proposal tokens used as verifier evidence.",
            "high",
        ),
        (
            "round_log_proposal_tokens_as_verifier",
            "Round-log draft tokens used as verifier evidence (valid as proposal only).",
            "critical",
        ),
    ]
    return tuple(
        L4VerifierEvidenceForbiddenSource(
            source_name=name,
            reason=reason,
            severity=sev,
        )
        for name, reason, sev in specs
    )


def build_l4_verifier_evidence_validation_rules() -> tuple[L4VerifierEvidenceValidationRule, ...]:
    specs: list[tuple[str, str, str, str]] = [
        (
            "explicit_evidence_required",
            "Verifier evidence must be explicit in trace fields.",
            "verifier_evidence_available set; source recorded when available.",
            "Implicit or inferred verifier evidence used.",
        ),
        (
            "full_kv_or_blocked",
            "Verifier evidence must be full-KV authoritative or explicitly blocked.",
            "verifier_evidence_is_full_kv=true with authoritative flag, or block_reason set.",
            "Partial or non-authoritative evidence treated as verifier truth.",
        ),
        (
            "proposal_verifier_distinct",
            "Proposal and verifier evidence must be separate fields.",
            "proposal_token_ids and verifier_evidence_token_ids are distinct field names.",
            "Same field or source used for both proposal and verifier.",
        ),
        (
            "missing_evidence_blocks",
            "Missing verifier evidence must block dry-run decision.",
            "verifier_evidence_available=false yields blocked_missing_verifier_evidence.",
            "Missing evidence treated as match.",
        ),
        (
            "no_committed_output_inference",
            "Verifier evidence cannot be inferred from committed output.",
            "No verifier field populated solely from committed tokens without full-KV mark.",
            "Committed output used as verifier without explicit marking.",
        ),
        (
            "no_accepted_count_inference",
            "Verifier evidence cannot be inferred from accepted counts alone.",
            "num_accepted or accepted_tokens not used as verifier token IDs.",
            "Accepted count alone implies verifier agreement.",
        ),
        (
            "immutable_token_ids",
            "Verifier evidence token IDs must be immutable/tuple-like.",
            "verifier_evidence_token_ids stored as fixed sequence.",
            "Mutable or recomputed token list without provenance.",
        ),
        (
            "source_recorded",
            "Verifier evidence source must be recorded.",
            "verifier_evidence_source non-empty when evidence available.",
            "Verifier tokens present without source provenance.",
        ),
        (
            "schema_version_recorded",
            "Trace schema version must be recorded.",
            "trace_schema_version matches design version.",
            "Missing or unknown schema version.",
        ),
        (
            "diagnostic_only_flag",
            "diagnostic_only flag must be true.",
            "diagnostic_only=true on every trace record.",
            "diagnostic_only false or absent.",
        ),
    ]
    return tuple(
        L4VerifierEvidenceValidationRule(
            rule_id=rid,
            description=desc,
            pass_condition=pass_cond,
            fail_condition=fail_cond,
        )
        for rid, desc, pass_cond, fail_cond in specs
    )


def _base_metadata(*, round_index: int = 0, proposal: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "proposal_token_ids": list(proposal),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_by": "l4_verifier_evidence_trace_schema_design",
        "diagnostic_only": True,
    }


def build_l4_verifier_evidence_trace_examples() -> tuple[L4VerifierEvidenceTraceExample, ...]:
    """Build design trace examples with expected validation and dry-run outcomes."""
    return (
        L4VerifierEvidenceTraceExample(
            example_id="complete_all_match_trace",
            description="Full verifier evidence; proposal matches verifier fully.",
            trace_record={
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
            expected_validation_passes=True,
            expected_dry_run_decision_status="all_match",
            notes="Complete trace enables trace-only all_match dry-run.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="complete_partial_match_trace",
            description="Verifier accepts prefix of proposal.",
            trace_record={
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
            expected_validation_passes=True,
            expected_dry_run_decision_status="partial_match",
            notes="Prefix match from explicit verifier fields.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="complete_first_mismatch_trace",
            description="First proposal token mismatches verifier.",
            trace_record={
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
            expected_validation_passes=True,
            expected_dry_run_decision_status="first_token_mismatch",
            notes="Empty matching prefix when first token mismatches.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="missing_verifier_evidence_trace",
            description="Proposal present; verifier evidence absent.",
            trace_record={
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
            expected_validation_passes=True,
            expected_dry_run_decision_status="blocked_missing_verifier_evidence",
            notes="Matches Phase 21E panel gap; dry-run blocks.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="verifier_exception_trace",
            description="Verifier raised exception; evidence blocked.",
            trace_record={
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
            expected_validation_passes=True,
            expected_dry_run_decision_status="blocked_missing_verifier_evidence",
            notes="Exception trace is valid schema but blocks dry-run.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="invalid_committed_tokens_as_verifier_trace",
            description="Committed tokens used as verifier without full-KV marking.",
            trace_record={
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
            expected_validation_passes=False,
            expected_dry_run_decision_status="invalid_trace",
            notes="Forbidden source; must fail schema validation.",
        ),
        L4VerifierEvidenceTraceExample(
            example_id="invalid_counts_only_trace",
            description="Only accepted counts; no verifier token IDs.",
            trace_record={
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
            expected_validation_passes=False,
            expected_dry_run_decision_status="invalid_trace",
            notes="Counts-only inference forbidden.",
        ),
    )


FORBIDDEN_VERIFIER_SOURCE_NAMES: frozenset[str] = frozenset(
    s.source_name for s in build_l4_verifier_evidence_forbidden_sources()
) | frozenset(
    {
        "committed_token_ids_unmarked",
        "accepted_token_counts_only",
        "committed_tokens",
        "baseline_tokens",
        "retokenized_generated_text",
        "guessed_token_ids",
        "compressed_draft_tokens",
        "shadow_top1_proposal_tokens",
        "round_log_proposal_tokens_as_verifier",
        PROPOSAL_SOURCE_ROUND_LOG,
        "exactkv_round_log_draft_tokens",
    },
)


def validate_verifier_evidence_trace_record(
    record: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Validate one future verifier evidence trace record (design-time only)."""
    errors: list[str] = []

    for field in ALL_SCHEMA_FIELDS:
        if field not in record:
            errors.append(f"missing field: {field}")

    if record.get("diagnostic_only") is not True:
        errors.append("diagnostic_only must be true")

    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        errors.append("trace_schema_version must match design version")

    proposal_source = str(record.get("proposal_source") or "")
    verifier_source = str(record.get("verifier_evidence_source") or "")

    if proposal_source == verifier_source and verifier_source:
        errors.append("proposal_source and verifier_evidence_source must differ")

    if verifier_source in FORBIDDEN_VERIFIER_SOURCE_NAMES:
        errors.append(f"forbidden verifier_evidence_source: {verifier_source}")

    if verifier_source == PROPOSAL_SOURCE_ROUND_LOG:
        errors.append("round-log draft tokens are proposal evidence, not verifier evidence")

    proposal_ids = record.get("proposal_token_ids") or []
    verifier_ids = record.get("verifier_evidence_token_ids") or []

    if (
        record.get("verifier_evidence_available") is True
        and not verifier_ids
        and not record.get("verifier_block_reason")
    ):
        errors.append("verifier_evidence_available but no token IDs or block reason")

    if record.get("verifier_evidence_available") is True:
        if record.get("verifier_evidence_is_full_kv") is not True:
            if verifier_source != "verifier_exception_or_block_reason":
                errors.append(
                    "available verifier evidence must be full_kv or exception/block source",
                )
        if not verifier_source:
            errors.append("verifier_evidence_source required when evidence available")

    if record.get("num_accepted") is not None and not verifier_ids:
        errors.append("cannot infer verifier evidence from num_accepted alone")

    if record.get("accepted_tokens") is not None and not verifier_ids:
        errors.append("cannot infer verifier evidence from accepted_tokens alone")

    if (
        proposal_ids
        and verifier_ids
        and proposal_ids == verifier_ids
        and proposal_source == PROPOSAL_SOURCE_ROUND_LOG
        and record.get("verifier_evidence_is_full_kv") is not True
    ):
        errors.append("proposal and verifier token IDs must not alias without full-KV mark")

    return len(errors) == 0, tuple(errors)


def validate_trace_example(example: L4VerifierEvidenceTraceExample) -> tuple[bool, tuple[str, ...]]:
    """Return whether trace validation outcome matches the example expectation."""
    actual_passes, errors = validate_verifier_evidence_trace_record(example.trace_record)
    if actual_passes == example.expected_validation_passes:
        return True, errors
    return False, (*errors, "validation result does not match expected_validation_passes")


def evaluate_l4_verifier_evidence_schema_decision(
    design: L4VerifierEvidenceTraceSchemaDesign,
) -> L4VerifierEvidenceTraceSchemaDecision:
    """Evaluate whether verifier evidence trace schema design is complete."""
    field_names = {f.field_name for f in design.schema_fields}
    fields_ok = (
        set(REQUIRED_VERIFIER_EVIDENCE_FIELDS) <= field_names
        and set(REQUIRED_METADATA_FIELDS) <= field_names
    )

    allowed_ok = any(
        s.source_name == "full_kv_verifier_output_tokens" for s in design.allowed_sources
    )

    forbidden_names = {s.source_name for s in design.forbidden_sources}
    forbidden_ok = (
        "committed_token_ids_unmarked" in forbidden_names
        and "accepted_token_counts_only" in forbidden_names
        and "baseline_generated_tokens" in forbidden_names
        and "round_log_proposal_tokens_as_verifier" in forbidden_names
    )

    rule_ids = {r.rule_id for r in design.validation_rules}
    rules_ok = (
        "explicit_evidence_required" in rule_ids
        and "proposal_verifier_distinct" in rule_ids
        and "missing_evidence_blocks" in rule_ids
        and "no_committed_output_inference" in rule_ids
        and "no_accepted_count_inference" in rule_ids
        and "diagnostic_only_flag" in rule_ids
    )

    examples_ok = True
    for ex in design.trace_examples:
        ok, _ = validate_trace_example(ex)
        if not ok:
            examples_ok = False
            break

    complete = fields_ok and allowed_ok and forbidden_ok and rules_ok and examples_ok

    if complete:
        return L4VerifierEvidenceTraceSchemaDecision(
            outcome=DESIGN_OUTCOME_COMPLETE,
            schema_scaffold_authorized=True,
            runtime_instrumentation_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21F,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21F,
            reason=(
                "Verifier evidence trace schema defines required fields, allowed/forbidden "
                "sources, validation rules, and examples; schema scaffold may begin; "
                "runtime instrumentation and commit blocked"
            ),
        )

    if not fields_ok or not rules_ok:
        return L4VerifierEvidenceTraceSchemaDecision(
            outcome=DESIGN_OUTCOME_INCOMPLETE,
            schema_scaffold_authorized=False,
            runtime_instrumentation_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21F,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21F,
            reason="verifier evidence trace schema design missing required fields or rules",
        )

    return L4VerifierEvidenceTraceSchemaDecision(
        outcome=DESIGN_OUTCOME_BLOCKED,
        schema_scaffold_authorized=False,
        runtime_instrumentation_authorized=False,
        runtime_commit_authorized=False,
        allowed_next_phase=RECOMMENDED_NEXT_PHASE_21F,
        forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21F,
        reason="verifier evidence trace schema design blocked by example validation",
    )


def build_l4_verifier_evidence_trace_schema_design() -> L4VerifierEvidenceTraceSchemaDesign:
    """Build the complete L4 verifier evidence trace schema design."""
    schema_fields = build_l4_verifier_evidence_schema_fields()
    allowed = build_l4_verifier_evidence_allowed_sources()
    forbidden = build_l4_verifier_evidence_forbidden_sources()
    rules = build_l4_verifier_evidence_validation_rules()
    examples = build_l4_verifier_evidence_trace_examples()

    partial = L4VerifierEvidenceTraceSchemaDesign(
        design_id=EXPERIMENT_107_ID,
        safety_level=L4_SAFETY_LEVEL,
        stage=STAGE,
        mode=MODE,
        schema_version=TRACE_SCHEMA_VERSION,
        schema_fields=schema_fields,
        allowed_sources=allowed,
        forbidden_sources=forbidden,
        validation_rules=rules,
        trace_examples=examples,
        decision=L4VerifierEvidenceTraceSchemaDecision(
            outcome=DESIGN_OUTCOME_BLOCKED,
            schema_scaffold_authorized=False,
            runtime_instrumentation_authorized=False,
            runtime_commit_authorized=False,
            allowed_next_phase=RECOMMENDED_NEXT_PHASE_21F,
            forbidden_next_phases=FORBIDDEN_NEXT_PHASES_21F,
            reason="pending evaluation",
        ),
    )
    decision = evaluate_l4_verifier_evidence_schema_decision(partial)
    return L4VerifierEvidenceTraceSchemaDesign(
        design_id=partial.design_id,
        safety_level=partial.safety_level,
        stage=partial.stage,
        mode=partial.mode,
        schema_version=partial.schema_version,
        schema_fields=partial.schema_fields,
        allowed_sources=partial.allowed_sources,
        forbidden_sources=partial.forbidden_sources,
        validation_rules=partial.validation_rules,
        trace_examples=partial.trace_examples,
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
        ("verifier_evidence_trace_schema_scaffold", "verifier evidence schema scaffold not implemented"),
        ("runtime_verifier_instrumentation", "runtime verifier evidence instrumentation not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp107_l4_verifier_evidence_trace_schema_design() -> dict[str, Any]:
    """Run Experiment 107 L4 verifier evidence trace schema design (no runtime changes)."""
    design = build_l4_verifier_evidence_trace_schema_design()
    decision = design.decision

    status = (
        "design_complete"
        if decision.outcome == DESIGN_OUTCOME_COMPLETE
        else "design_incomplete"
    )

    return {
        "experiment_id": EXPERIMENT_107_ID,
        "status": status,
        "phase": PHASE_21F,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": design.schema_version,
        "design_objects": {
            "L4VerifierEvidenceTraceSchemaDesign": design.to_dict(),
            "L4VerifierEvidenceTraceSchemaDecision": decision.to_dict(),
        },
        "schema_fields": [f.to_dict() for f in design.schema_fields],
        "allowed_sources": [s.to_dict() for s in design.allowed_sources],
        "forbidden_sources": [s.to_dict() for s in design.forbidden_sources],
        "validation_rules": [r.to_dict() for r in design.validation_rules],
        "trace_examples": [e.to_dict() for e in design.trace_examples],
        "design_decision": decision.to_dict(),
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21F,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21F),
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
            "Verifier evidence trace schema design only; not runtime instrumentation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Proposal and verifier evidence are separate fields.",
            "Missing verifier evidence blocks dry-run; never fabricates tokens.",
            "No model experiments; no performance/memory/serving claims.",
        ],
    }


def validate_exp107_report(report: dict[str, Any]) -> list[str]:
    """Validate Experiment 107 report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "schema_fields",
        "allowed_sources",
        "forbidden_sources",
        "validation_rules",
        "trace_examples",
        "design_decision",
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

    if report.get("experiment_id") != EXPERIMENT_107_ID:
        errors.append("experiment_id mismatch")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21F:
        errors.append(
            "allowed_next_phase must be phase21g_l4_verifier_evidence_trace_schema_scaffold",
        )

    forbidden = set(report.get("forbidden_next_phases") or [])
    if not set(FORBIDDEN_NEXT_PHASES_21F) <= forbidden:
        errors.append("missing required forbidden_next_phases")

    if report.get("runtime_instrumentation_authorized") is not False:
        errors.append("runtime_instrumentation_authorized must be false")

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
        errors.append(
            "design_decision.outcome must be verifier_evidence_trace_schema_design_complete",
        )

    if decision.get("runtime_instrumentation_authorized") is not False:
        errors.append("design_decision.runtime_instrumentation_authorized must be false")

    if decision.get("runtime_commit_authorized") is not False:
        errors.append("design_decision.runtime_commit_authorized must be false")

    field_names = {f["field_name"] for f in report.get("schema_fields") or []}
    if not set(REQUIRED_VERIFIER_EVIDENCE_FIELDS) <= field_names:
        errors.append("schema_fields missing required verifier evidence fields")

    allowed_names = {s["source_name"] for s in report.get("allowed_sources") or []}
    if "full_kv_verifier_output_tokens" not in allowed_names:
        errors.append("allowed_sources must include full_kv_verifier_output_tokens")

    forbidden_names = {s["source_name"] for s in report.get("forbidden_sources") or []}
    for req in (
        "committed_token_ids_unmarked",
        "accepted_token_counts_only",
        "baseline_generated_tokens",
        "round_log_proposal_tokens_as_verifier",
    ):
        if req not in forbidden_names:
            errors.append(f"forbidden_sources missing {req}")

    rule_ids = {r["rule_id"] for r in report.get("validation_rules") or []}
    for req in (
        "explicit_evidence_required",
        "proposal_verifier_distinct",
        "missing_evidence_blocks",
    ):
        if req not in rule_ids:
            errors.append(f"validation_rules missing {req}")

    for ex in report.get("trace_examples") or []:
        example = L4VerifierEvidenceTraceExample(
            example_id=str(ex.get("example_id", "")),
            description=str(ex.get("description", "")),
            trace_record=dict(ex.get("trace_record") or {}),
            expected_validation_passes=bool(ex.get("expected_validation_passes")),
            expected_dry_run_decision_status=str(
                ex.get("expected_dry_run_decision_status", ""),
            ),
            notes=str(ex.get("notes", "")),
        )
        ok, ex_errors = validate_trace_example(example)
        if not ok:
            errors.append(f"trace_example {example.example_id} expectation mismatch: {ex_errors}")

    return errors
