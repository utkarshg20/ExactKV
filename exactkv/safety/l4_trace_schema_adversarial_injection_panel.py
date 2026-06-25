"""L4 trace schema adversarial injection panel (Phase 21I / Exp 110).

Adversarial stress-testing for trace schema enforcement — not runtime instrumentation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_trace_only_dry_run_scaffold import evaluate_l4_trace_only_input
from exactkv.safety.l4_verifier_evidence_trace_schema_design import TRACE_SCHEMA_VERSION
from exactkv.safety.l4_verifier_evidence_trace_schema_scaffold import (
    convert_verifier_trace_to_l4_trace_only_input,
    validate_verifier_evidence_trace_record,
)
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_110_ID = "exp110_l4_trace_schema_adversarial_injection_panel"
DEFAULT_EXP110_REPORT = Path(
    "reports/experiment_110_l4_trace_schema_adversarial_injection_panel.json",
)
PHASE_21I = "21I"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "trace_schema_adversarial_injection_panel"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21I = "phase21j_l4_verifier_evidence_runtime_instrumentation_design"
FORBIDDEN_NEXT_PHASES_21I: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_verifier_instrumentation",
    "l4_stage3_verifier_mediated_dry_run",
)

PANEL_OUTCOME_COMPLETE = "adversarial_panel_complete"
PANEL_OUTCOME_INCOMPLETE = "adversarial_panel_incomplete"
PANEL_OUTCOME_BLOCKED = "adversarial_panel_blocked"

PANEL_CLASSIFICATIONS: tuple[str, ...] = (
    "pass",
    "blocked_missing_verifier_evidence",
    "invalid_trace",
    "detected_poisoning",
)

ADVERSARIAL_CATEGORIES: tuple[str, ...] = (
    "missing_field_attacks",
    "field_forgery_attacks",
    "structural_poisoning",
    "divergence_injection",
    "silent_failure_attempts",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

INTERPRETATION_NOTE = (
    "Adversarial injection panel is trace-only diagnostic; not commit authority."
)

PANEL_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
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
    },
)


@dataclass(frozen=True)
class L4AdversarialInjectionCase:
    """One adversarial trace injection case."""

    case_id: str
    category: str
    description: str
    record: dict[str, Any]
    expected_panel_classification: str
    expected_schema_valid: bool
    is_attack: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4AdversarialCaseResult:
    """Result of running one adversarial case through schema panel."""

    case_id: str
    category: str
    expected_panel_classification: str
    actual_panel_classification: str
    expected_schema_valid: bool
    actual_schema_valid: bool
    dry_run_status: str | None
    adversarial_test_passed: bool
    false_acceptance: bool
    validation_errors: tuple[str, ...]
    interpretation_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "validation_errors": list(self.validation_errors),
        }


@dataclass(frozen=True)
class L4AdversarialPanelValidationResult:
    """Validation outcome for Experiment 110 report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_metadata(*, proposal: tuple[int, ...] = ()) -> dict[str, Any]:
    return {
        "round_index": 0,
        "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
        "proposal_token_ids": list(proposal),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_by": "l4_trace_schema_adversarial_injection_panel",
        "diagnostic_only": True,
    }


def _valid_verifier_block(
    *,
    proposal: tuple[int, ...],
    verifier: tuple[int, ...],
    source: str = "full_kv_verifier_output_tokens",
) -> dict[str, Any]:
    return {
        "verifier_evidence_available": True,
        "verifier_evidence_source": source,
        "verifier_evidence_token_ids": list(verifier),
        "verifier_evidence_text": None,
        "verifier_evidence_is_full_kv": True,
        "verifier_evidence_is_authoritative": True,
        "verifier_checked_proposal_token_ids": list(proposal),
        "verifier_matching_prefix_token_ids": list(proposal),
        "verifier_rejected_suffix_token_ids": [],
        "verifier_first_mismatch_index": None,
        "verifier_decision_status": "all_match",
        "verifier_exception": None,
        "verifier_block_reason": None,
        "verifier_trace_complete": True,
    }


def build_adversarial_injection_cases() -> tuple[L4AdversarialInjectionCase, ...]:
    """Build adversarial injection panel cases across all required categories."""
    shared_ids = [10, 11, 12]
    cases: list[L4AdversarialInjectionCase] = []

    # --- 1. Missing Field Attacks ---
    missing_verifier = {**_base_metadata(proposal=(1, 2, 3))}
    cases.append(
        L4AdversarialInjectionCase(
            case_id="missing_verifier_fields_entirely",
            category="missing_field_attacks",
            description="Verifier block fields omitted entirely",
            record=missing_verifier,
            expected_panel_classification="invalid_trace",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    missing_proposal = {
        **_base_metadata(proposal=(1, 2, 3)),
        **_valid_verifier_block(proposal=(1, 2, 3), verifier=(1, 2, 3)),
    }
    del missing_proposal["proposal_token_ids"]
    cases.append(
        L4AdversarialInjectionCase(
            case_id="missing_proposal_token_ids",
            category="missing_field_attacks",
            description="proposal_token_ids field omitted entirely",
            record=missing_proposal,
            expected_panel_classification="invalid_trace",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    partial = {
        **_base_metadata(proposal=(1, 2)),
        **_valid_verifier_block(proposal=(1, 2), verifier=(1, 2)),
    }
    del partial["trace_schema_version"]
    del partial["diagnostic_only"]
    cases.append(
        L4AdversarialInjectionCase(
            case_id="partial_trace_metadata_corruption",
            category="missing_field_attacks",
            description="Required metadata fields removed",
            record=partial,
            expected_panel_classification="invalid_trace",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    # --- 2. Field Forgery Attacks ---
    cases.append(
        L4AdversarialInjectionCase(
            case_id="forgery_committed_tokens_as_verifier",
            category="field_forgery_attacks",
            description="Committed tokens injected as unmarked verifier evidence",
            record={
                **_base_metadata(proposal=(1, 2, 3)),
                **_valid_verifier_block(proposal=(1, 2, 3), verifier=(1, 2)),
                "verifier_evidence_source": "committed_token_ids_unmarked",
                "verifier_evidence_is_full_kv": False,
                "verifier_evidence_is_authoritative": False,
            },
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    cases.append(
        L4AdversarialInjectionCase(
            case_id="forgery_round_log_as_verifier_source",
            category="field_forgery_attacks",
            description="Round-log draft tokens mislabeled as verifier source",
            record={
                **_base_metadata(proposal=(1, 2, 3)),
                **_valid_verifier_block(proposal=(1, 2, 3), verifier=(1, 2, 3)),
                "verifier_evidence_source": PROPOSAL_SOURCE_ROUND_LOG,
            },
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    cases.append(
        L4AdversarialInjectionCase(
            case_id="forgery_proposal_verifier_alias",
            category="field_forgery_attacks",
            description="Proposal and verifier share same list object",
            record={
                **_base_metadata(proposal=(1, 2, 3)),
                "proposal_token_ids": shared_ids,
                "verifier_evidence_available": True,
                "verifier_evidence_source": "full_kv_verifier_output_tokens",
                "verifier_evidence_token_ids": shared_ids,
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
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    # --- 3. Structural Poisoning ---
    bad_version = {
        **_base_metadata(proposal=(1, 2)),
        **_valid_verifier_block(proposal=(1, 2), verifier=(1, 2)),
        "trace_schema_version": "poisoned_v99",
    }
    cases.append(
        L4AdversarialInjectionCase(
            case_id="poison_invalid_schema_version",
            category="structural_poisoning",
            description="Invalid trace_schema_version injected",
            record=bad_version,
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    no_diagnostic = {
        **_base_metadata(proposal=(1, 2)),
        **_valid_verifier_block(proposal=(1, 2), verifier=(1, 2)),
        "diagnostic_only": False,
    }
    cases.append(
        L4AdversarialInjectionCase(
            case_id="poison_diagnostic_only_false",
            category="structural_poisoning",
            description="diagnostic_only=false structural violation",
            record=no_diagnostic,
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    missing_created = {
        **_base_metadata(proposal=(1, 2)),
        **_valid_verifier_block(proposal=(1, 2), verifier=(1, 2)),
    }
    del missing_created["created_by"]
    cases.append(
        L4AdversarialInjectionCase(
            case_id="poison_missing_created_by",
            category="structural_poisoning",
            description="Missing created_by metadata field",
            record=missing_created,
            expected_panel_classification="invalid_trace",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    # --- 4. Divergence Injection (valid schema, correct dry-run) ---
    cases.append(
        L4AdversarialInjectionCase(
            case_id="divergence_first_token",
            category="divergence_injection",
            description="Forced first-token proposal/verifier mismatch",
            record={
                **_base_metadata(proposal=(9, 9, 9)),
                **_valid_verifier_block(proposal=(9, 9, 9), verifier=(1, 2, 3)),
                "verifier_evidence_source": "verifier_mismatch_evidence",
                "verifier_matching_prefix_token_ids": [],
                "verifier_rejected_suffix_token_ids": [9, 9, 9],
                "verifier_first_mismatch_index": 0,
                "verifier_decision_status": "first_token_mismatch",
            },
            expected_panel_classification="pass",
            expected_schema_valid=True,
            is_attack=False,
        ),
    )

    cases.append(
        L4AdversarialInjectionCase(
            case_id="divergence_mid_sequence",
            category="divergence_injection",
            description="Forced mid-sequence proposal/verifier mismatch",
            record={
                **_base_metadata(proposal=(1, 2, 9, 9)),
                **_valid_verifier_block(proposal=(1, 2, 9, 9), verifier=(1, 2, 3, 4)),
                "verifier_evidence_source": "verifier_matching_prefix_evidence",
                "verifier_matching_prefix_token_ids": [1, 2],
                "verifier_rejected_suffix_token_ids": [9, 9],
                "verifier_first_mismatch_index": 2,
                "verifier_decision_status": "partial_match",
            },
            expected_panel_classification="pass",
            expected_schema_valid=True,
            is_attack=False,
        ),
    )

    cases.append(
        L4AdversarialInjectionCase(
            case_id="divergence_full_match",
            category="divergence_injection",
            description="Full proposal/verifier agreement (control)",
            record={
                **_base_metadata(proposal=(5, 6, 7)),
                **_valid_verifier_block(proposal=(5, 6, 7), verifier=(5, 6, 7)),
            },
            expected_panel_classification="pass",
            expected_schema_valid=True,
            is_attack=False,
        ),
    )

    # --- 5. Silent Failure Attempts ---
    silent_empty = {
        **_base_metadata(proposal=(1, 2, 3)),
        "verifier_evidence_available": True,
        "verifier_evidence_source": "full_kv_verifier_output_tokens",
        "verifier_evidence_token_ids": [],
        "verifier_evidence_text": None,
        "verifier_evidence_is_full_kv": True,
        "verifier_evidence_is_authoritative": True,
        "verifier_checked_proposal_token_ids": [1, 2, 3],
        "verifier_matching_prefix_token_ids": [],
        "verifier_rejected_suffix_token_ids": [],
        "verifier_first_mismatch_index": None,
        "verifier_decision_status": "all_match",
        "verifier_exception": None,
        "verifier_block_reason": None,
        "verifier_trace_complete": True,
    }
    cases.append(
        L4AdversarialInjectionCase(
            case_id="silent_empty_verifier_with_available_true",
            category="silent_failure_attempts",
            description="verifier_evidence_available=true but token IDs empty",
            record=silent_empty,
            expected_panel_classification="invalid_trace",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    cases.append(
        L4AdversarialInjectionCase(
            case_id="silent_missing_verifier_no_block",
            category="silent_failure_attempts",
            description="Verifier absent; must block dry-run",
            record={
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
            expected_panel_classification="blocked_missing_verifier_evidence",
            expected_schema_valid=True,
            is_attack=True,
        ),
    )

    counts_only = {
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
    }
    cases.append(
        L4AdversarialInjectionCase(
            case_id="silent_counts_only_verifier",
            category="silent_failure_attempts",
            description="Accepted counts only without verifier token IDs",
            record=counts_only,
            expected_panel_classification="detected_poisoning",
            expected_schema_valid=False,
            is_attack=True,
        ),
    )

    return tuple(cases)


def _classify_panel_outcome(
    *,
    schema_valid: bool,
    dry_run_status: str | None,
    validation_errors: Sequence[str],
    category: str,
) -> str:
    """Map validation + dry-run to panel classification."""
    if not schema_valid:
        if category == "missing_field_attacks":
            return "invalid_trace"
        if category in ("field_forgery_attacks", "structural_poisoning", "silent_failure_attempts"):
            if any(
                kw in " ".join(validation_errors).lower()
                for kw in (
                    "forbidden",
                    "alias",
                    "differ",
                    "round-log",
                    "committed",
                    "num_accepted",
                    "diagnostic_only",
                    "trace_schema_version",
                )
            ):
                return "detected_poisoning"
        return "invalid_trace"

    if dry_run_status == "blocked_missing_verifier_evidence":
        return "blocked_missing_verifier_evidence"

    if dry_run_status in ("all_match", "partial_match", "first_token_mismatch"):
        return "pass"

    return "invalid_trace"


def execute_adversarial_case(case: L4AdversarialInjectionCase) -> L4AdversarialCaseResult:
    """Run one adversarial case through schema validation and dry-run panel."""
    record = dict(case.record)
    record["cell_id"] = case.case_id

    validation = validate_verifier_evidence_trace_record(record)
    schema_valid = validation.valid
    dry_run_status: str | None = None

    if schema_valid:
        conv = convert_verifier_trace_to_l4_trace_only_input(record)
        if conv.converted and conv.dry_run_input is not None:
            decision = evaluate_l4_trace_only_input(conv.dry_run_input)
            dry_run_status = decision.decision_status
        elif conv.decision_status:
            dry_run_status = conv.decision_status

    actual_classification = _classify_panel_outcome(
        schema_valid=schema_valid,
        dry_run_status=dry_run_status,
        validation_errors=validation.errors,
        category=case.category,
    )

    false_acceptance = schema_valid and not case.expected_schema_valid
    adversarial_passed = (
        case.expected_schema_valid == schema_valid
        and case.expected_panel_classification == actual_classification
        and not false_acceptance
    )

    return L4AdversarialCaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_panel_classification=case.expected_panel_classification,
        actual_panel_classification=actual_classification,
        expected_schema_valid=case.expected_schema_valid,
        actual_schema_valid=schema_valid,
        dry_run_status=dry_run_status,
        adversarial_test_passed=adversarial_passed,
        false_acceptance=false_acceptance,
        validation_errors=validation.errors,
        interpretation_note=INTERPRETATION_NOTE,
    )


def evaluate_panel_outcome(results: Sequence[L4AdversarialCaseResult]) -> str:
    """Return panel outcome from adversarial case results."""
    if not results:
        return PANEL_OUTCOME_BLOCKED

    all_passed = all(r.adversarial_test_passed for r in results)
    false_acceptances = sum(1 for r in results if r.false_acceptance)
    categories = {c for c in ADVERSARIAL_CATEGORIES}
    covered = {r.category for r in results}

    if false_acceptances > 0:
        return PANEL_OUTCOME_BLOCKED

    if all_passed and categories <= covered:
        return PANEL_OUTCOME_COMPLETE

    return PANEL_OUTCOME_INCOMPLETE


def _category_breakdown(
    results: Sequence[L4AdversarialCaseResult],
) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for cat in ADVERSARIAL_CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        passed = sum(1 for r in cat_results if r.adversarial_test_passed)
        breakdown[cat] = {
            "total_cases": len(cat_results),
            "passed_cases": passed,
            "pass_rate": passed / len(cat_results) if cat_results else 0.0,
        }
    return breakdown


def validate_exp110_adversarial_panel_report(
    report: dict[str, Any],
) -> L4AdversarialPanelValidationResult:
    """Validate Experiment 110 adversarial panel report."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "schema_version",
        "panel_outcome",
        "total_cases",
        "cases_passed",
        "cases_failed",
        "classification_summary",
        "failure_mode_breakdown",
        "category_breakdown",
        "adversarial_detection_rate",
        "invalid_trace_rejection_rate",
        "false_acceptance_rate",
        "schema_robustness_score",
        "case_results",
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

    if report.get("experiment_id") != EXPERIMENT_110_ID:
        errors.append("experiment_id mismatch")

    if report.get("panel_outcome") != PANEL_OUTCOME_COMPLETE:
        errors.append("panel_outcome must be adversarial_panel_complete")

    if report.get("false_acceptance_rate") != 0.0:
        errors.append("false_acceptance_rate must be 0")

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

    for cat in ADVERSARIAL_CATEGORIES:
        if cat not in (report.get("category_breakdown") or {}):
            errors.append(f"missing category breakdown: {cat}")

    for idx, case in enumerate(report.get("case_results") or []):
        if not case.get("adversarial_test_passed"):
            errors.append(f"case_results[{idx}] adversarial_test_passed false")
        if case.get("false_acceptance"):
            errors.append(f"case_results[{idx}] false_acceptance true")

    return L4AdversarialPanelValidationResult(
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
        if bid not in PANEL_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        ("runtime_verifier_instrumentation", "runtime verifier evidence instrumentation not implemented"),
        ("runtime_instrumentation_design", "runtime instrumentation design not complete"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def run_exp110_l4_trace_schema_adversarial_injection_panel() -> dict[str, Any]:
    """Run Experiment 110 L4 trace schema adversarial injection panel."""
    cases = build_adversarial_injection_cases()
    results = tuple(execute_adversarial_case(c) for c in cases)

    passed = sum(1 for r in results if r.adversarial_test_passed)
    failed = len(results) - passed

    classification_counts: dict[str, int] = {}
    for r in results:
        cls = r.actual_panel_classification
        classification_counts[cls] = classification_counts.get(cls, 0) + 1

    attack_results = [r for r in results if r.category != "divergence_injection"]
    attacks_detected = sum(
        1
        for r in attack_results
        if r.adversarial_test_passed and r.actual_panel_classification != "pass"
    )
    adversarial_detection_rate = (
        attacks_detected / len(attack_results) if attack_results else 1.0
    )

    invalid_expected = sum(1 for c in cases if not c.expected_schema_valid)
    invalid_rejected = sum(
        1 for r in results if not r.actual_schema_valid and not r.expected_schema_valid
    )
    invalid_trace_rejection_rate = (
        invalid_rejected / invalid_expected if invalid_expected else 1.0
    )

    false_acceptance_rate = sum(1 for r in results if r.false_acceptance) / len(results)

    schema_robustness_score = passed / len(results) if results else 0.0

    failure_modes: dict[str, int] = {}
    for r in results:
        if not r.adversarial_test_passed:
            key = r.actual_panel_classification
            failure_modes[key] = failure_modes.get(key, 0) + 1

    panel_outcome = evaluate_panel_outcome(results)
    status = (
        "panel_complete"
        if panel_outcome == PANEL_OUTCOME_COMPLETE
        else "panel_incomplete"
    )

    report = {
        "experiment_id": EXPERIMENT_110_ID,
        "status": status,
        "phase": PHASE_21I,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "panel_outcome": panel_outcome,
        "total_cases": len(results),
        "cases_passed": passed,
        "cases_failed": failed,
        "classification_summary": {
            "status_counts": classification_counts,
            "classifications": list(PANEL_CLASSIFICATIONS),
        },
        "failure_mode_breakdown": failure_modes,
        "category_breakdown": _category_breakdown(results),
        "adversarial_detection_rate": adversarial_detection_rate,
        "invalid_trace_rejection_rate": invalid_trace_rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "schema_robustness_score": schema_robustness_score,
        "case_results": [r.to_dict() for r in results],
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21I,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21I),
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
            "Adversarial injection panel only; not runtime instrumentation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Synthetic adversarial traces only; no model experiments.",
            "Stress-tests schema enforcement; not performance or serving.",
        ],
    }
    report["validation_result"] = validate_exp110_adversarial_panel_report(report).to_dict()
    return report


def validate_exp110_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 110 report."""
    return list(validate_exp110_adversarial_panel_report(report).errors)
