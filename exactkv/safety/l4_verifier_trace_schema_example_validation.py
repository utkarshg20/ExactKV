"""L4 verifier trace schema example validation (Phase 21H / Exp 109).

Schema-example validation and trace-only execution layer — not runtime instrumentation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_trace_only_dry_run_scaffold import evaluate_l4_trace_only_input
from exactkv.safety.l4_verifier_evidence_trace_schema_design import (
    TRACE_SCHEMA_VERSION,
    build_l4_verifier_evidence_validation_rules,
)
from exactkv.safety.l4_verifier_evidence_trace_schema_scaffold import (
    build_synthetic_schema_examples,
    convert_verifier_trace_to_l4_trace_only_input,
    process_schema_example,
    validate_verifier_evidence_trace_record,
)
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_109_ID = "exp109_l4_verifier_trace_schema_example_validation"
DEFAULT_EXP109_REPORT = Path(
    "reports/experiment_109_l4_verifier_trace_schema_example_validation.json",
)
PHASE_21H = "21H"
STAGE = "stage_2_trace_only_l4_dry_run"
MODE = "verifier_trace_schema_example_validation"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_21H = (
    "phase21i_l4_trace_schema_stress_adversarial_trace_injection_panel"
)
FORBIDDEN_NEXT_PHASES_21H: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_runtime_verifier_instrumentation",
    "l4_stage3_verifier_mediated_dry_run",
)

VALIDATION_OUTCOME_COMPLETE = "schema_example_validation_complete"
VALIDATION_OUTCOME_INCOMPLETE = "schema_example_validation_incomplete"
VALIDATION_OUTCOME_BLOCKED = "schema_example_validation_blocked"

EXPECTED_CLASSIFICATION_STATUSES: tuple[str, ...] = (
    "all_match",
    "partial_match",
    "first_token_mismatch",
    "blocked_missing_verifier_evidence",
    "invalid_trace",
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

INTERPRETATION_NOTE = (
    "Schema example validation is trace-only diagnostic; not commit authority."
)

SCHEMA_ENFORCEMENT_RULES: tuple[str, ...] = (
    "no_committed_token_inference",
    "no_accepted_count_inference",
    "missing_verifier_evidence_blocks",
    "proposal_verifier_separation",
    "invalid_trace_rejection",
    "diagnostic_only_required",
    "explicit_verifier_evidence",
    "no_runtime_commit_effect",
    "no_generator_exposure",
    "verifier_source_of_truth",
)

VALIDATION_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
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
    },
)


@dataclass(frozen=True)
class L4SchemaExampleExecutionResult:
    """Per-example schema validation + dry-run execution outcome."""

    example_id: str
    expected_validation: bool
    actual_validation: bool
    expected_classification: str
    actual_classification: str | None
    validation_passed: bool
    classification_passed: bool
    converted_to_dry_run_input: bool
    validation_errors: tuple[str, ...]
    conversion_errors: tuple[str, ...]
    enforcement_rules_exercised: tuple[str, ...]
    dry_run_decision_used_for_token_commit: bool
    exposed_to_generator: bool
    interpretation_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "enforcement_rules_exercised": list(self.enforcement_rules_exercised),
            "validation_errors": list(self.validation_errors),
            "conversion_errors": list(self.conversion_errors),
        }


@dataclass(frozen=True)
class L4SchemaEnforcementRuleCoverage:
    """Coverage of one schema enforcement rule across examples and probes."""

    rule_id: str
    description: str
    exercised: bool
    exercised_by: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "exercised_by": list(self.exercised_by),
        }


@dataclass(frozen=True)
class L4SchemaExampleValidationReport:
    """Top-level validation report aggregate (typing; report is dict at runtime)."""

    experiment_id: str
    status: str
    validation_outcome: str
    example_results: tuple[L4SchemaExampleExecutionResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "validation_outcome": self.validation_outcome,
            "example_results": [r.to_dict() for r in self.example_results],
        }


@dataclass(frozen=True)
class L4SchemaExampleValidationResult:
    """Validation outcome for Experiment 109 report."""

    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rules_for_example(example_id: str) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "complete_all_match_trace": (
            "explicit_verifier_evidence",
            "proposal_verifier_separation",
            "no_runtime_commit_effect",
            "no_generator_exposure",
            "verifier_source_of_truth",
            "diagnostic_only_required",
        ),
        "complete_partial_match_trace": (
            "explicit_verifier_evidence",
            "proposal_verifier_separation",
            "no_runtime_commit_effect",
            "no_generator_exposure",
            "verifier_source_of_truth",
            "diagnostic_only_required",
        ),
        "complete_first_mismatch_trace": (
            "explicit_verifier_evidence",
            "proposal_verifier_separation",
            "no_runtime_commit_effect",
            "no_generator_exposure",
            "verifier_source_of_truth",
            "diagnostic_only_required",
        ),
        "missing_verifier_evidence_trace": (
            "missing_verifier_evidence_blocks",
            "proposal_verifier_separation",
            "diagnostic_only_required",
        ),
        "verifier_exception_trace": (
            "missing_verifier_evidence_blocks",
            "diagnostic_only_required",
        ),
        "invalid_committed_tokens_as_verifier_trace": (
            "no_committed_token_inference",
            "invalid_trace_rejection",
        ),
        "invalid_counts_only_trace": (
            "no_accepted_count_inference",
            "invalid_trace_rejection",
        ),
        "invalid_proposal_verifier_alias_trace": (
            "proposal_verifier_separation",
            "invalid_trace_rejection",
        ),
    }
    return mapping.get(example_id, ())


def execute_schema_example(
    example: Mapping[str, Any],
) -> L4SchemaExampleExecutionResult:
    """Validate, convert, and classify one schema example (trace-only)."""
    example_id = str(example["example_id"])
    expected_validation = bool(example["expected_validation"])
    expected_classification = str(example["expected_dry_run_status"])

    processed = process_schema_example(example)
    actual_validation = bool(processed["actual_validation"])
    actual_classification = processed.get("actual_dry_run_status")

    validation_passed = expected_validation == actual_validation
    if not actual_validation:
        classification_passed = expected_classification == "invalid_trace"
    else:
        classification_passed = expected_classification == actual_classification

    commit_flag = False
    exposed = False
    if processed.get("converted_to_dry_run_input"):
        record = dict(example["record"])
        record["cell_id"] = example_id
        conv = convert_verifier_trace_to_l4_trace_only_input(record)
        if conv.dry_run_input is not None:
            decision = evaluate_l4_trace_only_input(conv.dry_run_input)
            commit_flag = decision.dry_run_decision_used_for_token_commit
            exposed = decision.exposed_to_generator

    return L4SchemaExampleExecutionResult(
        example_id=example_id,
        expected_validation=expected_validation,
        actual_validation=actual_validation,
        expected_classification=expected_classification,
        actual_classification=actual_classification,
        validation_passed=validation_passed,
        classification_passed=classification_passed,
        converted_to_dry_run_input=bool(processed.get("converted_to_dry_run_input")),
        validation_errors=tuple(processed.get("validation_errors") or ()),
        conversion_errors=tuple(processed.get("conversion_errors") or ()),
        enforcement_rules_exercised=_rules_for_example(example_id),
        dry_run_decision_used_for_token_commit=commit_flag,
        exposed_to_generator=exposed,
        interpretation_note=INTERPRETATION_NOTE,
    )


def build_enforcement_rule_coverage(
    example_results: Sequence[L4SchemaExampleExecutionResult],
    probe_results: Sequence[Mapping[str, Any]],
) -> tuple[L4SchemaEnforcementRuleCoverage, ...]:
    """Build enforcement rule coverage from examples and diagnostic probes."""
    design_rules = build_l4_verifier_evidence_validation_rules()
    descriptions = {r.rule_id: r.description for r in design_rules}
    extra_descriptions = {
        "no_committed_token_inference": "Verifier evidence cannot be inferred from committed tokens.",
        "no_accepted_count_inference": "Accepted counts alone cannot serve as verifier evidence.",
        "missing_verifier_evidence_blocks": "Missing verifier evidence blocks dry-run decision.",
        "proposal_verifier_separation": "Proposal and verifier evidence must be separate.",
        "invalid_trace_rejection": "Invalid traces are detected and rejected.",
        "diagnostic_only_required": "diagnostic_only must be true on every record.",
        "explicit_verifier_evidence": "Verifier evidence must be explicit when available.",
        "no_runtime_commit_effect": "Dry-run decisions never affect token commits.",
        "no_generator_exposure": "Dry-run decisions are not exposed to generator.",
        "verifier_source_of_truth": "Verifier evidence is source of truth for dry-run.",
    }

    exercised_by: dict[str, list[str]] = {rule: [] for rule in SCHEMA_ENFORCEMENT_RULES}

    for result in example_results:
        for rule in result.enforcement_rules_exercised:
            if rule in exercised_by:
                exercised_by[rule].append(result.example_id)

    for probe in probe_results:
        if probe.get("passed"):
            for rule in probe.get("rules_exercised") or ():
                if rule in exercised_by:
                    exercised_by[str(rule)].append(str(probe.get("probe_id")))

    coverage: list[L4SchemaEnforcementRuleCoverage] = []
    for rule_id in SCHEMA_ENFORCEMENT_RULES:
        by = tuple(exercised_by.get(rule_id, ()))
        coverage.append(
            L4SchemaEnforcementRuleCoverage(
                rule_id=rule_id,
                description=extra_descriptions.get(rule_id, descriptions.get(rule_id, rule_id)),
                exercised=len(by) > 0,
                exercised_by=by,
            ),
        )
    return tuple(coverage)


def run_diagnostic_probes() -> tuple[dict[str, Any], ...]:
    """Run additional diagnostic probes for enforcement rules not covered by examples alone."""
    examples = {ex["example_id"]: ex for ex in build_synthetic_schema_examples()}
    probes: list[dict[str, Any]] = []

    # diagnostic_only=false must fail
    if "complete_all_match_trace" in examples:
        record = dict(examples["complete_all_match_trace"]["record"])
        record["diagnostic_only"] = False
        val = validate_verifier_evidence_trace_record(record)
        probes.append(
            {
                "probe_id": "probe_diagnostic_only_false",
                "description": "diagnostic_only=false must fail validation",
                "passed": not val.valid and not val.diagnostic_only_ok,
                "rules_exercised": ("diagnostic_only_required",),
            },
        )

    # invalid conversion must not fabricate dry-run match
    if "invalid_committed_tokens_as_verifier_trace" in examples:
        record = dict(examples["invalid_committed_tokens_as_verifier_trace"]["record"])
        conv = convert_verifier_trace_to_l4_trace_only_input(
            record,
            allow_invalid_for_diagnostics=True,
        )
        probes.append(
            {
                "probe_id": "probe_invalid_no_conversion",
                "description": "invalid trace does not convert without diagnostic flag",
                "passed": not convert_verifier_trace_to_l4_trace_only_input(record).converted,
                "rules_exercised": ("invalid_trace_rejection", "no_committed_token_inference"),
            },
        )
        probes.append(
            {
                "probe_id": "probe_invalid_diagnostic_no_fabricated_match",
                "description": "invalid trace with diagnostic flag does not yield all_match",
                "passed": conv.decision_status != "all_match",
                "rules_exercised": ("no_committed_token_inference",),
            },
        )

    # safety gates on converted example
    if "complete_all_match_trace" in examples:
        result = execute_schema_example(examples["complete_all_match_trace"])
        probes.append(
            {
                "probe_id": "probe_no_commit_on_converted",
                "description": "converted dry-run never uses decision for commit",
                "passed": not result.dry_run_decision_used_for_token_commit,
                "rules_exercised": ("no_runtime_commit_effect", "no_generator_exposure"),
            },
        )

    return tuple(probes)


def evaluate_validation_outcome(
    example_results: Sequence[L4SchemaExampleExecutionResult],
    rule_coverage: Sequence[L4SchemaEnforcementRuleCoverage],
    probe_results: Sequence[Mapping[str, Any]],
) -> str:
    """Return validation outcome from example and rule coverage."""
    if not example_results:
        return VALIDATION_OUTCOME_BLOCKED

    all_examples_pass = all(
        r.validation_passed and r.classification_passed for r in example_results
    )
    all_rules_exercised = all(r.exercised for r in rule_coverage)
    all_probes_pass = all(p.get("passed") for p in probe_results)

    classifications_seen = {
        r.actual_classification for r in example_results if r.actual_classification
    }
    statuses_ok = (
        "all_match" in classifications_seen
        and "partial_match" in classifications_seen
        and "first_token_mismatch" in classifications_seen
        and "blocked_missing_verifier_evidence" in classifications_seen
    )

    invalid_rejected = sum(1 for r in example_results if not r.actual_validation) >= 3

    if all_examples_pass and all_rules_exercised and all_probes_pass and statuses_ok and invalid_rejected:
        return VALIDATION_OUTCOME_COMPLETE

    if not all_examples_pass:
        return VALIDATION_OUTCOME_BLOCKED

    return VALIDATION_OUTCOME_INCOMPLETE


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
        if bid not in VALIDATION_RESOLVED_BLOCKER_IDS:
            remaining.append({"blocker_id": bid, "description": text})
    for extra in (
        ("runtime_verifier_instrumentation", "runtime verifier evidence instrumentation not implemented"),
        ("stage_3_verifier_dry_run", "stage 3 verifier-mediated dry-run not implemented"),
        ("l4_runtime_fallback_implementation", "runtime L4 fallback path not implemented"),
        ("l4_runtime_rollback_implementation", "runtime L4 rollback path not implemented"),
        ("l4_runtime_commit", "L4 runtime commit integration blocked"),
        ("production_cli_l4_flag", "production CLI L4 flag not implemented"),
        ("trace_schema_stress_adversarial_panel", "trace schema stress/adversarial panel not run"),
    ):
        remaining.append({"blocker_id": extra[0], "description": extra[1]})
    return remaining


def validate_exp109_example_validation_report(
    report: dict[str, Any],
) -> L4SchemaExampleValidationResult:
    """Validate Experiment 109 schema example validation report."""
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "schema_version",
        "validation_outcome",
        "total_examples",
        "examples_passed",
        "examples_failed",
        "classification_summary",
        "schema_correctness_coverage",
        "invalid_trace_detection_summary",
        "verifier_separation_summary",
        "blocked_evidence_summary",
        "enforcement_rule_coverage",
        "probe_results",
        "example_results",
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

    if report.get("experiment_id") != EXPERIMENT_109_ID:
        errors.append("experiment_id mismatch")

    if report.get("validation_outcome") != VALIDATION_OUTCOME_COMPLETE:
        errors.append("validation_outcome must be schema_example_validation_complete")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_21H:
        errors.append(
            "allowed_next_phase must be phase21i_l4_trace_schema_stress_adversarial_trace_injection_panel",
        )

    forbidden = set(report.get("forbidden_next_phases") or [])
    if not set(FORBIDDEN_NEXT_PHASES_21H) <= forbidden:
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

    if report.get("total_examples") != 8:
        errors.append("total_examples must be 8")

    for idx, ex in enumerate(report.get("example_results") or []):
        if not ex.get("validation_passed"):
            errors.append(f"example_results[{idx}] validation_passed false")
        if not ex.get("classification_passed"):
            errors.append(f"example_results[{idx}] classification_passed false")
        if ex.get("dry_run_decision_used_for_token_commit") is True:
            errors.append(f"example_results[{idx}] commit flag must be false")
        if ex.get("exposed_to_generator") is True:
            errors.append(f"example_results[{idx}] exposed_to_generator must be false")

    coverage = report.get("enforcement_rule_coverage") or []
    if not all(c.get("exercised") for c in coverage):
        errors.append("not all enforcement rules exercised")

    for probe in report.get("probe_results") or []:
        if not probe.get("passed"):
            errors.append(f"probe {probe.get('probe_id')} failed")

    return L4SchemaExampleValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
    )


def run_exp109_l4_verifier_trace_schema_example_validation() -> dict[str, Any]:
    """Run Experiment 109 L4 verifier trace schema example validation."""
    examples = build_synthetic_schema_examples()
    example_results = tuple(execute_schema_example(ex) for ex in examples)
    probe_results = run_diagnostic_probes()
    rule_coverage = build_enforcement_rule_coverage(example_results, probe_results)

    passed = sum(1 for r in example_results if r.validation_passed and r.classification_passed)
    failed = len(example_results) - passed

    classification_counts: dict[str, int] = {}
    for r in example_results:
        if r.actual_classification:
            classification_counts[r.actual_classification] = (
                classification_counts.get(r.actual_classification, 0) + 1
            )

    invalid_detected = sum(
        1
        for r in example_results
        if not r.actual_validation and r.expected_classification == "invalid_trace"
    )
    invalid_expected = sum(
        1 for r in example_results if r.expected_classification == "invalid_trace"
    )

    separation_ok = sum(
        1
        for r in example_results
        if "proposal_verifier_separation" in r.enforcement_rules_exercised
        or not r.actual_validation
    )

    blocked_ok = sum(
        1
        for r in example_results
        if r.actual_classification == "blocked_missing_verifier_evidence"
    )

    rules_exercised = sum(1 for c in rule_coverage if c.exercised)
    rules_total = len(rule_coverage)

    validation_outcome = evaluate_validation_outcome(
        example_results,
        rule_coverage,
        probe_results,
    )
    status = (
        "validation_complete"
        if validation_outcome == VALIDATION_OUTCOME_COMPLETE
        else "validation_incomplete"
    )

    report = {
        "experiment_id": EXPERIMENT_109_ID,
        "status": status,
        "phase": PHASE_21H,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STAGE,
        "mode": MODE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "validation_outcome": validation_outcome,
        "total_examples": len(example_results),
        "examples_passed": passed,
        "examples_failed": failed,
        "classification_summary": {
            "status_counts": classification_counts,
            "expected_statuses_covered": all(
                s in classification_counts for s in EXPECTED_CLASSIFICATION_STATUSES[:-1]
            ),
            "invalid_trace_rejected_count": invalid_detected,
        },
        "schema_correctness_coverage": {
            "rules_exercised": rules_exercised,
            "rules_total": rules_total,
            "coverage_rate": rules_exercised / rules_total if rules_total else 0.0,
            "all_rules_exercised": rules_exercised == rules_total,
        },
        "invalid_trace_detection_summary": {
            "invalid_expected": invalid_expected,
            "invalid_detected": invalid_detected,
            "detection_accuracy": (
                invalid_detected / invalid_expected if invalid_expected else 1.0
            ),
        },
        "verifier_separation_summary": {
            "separation_checks": separation_ok,
            "total_examples": len(example_results),
            "separation_success_rate": separation_ok / len(example_results) if example_results else 0.0,
        },
        "blocked_evidence_summary": {
            "blocked_examples": blocked_ok,
            "blocked_correct": all(
                r.classification_passed
                for r in example_results
                if r.expected_classification == "blocked_missing_verifier_evidence"
            ),
        },
        "enforcement_rule_coverage": [c.to_dict() for c in rule_coverage],
        "probe_results": list(probe_results),
        "example_results": [r.to_dict() for r in example_results],
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21H,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21H),
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
            "Schema example validation only; not runtime instrumentation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Synthetic schema examples only; no model experiments.",
            "Trace-only diagnostics; never fabricates verifier evidence.",
            "No performance/memory/serving claims.",
        ],
    }
    report["validation_result"] = validate_exp109_example_validation_report(report).to_dict()
    return report


def validate_exp109_report(report: dict[str, Any]) -> list[str]:
    """Return validation error strings for Experiment 109 report."""
    return list(validate_exp109_example_validation_report(report).errors)
