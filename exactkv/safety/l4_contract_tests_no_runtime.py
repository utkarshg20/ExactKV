"""L4 contract tests with no runtime integration (Phase 20C / Exp 100).

Pure synthetic contract evaluator only — must not be imported by runtime generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_verifier_mediated_design_spec import (
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    build_l4_claim_boundaries,
)
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_100_ID = "exp100_l4_contract_tests_no_runtime"
DEFAULT_EXP100_REPORT = Path(
    "reports/experiment_100_l4_contract_tests_no_runtime.json",
)
PHASE_20C = "20C"
L4_SAFETY_LEVEL = "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"

RECOMMENDED_NEXT_PHASE_20C = "phase20d_l4_integration_plan_review"
FORBIDDEN_NEXT_PHASES_20C: tuple[str, ...] = (
    "phase20c_l4_runtime_implementation",
    "phase20d_l4_runtime_implementation",
    "cuda_backend_implementation",
    "vllm_integration",
    "lmcache_integration",
    "performance_benchmark",
    "memory_benchmark",
)

CASE_STATUS_PASS = "pass"
CASE_STATUS_FAIL = "fail"

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES

CONTRACT_TEST_RESOLVED_BLOCKER_IDS: frozenset[str] = frozenset(
    {"l4_synthetic_contract_tests_no_runtime"},
)


@dataclass(frozen=True)
class L4SyntheticProposal:
    """Synthetic draft proposal tokens (contract test only)."""

    token_ids: tuple[int, ...]
    source: str = "synthetic_l4_contract_test"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SyntheticVerifierEvidence:
    """Synthetic full-verifier evidence (source of truth in contract tests)."""

    token_ids: tuple[int, ...]
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SyntheticContractCase:
    """Single synthetic L4 contract test case."""

    case_id: str
    description: str
    proposal_token_ids: tuple[int, ...]
    verifier_token_ids: tuple[int, ...]
    expected_accepted_prefix: tuple[int, ...]
    expected_rejected_suffix: tuple[int, ...]
    proposal_exception: bool = False
    verifier_evidence_missing: bool = False
    hidden_divergence_attempt: bool = False
    direct_commit_attempt: bool = False
    expected_status: str = CASE_STATUS_PASS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SyntheticContractDecision:
    """Contract decision record for a synthetic case."""

    accepted_prefix: tuple[int, ...]
    rejected_suffix: tuple[int, ...]
    verifier_source_of_truth: bool
    direct_commit_rejected: bool
    hidden_divergence_detected: bool
    fallback_required: bool
    fallback_triggered: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SyntheticTrace:
    """Trace of contract evaluation decisions (no runtime objects)."""

    proposal_token_ids: tuple[int, ...]
    verifier_token_ids: tuple[int, ...]
    accepted_prefix: tuple[int, ...]
    rejected_suffix: tuple[int, ...]
    decision_steps: tuple[str, ...]
    fallback_triggered: bool
    direct_commit_attempt_blocked: bool
    hidden_divergence_blocked: bool
    trace_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class L4SyntheticContractResult:
    """Result of evaluating one synthetic contract case."""

    case_id: str
    expected_status: str
    actual_status: str
    case_passed: bool
    decision: L4SyntheticContractDecision
    trace: L4SyntheticTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "case_passed": self.case_passed,
            "decision": self.decision.to_dict(),
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class L4ContractTestSuiteResult:
    """Aggregated L4 contract test suite outcome."""

    total_cases: int
    passing_cases: int
    failing_cases: int
    expected_fail_cases: int
    unexpected_fail_cases: int
    fallback_cases: int
    hidden_divergence_failures_detected: int
    direct_commit_failures_detected: int
    trace_complete_cases: int
    suite_status: str
    all_expected_pass_cases_passed: bool
    expected_fail_cases_failed_correctly: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def evaluate_l4_synthetic_contract_case(
    case: L4SyntheticContractCase,
) -> L4SyntheticContractResult:
    """Evaluate one synthetic L4 contract case (pure, no runtime generation)."""
    steps: list[str] = []
    failure_reasons: list[str] = []
    fallback_triggered = False
    direct_commit_rejected = False
    hidden_divergence_detected = False
    verifier_source_of_truth = True

    proposal = case.proposal_token_ids
    verifier = case.verifier_token_ids
    accepted: tuple[int, ...] = ()
    rejected: tuple[int, ...] = ()

    steps.append("contract_evaluator_started")

    if case.direct_commit_attempt:
        steps.append("direct_commit_attempt_detected")
        direct_commit_rejected = True
        failure_reasons.append("direct_commit_attempt_blocked")
        actual_status = CASE_STATUS_FAIL
        trace = L4SyntheticTrace(
            proposal_token_ids=proposal,
            verifier_token_ids=verifier,
            accepted_prefix=(),
            rejected_suffix=proposal,
            decision_steps=tuple(steps),
            fallback_triggered=False,
            direct_commit_attempt_blocked=True,
            hidden_divergence_blocked=False,
            trace_complete=True,
        )
        decision = L4SyntheticContractDecision(
            accepted_prefix=(),
            rejected_suffix=proposal,
            verifier_source_of_truth=True,
            direct_commit_rejected=True,
            hidden_divergence_detected=False,
            fallback_required=False,
            fallback_triggered=False,
            failure_reasons=tuple(failure_reasons),
        )
        return L4SyntheticContractResult(
            case_id=case.case_id,
            expected_status=case.expected_status,
            actual_status=actual_status,
            case_passed=actual_status == case.expected_status,
            decision=decision,
            trace=trace,
        )

    if case.hidden_divergence_attempt:
        steps.append("hidden_divergence_attempt_detected")
        hidden_divergence_detected = True
        failure_reasons.append("hidden_divergence_blocked")
        actual_status = CASE_STATUS_FAIL
        trace = L4SyntheticTrace(
            proposal_token_ids=proposal,
            verifier_token_ids=verifier,
            accepted_prefix=(),
            rejected_suffix=proposal,
            decision_steps=tuple(steps),
            fallback_triggered=False,
            direct_commit_attempt_blocked=False,
            hidden_divergence_blocked=True,
            trace_complete=True,
        )
        decision = L4SyntheticContractDecision(
            accepted_prefix=(),
            rejected_suffix=proposal,
            verifier_source_of_truth=True,
            direct_commit_rejected=False,
            hidden_divergence_detected=True,
            fallback_required=False,
            fallback_triggered=False,
            failure_reasons=tuple(failure_reasons),
        )
        return L4SyntheticContractResult(
            case_id=case.case_id,
            expected_status=case.expected_status,
            actual_status=actual_status,
            case_passed=actual_status == case.expected_status,
            decision=decision,
            trace=trace,
        )

    if case.proposal_exception:
        steps.append("proposal_exception_detected")
        steps.append("fallback_triggered")
        fallback_triggered = True
        accepted = ()
        rejected = ()
        actual_status = CASE_STATUS_PASS
        trace = L4SyntheticTrace(
            proposal_token_ids=proposal,
            verifier_token_ids=verifier,
            accepted_prefix=accepted,
            rejected_suffix=rejected,
            decision_steps=tuple(steps),
            fallback_triggered=True,
            direct_commit_attempt_blocked=False,
            hidden_divergence_blocked=False,
            trace_complete=True,
        )
        decision = L4SyntheticContractDecision(
            accepted_prefix=accepted,
            rejected_suffix=rejected,
            verifier_source_of_truth=True,
            direct_commit_rejected=False,
            hidden_divergence_detected=False,
            fallback_required=True,
            fallback_triggered=True,
            failure_reasons=(),
        )
        return L4SyntheticContractResult(
            case_id=case.case_id,
            expected_status=case.expected_status,
            actual_status=actual_status,
            case_passed=actual_status == case.expected_status,
            decision=decision,
            trace=trace,
        )

    if case.verifier_evidence_missing:
        steps.append("verifier_evidence_missing")
        steps.append("fallback_triggered")
        fallback_triggered = True
        accepted = ()
        rejected = ()
        actual_status = CASE_STATUS_PASS
        trace = L4SyntheticTrace(
            proposal_token_ids=proposal,
            verifier_token_ids=(),
            accepted_prefix=accepted,
            rejected_suffix=rejected,
            decision_steps=tuple(steps),
            fallback_triggered=True,
            direct_commit_attempt_blocked=False,
            hidden_divergence_blocked=False,
            trace_complete=True,
        )
        decision = L4SyntheticContractDecision(
            accepted_prefix=accepted,
            rejected_suffix=rejected,
            verifier_source_of_truth=True,
            direct_commit_rejected=False,
            hidden_divergence_detected=False,
            fallback_required=True,
            fallback_triggered=True,
            failure_reasons=(),
        )
        return L4SyntheticContractResult(
            case_id=case.case_id,
            expected_status=case.expected_status,
            actual_status=actual_status,
            case_passed=actual_status == case.expected_status,
            decision=decision,
            trace=trace,
        )

    steps.append("verifier_evidence_used_as_source_of_truth")
    accepted = _longest_matching_prefix(proposal, verifier)
    rejected = proposal[len(accepted) :]
    steps.append(f"accepted_prefix_length={len(accepted)}")
    steps.append(f"rejected_suffix_length={len(rejected)}")

    prefix_ok = accepted == case.expected_accepted_prefix
    suffix_ok = rejected == case.expected_rejected_suffix
    if not prefix_ok:
        failure_reasons.append("accepted_prefix_mismatch")
    if not suffix_ok:
        failure_reasons.append("rejected_suffix_mismatch")

    actual_status = CASE_STATUS_PASS if prefix_ok and suffix_ok else CASE_STATUS_FAIL

    trace = L4SyntheticTrace(
        proposal_token_ids=proposal,
        verifier_token_ids=verifier,
        accepted_prefix=accepted,
        rejected_suffix=rejected,
        decision_steps=tuple(steps),
        fallback_triggered=False,
        direct_commit_attempt_blocked=False,
        hidden_divergence_blocked=False,
        trace_complete=True,
    )
    decision = L4SyntheticContractDecision(
        accepted_prefix=accepted,
        rejected_suffix=rejected,
        verifier_source_of_truth=verifier_source_of_truth,
        direct_commit_rejected=False,
        hidden_divergence_detected=False,
        fallback_required=False,
        fallback_triggered=False,
        failure_reasons=tuple(failure_reasons),
    )
    return L4SyntheticContractResult(
        case_id=case.case_id,
        expected_status=case.expected_status,
        actual_status=actual_status,
        case_passed=actual_status == case.expected_status,
        decision=decision,
        trace=trace,
    )


def build_default_l4_contract_test_suite() -> tuple[L4SyntheticContractCase, ...]:
    """Build the default synthetic L4 contract test cases."""
    return (
        L4SyntheticContractCase(
            case_id="all_match_accept_all",
            description="Proposal fully matches verifier; accept entire proposal.",
            proposal_token_ids=(1, 2, 3, 4),
            verifier_token_ids=(1, 2, 3, 4),
            expected_accepted_prefix=(1, 2, 3, 4),
            expected_rejected_suffix=(),
            expected_status=CASE_STATUS_PASS,
        ),
        L4SyntheticContractCase(
            case_id="partial_match_accept_prefix",
            description="Proposal diverges after shared prefix; accept longest match only.",
            proposal_token_ids=(1, 2, 9, 9),
            verifier_token_ids=(1, 2, 3, 4),
            expected_accepted_prefix=(1, 2),
            expected_rejected_suffix=(9, 9),
            expected_status=CASE_STATUS_PASS,
        ),
        L4SyntheticContractCase(
            case_id="first_token_mismatch_accept_none",
            description="First-token mismatch yields empty accepted prefix.",
            proposal_token_ids=(9, 9, 9),
            verifier_token_ids=(1, 2, 3),
            expected_accepted_prefix=(),
            expected_rejected_suffix=(9, 9, 9),
            expected_status=CASE_STATUS_PASS,
        ),
        L4SyntheticContractCase(
            case_id="proposal_exception_fallback",
            description="Proposal exception triggers fallback without acceptance.",
            proposal_token_ids=(1, 2, 3),
            verifier_token_ids=(1, 2, 3),
            expected_accepted_prefix=(),
            expected_rejected_suffix=(),
            proposal_exception=True,
            expected_status=CASE_STATUS_PASS,
        ),
        L4SyntheticContractCase(
            case_id="missing_verifier_evidence_fallback",
            description="Missing verifier evidence triggers fallback.",
            proposal_token_ids=(1, 2, 3),
            verifier_token_ids=(1, 2, 3),
            expected_accepted_prefix=(),
            expected_rejected_suffix=(),
            verifier_evidence_missing=True,
            expected_status=CASE_STATUS_PASS,
        ),
        L4SyntheticContractCase(
            case_id="hidden_divergence_attempt_fails",
            description="Hidden divergence attempt must fail contract evaluation.",
            proposal_token_ids=(1, 2, 3),
            verifier_token_ids=(1, 2, 3),
            expected_accepted_prefix=(),
            expected_rejected_suffix=(),
            hidden_divergence_attempt=True,
            expected_status=CASE_STATUS_FAIL,
        ),
        L4SyntheticContractCase(
            case_id="direct_commit_attempt_fails",
            description="Direct proposal commit attempt must fail contract evaluation.",
            proposal_token_ids=(1, 2, 3),
            verifier_token_ids=(1, 2, 3),
            expected_accepted_prefix=(),
            expected_rejected_suffix=(),
            direct_commit_attempt=True,
            expected_status=CASE_STATUS_FAIL,
        ),
    )


def run_l4_contract_test_suite(
    cases: tuple[L4SyntheticContractCase, ...] | None = None,
) -> tuple[L4SyntheticContractResult, ...]:
    """Run the L4 contract test suite on synthetic cases."""
    suite_cases = cases if cases is not None else build_default_l4_contract_test_suite()
    return tuple(evaluate_l4_synthetic_contract_case(c) for c in suite_cases)


def _aggregate_suite_results(
    results: tuple[L4SyntheticContractResult, ...],
    cases: tuple[L4SyntheticContractCase, ...],
) -> L4ContractTestSuiteResult:
    case_by_id = {c.case_id: c for c in cases}
    passing = sum(1 for r in results if r.case_passed)
    failing = len(results) - passing

    expected_fail_cases = sum(
        1 for c in cases if c.expected_status == CASE_STATUS_FAIL
    )
    expected_fail_failed_correctly = all(
        r.actual_status == CASE_STATUS_FAIL and r.case_passed
        for r in results
        if case_by_id[r.case_id].expected_status == CASE_STATUS_FAIL
    )
    all_expected_pass_passed = all(
        r.case_passed
        for r in results
        if case_by_id[r.case_id].expected_status == CASE_STATUS_PASS
    )

    unexpected_fail = sum(
        1
        for r in results
        if not r.case_passed
        and case_by_id[r.case_id].expected_status == CASE_STATUS_PASS
    )

    fallback_cases = sum(1 for r in results if r.decision.fallback_triggered)
    hidden_div = sum(
        1 for r in results if r.decision.hidden_divergence_detected
    )
    direct_commit = sum(1 for r in results if r.decision.direct_commit_rejected)
    trace_complete = sum(1 for r in results if r.trace.trace_complete)

    suite_ok = (
        all_expected_pass_passed
        and expected_fail_failed_correctly
        and trace_complete == len(results)
    )
    suite_status = "contract_tests_complete" if suite_ok else "contract_tests_incomplete"

    return L4ContractTestSuiteResult(
        total_cases=len(results),
        passing_cases=passing,
        failing_cases=failing,
        expected_fail_cases=expected_fail_cases,
        unexpected_fail_cases=unexpected_fail,
        fallback_cases=fallback_cases,
        hidden_divergence_failures_detected=hidden_div,
        direct_commit_failures_detected=direct_commit,
        trace_complete_cases=trace_complete,
        suite_status=suite_status,
        all_expected_pass_cases_passed=all_expected_pass_passed,
        expected_fail_cases_failed_correctly=expected_fail_failed_correctly,
    )


def _remaining_implementation_blockers() -> list[dict[str, Any]]:
    resolved_by_contract_tests = CONTRACT_TEST_RESOLVED_BLOCKER_IDS | {
        "explicit_l4_design_spec",
        "verifier_mediated_acceptance_contract",
        "rollback_behavior_defined",
        "l4_test_matrix_defined",
        "l4_opt_in_flag_designed",
    }
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
        if bid not in resolved_by_contract_tests:
            remaining.append({"blocker_id": bid, "description": text})
    remaining.append(
        {
            "blocker_id": "l4_runtime_fallback_implementation",
            "description": "runtime fallback path not yet implemented for L4",
        },
    )
    remaining.append(
        {
            "blocker_id": "l4_runtime_rollback_implementation",
            "description": "runtime rollback path not yet implemented for L4",
        },
    )
    return remaining


def run_exp100_l4_contract_tests_no_runtime() -> dict[str, Any]:
    """Run Experiment 100 L4 contract tests (no runtime integration)."""
    cases = build_default_l4_contract_test_suite()
    results = run_l4_contract_test_suite(cases)
    suite = _aggregate_suite_results(results, cases)

    verifier_source_of_truth = all(
        r.decision.verifier_source_of_truth for r in results
    )
    direct_commit_rejected = all(
        not r.decision.direct_commit_rejected or r.actual_status == CASE_STATUS_FAIL
        for r in results
    )
    hidden_divergence_detected = any(
        r.decision.hidden_divergence_detected for r in results
    )

    status = (
        "contract_tests_complete"
        if suite.suite_status == "contract_tests_complete"
        else "contract_tests_incomplete"
    )

    return {
        "experiment_id": EXPERIMENT_100_ID,
        "status": status,
        "phase": PHASE_20C,
        "safety_level": L4_SAFETY_LEVEL,
        "contract_cases": [c.to_dict() for c in cases],
        "contract_results": [r.to_dict() for r in results],
        "suite_summary": suite.to_dict(),
        "verifier_source_of_truth": verifier_source_of_truth,
        "direct_commit_attempts_rejected": direct_commit_rejected,
        "hidden_divergence_attempts_detected": hidden_divergence_detected,
        "fallback_cases": suite.fallback_cases,
        "trace_completeness_summary": {
            "trace_complete_cases": suite.trace_complete_cases,
            "total_cases": suite.total_cases,
            "all_traces_complete": suite.trace_complete_cases == suite.total_cases,
        },
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "runtime_generation_path_modified": False,
        "l4_runtime_implementation_added": False,
        "cli_opt_in_added": False,
        "model_experiments_run": False,
        "proposal_used_for_token_commit": False,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_20C,
        "forbidden_next_phase": FORBIDDEN_NEXT_PHASES_20C[0],
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_20C),
        "implementation_blockers_remaining": _remaining_implementation_blockers(),
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "No-runtime L4 contract testing only; not L4 runtime implementation.",
            "ExactKVGenerator and default runtime unchanged.",
            "Synthetic token sequences only; no model experiments.",
            "Runtime fallback and rollback behavior not implemented.",
            "Performance, memory, and serving claims remain forbidden.",
        ],
    }


def validate_exp100_report(report: dict[str, Any]) -> list[str]:
    """Validate Experiment 100 report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "safety_level",
        "contract_cases",
        "suite_summary",
        "verifier_source_of_truth",
        "direct_commit_attempts_rejected",
        "hidden_divergence_attempts_detected",
        "fallback_cases",
        "trace_completeness_summary",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "runtime_generation_path_modified",
        "l4_runtime_implementation_added",
        "cli_opt_in_added",
        "model_experiments_run",
        "proposal_used_for_token_commit",
        "allowed_next_phase",
        "forbidden_next_phase",
        "implementation_blockers_remaining",
        "claim_boundaries",
        "no_performance_claims_note",
        "limitations",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_100_ID:
        errors.append("experiment_id mismatch")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_20C:
        errors.append("allowed_next_phase must be phase20d_l4_integration_plan_review")

    forbidden = report.get("forbidden_next_phase")
    if forbidden not in FORBIDDEN_NEXT_PHASES_20C:
        errors.append("forbidden_next_phase must be a known forbidden phase")

    for flag in (
        "exactkv_generator_modified",
        "default_runtime_changed",
        "runtime_generation_path_modified",
        "l4_runtime_implementation_added",
        "cli_opt_in_added",
        "model_experiments_run",
        "proposal_used_for_token_commit",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    if report.get("verifier_source_of_truth") is not True:
        errors.append("verifier_source_of_truth must be true")

    if report.get("direct_commit_attempts_rejected") is not True:
        errors.append("direct_commit_attempts_rejected must be true")

    suite = report.get("suite_summary") or {}
    if suite.get("suite_status") != "contract_tests_complete":
        errors.append("suite_summary.suite_status must be contract_tests_complete")

    case_ids = {c["case_id"] for c in report.get("contract_cases") or []}
    required_cases = {
        "all_match_accept_all",
        "partial_match_accept_prefix",
        "first_token_mismatch_accept_none",
        "proposal_exception_fallback",
        "missing_verifier_evidence_fallback",
        "hidden_divergence_attempt_fails",
        "direct_commit_attempt_fails",
    }
    if not required_cases <= case_ids:
        errors.append("missing required synthetic contract cases")

    return errors
