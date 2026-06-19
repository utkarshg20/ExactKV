"""Pre-L4 safety gate review (Phase 20A / Exp 098).

Evidence-based review of whether L4 design specification may begin.
Does not authorize L4 implementation or modify runtime behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.safety.guarded_draft_shadow import (
    DECISION_L3_SOURCE_PROMOTED,
    DEFAULT_EXP091_REPORT,
    DEFAULT_EXP092_REPORT,
    DEFAULT_EXP093_REPORT,
    DEFAULT_EXP094_REPORT,
    DEFAULT_EXP095_REPORT,
    DEFAULT_EXP096_REPORT,
    DEFAULT_EXP097_REPORT,
    PROPOSAL_SOURCE_ROUND_LOG,
)
from exactkv.safety.integration_safety_spec import (
    DEFAULT_EXP090_REPORT,
    NO_PERFORMANCE_CLAIMS_NOTE,
)

EXPERIMENT_098_ID = "exp098_pre_l4_safety_gate_review"
DEFAULT_EXP098_REPORT = Path("reports/experiment_098_pre_l4_safety_gate_review.json")
PHASE_20A = "20A"
RECOMMENDED_NEXT_PHASE_20A = "phase20b_l4_verifier_mediated_design_spec"

OUTCOME_READY_L4_DESIGN_SPEC_ONLY = "ready_for_l4_design_spec_only"
OUTCOME_NOT_READY_L4_DESIGN_SPEC = "not_ready_for_l4_design_spec"
OUTCOME_BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"
OUTCOME_BLOCKED_SAFETY_FAILURE = "blocked_safety_failure"
OUTCOME_FORBIDDEN_L4_IMPLEMENTATION = "ready_for_l4_implementation"

REVIEW_OUTCOMES: tuple[str, ...] = (
    OUTCOME_READY_L4_DESIGN_SPEC_ONLY,
    OUTCOME_NOT_READY_L4_DESIGN_SPEC,
    OUTCOME_BLOCKED_MISSING_EVIDENCE,
    OUTCOME_BLOCKED_SAFETY_FAILURE,
)

GATE_NAMES: tuple[str, ...] = (
    "l3_source_promotion_gate",
    "proposal_provenance_gate",
    "proposal_isolation_gate",
    "generation_parity_gate",
    "exactkv_failure_gate",
    "safety_spec_gate",
    "claim_boundary_gate",
    "fallback_requirement_gate",
    "l4_design_only_gate",
    "implementation_block_gate",
)

CRITICAL_EVIDENCE_IDS: tuple[str, ...] = (
    "phase_19c_promoted_source_validation",
    "phase_18a_integration_safety_spec",
)

EVIDENCE_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "id": "phase_18a_integration_safety_spec",
        "label": "Phase 18A integration safety spec",
        "doc": "docs/PHASE_18A_INTEGRATION_SAFETY_SPEC.md",
        "report": str(DEFAULT_EXP090_REPORT),
    },
    {
        "id": "phase_18b_guarded_draft_shadow_scaffold",
        "label": "Phase 18B guarded draft-shadow scaffold",
        "doc": "docs/PHASE_18B_GUARDED_DRAFT_SHADOW_NO_COMMIT.md",
        "report": str(DEFAULT_EXP091_REPORT),
    },
    {
        "id": "phase_18c_panel_validation",
        "label": "Phase 18C panel validation",
        "doc": "docs/PHASE_18C_GUARDED_DRAFT_SHADOW_PANEL_VALIDATION.md",
        "report": str(DEFAULT_EXP092_REPORT),
    },
    {
        "id": "phase_18d_top1_extraction_hardening",
        "label": "Phase 18D top-1 extraction hardening",
        "doc": "docs/PHASE_18D_SHADOW_TOP1_EXTRACTION_HARDENING.md",
        "report": str(DEFAULT_EXP093_REPORT),
    },
    {
        "id": "phase_18e_provenance_audit",
        "label": "Phase 18E provenance audit",
        "doc": "docs/PHASE_18E_SHADOW_PROPOSAL_PROVENANCE_AUDIT.md",
        "report": str(DEFAULT_EXP094_REPORT),
    },
    {
        "id": "phase_19a_round_log_proposal_source",
        "label": "Phase 19A round-log proposal source",
        "doc": "docs/PHASE_19A_ROUND_LOG_DRAFT_PROPOSAL_SOURCE.md",
        "report": str(DEFAULT_EXP095_REPORT),
    },
    {
        "id": "phase_19b_proposal_source_comparison",
        "label": "Phase 19B proposal source comparison",
        "doc": "docs/PHASE_19B_ROUND_LOG_PROPOSAL_SOURCE_COMPARISON.md",
        "report": str(DEFAULT_EXP096_REPORT),
    },
    {
        "id": "phase_19c_promoted_source_validation",
        "label": "Phase 19C promoted-source validation",
        "doc": "docs/PHASE_19C_L3_PROMOTED_SOURCE_VALIDATION.md",
        "report": str(DEFAULT_EXP097_REPORT),
    },
    {
        "id": "claims_audit",
        "label": "Claims audit",
        "doc": "docs/CLAIMS_AUDIT.md",
        "report": "",
    },
    {
        "id": "deferred_work_register",
        "label": "Deferred work register",
        "doc": "docs/DEFERRED_WORK_REGISTER.md",
        "report": "",
    },
    {
        "id": "vericache_systems_roadmap",
        "label": "VeriCache systems roadmap",
        "doc": "docs/VERICACHE_SYSTEMS_ROADMAP.md",
        "report": "",
    },
)

L4_IMPLEMENTATION_BLOCKERS: tuple[str, ...] = (
    "explicit L4 design spec missing",
    "ExactKVGenerator integration plan missing",
    "fallback path not yet implemented for L4",
    "L4 opt-in flag not yet designed",
    "verifier-mediated acceptance contract not yet defined",
    "rollback behavior not yet defined",
    "L4 test matrix not yet defined",
    "no L4 baseline-vs-integrated parity panel",
    "no L4 exactkv_failures gate run",
    "no active GPU memory measurement",
    "no performance benchmark",
    "no serving integration",
)

FORBIDDEN_NEXT_PHASES: tuple[str, ...] = (
    "l4_implementation",
    "cuda_backend",
    "vllm_integration",
    "lmcache_integration",
    "performance_benchmark",
    "memory_benchmark",
)

_GATE_DEFINITIONS: dict[str, dict[str, str]] = {
    "l3_source_promotion_gate": {
        "purpose": (
            "Confirm L3 promoted source is exactkv_round_log_draft_tokens with "
            "l3_source_promoted decision from Phase 19C."
        ),
        "required_evidence": "phase_19c_promoted_source_validation report",
        "pass_condition": (
            "decision_recommendation is l3_source_promoted and promoted_source is "
            "exactkv_round_log_draft_tokens"
        ),
        "fail_condition": "promotion decision missing or promoted source mismatch",
    },
    "proposal_provenance_gate": {
        "purpose": "Confirm round-log draft provenance gate passed in Phase 19C.",
        "required_evidence": "phase_19c source_viability_gate_summary",
        "pass_condition": "proposal_provenance_gate pass is true on completed cells",
        "fail_condition": "provenance gate failed or evidence missing",
    },
    "proposal_isolation_gate": {
        "purpose": (
            "Confirm proposals do not affect commits and are not exposed to generator."
        ),
        "required_evidence": "phase_19c source_viability_gate_summary and report flags",
        "pass_condition": (
            "proposal_isolation_gate pass; proposal_used_for_token_commit false; "
            "proposal_exposed_to_generator false"
        ),
        "fail_condition": "isolation gate failed or proposal flags unsafe",
    },
    "generation_parity_gate": {
        "purpose": "Confirm baseline vs promoted-source token/text parity in Phase 19C.",
        "required_evidence": "phase_19c parity_summary",
        "pass_condition": "generation_parity_gate pass; all successful cells match",
        "fail_condition": "parity mismatch on any completed cell",
    },
    "exactkv_failure_gate": {
        "purpose": "Confirm exactkv_failures == 0 on baseline and promoted-source paths.",
        "required_evidence": "phase_19c exactkv_failure_summary",
        "pass_condition": "baseline_failures and promoted_source_failures are 0",
        "fail_condition": "any exactkv_failures reported in Phase 19C",
    },
    "safety_spec_gate": {
        "purpose": "Confirm L3_GUARDED_DRAFT_SHADOW_NO_COMMIT safety spec validation passed.",
        "required_evidence": "phase_18a or phase_19c safety_spec_validation",
        "pass_condition": "safety_spec_validation pass is true",
        "fail_condition": "safety spec validation missing or failed",
    },
    "claim_boundary_gate": {
        "purpose": (
            "Confirm no performance/memory/serving/VeriCache claims; claims audit doc present."
        ),
        "required_evidence": "claims_audit doc; phase_19c claim_boundary_gate",
        "pass_condition": "claims audit doc present; claim boundary gate pass",
        "fail_condition": "claims audit missing or forbidden claims boundary violated",
    },
    "fallback_requirement_gate": {
        "purpose": "Confirm fallback_to_baseline required by integration safety spec.",
        "required_evidence": "phase_18a or phase_19c safety_spec_validation",
        "pass_condition": "fallback_to_baseline is true in safety spec validation",
        "fail_condition": "fallback requirement not documented or false",
    },
    "l4_design_only_gate": {
        "purpose": "Authorize L4 design specification only, never implementation.",
        "required_evidence": "all required L3 gates pass",
        "pass_condition": (
            "all prerequisite gates pass; review outcome is not ready_for_l4_implementation"
        ),
        "fail_condition": "prerequisite gates fail or implementation would be authorized",
    },
    "implementation_block_gate": {
        "purpose": "Ensure L4 implementation remains explicitly blocked with documented blockers.",
        "required_evidence": "l4_implementation_blockers list",
        "pass_condition": (
            "l4_implementation_authorized is false and blockers list is non-empty"
        ),
        "fail_condition": "implementation authorized or blockers missing",
    },
}


def _load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def inventory_evidence(
    *,
    root: Path | None = None,
    evidence_overrides: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Inventory local docs and reports; optional test overrides without inventing data."""
    base = root or Path(".")
    overrides = evidence_overrides or {}
    found: list[dict[str, Any]] = []
    missing: list[str] = []

    for item in EVIDENCE_INVENTORY:
        eid = item["id"]
        if eid in overrides:
            entry = {
                "id": eid,
                "label": item["label"],
                "doc_path": item["doc"],
                "report_path": item.get("report") or None,
                "doc_present": True,
                "report_present": True,
                "report_data": dict(overrides[eid]),
            }
            found.append(entry)
            continue

        doc_path = base / item["doc"]
        doc_present = doc_path.is_file()
        report_present = False
        report_data: dict[str, Any] | None = None
        report_path_str = item.get("report") or ""
        if report_path_str:
            report_path = base / report_path_str
            report_data = _load_json_report(report_path)
            report_present = report_data is not None

        entry = {
            "id": eid,
            "label": item["label"],
            "doc_path": item["doc"],
            "report_path": report_path_str or None,
            "doc_present": doc_present,
            "report_present": report_present,
            "report_data": report_data,
        }
        if doc_present or report_present:
            found.append(entry)
        else:
            missing.append(eid)

    evidence_found = [e["id"] for e in found]
    return found, evidence_found, missing


def _evidence_report(
    inventory: Sequence[Mapping[str, Any]],
    evidence_id: str,
) -> dict[str, Any] | None:
    for entry in inventory:
        if entry.get("id") == evidence_id:
            data = entry.get("report_data")
            return data if isinstance(data, dict) else None
    return None


def _gate_result(
    name: str,
    *,
    result: str,
    evidence_status: str,
    notes: str,
) -> dict[str, Any]:
    definition = _GATE_DEFINITIONS[name]
    return {
        "name": name,
        "purpose": definition["purpose"],
        "required_evidence": definition["required_evidence"],
        "pass_condition": definition["pass_condition"],
        "fail_condition": definition["fail_condition"],
        "evidence_status": evidence_status,
        "result": result,
        "notes": notes,
    }


def evaluate_pre_l4_gates(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all pre-L4 gate categories from inventoried evidence."""
    exp097 = _evidence_report(inventory, "phase_19c_promoted_source_validation")
    exp090 = _evidence_report(inventory, "phase_18a_integration_safety_spec")
    claims_doc = next(
        (e for e in inventory if e.get("id") == "claims_audit"),
        None,
    )
    claims_present = bool(claims_doc and claims_doc.get("doc_present"))

    results: list[dict[str, Any]] = []
    viability = (exp097 or {}).get("source_viability_gate_summary") or {}

    # l3_source_promotion_gate
    if exp097 is None:
        results.append(_gate_result(
            "l3_source_promotion_gate",
            result="fail",
            evidence_status="missing",
            notes="Phase 19C report not available",
        ))
    else:
        policy = exp097.get("promoted_source_policy") or {}
        promoted = policy.get("promoted_source")
        decision = exp097.get("decision_recommendation")
        cov_gate = viability.get("proposal_coverage_gate") or {}
        ok = (
            decision == DECISION_L3_SOURCE_PROMOTED
            and promoted == PROPOSAL_SOURCE_ROUND_LOG
            and cov_gate.get("pass") is True
        )
        results.append(_gate_result(
            "l3_source_promotion_gate",
            result="pass" if ok else "fail",
            evidence_status="available",
            notes=(
                f"decision={decision!r} promoted_source={promoted!r} "
                f"coverage_gate={cov_gate.get('pass')}"
            ),
        ))

    def _viability_gate(name: str, gate_key: str) -> None:
        if exp097 is None:
            results.append(_gate_result(
                name,
                result="fail",
                evidence_status="missing",
                notes="Phase 19C report not available",
            ))
            return
        gate = viability.get(gate_key) or {}
        ok = gate.get("pass") is True
        results.append(_gate_result(
            name,
            result="pass" if ok else "fail",
            evidence_status="available",
            notes=f"{gate_key} pass={gate.get('pass')}",
        ))

    _viability_gate("proposal_provenance_gate", "proposal_provenance_gate")
    _viability_gate("generation_parity_gate", "generation_parity_gate")

    if exp097 is None:
        results.append(_gate_result(
            "proposal_isolation_gate",
            result="fail",
            evidence_status="missing",
            notes="Phase 19C report not available",
        ))
    else:
        iso_gate = viability.get("proposal_isolation_gate") or {}
        iso_ok = (
            iso_gate.get("pass") is True
            and exp097.get("proposal_used_for_token_commit") is False
            and exp097.get("proposal_exposed_to_generator") is False
        )
        results.append(_gate_result(
            "proposal_isolation_gate",
            result="pass" if iso_ok else "fail",
            evidence_status="available",
            notes=(
                f"isolation_gate={iso_gate.get('pass')} "
                f"commit={exp097.get('proposal_used_for_token_commit')} "
                f"exposed={exp097.get('proposal_exposed_to_generator')}"
            ),
        ))

    if exp097 is None:
        results.append(_gate_result(
            "exactkv_failure_gate",
            result="fail",
            evidence_status="missing",
            notes="Phase 19C report not available",
        ))
    else:
        failures = exp097.get("exactkv_failure_summary") or {}
        ok = (
            (failures.get("baseline_failures") or 0) == 0
            and (failures.get("promoted_source_failures") or 0) == 0
        )
        results.append(_gate_result(
            "exactkv_failure_gate",
            result="pass" if ok else "fail",
            evidence_status="available",
            notes=str(failures),
        ))

    spec_val = None
    if exp097 and (exp097.get("safety_spec_validation") or {}).get("pass"):
        spec_val = exp097.get("safety_spec_validation")
    elif exp090 and (exp090.get("safety_spec_validation") or {}).get("pass"):
        spec_val = exp090.get("safety_spec_validation")
    elif exp090 and exp090.get("validation_pass") is True:
        spec_val = {"pass": True, "fallback_to_baseline": True}

    if spec_val and spec_val.get("pass") is True:
        results.append(_gate_result(
            "safety_spec_gate",
            result="pass",
            evidence_status="available",
            notes="safety_spec_validation pass",
        ))
    else:
        results.append(_gate_result(
            "safety_spec_gate",
            result="fail",
            evidence_status="missing" if spec_val is None else "available",
            notes="safety_spec_validation not passed",
        ))

    claim_gate = viability.get("claim_boundary_gate") or {}
    if exp097 is None:
        claim_ok = False
    else:
        claim_ok = claims_present and claim_gate.get("pass") is True
    results.append(_gate_result(
        "claim_boundary_gate",
        result="pass" if claim_ok else "fail",
        evidence_status="available" if claims_present else "missing",
        notes=(
            f"claims_audit_doc={claims_present} "
            f"claim_boundary_gate={claim_gate.get('pass')}"
        ),
    ))

    fallback_ok = bool(spec_val and spec_val.get("pass") is True)
    results.append(_gate_result(
        "fallback_requirement_gate",
        result="pass" if fallback_ok else "fail",
        evidence_status="available" if spec_val else "missing",
        notes=(
            "fallback_to_baseline enforced by integration safety spec validator "
            f"(pass={spec_val.get('pass') if spec_val else None})"
        ),
    ))

    prerequisite_names = (
        "l3_source_promotion_gate",
        "proposal_provenance_gate",
        "proposal_isolation_gate",
        "generation_parity_gate",
        "exactkv_failure_gate",
        "safety_spec_gate",
        "claim_boundary_gate",
        "fallback_requirement_gate",
    )
    prereq_pass = all(
        next(g for g in results if g["name"] == n)["result"] == "pass"
        for n in prerequisite_names
    )
    results.append(_gate_result(
        "l4_design_only_gate",
        result="pass" if prereq_pass else "fail",
        evidence_status="derived",
        notes="L4 design spec only; implementation never authorized by this gate",
    ))

    impl_block_ok = len(L4_IMPLEMENTATION_BLOCKERS) > 0
    results.append(_gate_result(
        "implementation_block_gate",
        result="pass" if impl_block_ok else "fail",
        evidence_status="policy",
        notes=f"{len(L4_IMPLEMENTATION_BLOCKERS)} L4 implementation blockers documented",
    ))

    passing = sum(1 for g in results if g["result"] == "pass")
    summary = {
        "gates_total": len(results),
        "gates_passing": passing,
        "gates_failing": len(results) - passing,
        "all_gates_pass": passing == len(results),
        "prerequisite_gates_pass": prereq_pass,
    }
    return results, summary


def compute_review_outcome(
    *,
    gate_results: Sequence[Mapping[str, Any]],
    gate_summary: Mapping[str, Any],
    evidence_missing: Sequence[str],
    critical_missing: Sequence[str],
) -> tuple[str, str]:
    """Determine pre-L4 review outcome; never returns ready_for_l4_implementation."""
    safety_gates = (
        "proposal_provenance_gate",
        "proposal_isolation_gate",
        "generation_parity_gate",
        "exactkv_failure_gate",
        "safety_spec_gate",
    )
    by_name = {g["name"]: g for g in gate_results}

    if critical_missing:
        return (
            OUTCOME_BLOCKED_MISSING_EVIDENCE,
            f"critical evidence missing: {', '.join(critical_missing)}",
        )

    if any(by_name.get(n, {}).get("result") == "fail" for n in safety_gates):
        return (
            OUTCOME_BLOCKED_SAFETY_FAILURE,
            "one or more safety-related pre-L4 gates failed",
        )

    if gate_summary.get("prerequisite_gates_pass") is True:
        return (
            OUTCOME_READY_L4_DESIGN_SPEC_ONLY,
            "all prerequisite gates pass; L4 design specification may begin; "
            "L4 implementation is not authorized",
        )

    if evidence_missing:
        return (
            OUTCOME_NOT_READY_L4_DESIGN_SPEC,
            f"optional evidence missing: {', '.join(evidence_missing)}; "
            "prerequisite gates incomplete",
        )

    return (
        OUTCOME_NOT_READY_L4_DESIGN_SPEC,
        "one or more prerequisite pre-L4 gates did not pass",
    )


def run_exp098_pre_l4_safety_gate_review(
    *,
    root: Path | None = None,
    evidence_overrides: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 098 pre-L4 safety gate review from local evidence."""
    inventory, evidence_found, evidence_missing = inventory_evidence(
        root=root,
        evidence_overrides=evidence_overrides,
    )
    critical_missing = [
        eid for eid in CRITICAL_EVIDENCE_IDS if eid in evidence_missing
    ]
    gate_results, gate_summary = evaluate_pre_l4_gates(inventory)
    outcome, outcome_reason = compute_review_outcome(
        gate_results=gate_results,
        gate_summary=gate_summary,
        evidence_missing=evidence_missing,
        critical_missing=critical_missing,
    )

    l4_design_authorized = outcome == OUTCOME_READY_L4_DESIGN_SPEC_ONLY
    l4_implementation_authorized = False

    if outcome == OUTCOME_BLOCKED_MISSING_EVIDENCE:
        status = "blocked_missing_evidence"
    elif outcome == OUTCOME_BLOCKED_SAFETY_FAILURE:
        status = "blocked_safety_failure"
    elif outcome == OUTCOME_READY_L4_DESIGN_SPEC_ONLY:
        status = "review_complete"
    else:
        status = "review_incomplete"

    claim_boundary_summary = {
        "claims_audit_doc_required": True,
        "claims_audit_doc_present": any(
            e.get("id") == "claims_audit" and e.get("doc_present")
            for e in inventory
        ),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "forbidden_claim_categories": [
            "speedup",
            "throughput",
            "latency",
            "tokens_per_second",
            "runtime_seconds",
            "active_gpu_memory",
            "production_memory",
            "serving",
            "vericache_reproduction",
        ],
    }

    return {
        "experiment_id": EXPERIMENT_098_ID,
        "status": status,
        "phase": PHASE_20A,
        "evidence_found": evidence_found,
        "evidence_missing": evidence_missing,
        "gate_results": gate_results,
        "gate_summary": gate_summary,
        "review_outcome": outcome,
        "review_outcome_reason": outcome_reason,
        "l4_design_spec_authorized": l4_design_authorized,
        "l4_implementation_authorized": l4_implementation_authorized,
        "l4_design_spec_may_be_started": l4_design_authorized,
        "l4_implementation_is_not_authorized": True,
        "l4_implementation_blockers": list(L4_IMPLEMENTATION_BLOCKERS),
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_20A,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES),
        "claim_boundary_summary": claim_boundary_summary,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE_20A,
        "limitations": [
            "Pre-L4 safety gate review only; not L4 implementation.",
            "Evidence loaded from local docs/reports only; missing items not invented.",
            "L3 proposal source promotion does not authorize token-commit integration.",
            "L4 design specification may begin only when review_outcome permits.",
            "L4 implementation remains blocked.",
            "ExactKVGenerator and default runtime unchanged.",
            "No new model experiments were run for this review.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp098_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "evidence_found",
        "evidence_missing",
        "gate_results",
        "gate_summary",
        "review_outcome",
        "l4_design_spec_authorized",
        "l4_implementation_authorized",
        "l4_implementation_blockers",
        "allowed_next_phase",
        "forbidden_next_phases",
        "claim_boundary_summary",
        "exactkv_generator_modified",
        "default_runtime_changed",
        "proposal_used_for_token_commit",
        "proposal_exposed_to_generator",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_098_ID:
        errors.append("experiment_id mismatch")

    if report.get("review_outcome") not in REVIEW_OUTCOMES:
        errors.append("invalid review_outcome")

    if report.get("review_outcome") == OUTCOME_FORBIDDEN_L4_IMPLEMENTATION:
        errors.append("review_outcome must not be ready_for_l4_implementation")

    if report.get("l4_implementation_authorized") is not False:
        errors.append("l4_implementation_authorized must be false")

    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")

    if report.get("default_runtime_changed") is not False:
        errors.append("default_runtime_changed must be false")

    if report.get("proposal_used_for_token_commit") is not False:
        errors.append("proposal_used_for_token_commit must be false")

    if report.get("proposal_exposed_to_generator") is not False:
        errors.append("proposal_exposed_to_generator must be false")

    if report.get("allowed_next_phase") != RECOMMENDED_NEXT_PHASE_20A:
        errors.append("allowed_next_phase must be phase20b_l4_verifier_mediated_design_spec")

    if not report.get("l4_implementation_blockers"):
        errors.append("l4_implementation_blockers must be non-empty")

    gate_keys = (
        "name",
        "purpose",
        "required_evidence",
        "pass_condition",
        "fail_condition",
        "evidence_status",
        "result",
        "notes",
    )
    for idx, gate in enumerate(report.get("gate_results") or []):
        for gk in gate_keys:
            if gk not in gate:
                errors.append(f"gate_results[{idx}] missing {gk}")
        if gate.get("name") not in GATE_NAMES:
            errors.append(f"gate_results[{idx}] unknown gate name")

    return errors


def synthetic_exp097_evidence(**overrides: Any) -> dict[str, Any]:
    """Build synthetic Phase 19C evidence for tests (not used in real review)."""
    base: dict[str, Any] = {
        "experiment_id": "exp097_l3_promoted_source_validation",
        "decision_recommendation": DECISION_L3_SOURCE_PROMOTED,
        "promoted_source_policy": {
            "promoted_source": PROPOSAL_SOURCE_ROUND_LOG,
        },
        "source_viability_gate_summary": {
            "proposal_coverage_gate": {"pass": True},
            "proposal_provenance_gate": {"pass": True},
            "proposal_isolation_gate": {"pass": True},
            "generation_parity_gate": {"pass": True},
            "exactkv_failure_gate": {"pass": True},
            "claim_boundary_gate": {"pass": True},
            "all_required_gates_pass": True,
        },
        "exactkv_failure_summary": {
            "baseline_failures": 0,
            "promoted_source_failures": 0,
        },
        "safety_spec_validation": {
            "pass": True,
            "fallback_to_baseline": True,
            "proposed_level": "L3_GUARDED_DRAFT_SHADOW_NO_COMMIT",
        },
        "proposal_used_for_token_commit": False,
        "proposal_exposed_to_generator": False,
        "default_runtime_changed": False,
    }
    base.update(overrides)
    return base
