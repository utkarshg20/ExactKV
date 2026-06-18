"""Integration design review (Phase 17D / Exp 089).

Claim-safe review of integration levels, gates, and risks after Phase 17.
Design and decision phase only — no runtime changes or new experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exactkv.attention.phase16_closeout import (
    ALLOWED_CLAIMS,
    FORBIDDEN_CLAIMS,
    FUTURE_DEFERRED_CLAIMS,
    TOPK_INTERPRETATION_NOTE,
)

EXPERIMENT_089_ID = "exp089_integration_design_review"
DEFAULT_EXP089_REPORT = Path("reports/experiment_089_integration_design_review.json")
PHASE_17D = "17D"

RECOMMENDED_NEXT_PHASE = "phase18a_integration_safety_spec"
RECOMMENDED_NEXT_PHASE_REASON = (
    "The project should define the exact safety contract before implementing any "
    "draft-shadow or verifier-mediated token-commit path. L3–L4 work without a "
    "written integration safety spec risks shadow influence on commits, verifier "
    "bypass, or hidden default-runtime changes."
)

CURRENT_IMPLEMENTED_LEVEL = "L2_live_diagnostic_observer"

SOURCE_DOCS: tuple[str, ...] = (
    "docs/PHASE_16_CLOSEOUT.md",
    "docs/PHASE_17_CLAIM_SAFE_DEMO.md",
    "docs/PHASE_17B_BROADER_MODEL_VALIDATION.md",
    "docs/PHASE_17C_LONG_CONTEXT_VALIDATION.md",
    "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
    "docs/EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md",
    "docs/CLAIMS_AUDIT.md",
    "docs/DEFERRED_WORK_REGISTER.md",
    "docs/VERICACHE_SYSTEMS_ROADMAP.md",
)

OPTIONAL_REPORTS: tuple[str, ...] = (
    "reports/experiment_085_phase16_closeout_summary.json",
    "reports/experiment_086_claim_safe_demo_packaging.json",
    "reports/experiment_084_guarded_decode_time_shadow_panel.json",
    "reports/experiment_087_broader_model_validation_panel.json",
    "reports/experiment_088_long_context_validation_panel.json",
)

NO_PERFORMANCE_CLAIMS_NOTE = (
    "No speed, throughput, latency, serving, measured active GPU memory, "
    "or production-memory claim is made."
)

GATE_POLICY_BEFORE_TOKEN_COMMIT: tuple[dict[str, str], ...] = (
    {
        "gate_id": "baseline_vs_integrated_token_parity",
        "description": (
            "Baseline-vs-integrated generated token IDs must match on fixed greedy "
            "settings before any L4 token-commit research is allowed."
        ),
    },
    {
        "gate_id": "exactkv_failures_zero",
        "description": "exactkv_failures must remain 0 on gated panels.",
    },
    {
        "gate_id": "shadow_cannot_bypass_verifier",
        "description": (
            "Shadow or compressed-draft output cannot bypass full verification; "
            "full verifier remains source of truth."
        ),
    },
    {
        "gate_id": "full_verifier_source_of_truth",
        "description": (
            "Committed tokens must come only from the full verifier path unless "
            "an explicitly approved spec documents otherwise."
        ),
    },
    {
        "gate_id": "fallback_restores_baseline",
        "description": (
            "A fallback path must restore existing generation behavior when "
            "integration is disabled or fails."
        ),
    },
    {
        "gate_id": "deterministic_test_harness",
        "description": (
            "Deterministic test harness with parity and safety-gate checks "
            "required before merge."
        ),
    },
    {
        "gate_id": "claim_audit_pass",
        "description": "Public claims audit must pass after any integration change.",
    },
    {
        "gate_id": "no_perf_claims_without_measurement",
        "description": (
            "No performance or memory claims without explicit measurement "
            "infrastructure and scoped reporting."
        ),
    },
    {
        "gate_id": "no_serving_claims_without_backend",
        "description": (
            "No production or serving claims without backend validation panels."
        ),
    },
    {
        "gate_id": "no_broad_model_claim_from_small_panel",
        "description": (
            "Small validation panels cannot support broad model-family or "
            "long-context support claims."
        ),
    },
)

RISK_REGISTER: tuple[dict[str, str], ...] = (
    {
        "risk_id": "shadow_influences_token_commit",
        "description": "Shadow diagnostics accidentally influence token commit.",
        "severity": "critical",
        "mitigation": (
            "GuardedDecodeTimeShadowObserver ignores return values; L3+ requires "
            "integration safety spec and parity gates before any commit path."
        ),
        "current_status": "mitigated_in_L2",
    },
    {
        "risk_id": "verifier_bypass",
        "description": "Compressed draft or shadow logits bypass full verification.",
        "severity": "critical",
        "mitigation": (
            "L4 not implemented; gate policy requires full verifier as source of truth."
        ),
        "current_status": "not_applicable_until_L4",
    },
    {
        "risk_id": "hidden_default_runtime_change",
        "description": "Integration changes default generation without opt-in.",
        "severity": "high",
        "mitigation": (
            "ExactKVGenerator unchanged through Phase 17; future work requires "
            "explicit flags and safety_gate default_runtime_changed=false."
        ),
        "current_status": "mitigated",
    },
    {
        "risk_id": "callbacks_mutate_generation_state",
        "description": "Observer callbacks mutate generation or KV state.",
        "severity": "high",
        "mitigation": (
            "Post-commit snapshots only; guarded observer swallows exceptions; "
            "generation_modified_by_decode_time_shadow=false in panels."
        ),
        "current_status": "mitigated_in_L2",
    },
    {
        "risk_id": "topk_misrepresented_as_exactness",
        "description": "Top-k agreement cited as exactness guarantee.",
        "severity": "medium",
        "mitigation": TOPK_INTERPRETATION_NOTE,
        "current_status": "documented_in_claim_freeze",
    },
    {
        "risk_id": "small_panel_overstated",
        "description": "Small-panel results overstated as general support.",
        "severity": "medium",
        "mitigation": (
            "Panel-scoped and context-length-scoped claim notes in Exp 087/088; "
            "gate policy forbids broad claims from small panels."
        ),
        "current_status": "mitigated_in_docs",
    },
    {
        "risk_id": "performance_claims_without_measurement",
        "description": "Speed or throughput claims without measurement.",
        "severity": "high",
        "mitigation": (
            "Forbidden claims list; no runtime_seconds in Phase 17 panels; "
            "audit_public_claims.py in CI."
        ),
        "current_status": "mitigated",
    },
    {
        "risk_id": "memory_claims_without_active_measurement",
        "description": "GPU or production memory savings claimed without measurement.",
        "severity": "high",
        "mitigation": (
            "Active memory savings explicitly forbidden; deferred work register "
            "lists compressed-active KV as future only."
        ),
        "current_status": "mitigated",
    },
    {
        "risk_id": "vericache_reproduction_overclaim",
        "description": "VeriCache throughput or serving results implied.",
        "severity": "medium",
        "mitigation": (
            "Explicit forbidden claims; VeriCache roadmap documents gap; "
            "ExactKV does not reproduce VeriCache results."
        ),
        "current_status": "mitigated_in_docs",
    },
    {
        "risk_id": "production_serving_overclaim",
        "description": "Production serving support implied from diagnostic work.",
        "severity": "high",
        "mitigation": (
            "vLLM/LMCache direct integration no-go; L5 deferred; serving claims forbidden."
        ),
        "current_status": "mitigated",
    },
)

DEFERRED_WORK_ITEMS: tuple[dict[str, str], ...] = (
    {"item": "L3 guarded draft shadow (no commit)", "status": "deferred", "phase": "18+"},
    {"item": "L4 verifier-mediated compressed draft", "status": "deferred", "phase": "18+"},
    {"item": "L5 CUDA/Triton/vLLM/LMCache/serving", "status": "deferred", "phase": "future"},
    {"item": "CUDA/Triton streaming attention kernels", "status": "deferred", "phase": "future"},
    {"item": "Direct vLLM integration", "status": "no-go", "phase": "V11+"},
    {"item": "Direct LMCache integration", "status": "no-go", "phase": "V11+"},
    {"item": "Measured active GPU memory savings", "status": "deferred", "phase": "future"},
    {"item": "Production serving", "status": "deferred", "phase": "future"},
    {"item": "Real compressed-attention token commit path", "status": "deferred", "phase": "18+"},
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _inventory_sources(root: Path) -> tuple[list[str], list[str], dict[str, str]]:
    found: list[str] = []
    missing: list[str] = []
    contents: dict[str, str] = {}
    for rel in SOURCE_DOCS:
        path = root / rel
        if path.is_file():
            found.append(rel)
            try:
                contents[rel] = path.read_text()
            except OSError:
                contents[rel] = ""
        else:
            missing.append(rel)
    return found, missing, contents


def _inventory_reports(root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    loaded: dict[str, Any] = {}
    found: list[str] = []
    missing: list[str] = []
    for rel in OPTIONAL_REPORTS:
        path = root / rel
        if path.is_file():
            found.append(rel)
            data = _load_json(path)
            if data is not None:
                loaded[rel] = data
        else:
            missing.append(rel)
    return loaded, found, missing


def _safe_report_metric(report: dict[str, Any] | None, *keys: str, default: str = "not_loaded") -> str:
    if not report:
        return default
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return str(cur)


def _build_integration_levels(
    *,
    docs_found: list[str],
    reports: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    exp084 = reports.get("reports/experiment_084_guarded_decode_time_shadow_panel.json")
    exp088 = reports.get("reports/experiment_088_long_context_validation_panel.json")
    exp087 = reports.get("reports/experiment_087_broader_model_validation_panel.json")

    exp084_evidence = (
        f"Exp 084 report: status={_safe_report_metric(exp084, 'status')}; "
        f"token parity {_safe_report_metric(exp084, 'baseline_vs_guarded_token_match_cells')}/"
        f"{_safe_report_metric(exp084, 'total_cells')}."
        if exp084
        else (
            "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md present."
            if "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md" in docs_found
            else "Exp 084 evidence missing."
        )
    )
    exp088_evidence = (
        f"Exp 088 report: status={_safe_report_metric(exp088, 'status')}; "
        f"token parity {_safe_report_metric(exp088, 'baseline_vs_guarded_token_match_cells')}/"
        f"{_safe_report_metric(exp088, 'total_cells')}."
        if exp088
        else (
            "docs/EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md present."
            if "docs/EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md" in docs_found
            else "Exp 088 evidence missing."
        )
    )
    exp087_evidence = (
        f"Exp 087 report: status={_safe_report_metric(exp087, 'status')}; "
        f"models loaded {len(exp087.get('models_loaded', []))}."
        if exp087
        else (
            "docs/PHASE_17B_BROADER_MODEL_VALIDATION.md present."
            if "docs/PHASE_17B_BROADER_MODEL_VALIDATION.md" in docs_found
            else "Exp 087 evidence missing."
        )
    )

    return {
        "L0_demo_only": {
            "level_id": "L0_demo_only",
            "title": "Claim-safe documentation and demo packaging",
            "status": "implemented",
            "phase": "17A",
            "evidence": [
                "docs/PHASE_17_CLAIM_SAFE_DEMO.md",
                "docs/PHASE_17_DEMO_SCRIPT.md",
                "exactkv/demo/phase17_claim_safe_demo.py",
            ],
            "implementation_risk": "low",
            "claim_risk": "low if panel-scoped language preserved",
            "required_gates": ["claim_audit_pass"],
            "allowed_claims": [
                "ExactKV has claim-safe demo packaging from Phase 16 evidence.",
            ],
            "forbidden_claims": [
                "ExactKV improves speed.",
                "ExactKV is production-ready.",
            ],
        },
        "L1_external_shadow_observer": {
            "level_id": "L1_external_shadow_observer",
            "title": "External post-hoc generation-shadow observer",
            "status": "implemented",
            "phase": "16K–16O",
            "evidence": [
                "docs/EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md",
                "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
                "docs/EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md",
            ],
            "implementation_risk": "low — runs after generation",
            "claim_risk": "medium if shadow mislabeled as live integration",
            "required_gates": ["claim_audit_pass", "no_broad_model_claim_from_small_panel"],
            "allowed_claims": [
                "ExactKV has an external generation-shadow observer.",
            ],
            "forbidden_claims": [
                "Streaming attention is integrated into token commit.",
            ],
        },
        "L2_live_diagnostic_observer": {
            "level_id": "L2_live_diagnostic_observer",
            "title": "Live observer and guarded decode-time diagnostic shadow",
            "status": "implemented",
            "phase": "16P–17C",
            "evidence": [
                exp084_evidence,
                exp087_evidence,
                exp088_evidence,
                "docs/PHASE_16_CLOSEOUT.md",
            ],
            "implementation_risk": "medium — callbacks during decode but guarded",
            "claim_risk": "high if parity panels overstated",
            "required_gates": [
                "baseline_vs_integrated_token_parity",
                "exactkv_failures_zero",
                "shadow_cannot_bypass_verifier",
                "deterministic_test_harness",
            ],
            "allowed_claims": list(ALLOWED_CLAIMS),
            "forbidden_claims": [
                "Streaming attention is integrated into token commit.",
                "Shadow logits are exactness guarantees.",
            ],
        },
        "L3_guarded_draft_shadow_no_commit": {
            "level_id": "L3_guarded_draft_shadow_no_commit",
            "title": "Guarded draft shadow diagnostics during generation (no commit)",
            "status": "not_implemented",
            "phase": "future",
            "evidence": ["Deferred — requires integration safety spec first."],
            "implementation_risk": "high — runs during generation with compressed attention",
            "claim_risk": "high — easy to conflate with token commit",
            "required_gates": [
                gate["gate_id"] for gate in GATE_POLICY_BEFORE_TOKEN_COMMIT
            ],
            "allowed_claims": [
                "Diagnostic compressed-attention draft replay in test panels only.",
            ],
            "forbidden_claims": [
                "Streaming attention is integrated into token commit.",
                "ExactKV improves throughput.",
            ],
        },
        "L4_verifier_mediated_compressed_draft": {
            "level_id": "L4_verifier_mediated_compressed_draft",
            "title": "Verifier-mediated compressed draft (full verifier source of truth)",
            "status": "not_implemented",
            "phase": "future",
            "evidence": ["Not started — token-commit path research blocked until gates pass."],
            "implementation_risk": "critical",
            "claim_risk": "critical",
            "required_gates": [
                gate["gate_id"] for gate in GATE_POLICY_BEFORE_TOKEN_COMMIT
            ],
            "allowed_claims": [
                "Panel-scoped parity on fixed greedy settings after explicit approval.",
            ],
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
        },
        "L5_real_backend_integration": {
            "level_id": "L5_real_backend_integration",
            "title": "CUDA/Triton/vLLM/LMCache/serving backend integration",
            "status": "deferred",
            "phase": "future",
            "evidence": [
                "docs/DEFERRED_WORK_REGISTER.md — vLLM/LMCache direct integration no-go",
                "docs/VERICACHE_SYSTEMS_ROADMAP.md",
            ],
            "implementation_risk": "critical",
            "claim_risk": "critical",
            "required_gates": [
                gate["gate_id"] for gate in GATE_POLICY_BEFORE_TOKEN_COMMIT
            ],
            "allowed_claims": [
                "Feasibility probes and sidecar diagnostics only when documented.",
            ],
            "forbidden_claims": [
                "ExactKV reproduces VeriCache serving results.",
                "ExactKV reproduces VeriCache throughput results.",
                "Production serving supported.",
            ],
        },
    }


def _build_evidence_summary(
    *,
    docs_found: list[str],
    docs_missing: list[str],
    reports: dict[str, Any],
    reports_found: list[str],
    reports_missing: list[str],
) -> dict[str, Any]:
    closeout_present = "docs/PHASE_16_CLOSEOUT.md" in docs_found
    return {
        "phase_16_closeout": {
            "doc_present": closeout_present,
            "note": (
                "Phase 16 complete with claim freeze; 19 experiment steps."
                if closeout_present
                else "PHASE_16_CLOSEOUT.md missing."
            ),
        },
        "phase_17a_demo": {
            "doc_present": "docs/PHASE_17_CLAIM_SAFE_DEMO.md" in docs_found,
        },
        "phase_17b_validation": {
            "doc_present": "docs/PHASE_17B_BROADER_MODEL_VALIDATION.md" in docs_found,
            "report_loaded": (
                "reports/experiment_087_broader_model_validation_panel.json" in reports
            ),
        },
        "phase_17c_long_context": {
            "doc_present": "docs/PHASE_17C_LONG_CONTEXT_VALIDATION.md" in docs_found,
            "report_loaded": (
                "reports/experiment_088_long_context_validation_panel.json" in reports
            ),
        },
        "exp084_guarded_panel": {
            "doc_present": (
                "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md" in docs_found
            ),
            "report_loaded": (
                "reports/experiment_084_guarded_decode_time_shadow_panel.json" in reports
            ),
        },
        "claims_audit": {
            "doc_present": "docs/CLAIMS_AUDIT.md" in docs_found,
        },
        "deferred_work_register": {
            "doc_present": "docs/DEFERRED_WORK_REGISTER.md" in docs_found,
        },
        "vericache_roadmap": {
            "doc_present": "docs/VERICACHE_SYSTEMS_ROADMAP.md" in docs_found,
        },
        "reports_found": reports_found,
        "reports_missing": reports_missing,
        "docs_missing_count": len(docs_missing),
    }


def run_exp089_integration_design_review(
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build Experiment 089 integration design review report."""
    base = root or Path(".")
    docs_found, docs_missing, _contents = _inventory_sources(base)
    reports, reports_found, reports_missing = _inventory_reports(base)
    integration_levels = _build_integration_levels(
        docs_found=docs_found,
        reports=reports,
    )
    evidence_summary = _build_evidence_summary(
        docs_found=docs_found,
        docs_missing=docs_missing,
        reports=reports,
        reports_found=reports_found,
        reports_missing=reports_missing,
    )

    future_levels = [
        lid for lid, level in integration_levels.items()
        if level["status"] in ("not_implemented", "deferred")
    ]
    implemented_levels = [
        lid for lid, level in integration_levels.items()
        if level["status"] == "implemented"
    ]

    if docs_missing and len(docs_found) == 0:
        status = "blocked"
    elif docs_missing:
        status = "review_partial"
    else:
        status = "review_complete"

    return {
        "experiment_id": EXPERIMENT_089_ID,
        "status": status,
        "phase": PHASE_17D,
        "source_docs_found": docs_found,
        "source_docs_missing": docs_missing,
        "reports_found": reports_found,
        "reports_missing": reports_missing,
        "integration_levels": integration_levels,
        "current_implemented_level": CURRENT_IMPLEMENTED_LEVEL,
        "implemented_levels": implemented_levels,
        "future_levels": future_levels,
        "evidence_summary": evidence_summary,
        "gate_policy_before_token_commit_changes": [
            dict(gate) for gate in GATE_POLICY_BEFORE_TOKEN_COMMIT
        ],
        "risk_register": [dict(risk) for risk in RISK_REGISTER],
        "allowed_claims": list(ALLOWED_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "future_deferred_claims": list(FUTURE_DEFERRED_CLAIMS),
        "deferred_work": [dict(item) for item in DEFERRED_WORK_ITEMS],
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "recommended_next_phase_reason": RECOMMENDED_NEXT_PHASE_REASON,
        "topk_interpretation_note": TOPK_INTERPRETATION_NOTE,
        "limitations": [
            "Integration design review only — no implementation in Phase 17D.",
            "ExactKV default generation and ExactKVGenerator unchanged.",
            "Guarded shadow remains diagnostic-only in the implemented path (L2).",
            "Streaming attention is not integrated into token commit.",
            "Before any token-commit path changes, full verification must remain source of truth.",
            "No new model experiments were run for this review.",
        ],
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
    }


def validate_exp089_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "source_docs_found",
        "source_docs_missing",
        "integration_levels",
        "evidence_summary",
        "gate_policy_before_token_commit_changes",
        "risk_register",
        "allowed_claims",
        "forbidden_claims",
        "deferred_work",
        "recommended_next_phase",
        "recommended_next_phase_reason",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_089_ID:
        errors.append("experiment_id mismatch")

    levels = report.get("integration_levels") or {}
    for level_id in (
        "L0_demo_only",
        "L1_external_shadow_observer",
        "L2_live_diagnostic_observer",
        "L3_guarded_draft_shadow_no_commit",
        "L4_verifier_mediated_compressed_draft",
        "L5_real_backend_integration",
    ):
        if level_id not in levels:
            errors.append(f"missing integration level: {level_id}")
        else:
            for field in (
                "status",
                "evidence",
                "implementation_risk",
                "claim_risk",
                "required_gates",
                "allowed_claims",
                "forbidden_claims",
            ):
                if field not in levels[level_id]:
                    errors.append(f"{level_id} missing {field}")

    if report.get("recommended_next_phase") != RECOMMENDED_NEXT_PHASE:
        errors.append("recommended_next_phase must be phase18a_integration_safety_spec")

    required_risks = {
        "shadow_influences_token_commit",
        "verifier_bypass",
        "hidden_default_runtime_change",
        "callbacks_mutate_generation_state",
        "topk_misrepresented_as_exactness",
        "small_panel_overstated",
        "performance_claims_without_measurement",
        "memory_claims_without_active_measurement",
        "vericache_reproduction_overclaim",
        "production_serving_overclaim",
    }
    found_risks = {r.get("risk_id") for r in report.get("risk_register", [])}
    for rid in required_risks - found_risks:
        errors.append(f"missing risk: {rid}")

    if list(report.get("allowed_claims") or []) != list(ALLOWED_CLAIMS):
        errors.append("allowed_claims must match Phase 16 claim freeze")

    if list(report.get("forbidden_claims") or []) != list(FORBIDDEN_CLAIMS):
        errors.append("forbidden_claims must match Phase 16 claim freeze")

    gates = report.get("gate_policy_before_token_commit_changes") or []
    if not gates:
        errors.append("gate_policy_before_token_commit_changes must be non-empty")

    return errors
