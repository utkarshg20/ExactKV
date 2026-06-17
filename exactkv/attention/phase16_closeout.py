"""Phase 16 closeout summary and claim freeze (Phase 16T / Exp 085).

Aggregates evidence from Phase 16 experiments (066–084) without adding new
shadow or integration functionality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

EXPERIMENT_085_ID = "exp085_phase16_closeout_summary"
DEFAULT_EXP085_REPORT = Path("reports/experiment_085_phase16_closeout_summary.json")

PHASE16_COMPLETED_STEPS = 19
LAST_COMPLETED_STEP = "16S"
PHASE16_STATUS = "complete"
RECOMMENDED_STOP = True
RECOMMENDED_NEXT_PHASE = "phase17_claim_safe_demo_packaging"

TOPK_INTERPRETATION_NOTE = (
    "Top-k agreement metrics are supplementary diagnostics only. They do not "
    "constitute an exactness guarantee and must not be cited as proof of exact "
    "generation preservation."
)

ALLOWED_CLAIMS: tuple[str, ...] = (
    "ExactKV has offline streaming-attention diagnostics.",
    "ExactKV has Qwen2.5 offline attention shadow probes.",
    "ExactKV has a diagnostic tolerance policy.",
    "ExactKV has an external generation-shadow observer.",
    "ExactKV has an opt-in live round observer.",
    "ExactKV has a guarded decode-time shadow dry-run.",
    "In tested Phase 16 panels, guarded shadow did not change generated tokens.",
    "In tested Phase 16 panels, ExactKV failures remained zero.",
)

FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "ExactKV improves speed.",
    "ExactKV improves throughput.",
    "ExactKV reduces latency.",
    "ExactKV reduces active GPU memory.",
    "ExactKV provides production memory savings.",
    "ExactKV reproduces VeriCache serving results.",
    "ExactKV reproduces VeriCache throughput results.",
    "Streaming attention is integrated into token commit.",
    "Shadow logits are exactness guarantees.",
    "Top-k agreement proves exactness.",
    "The system is production-ready.",
)

FUTURE_DEFERRED_CLAIMS: tuple[str, ...] = (
    "CUDA/Triton streaming attention kernels.",
    "vLLM integration.",
    "LMCache integration.",
    "measured active GPU memory savings.",
    "production serving.",
    "broader model-family validation.",
    "longer-context validation.",
    "real compressed-attention token commit path.",
)

PHASE_16_EXPERIMENTS: tuple[dict[str, str], ...] = (
    {"phase": "16A", "exp": "066", "doc": "docs/EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md",
     "report": "reports/experiment_066_streaming_quant_attention_feasibility.json"},
    {"phase": "16B", "exp": "067", "doc": "docs/EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md",
     "report": "reports/experiment_067_hf_single_layer_attention_drift.json"},
    {"phase": "16C", "exp": "068", "doc": "docs/EXPERIMENT_068_QWEN_ROPE_LONG_CONTEXT_ATTENTION_PROBE.md",
     "report": "reports/experiment_068_qwen_rope_long_context_attention_probe.json"},
    {"phase": "16D", "exp": "069", "doc": "docs/EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md",
     "report": "reports/experiment_069_multilayer_attention_drift_accumulation.json"},
    {"phase": "16E", "exp": "070", "doc": "docs/EXPERIMENT_070_STREAMING_MULTILAYER_NUMERICS_AUDIT.md",
     "report": "reports/experiment_070_streaming_multilayer_numerics_audit.json"},
    {"phase": "16F", "exp": "071", "doc": "docs/EXPERIMENT_071_FULL_PREFIX_LOGIT_DRIFT_SMOKE.md",
     "report": "reports/experiment_071_full_prefix_logit_drift_smoke.json"},
    {"phase": "16G", "exp": "072", "doc": "docs/EXPERIMENT_072_FULL_DEPTH_DIVERGENCE_TRACE.md",
     "report": "reports/experiment_072_full_depth_divergence_trace.json"},
    {"phase": "16H", "exp": "073", "doc": "docs/EXPERIMENT_073_QWEN_FAMILY_DIVERGENCE_PANEL.md",
     "report": "reports/experiment_073_qwen_family_divergence_panel.json"},
    {"phase": "16I", "exp": "074", "doc": "docs/EXPERIMENT_074_ATTENTION_TOLERANCE_POLICY_PANEL.md",
     "report": "reports/experiment_074_attention_tolerance_policy_panel.json"},
    {"phase": "16J", "exp": "075", "doc": "docs/EXPERIMENT_075_GENERATION_SHADOW_WIRING_REVIEW.md",
     "report": "reports/experiment_075_generation_shadow_wiring_review.json"},
    {"phase": "16K", "exp": "076", "doc": "docs/EXPERIMENT_076_GENERATION_SHADOW_OBSERVER_SMOKE.md",
     "report": "reports/experiment_076_generation_shadow_observer_smoke.json"},
    {"phase": "16L", "exp": "077", "doc": "docs/EXPERIMENT_077_GENERATION_SHADOW_PROMPT_PLUS_GENERATED_PANEL.md",
     "report": "reports/experiment_077_generation_shadow_prompt_plus_generated_panel.json"},
    {"phase": "16M", "exp": "078", "doc": "docs/EXPERIMENT_078_GENERATION_SHADOW_EXPANDED_PANEL.md",
     "report": "reports/experiment_078_generation_shadow_expanded_panel.json"},
    {"phase": "16N", "exp": "079", "doc": "docs/EXPERIMENT_079_DECODE_PREFIX_LADDER_SHADOW_OBSERVER.md",
     "report": "reports/experiment_079_decode_prefix_ladder_shadow_observer.json"},
    {"phase": "16O", "exp": "080", "doc": "docs/EXPERIMENT_080_ROUND_LOG_SHADOW_OBSERVER.md",
     "report": "reports/experiment_080_round_log_shadow_observer.json"},
    {"phase": "16P", "exp": "081", "doc": "docs/EXPERIMENT_081_LIVE_ROUND_OBSERVER_SMOKE.md",
     "report": "reports/experiment_081_live_round_observer_smoke.json"},
    {"phase": "16Q", "exp": "082", "doc": "docs/EXPERIMENT_082_LIVE_OBSERVER_SHADOW_PANEL.md",
     "report": "reports/experiment_082_live_observer_shadow_panel.json"},
    {"phase": "16R", "exp": "083", "doc": "docs/EXPERIMENT_083_GUARDED_DECODE_TIME_SHADOW_SMOKE.md",
     "report": "reports/experiment_083_guarded_decode_time_shadow_smoke.json"},
    {"phase": "16S", "exp": "084", "doc": "docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md",
     "report": "reports/experiment_084_guarded_decode_time_shadow_panel.json"},
)

PHASE_16_META_DOCS: tuple[str, ...] = (
    "docs/PHASE_16_CLOSEOUT.md",
    "docs/CLAIMS_AUDIT.md",
    "docs/DEFERRED_WORK_REGISTER.md",
    "docs/VERICACHE_SYSTEMS_ROADMAP.md",
    "docs/EXPERIMENT_085_PHASE16_CLOSEOUT_SUMMARY.md",
)


def _load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _inventory_paths(
    root: Path,
    *,
    reports_root: Path | None = None,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, Any]]:
    """Return reports_found, reports_missing, docs_found, docs_missing, loaded_reports."""
    reports_found: list[str] = []
    reports_missing: list[str] = []
    docs_found: list[str] = []
    docs_missing: list[str] = []
    loaded: dict[str, Any] = {}

    base = reports_root or root
    for entry in PHASE_16_EXPERIMENTS:
        rel_report = entry["report"]
        report_path = base / rel_report if not Path(rel_report).is_absolute() else Path(rel_report)
        if not report_path.is_file():
            report_path = root / rel_report
        if report_path.is_file():
            reports_found.append(rel_report)
            data = _load_json_report(report_path)
            if data is not None:
                loaded[entry["exp"]] = data
        else:
            reports_missing.append(rel_report)

        rel_doc = entry["doc"]
        doc_path = root / rel_doc
        if doc_path.is_file():
            docs_found.append(rel_doc)
        else:
            docs_missing.append(rel_doc)

    for rel_doc in PHASE_16_META_DOCS:
        doc_path = root / rel_doc
        if doc_path.is_file():
            if rel_doc not in docs_found:
                docs_found.append(rel_doc)
        else:
            docs_missing.append(rel_doc)

    return reports_found, reports_missing, docs_found, docs_missing, loaded


def _section(
    *,
    experiments: Sequence[str],
    strongest_evidence: str,
    limitations: str,
    allowed_claims: Sequence[str],
    forbidden_claims: Sequence[str],
    next_step_implications: str,
) -> dict[str, Any]:
    return {
        "experiments_covered": list(experiments),
        "strongest_evidence": strongest_evidence,
        "limitations": limitations,
        "allowed_claims": list(allowed_claims),
        "forbidden_claims": list(forbidden_claims),
        "next_step_implications": next_step_implications,
    }


def _build_evidence_summary(loaded: dict[str, Any]) -> dict[str, Any]:
    exp084 = loaded.get("084") or {}
    exp082 = loaded.get("082") or {}
    exp081 = loaded.get("081") or {}

    parity_note = (
        "Exp 084 (32 cells): baseline-vs-guarded token/text match 32/32; "
        "decode-time vs post-hoc shadow match 32/32; safety gates 32/32 OK."
        if exp084
        else "Exp 084 doc summary: 32/32 parity, 53/53 decode-time callbacks, safety gates OK."
    )

    return {
        "attention_feasibility": _section(
            experiments=["066"],
            strongest_evidence=(
                "Tensor-level streaming≈materialized int8 reference attention on synthetic Q/K/V; "
                "theoretical memory accounting without inference integration."
            ),
            limitations="Synthetic tensors only; not wired into ExactKV generation.",
            allowed_claims=["ExactKV has offline streaming-attention diagnostics."],
            forbidden_claims=[
                "Streaming attention is integrated into token commit.",
                "ExactKV improves throughput.",
            ],
            next_step_implications="Foundation for HF probes; no token-commit path.",
        ),
        "qwen_rope_gqa": _section(
            experiments=["067", "068"],
            strongest_evidence=(
                "HF-derived Qwen2.5 single-layer probes with RoPE/GQA extraction fidelity "
                "and long-context chunked streaming cells."
            ),
            limitations="Single-layer slices; offline only.",
            allowed_claims=["ExactKV has Qwen2.5 offline attention shadow probes."],
            forbidden_claims=["ExactKV reproduces VeriCache serving results."],
            next_step_implications="Enabled multi-layer drift accumulation (16D).",
        ),
        "multi_layer_drift": _section(
            experiments=["069", "070", "071", "072", "073"],
            strongest_evidence=(
                "Multi-layer drift accumulation, numerics audit, full-prefix logit drift, "
                "full-depth divergence trace, and Qwen-family divergence panel."
            ),
            limitations="Offline replay; depth-aware tolerance is diagnostic not exactness proof.",
            allowed_claims=["ExactKV has Qwen2.5 offline attention shadow probes."],
            forbidden_claims=["Top-k agreement proves exactness."],
            next_step_implications="Motivated formal tolerance policy (16I).",
        ),
        "tolerance_policy": _section(
            experiments=["074"],
            strongest_evidence=(
                "AttentionTolerancePolicy classifies offline streaming vs materialized "
                "results with depth-aware local-alignment vs free-running accumulation."
            ),
            limitations="Policy applies to offline shadow cells only.",
            allowed_claims=["ExactKV has a diagnostic tolerance policy."],
            forbidden_claims=["Shadow logits are exactness guarantees."],
            next_step_implications="Used in generation-shadow and decode-time shadow panels.",
        ),
        "generation_shadow_external": _section(
            experiments=["075", "076", "077", "078"],
            strongest_evidence=(
                "Wiring review plus external post-hoc generation-shadow observer panels "
                "with fixed-sequence replay and expanded prompt/compressor coverage."
            ),
            limitations="Shadow runs after generation; not decode-time integration.",
            allowed_claims=["ExactKV has an external generation-shadow observer."],
            forbidden_claims=["Streaming attention is integrated into token commit."],
            next_step_implications="Led to round-log and live-observer tracks.",
        ),
        "round_log_observer": _section(
            experiments=["079", "080"],
            strongest_evidence=(
                "Decode-prefix ladder and ExactKV round-log post-hoc shadow at "
                "round boundaries without generator hooks (080)."
            ),
            limitations="Post-hoc only; round data from result traces.",
            allowed_claims=["ExactKV has an external generation-shadow observer."],
            forbidden_claims=["Shadow logits are exactness guarantees."],
            next_step_implications="Validated round-boundary replay before live observer.",
        ),
        "live_round_observer": _section(
            experiments=["081", "082"],
            strongest_evidence=(
                f"Opt-in LiveRoundObserver with baseline parity "
                f"({exp081.get('baseline_vs_observer_token_match_cells', '16')}/"
                f"{exp081.get('total_cells', '16')} token match when report present); "
                f"Exp 082 live+post-hoc shadow panel."
                if exp081 or exp082
                else "Exp 081/082 docs: observer parity and snapshot-vs-round-log agreement."
            ),
            limitations="Default runtime unchanged; observer return values ignored.",
            allowed_claims=["ExactKV has an opt-in live round observer."],
            forbidden_claims=["ExactKV improves latency."],
            next_step_implications="Enabled guarded decode-time shadow dry-run.",
        ),
        "guarded_decode_time_shadow": _section(
            experiments=["083", "084"],
            strongest_evidence=parity_note,
            limitations=(
                "Diagnostic-only callback shadow; not streaming-attention token-commit "
                "integration."
            ),
            allowed_claims=[
                "ExactKV has a guarded decode-time shadow dry-run.",
                "In tested Phase 16 panels, guarded shadow did not change generated tokens.",
            ],
            forbidden_claims=[
                "Streaming attention is integrated into token commit.",
                "ExactKV improves speed.",
            ],
            next_step_implications="Phase 16 stop; Phase 17 demo packaging recommended.",
        ),
        "safety_gates": _section(
            experiments=["081", "082", "083", "084"],
            strongest_evidence=(
                "Consistent safety gates: shadow/observer cannot affect token commits; "
                "default runtime unchanged; observer return values ignored."
            ),
            limitations="Tested panels only; not all models/contexts.",
            allowed_claims=[
                "In tested Phase 16 panels, guarded shadow did not change generated tokens.",
            ],
            forbidden_claims=["The system is production-ready."],
            next_step_implications="Claim freeze before any Phase 17 work.",
        ),
        "exactkv_failures": _section(
            experiments=["076", "078", "080", "081", "082", "083", "084"],
            strongest_evidence=(
                "Generation panels report exactkv_failures == 0 on tested cells "
                "(including Exp 084 32/32)."
            ),
            limitations="Small deterministic panels; not universal exactness proof.",
            allowed_claims=[
                "In tested Phase 16 panels, ExactKV failures remained zero.",
            ],
            forbidden_claims=["Top-k agreement proves exactness."],
            next_step_implications="Cite panel scope when claiming exactkv_failures.",
        ),
        "claim_boundaries": _section(
            experiments=[e["exp"] for e in PHASE_16_EXPERIMENTS],
            strongest_evidence="Phase 16T claim freeze table consolidates allowed/forbidden/deferred claims.",
            limitations="Claims must cite specific experiment scope.",
            allowed_claims=list(ALLOWED_CLAIMS),
            forbidden_claims=list(FORBIDDEN_CLAIMS),
            next_step_implications="Phase 17 must respect claim freeze.",
        ),
        "deferred_work": _section(
            experiments=[],
            strongest_evidence=(
                "DEFERRED_WORK_REGISTER and VERICACHE_SYSTEMS_ROADMAP list vLLM, "
                "CUDA/Triton kernels, serving, and memory savings as deferred."
            ),
            limitations="Deferred items are not blocked forever; require explicit approval.",
            allowed_claims=[],
            forbidden_claims=list(FORBIDDEN_CLAIMS),
            next_step_implications="Phase 17 demo packaging preferred over integration.",
        ),
        "recommended_next_phase": _section(
            experiments=[],
            strongest_evidence=(
                "Phase 16 evidence is strong for claim-safe demo/story but insufficient "
                "for performance or production-serving claims."
            ),
            limitations="Phase 17 not implemented in 16T.",
            allowed_claims=[],
            forbidden_claims=[],
            next_step_implications=(
                f"Recommended: {RECOMMENDED_NEXT_PHASE} — package diagnostics into "
                "claim-safe demo narrative without new integration."
            ),
        ),
    }


def _aggregate_exactkv_failures(loaded: dict[str, Any]) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for exp_id in ("076", "078", "080", "081", "082", "083", "084"):
        report = loaded.get(exp_id)
        if not report:
            continue
        summary = report.get("exactkv_failure_summary")
        if isinstance(summary, dict):
            panels.append({"experiment": exp_id, **summary})
    return {
        "panels_with_data": len(panels),
        "panel_summaries": panels,
        "phase16_generation_panels_exactkv_failures_zero": all(
            (p.get("baseline_failures") or 0) == 0
            and (p.get("observer_failures") or p.get("guarded_failures") or 0) == 0
            for p in panels
        ) if panels else None,
        "note": "Summarized from available local JSON reports; missing reports omitted.",
    }


def _aggregate_safety_gates(loaded: dict[str, Any]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for exp_id in ("081", "082", "083", "084"):
        report = loaded.get(exp_id)
        if not report:
            continue
        sg = report.get("safety_gate_summary")
        if isinstance(sg, dict):
            summaries.append({"experiment": exp_id, **sg})
        elif report.get("observer_used_for_token_commit") is False:
            summaries.append({
                "experiment": exp_id,
                "observer_used_for_token_commit": False,
                "shadow_used_for_token_commit": report.get("shadow_used_for_token_commit"),
                "decode_time_shadow_used_for_token_commit": report.get(
                    "decode_time_shadow_used_for_token_commit",
                ),
            })
    exp084 = loaded.get("084") or {}
    return {
        "panel_summaries": summaries,
        "exp084_cells_all_gates_ok": (exp084.get("safety_gate_summary") or {}).get(
            "cells_all_gates_ok",
        ),
        "critical_safety_result": (
            "Guarded decode-time shadow did not affect token commits in tested panels; "
            "default runtime unchanged."
        ),
    }


def _build_claim_freeze() -> dict[str, Any]:
    return {
        "allowed_claims": list(ALLOWED_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "future_deferred_claims": list(FUTURE_DEFERRED_CLAIMS),
        "phase17_options": {
            "phase17_claim_safe_demo_packaging": {
                "recommended": True,
                "justification": (
                    "Phase 16 evidence supports a claim-safe demo/story without "
                    "performance or serving claims."
                ),
            },
            "phase17_deeper_integration_design": {
                "recommended": False,
                "justification": "Requires explicit approval; not justified by Phase 16 safety evidence alone.",
            },
            "phase17_broader_model_validation": {
                "recommended": False,
                "justification": "Useful later; Phase 16 Qwen-focused panels already sufficient for closeout.",
            },
            "phase17_cuda_memory_backend_research": {
                "recommended": False,
                "justification": "Deferred; no measured GPU memory savings claim from Phase 16.",
            },
        },
    }


def run_exp085_phase16_closeout_summary(
    *,
    root: Path | None = None,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    """Build Experiment 085 Phase 16 closeout summary from local evidence."""
    repo_root = root or Path(".")
    reports_found, reports_missing, docs_found, docs_missing, loaded = _inventory_paths(
        repo_root, reports_root=reports_root,
    )

    experiments_covered = [e["exp"] for e in PHASE_16_EXPERIMENTS]
    evidence = _build_evidence_summary(loaded)
    claim_freeze = _build_claim_freeze()

    status = "complete" if not docs_missing else "complete_with_missing_reports"
    if len(docs_missing) > len(PHASE_16_META_DOCS):
        status = "partial"

    return {
        "experiment_id": EXPERIMENT_085_ID,
        "status": status,
        "phase16_status": PHASE16_STATUS,
        "phase16_completed_steps": PHASE16_COMPLETED_STEPS,
        "last_completed_step": LAST_COMPLETED_STEP,
        "experiments_covered": experiments_covered,
        "reports_found": reports_found,
        "reports_missing": reports_missing,
        "docs_found": docs_found,
        "docs_missing": docs_missing,
        "evidence_summary": evidence,
        "claim_freeze": claim_freeze,
        "safety_gate_summary": _aggregate_safety_gates(loaded),
        "exactkv_failure_summary": _aggregate_exactkv_failures(loaded),
        "topk_interpretation_note": TOPK_INTERPRETATION_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS) + list(FORBIDDEN_ATTENTION_CLAIMS),
        "deferred_work": list(FUTURE_DEFERRED_CLAIMS),
        "recommended_stop": RECOMMENDED_STOP,
        "recommended_next_phase": RECOMMENDED_NEXT_PHASE,
        "critical_safety_result": (
            "Guarded diagnostic shadow infrastructure validated without changing "
            "generated tokens in tested panels."
        ),
        "critical_limitation": (
            "Not streaming-attention token-commit integration; no performance or "
            "serving claims."
        ),
        "limitations": [
            "Phase 16 is complete; no new shadow/integration functionality in 16T.",
            "Evidence inventory uses local reports when present and docs otherwise.",
            "Top-k agreement is supplementary only.",
            "ExactKV does not reproduce VeriCache throughput or serving results.",
            "Phase 17 should begin only after the Phase 16 claim freeze is committed.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
    }


def validate_exp085_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "phase16_status",
        "phase16_completed_steps",
        "experiments_covered",
        "reports_found",
        "reports_missing",
        "docs_found",
        "docs_missing",
        "evidence_summary",
        "claim_freeze",
        "safety_gate_summary",
        "exactkv_failure_summary",
        "topk_interpretation_note",
        "forbidden_claims",
        "deferred_work",
        "recommended_stop",
        "recommended_next_phase",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_085_ID:
        errors.append("experiment_id mismatch")

    if report.get("phase16_status") != "complete":
        errors.append("phase16_status must be complete")

    if report.get("phase16_completed_steps") != PHASE16_COMPLETED_STEPS:
        errors.append(f"phase16_completed_steps must be {PHASE16_COMPLETED_STEPS}")

    if report.get("recommended_stop") is not True:
        errors.append("recommended_stop must be true")

    if report.get("recommended_next_phase") != RECOMMENDED_NEXT_PHASE:
        errors.append("recommended_next_phase mismatch")

    cf = report.get("claim_freeze") or {}
    for claim in ALLOWED_CLAIMS:
        if claim not in (cf.get("allowed_claims") or []):
            errors.append(f"missing allowed claim: {claim}")

    for claim in FORBIDDEN_CLAIMS:
        if claim not in (cf.get("forbidden_claims") or []):
            errors.append(f"missing forbidden claim: {claim}")

    note = (report.get("topk_interpretation_note") or "").lower()
    if "supplementary" not in note or "exactness guarantee" not in note:
        errors.append("topk_interpretation_note must mark supplementary-only status")

    return errors
