"""L4 runtime coupling stress panel (Phase 21N / Exp 115).

Multi-model stress expansion over Exp 114 inference-driven trace coupling.
Trace-only diagnostics — no L4 commit or generator modifications.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.safety.integration_safety_spec import NO_PERFORMANCE_CLAIMS_NOTE
from exactkv.safety.l4_runtime_coupling import (
    L4_SAFETY_LEVEL,
    MODE,
    STAGE,
    STAGE3_DECISIONS,
    FORBIDDEN_CLAIM_PHRASES,
    ModelOutputRecord,
    build_panel_cell_from_model_record,
    build_runtime_trace_records,
    build_synthetic_model_output,
    cache_key_for_cell,
    load_model_outputs,
)
from exactkv.safety.l4_verifier_mediated_design_spec import build_l4_claim_boundaries
from exactkv.safety.pre_l4_gate_review import L4_IMPLEMENTATION_BLOCKERS

EXPERIMENT_115_ID = "exp115_l4_runtime_coupling_stress_panel"
DEFAULT_EXP115_REPORT = Path(
    "reports/experiment_115_l4_runtime_coupling_stress_panel.json",
)
PHASE_21N = "21N"
STRESS_STAGE = "runtime_coupling_stress_expansion"
STRESS_MODE = "multi_model_trace_consistency_panel"

RECOMMENDED_NEXT_PHASE_21N = "phase21o_instability_regime_extraction"
FORBIDDEN_NEXT_PHASES_21N: tuple[str, ...] = (
    "l4_runtime_commit_implementation",
    "l4_default_runtime_modification",
    "l4_verifier_in_loop_execution",
)

DEFAULT_STRESS_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
DEFAULT_STRESS_COMPRESSORS: tuple[str, ...] = (
    "noop",
    "int8",
    "int4_sim",
    "k8_v4_sim",
)
DEFAULT_STRESS_MAX_NEW_TOKENS: tuple[int, ...] = (4, 8, 16)
DEFAULT_STRESS_PROMPT_COUNT = 6
EXPECTED_STRESS_CELL_COUNT = 144

DIVERGENCE_DECISIONS: frozenset[str] = frozenset(
    {"REJECT", "INVALID_TRACE", "BLOCK_MISSING_EVIDENCE"},
)

PANEL_OUTCOME_COMPLETE = "stress_panel_complete"
PANEL_OUTCOME_INCOMPLETE = "stress_panel_incomplete"
PANEL_OUTCOME_BLOCKED = "stress_panel_blocked"


@dataclass(frozen=True)
class CrossModelConsistencyMetrics:
    cross_model_agreement_rate: float
    cross_model_prefix_stability: float
    cross_model_failure_delta: float
    comparable_groups: int
    agreement_groups: int
    prefix_stable_groups: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressPanelValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRESS_PANEL_PROMPTS: tuple[tuple[str, str], ...] = (
    ("p0_capital_france", "The capital of France is"),
    ("p1_simple_math", "Two plus two equals"),
    ("p2_json_tool", 'Complete JSON: {"name": "get_weather", "city":'),
    ("p3_code_fn", "def add(a, b):\n    return"),
    ("p4_factual", "The largest planet in our solar system is"),
    ("p5_structured", 'Output only: {"status": "ok", "value":'),
)


def default_stress_panel_prompts(
    max_prompts: int = DEFAULT_STRESS_PROMPT_COUNT,
) -> list[tuple[str, str]]:
    """Six deterministic prompts for stress panel."""
    return list(STRESS_PANEL_PROMPTS[:max_prompts])


def _primary_trace_summary(cell: Mapping[str, Any]) -> dict[str, Any] | None:
    records = cell.get("trace_records") or []
    if not records:
        return None
    rec = records[0]
    return {
        "proposal_token_ids": tuple(rec.get("proposal_tokens") or ()),
        "decision": str(rec.get("decision") or ""),
        "prefix_length": int(rec.get("prefix_length") or 0),
        "mismatch_index": rec.get("mismatch_index"),
    }


def _group_key(cell: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(cell.get("prompt_id") or ""),
        str(cell.get("compressor") or ""),
        int(cell.get("max_new_tokens") or 0),
    )


def _is_divergent_cell(cell: Mapping[str, Any]) -> bool:
    if (cell.get("exactkv_failures") or 0) > 0:
        return True
    for decision in cell.get("decisions") or []:
        if decision in DIVERGENCE_DECISIONS:
            return True
    return False


def compute_cross_model_consistency_metrics(
    cells: Sequence[Mapping[str, Any]],
) -> CrossModelConsistencyMetrics:
    """Compare proposal/decision/prefix across models per (prompt, compressor, length)."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        if not cell.get("generation_completed"):
            continue
        summary = _primary_trace_summary(cell)
        if summary is None:
            continue
        key = _group_key(cell)
        groups.setdefault(key, []).append(
            {
                "model_name": cell.get("model_name"),
                "summary": summary,
                "divergent": _is_divergent_cell(cell),
            },
        )

    comparable = 0
    agreement = 0
    prefix_stable = 0
    failure_deltas: list[float] = []

    for entries in groups.values():
        if len(entries) < 2:
            continue
        comparable += 1
        decisions = {e["summary"]["decision"] for e in entries}
        prefixes = {e["summary"]["prefix_length"] for e in entries}
        if len(decisions) == 1:
            agreement += 1
        if len(prefixes) == 1:
            prefix_stable += 1
        fail_rates = [1.0 if e["divergent"] else 0.0 for e in entries]
        failure_deltas.append(max(fail_rates) - min(fail_rates))

    n = comparable or 1
    return CrossModelConsistencyMetrics(
        cross_model_agreement_rate=agreement / n,
        cross_model_prefix_stability=prefix_stable / n,
        cross_model_failure_delta=(
            sum(failure_deltas) / len(failure_deltas) if failure_deltas else 0.0
        ),
        comparable_groups=comparable,
        agreement_groups=agreement,
        prefix_stable_groups=prefix_stable,
    )


def compute_divergence_heatmap(
    cells: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
) -> dict[str, Any]:
    """Divergence rate matrix divergence[model][compressor][length]."""
    heatmap: dict[str, dict[str, dict[str, float]]] = {
        m: {c: {} for c in compressors} for m in models
    }
    counts: dict[tuple[str, str, int], tuple[int, int]] = {}

    for cell in cells:
        key = (
            str(cell.get("model_name") or ""),
            str(cell.get("compressor") or ""),
            int(cell.get("max_new_tokens") or 0),
        )
        total, div = counts.get(key, (0, 0))
        total += 1
        if _is_divergent_cell(cell):
            div += 1
        counts[key] = (total, div)

    for model in models:
        for compressor in compressors:
            for mnt in max_new_tokens_values:
                total, div = counts.get((model, compressor, mnt), (0, 0))
                rate = (div / total) if total else 0.0
                heatmap[model][compressor][str(mnt)] = rate

    return {
        "divergence": heatmap,
        "divergence_counts": {
            f"{m}|{c}|{mnt}": {"total": counts.get((m, c, mnt), (0, 0))[0],
                               "divergent": counts.get((m, c, mnt), (0, 0))[1]}
            for m in models
            for c in compressors
            for mnt in max_new_tokens_values
        },
    }


def compute_verifier_stability_score(
    cells: Sequence[Mapping[str, Any]],
) -> float:
    """Fraction of cross-model groups with identical verifier decisions."""
    metrics = compute_cross_model_consistency_metrics(cells)
    if metrics.comparable_groups == 0:
        return 1.0
    return metrics.cross_model_agreement_rate


def compute_proposal_instability_rate(
    cells: Sequence[Mapping[str, Any]],
) -> float:
    """Fraction of cross-model groups where proposal_token_ids differ."""
    groups: dict[tuple[str, str, int], list[tuple[int, ...]]] = {}
    for cell in cells:
        if not cell.get("generation_completed"):
            continue
        summary = _primary_trace_summary(cell)
        if summary is None:
            continue
        groups.setdefault(_group_key(cell), []).append(summary["proposal_token_ids"])

    unstable = 0
    comparable = 0
    for proposals in groups.values():
        if len(proposals) < 2:
            continue
        comparable += 1
        if len(set(proposals)) > 1:
            unstable += 1
    if comparable == 0:
        return 0.0
    return unstable / comparable


def collect_failure_conditions(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Track detected failure conditions without resolving them."""
    missing_verifier = 0
    proposal_mismatch = 0
    cross_model_spikes = 0
    compressor_divergence: dict[str, int] = {c: 0 for c in DEFAULT_STRESS_COMPRESSORS}
    length_instability: dict[str, int] = {str(v): 0 for v in DEFAULT_STRESS_MAX_NEW_TOKENS}

    for cell in cells:
        for decision in cell.get("decisions") or []:
            if decision == "BLOCK_MISSING_EVIDENCE":
                missing_verifier += 1
            if decision == "REJECT":
                proposal_mismatch += 1
        comp = str(cell.get("compressor") or "")
        if comp in compressor_divergence and _is_divergent_cell(cell):
            compressor_divergence[comp] += 1
        mnt = str(cell.get("max_new_tokens") or "")
        if mnt in length_instability and _is_divergent_cell(cell):
            length_instability[mnt] += 1

    metrics = compute_cross_model_consistency_metrics(cells)
    if metrics.cross_model_agreement_rate < 0.5 and metrics.comparable_groups > 0:
        cross_model_spikes += 1

    return {
        "missing_verifier_evidence": missing_verifier,
        "proposal_verifier_mismatch": proposal_mismatch,
        "cross_model_instability_spikes": cross_model_spikes,
        "compressor_induced_divergence": compressor_divergence,
        "length_dependent_instability": length_instability,
    }


def build_deterministic_stress_generation_fn(
    *,
    vary_by_model: bool = True,
) -> Callable[..., dict[str, Any]]:
    """Build deterministic generation_fn for 144-cell replay tests."""

    def _generation_fn(**kwargs: object) -> dict[str, Any]:
        model_id = str(kwargs.get("model_id") or "model")
        prompt_id = str(kwargs.get("prompt_id") or "p0")
        compressor = str(kwargs.get("compressor_name") or "noop")
        max_new_tokens = int(kwargs.get("max_new_tokens") or 4)

        seed_parts = (prompt_id, compressor, max_new_tokens)
        if vary_by_model:
            seed_parts = (model_id, *seed_parts)
        seed = abs(hash(seed_parts)) % 10_000

        draft_len = min(max_new_tokens, 4)
        draft = tuple(1000 + seed + i for i in range(draft_len))
        if compressor in ("int4_sim", "k8_v4_sim") and seed % 3 == 0:
            verifier = draft[:-1] + (draft[-1] + 7,)
        else:
            verifier = draft

        return build_synthetic_model_output(
            prompt_id=prompt_id,
            compressor=compressor,
            draft_tokens=draft,
            verifier_tokens=verifier,
        )

    return _generation_fn


def build_deterministic_stress_cache(
    *,
    models: Sequence[str],
    prompts: Sequence[tuple[str, str]],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    vary_by_model: bool = True,
) -> dict[str, dict[str, Any]]:
    """Pre-build cached outputs for full stress grid."""
    gen_fn = build_deterministic_stress_generation_fn(vary_by_model=vary_by_model)
    cache: dict[str, dict[str, Any]] = {}
    for model in models:
        for prompt_id, prompt_text in prompts:
            for compressor in compressors:
                for mnt in max_new_tokens_values:
                    key = cache_key_for_cell(
                        prompt_id, compressor, mnt, model_name=model,
                    )
                    cache[key] = gen_fn(
                        model_id=model,
                        prompt_id=prompt_id,
                        prompt=prompt_text,
                        compressor_name=compressor,
                        max_new_tokens=mnt,
                    )
    return cache


def _stress_safety_gates() -> dict[str, bool]:
    return {
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "l4_runtime_commit_authorized": False,
        "proposal_used_for_token_commit": False,
        "verifier_is_source_of_truth": True,
        "trace_only": True,
        "dry_run_decision_used_for_token_commit": False,
        "exposed_to_generator": False,
        "rollback_execution_performed": False,
    }


def run_stress_panel_cells(
    *,
    models: Sequence[str],
    prompts: Sequence[tuple[str, str]],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    device: str = "cpu",
    dtype: str = "float32",
    draft_len: int = 4,
    local_files_only: bool = False,
    cached_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    allow_model_blocked: bool = True,
    deterministic_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[ModelOutputRecord]]:
    """Execute full stress grid and return panel cells + model records."""
    if deterministic_mode and generation_fn is None and cached_outputs is None:
        cached_outputs = build_deterministic_stress_cache(
            models=models,
            prompts=prompts,
            compressors=compressors,
            max_new_tokens_values=max_new_tokens_values,
        )

    all_records: list[ModelOutputRecord] = []
    for model_name in models:
        for mnt in max_new_tokens_values:
            model_cache: dict[str, Mapping[str, Any]] | None = None
            if cached_outputs is not None:
                model_cache = {
                    cache_key_for_cell(pid, comp, mnt): cached_outputs[
                        cache_key_for_cell(pid, comp, mnt, model_name=model_name)
                    ]
                    for pid, _ in prompts
                    for comp in compressors
                    if cache_key_for_cell(pid, comp, mnt, model_name=model_name)
                    in cached_outputs
                }
            records = load_model_outputs(
                model_name,
                prompts,
                mnt,
                compressors=compressors,
                device=device,
                dtype=dtype,
                draft_len=draft_len,
                local_files_only=local_files_only,
                cached_outputs=model_cache,
                generation_fn=generation_fn,
                runtime_loader=runtime_loader,
                allow_model_blocked=allow_model_blocked,
            )
            all_records.extend(records)

    cells = [build_panel_cell_from_model_record(r) for r in all_records]
    return cells, all_records


def validate_exp115_stress_panel_report(
    report: Mapping[str, Any],
) -> StressPanelValidationResult:
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "panel_outcome",
        "total_cells",
        "expected_cells",
        "cross_model_metrics",
        "divergence_heatmap",
        "verifier_stability_score",
        "proposal_instability_rate",
        "cells",
        "runtime_commit_authorized",
        "l4_activation",
        "safety_gate_summary",
        "trace_only",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_115_ID:
        errors.append("experiment_id mismatch")

    for flag in (
        "runtime_commit_authorized",
        "l4_activation",
        "exactkv_generator_modified",
        "default_runtime_changed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    if report.get("trace_only") is not True:
        errors.append("trace_only must be true")

    if report.get("safety_gate_summary", {}).get("proposal_used_for_token_commit") is not False:
        errors.append("proposal_used_for_token_commit must be false")

    expected = int(report.get("expected_cells") or EXPECTED_STRESS_CELL_COUNT)
    if int(report.get("total_cells") or 0) != expected:
        errors.append(f"total_cells must be {expected}")

    for cell in report.get("cells") or []:
        for rec in cell.get("trace_records") or []:
            if rec.get("dry_run_decision_used_for_token_commit") is not False:
                errors.append("dry_run_decision_used_for_token_commit must be false")

    return StressPanelValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def run_exp115_l4_runtime_coupling_stress_panel(
    *,
    models: Sequence[str] = DEFAULT_STRESS_MODELS,
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = DEFAULT_STRESS_PROMPT_COUNT,
    compressors: Sequence[str] = DEFAULT_STRESS_COMPRESSORS,
    max_new_tokens_values: Sequence[int] = DEFAULT_STRESS_MAX_NEW_TOKENS,
    device: str = "cpu",
    dtype: str = "float32",
    draft_len: int = 4,
    local_files_only: bool = False,
    cached_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    generation_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    allow_model_blocked: bool = True,
    deterministic_mode: bool = False,
) -> dict[str, Any]:
    """Run Phase 21N multi-model L4 runtime coupling stress panel."""
    prompt_panel = (
        list(prompts) if prompts is not None else default_stress_panel_prompts(max_prompts)
    )
    mnt_values = list(max_new_tokens_values)
    model_list = list(models)
    compressor_list = list(compressors)

    expected_cells = len(model_list) * len(prompt_panel) * len(compressor_list) * len(
        mnt_values,
    )

    cells, _records = run_stress_panel_cells(
        models=model_list,
        prompts=prompt_panel,
        compressors=compressor_list,
        max_new_tokens_values=mnt_values,
        device=device,
        dtype=dtype,
        draft_len=draft_len,
        local_files_only=local_files_only,
        cached_outputs=cached_outputs,
        generation_fn=generation_fn,
        runtime_loader=runtime_loader,
        allow_model_blocked=allow_model_blocked,
        deterministic_mode=deterministic_mode,
    )

    cross_metrics = compute_cross_model_consistency_metrics(cells)
    heatmap = compute_divergence_heatmap(
        cells,
        models=model_list,
        compressors=compressor_list,
        max_new_tokens_values=mnt_values,
    )
    verifier_stability = compute_verifier_stability_score(cells)
    proposal_instability = compute_proposal_instability_rate(cells)
    failure_conditions = collect_failure_conditions(cells)

    completed = sum(1 for c in cells if c.get("generation_completed"))
    trace_total = sum(c.get("trace_record_count", 0) for c in cells)
    decision_counts: dict[str, int] = {d: 0 for d in STAGE3_DECISIONS}
    for cell in cells:
        for decision in cell.get("decisions") or []:
            decision_counts[str(decision)] = decision_counts.get(str(decision), 0) + 1

    model_experiments_run = any(
        c.get("ingestion_source") in ("huggingface", "generation_fn") for c in cells
    )

    safety = _stress_safety_gates()
    runtime_violations = any(
        rec.get("dry_run_decision_used_for_token_commit")
        for c in cells
        for rec in c.get("trace_records") or []
    )

    if len(cells) != expected_cells:
        status = "failed"
        panel_outcome = PANEL_OUTCOME_INCOMPLETE
    elif runtime_violations:
        status = "failed"
        panel_outcome = PANEL_OUTCOME_INCOMPLETE
    elif completed == 0:
        status = "blocked"
        panel_outcome = PANEL_OUTCOME_BLOCKED
    elif completed == len(cells) and trace_total > 0:
        status = "stress_panel_complete"
        panel_outcome = PANEL_OUTCOME_COMPLETE
    else:
        status = "stress_panel_partial"
        panel_outcome = PANEL_OUTCOME_INCOMPLETE

    report = {
        "experiment_id": EXPERIMENT_115_ID,
        "status": status,
        "panel_outcome": panel_outcome,
        "phase": PHASE_21N,
        "safety_level": L4_SAFETY_LEVEL,
        "stage": STRESS_STAGE,
        "mode": STRESS_MODE,
        "models": model_list,
        "device": device,
        "dtype": dtype,
        "compressors": compressor_list,
        "max_new_tokens_values": mnt_values,
        "prompt_count": len(prompt_panel),
        "expected_cells": expected_cells,
        "total_cells": len(cells),
        "successful_generation_cells": completed,
        "trace_records_total": trace_total,
        "decision_status_counts": decision_counts,
        "cross_model_metrics": cross_metrics.to_dict(),
        "divergence_heatmap": heatmap,
        "verifier_stability_score": verifier_stability,
        "proposal_instability_rate": proposal_instability,
        "failure_conditions_detected": failure_conditions,
        "cells": cells,
        "safety_gate_summary": safety,
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "exactkv_generator_modified": False,
        "default_runtime_changed": False,
        "generation_logic_changed": False,
        "production_cli_modified": False,
        "trace_only": True,
        "model_experiments_run": model_experiments_run,
        "deterministic_mode": deterministic_mode,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21N,
        "forbidden_next_phases": list(FORBIDDEN_NEXT_PHASES_21N),
        "implementation_blockers_remaining": [
            {"blocker_id": "l4_runtime_commit", "description": text}
            for text in L4_IMPLEMENTATION_BLOCKERS
            if "commit" in text.lower()
        ][:6],
        "claim_boundaries": build_l4_claim_boundaries().to_dict(),
        "no_performance_claims_note": NO_PERFORMANCE_CLAIMS_NOTE,
        "limitations": [
            "Stress panel measures verifier+proposal stability under scaling; not correctness improvement.",
            "ExactKVGenerator and default runtime unchanged.",
            "Trace-only diagnostics; failure conditions detected not resolved.",
            "No L4 commit; no token commits from dry-run decisions.",
            "No speed, throughput, latency, serving, or memory claims.",
        ],
    }
    report["validation_result"] = validate_exp115_stress_panel_report(report).to_dict()
    return report


def validate_exp115_report(report: Mapping[str, Any]) -> list[str]:
    return list(validate_exp115_stress_panel_report(report).errors)
