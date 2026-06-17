"""Broader model validation panel (Phase 17B / Exp 087).

Runs guarded decode-time shadow diagnostics across a small Qwen model panel.
Not a benchmark suite, performance test, or production validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from exactkv.attention.decode_time_shadow_observer import (
    PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    _aggregate_safety_gate_summary,
    _build_guarded_panel_cell,
    _cell_safety_gates,
    _safety_gates_ok,
    default_exp083_prompts,
)
from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS

EXPERIMENT_087_ID = "exp087_broader_model_validation_panel"
DEFAULT_EXP087_REPORT = Path("reports/experiment_087_broader_model_validation_panel.json")
PHASE_17B = "17B"

DEFAULT_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
OPTIONAL_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
DEFAULT_COMPRESSORS: tuple[str, ...] = ("noop", "int8")

CLAIM_SCOPE_NOTE = (
    "Results are model-scoped and panel-scoped only. Passing this panel does not "
    "prove general model-family support, production readiness, speed, memory savings, "
    "serving capability, or VeriCache reproduction."
)


def resolve_model_panel(
    *,
    model_ids: Sequence[str] | None = None,
    include_optional_models: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Return (default_requested, optional_requested, panel_to_run)."""
    if model_ids is not None:
        panel = list(model_ids)
        return panel, [], panel

    default = list(DEFAULT_MODEL_IDS)
    optional = list(OPTIONAL_MODEL_IDS) if include_optional_models else []
    return default, optional, default + optional


def _exp087_safety_gates(
    *,
    baseline_completed: bool,
    guarded_completed: bool,
    tok_match: bool,
    txt_match: bool,
) -> dict[str, bool]:
    return {
        **_cell_safety_gates(),
        "baseline_generation_completed": baseline_completed,
        "guarded_shadow_generation_completed": guarded_completed,
        "baseline_vs_guarded_token_match": tok_match,
        "baseline_vs_guarded_text_match": txt_match,
    }


def _exp087_safety_gates_ok(gates: dict[str, bool]) -> bool:
    for key in (
        "baseline_generation_completed",
        "guarded_shadow_generation_completed",
        "baseline_vs_guarded_token_match",
        "baseline_vs_guarded_text_match",
    ):
        if gates.get(key) is not True:
            return False
    return _safety_gates_ok(gates)


def _finalize_cell(cell: dict[str, Any], *, model_id: str) -> tuple[dict[str, Any], bool]:
    """Attach model_id, extended safety gates, and post-hoc match flag."""
    gates = _exp087_safety_gates(
        baseline_completed=bool(cell.get("baseline_generation_completed")),
        guarded_completed=bool(cell.get("guarded_shadow_generation_completed")),
        tok_match=bool(cell.get("baseline_vs_guarded_token_match")),
        txt_match=bool(cell.get("baseline_vs_guarded_text_match")),
    )
    posthoc = cell.get("posthoc_comparison_summary") or {}
    dt_match = bool(posthoc.get("all_match")) if posthoc.get("total_rounds") else None
    out = {
        **cell,
        "model_id": model_id,
        "decode_time_vs_posthoc_shadow_match": dt_match,
        "safety_gates": gates,
    }
    blockers = list(out.get("blockers") or [])
    failed = not _exp087_safety_gates_ok(gates)
    if failed and "safety_gate_failed" not in blockers:
        blockers.append("safety_gate_failed")
    out["blockers"] = blockers
    return out, failed


def run_exp087_broader_model_validation_panel(
    *,
    model_ids: Sequence[str] | None = None,
    include_optional_models: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    prompts: Sequence[tuple[str, str]] | None = None,
    max_prompts: int = 2,
    max_new_tokens: int = 4,
    compressors_requested: Sequence[str] = DEFAULT_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_shadow_fail: bool = True,
    allow_model_blocked: bool = True,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    guarded_generation_fn: Callable[..., dict[str, Any]] | None = None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Experiment 087 broader model validation panel."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    default_req, optional_req, panel = resolve_model_panel(
        model_ids=model_ids,
        include_optional_models=include_optional_models,
    )
    prompt_panel = list(prompts) if prompts is not None else default_exp083_prompts()[:max_prompts]
    runnable, _blocked_comp = resolve_panel_compressors(compressors_requested)

    blockers: list[str] = []
    if not runnable:
        blockers.append("no compressors runnable via get_compressor API")

    model_results: list[dict[str, Any]] = []
    models_loaded: list[str] = []
    models_blocked: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    token_match = 0
    text_match = 0
    successful_cells = 0
    blocked_cells = 0
    failed_cells = 0
    dt_callbacks = 0
    dt_success = 0
    dt_exceptions = 0
    dt_vs_posthoc_match = 0
    dt_vs_posthoc_total = 0

    default_models_blocked = 0

    for model_id in panel:
        runtime: Any | None = None
        load_blockers: list[str] = []
        if baseline_generation_fn is None or guarded_generation_fn is None:
            try:
                if runtime_loader is not None:
                    runtime = runtime_loader(
                        model_id=model_id,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files_only,
                    )
                else:
                    from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
                    from exactkv.runtime.model_runtime import ModelRuntime

                    runtime = ModelRuntime(model_id, device=device, dtype=dtype)
            except Exception as exc:  # noqa: BLE001
                load_blockers.append(f"{type(exc).__name__}: {exc}")

        if runtime is None and (baseline_generation_fn is None or guarded_generation_fn is None):
            reason = "; ".join(load_blockers) or "model load failed"
            if model_id in default_req:
                default_models_blocked += 1
            models_blocked.append({"model_id": model_id, "blocked_reason": reason})
            model_results.append({
                "model_id": model_id,
                "model_status": "blocked",
                "blocked_reason": reason,
                "cells": [],
                "model_level_summary": {
                    "total_cells": 0,
                    "successful_cells": 0,
                    "blocked_cells": 0,
                },
            })
            continue

        models_loaded.append(model_id)
        model_cells: list[dict[str, Any]] = []
        model_ok = 0
        model_blocked = 0

        for prompt_id, prompt_text in prompt_panel:
            preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
            b_fn = (
                (lambda mid: lambda **kw: baseline_generation_fn(model_id=mid, **kw))(model_id)
                if baseline_generation_fn is not None
                else None
            )
            g_fn = (
                (lambda mid: lambda **kw: guarded_generation_fn(model_id=mid, **kw))(model_id)
                if guarded_generation_fn is not None
                else None
            )
            for compressor in runnable:
                raw_cell, _failed_raw, _dt_shadow_cells = _build_guarded_panel_cell(
                    prompt_id=prompt_id,
                    prompt_preview=preview,
                    compressor=compressor,
                    max_new_tokens=max_new_tokens,
                    runtime=runtime,
                    draft_len=draft_len,
                    global_blockers=blockers + load_blockers,
                    baseline_generation_fn=b_fn,
                    guarded_generation_fn=g_fn,
                    shadow_diagnostic_fn=shadow_diagnostic_fn,
                    allow_shadow_fail=allow_shadow_fail,
                )
                cell, failed = _finalize_cell(raw_cell, model_id=model_id)
                model_cells.append(cell)
                all_cells.append(cell)

                if cell["baseline_vs_guarded_token_match"]:
                    token_match += 1
                if cell["baseline_vs_guarded_text_match"]:
                    text_match += 1

                dt_callbacks += cell["decode_time_shadow_callback_count"]
                dt_success += cell["decode_time_shadow_successful_callbacks"]
                dt_exceptions += cell["decode_time_shadow_exception_callbacks"]

                if cell.get("decode_time_vs_posthoc_shadow_match") is not None:
                    dt_vs_posthoc_total += 1
                    if cell["decode_time_vs_posthoc_shadow_match"]:
                        dt_vs_posthoc_match += 1

                completed = (
                    cell["baseline_generation_completed"]
                    and cell["guarded_shadow_generation_completed"]
                )
                if completed and not failed:
                    successful_cells += 1
                    model_ok += 1
                elif not completed:
                    blocked_cells += 1
                    model_blocked += 1
                if failed:
                    failed_cells += 1

        model_results.append({
            "model_id": model_id,
            "model_status": "loaded",
            "blocked_reason": None,
            "cells": model_cells,
            "model_level_summary": {
                "total_cells": len(model_cells),
                "successful_cells": model_ok,
                "blocked_cells": model_blocked,
                "token_match_cells": sum(
                    1 for c in model_cells if c.get("baseline_vs_guarded_token_match")
                ),
                "text_match_cells": sum(
                    1 for c in model_cells if c.get("baseline_vs_guarded_text_match")
                ),
            },
        })

    total = len(all_cells)
    all_default_blocked = default_models_blocked == len(default_req) and len(default_req) > 0

    if all_default_blocked:
        status = "blocked" if allow_model_blocked else "failed"
        blockers.append("all_default_models_blocked")
    elif failed_cells > 0:
        status = "failed"
    elif successful_cells == total and total > 0:
        status = "diagnostic_complete"
    elif successful_cells > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    sg = _aggregate_safety_gate_summary(all_cells)
    # Re-aggregate with exp087 gates
    sg087_ok = sum(1 for c in all_cells if _exp087_safety_gates_ok(c.get("safety_gates") or {}))

    return {
        "experiment_id": EXPERIMENT_087_ID,
        "status": status,
        "phase": PHASE_17B,
        "device": device,
        "dtype": dtype,
        "default_models_requested": default_req,
        "optional_models_requested": optional_req,
        "models_loaded": models_loaded,
        "models_blocked": models_blocked,
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens": max_new_tokens,
        "total_cells": total,
        "successful_cells": successful_cells,
        "blocked_cells": blocked_cells,
        "baseline_vs_guarded_token_match_cells": token_match,
        "baseline_vs_guarded_text_match_cells": text_match,
        "exactkv_failure_summary": {
            "baseline_failures": sum(
                1 for c in all_cells if (c.get("exactkv_failures_baseline") or 0) > 0
            ),
            "guarded_failures": sum(
                1 for c in all_cells if (c.get("exactkv_failures_guarded") or 0) > 0
            ),
        },
        "decode_time_shadow_callback_summary": {
            "callback_count": dt_callbacks,
            "successful_callbacks": dt_success,
            "exception_callbacks": dt_exceptions,
        },
        "decode_time_vs_posthoc_shadow_match_summary": {
            "cells_with_comparison": dt_vs_posthoc_total,
            "cells_matching": dt_vs_posthoc_match,
            "cells_mismatching": dt_vs_posthoc_total - dt_vs_posthoc_match,
        },
        "safety_gate_summary": {
            "cells_all_gates_ok": sg087_ok,
            "cells_with_gate_failure": total - sg087_ok,
            **sg,
        },
        "model_results": model_results,
        "claim_scope_note": CLAIM_SCOPE_NOTE,
        "claim_note": CLAIM_SCOPE_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "guarded_decode_time_shadow_cli_flag": PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        "blockers": blockers,
        "limitations": [
            "Broader model validation for diagnostic guarded shadow only.",
            "Not production model-family support or benchmark suite.",
            "Panel-scoped results; top-k agreement supplementary only.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
    }


def validate_exp087_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "device",
        "dtype",
        "default_models_requested",
        "optional_models_requested",
        "models_loaded",
        "models_blocked",
        "compressors_requested",
        "compressors_run",
        "total_cells",
        "successful_cells",
        "blocked_cells",
        "baseline_vs_guarded_token_match_cells",
        "baseline_vs_guarded_text_match_cells",
        "exactkv_failure_summary",
        "decode_time_shadow_callback_summary",
        "decode_time_vs_posthoc_shadow_match_summary",
        "safety_gate_summary",
        "claim_scope_note",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "model_results",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_087_ID:
        errors.append("experiment_id mismatch")

    if CLAIM_SCOPE_NOTE not in (report.get("claim_scope_note") or ""):
        errors.append("claim_scope_note must include panel-scoped wording")

    for idx, mr in enumerate(report.get("model_results", [])):
        for ck in ("model_id", "model_status", "cells", "model_level_summary"):
            if ck not in mr:
                errors.append(f"model_results[{idx}] missing {ck}")
        for cidx, cell in enumerate(mr.get("cells") or []):
            gates = cell.get("safety_gates") or {}
            if gates.get("decode_time_shadow_used_for_token_commit") is not False:
                errors.append(
                    f"model_results[{idx}].cells[{cidx}] "
                    "decode_time_shadow_used_for_token_commit must be false",
                )

    return errors
