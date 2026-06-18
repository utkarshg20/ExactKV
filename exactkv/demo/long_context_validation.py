"""Longer-context guarded-shadow validation panel (Phase 17C / Exp 088).

Validates guarded decode-time shadow on deterministic long prompts.
Not a benchmark, production validation, or performance test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from exactkv.attention.decode_time_shadow_observer import (
    PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    _aggregate_safety_gate_summary,
    _build_guarded_panel_cell,
)
from exactkv.attention.generation_shadow_review import SHADOW_FORBIDDEN_CLAIMS
from exactkv.demo.broader_model_validation import (
    _exp087_safety_gates,
    _exp087_safety_gates_ok,
)

EXPERIMENT_088_ID = "exp088_long_context_validation_panel"
DEFAULT_EXP088_REPORT = Path("reports/experiment_088_long_context_validation_panel.json")
PHASE_17C = "17C"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_CONTEXT_TOKENS: tuple[int, ...] = (128, 256, 512)
DEFAULT_PROMPT_FAMILIES: tuple[str, ...] = ("factual", "structured", "code")
DEFAULT_COMPRESSORS: tuple[str, ...] = ("noop", "int8")

FAMILY_FILLERS: dict[str, str] = {
    "factual": (
        "Paris is the capital of France. The Earth orbits the Sun. "
        "Water boils at one hundred degrees Celsius at sea level. "
        "ExactKV long-context validation uses deterministic factual filler. "
    ),
    "structured": (
        '{"experiment": "exp088", "mode": "long_context", "family": "structured", '
        '"index": 0, "note": "deterministic JSON-like filler for panel validation"} '
    ),
    "code": (
        "def segment(n: int) -> int:\n    return n + 1\n"
        "# ExactKV long-context code-like filler segment\n"
    ),
}

CLAIM_SCOPE_NOTE = (
    "Results are model-scoped, panel-scoped, and context-length-scoped only. "
    "Passing this panel does not prove general long-context support, production "
    "readiness, speed, memory savings, serving capability, or VeriCache reproduction."
)


def resolve_model_panel(
    *,
    model_id: str | None = None,
    include_instruct: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Return (default_requested, optional_requested, panel_to_run)."""
    if model_id is not None:
        return [model_id], [], [model_id]
    default = [DEFAULT_MODEL_ID]
    optional = [INSTRUCT_MODEL_ID] if include_instruct else []
    return default, optional, default + optional


def generate_family_long_prompt(
    family: str,
    target_context_tokens: int,
    *,
    tokenizer: Any | None = None,
    tokenize_fn: Callable[[str], list[int]] | None = None,
    decode_fn: Callable[[list[int]], str] | None = None,
    max_iterations: int = 20_000,
) -> tuple[str, int]:
    """Build deterministic long prompt; return text and actual token count."""
    if family not in FAMILY_FILLERS:
        raise ValueError(f"unknown prompt family: {family}")
    if target_context_tokens <= 0:
        raise ValueError("target_context_tokens must be positive")

    filler = FAMILY_FILLERS[family]
    if tokenizer is not None:
        from exactkv.attention.hf_single_layer_probe import generate_long_prompt_text

        return generate_long_prompt_text(
            tokenizer, target_context_tokens, filler=filler,
        )

    if tokenize_fn is None:
        raise ValueError("tokenizer or tokenize_fn required")

    text = filler.strip()
    for i in range(max_iterations):
        token_ids = tokenize_fn(text)
        if len(token_ids) >= target_context_tokens:
            truncated = token_ids[:target_context_tokens]
            if decode_fn is not None:
                decoded = decode_fn(truncated)
                actual = len(tokenize_fn(decoded))
                return decoded, actual
            while len(tokenize_fn(text)) > target_context_tokens and len(text) > 1:
                text = text[:-4]
            return text, len(tokenize_fn(text))
        text += f" {filler} segment_{i}."
    raise ValueError(f"could not reach target token length {target_context_tokens}")


def build_panel_cells(
    *,
    target_context_tokens: Sequence[int],
    prompt_families: Sequence[str],
    compressors: Sequence[str],
    tokenizer: Any | None = None,
    tokenize_fn: Callable[[str], list[int]] | None = None,
    decode_fn: Callable[[list[int]], str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate panel cell specs without running generation."""
    specs: list[dict[str, Any]] = []
    for target in target_context_tokens:
        for family in prompt_families:
            text, actual = generate_family_long_prompt(
                family,
                target,
                tokenizer=tokenizer,
                tokenize_fn=tokenize_fn,
                decode_fn=decode_fn,
            )
            prompt_id = f"{family}_{target}"
            preview = text if len(text) <= 80 else text[:77] + "..."
            for compressor in compressors:
                specs.append({
                    "prompt_id": prompt_id,
                    "prompt_preview": preview,
                    "prompt_text": text,
                    "prompt_family": family,
                    "target_context_tokens": target,
                    "actual_prompt_token_count": actual,
                    "compressor": compressor,
                })
    return specs


def _finalize_long_context_cell(
    cell: dict[str, Any],
    *,
    model_id: str,
    prompt_family: str,
    target_context_tokens: int,
    actual_prompt_token_count: int,
) -> tuple[dict[str, Any], bool]:
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
        "prompt_family": prompt_family,
        "target_context_tokens": target_context_tokens,
        "actual_prompt_token_count": actual_prompt_token_count,
        "decode_time_vs_posthoc_shadow_match": dt_match,
        "safety_gates": gates,
    }
    blockers = list(out.get("blockers") or [])
    failed = not _exp087_safety_gates_ok(gates)
    if failed and "safety_gate_failed" not in blockers:
        blockers.append("safety_gate_failed")
    out["blockers"] = blockers
    return out, failed


def _aggregate_context_length_summary(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, dict[str, int]] = {}
    for cell in cells:
        key = str(cell.get("target_context_tokens", ""))
        bucket = by_target.setdefault(
            key,
            {
                "cells": 0,
                "token_match_cells": 0,
                "text_match_cells": 0,
                "successful_cells": 0,
            },
        )
        bucket["cells"] += 1
        if cell.get("baseline_vs_guarded_token_match"):
            bucket["token_match_cells"] += 1
        if cell.get("baseline_vs_guarded_text_match"):
            bucket["text_match_cells"] += 1
        if (
            cell.get("baseline_generation_completed")
            and cell.get("guarded_shadow_generation_completed")
            and _exp087_safety_gates_ok(cell.get("safety_gates") or {})
        ):
            bucket["successful_cells"] += 1
    return by_target


def run_exp088_long_context_validation_panel(
    *,
    model_id: str | None = None,
    include_instruct: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    target_context_tokens: Sequence[int] = DEFAULT_TARGET_CONTEXT_TOKENS,
    prompt_families: Sequence[str] = DEFAULT_PROMPT_FAMILIES,
    max_new_tokens: int = 4,
    compressors_requested: Sequence[str] = DEFAULT_COMPRESSORS,
    draft_len: int = 4,
    local_files_only: bool = False,
    allow_shadow_fail: bool = True,
    allow_model_blocked: bool = True,
    max_cells: int | None = None,
    baseline_generation_fn: Callable[..., dict[str, Any]] | None = None,
    guarded_generation_fn: Callable[..., dict[str, Any]] | None = None,
    shadow_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
    runtime_loader: Callable[..., Any] | None = None,
    tokenize_fn: Callable[[str], list[int]] | None = None,
    panel_specs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 088 longer-context validation panel."""
    from exactkv.attention.generation_shadow_observer import resolve_panel_compressors

    default_req, optional_req, panel_models = resolve_model_panel(
        model_id=model_id,
        include_instruct=include_instruct,
    )
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

    default_model_blocked = False

    for mid in panel_models:
        runtime: Any | None = None
        load_blockers: list[str] = []
        tokenizer_obj: Any | None = None

        if baseline_generation_fn is None or guarded_generation_fn is None:
            try:
                if runtime_loader is not None:
                    runtime = runtime_loader(
                        model_id=mid,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files_only,
                    )
                else:
                    from exactkv.runtime.exactkv_generator import ExactKVGenerator  # noqa: F401
                    from exactkv.runtime.model_runtime import ModelRuntime

                    runtime = ModelRuntime(mid, device=device, dtype=dtype)
                tokenizer_obj = getattr(runtime, "tokenizer", None)
            except Exception as exc:  # noqa: BLE001
                load_blockers.append(f"{type(exc).__name__}: {exc}")

        if runtime is None and (baseline_generation_fn is None or guarded_generation_fn is None):
            reason = "; ".join(load_blockers) or "model load failed"
            if mid == DEFAULT_MODEL_ID:
                default_model_blocked = True
            models_blocked.append({"model_id": mid, "blocked_reason": reason})
            model_results.append({
                "model_id": mid,
                "model_status": "blocked",
                "blocked_reason": reason,
                "cells": [],
                "model_level_summary": {"total_cells": 0, "successful_cells": 0, "failed_cells": 0},
            })
            continue

        if panel_specs is not None:
            specs = list(panel_specs)
        else:
            build_tok_fn = tokenize_fn
            if tokenizer_obj is None and build_tok_fn is None:
                build_tok_fn = lambda t: list(range(len(t.split())))  # noqa: E731
            specs = build_panel_cells(
                target_context_tokens=target_context_tokens,
                prompt_families=prompt_families,
                compressors=runnable,
                tokenizer=tokenizer_obj,
                tokenize_fn=build_tok_fn,
            )
        if max_cells is not None:
            specs = specs[:max_cells]

        models_loaded.append(mid)
        model_cells: list[dict[str, Any]] = []
        model_ok = 0
        model_fail = 0

        b_fn = (
            (lambda m: lambda **kw: baseline_generation_fn(model_id=m, **kw))(mid)
            if baseline_generation_fn is not None
            else None
        )
        g_fn = (
            (lambda m: lambda **kw: guarded_generation_fn(model_id=m, **kw))(mid)
            if guarded_generation_fn is not None
            else None
        )

        for spec in specs:
            raw_cell, _raw_failed, _dt = _build_guarded_panel_cell(
                prompt_id=spec["prompt_id"],
                prompt_preview=spec["prompt_text"],
                compressor=spec["compressor"],
                max_new_tokens=max_new_tokens,
                runtime=runtime,
                draft_len=draft_len,
                global_blockers=blockers + load_blockers,
                baseline_generation_fn=b_fn,
                guarded_generation_fn=g_fn,
                shadow_diagnostic_fn=shadow_diagnostic_fn,
                allow_shadow_fail=allow_shadow_fail,
            )
            cell, failed = _finalize_long_context_cell(
                raw_cell,
                model_id=mid,
                prompt_family=spec["prompt_family"],
                target_context_tokens=spec["target_context_tokens"],
                actual_prompt_token_count=spec["actual_prompt_token_count"],
            )
            cell["prompt_preview"] = spec["prompt_preview"]
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
            if failed:
                failed_cells += 1
                model_fail += 1

        model_results.append({
            "model_id": mid,
            "model_status": "loaded",
            "blocked_reason": None,
            "cells": model_cells,
            "model_level_summary": {
                "total_cells": len(model_cells),
                "successful_cells": model_ok,
                "failed_cells": model_fail,
                "token_match_cells": sum(
                    1 for c in model_cells if c.get("baseline_vs_guarded_token_match")
                ),
            },
        })

    total = len(all_cells)

    if default_model_blocked and DEFAULT_MODEL_ID in default_req:
        status = "blocked" if allow_model_blocked else "failed"
        blockers.append("default_model_blocked")
    elif failed_cells > 0:
        status = "failed"
    elif successful_cells == total and total > 0:
        status = "diagnostic_complete"
    elif successful_cells > 0:
        status = "diagnostic_partial"
    else:
        status = "blocked"

    sg_ok = sum(1 for c in all_cells if _exp087_safety_gates_ok(c.get("safety_gates") or {}))

    return {
        "experiment_id": EXPERIMENT_088_ID,
        "status": status,
        "phase": PHASE_17C,
        "device": device,
        "dtype": dtype,
        "models_requested": panel_models,
        "default_models_requested": default_req,
        "optional_models_requested": optional_req,
        "models_loaded": models_loaded,
        "models_blocked": models_blocked,
        "target_context_tokens": list(target_context_tokens),
        "prompt_families": list(prompt_families),
        "compressors_requested": list(compressors_requested),
        "compressors_run": runnable,
        "max_new_tokens": max_new_tokens,
        "total_cells": total,
        "successful_cells": successful_cells,
        "failed_cells": failed_cells,
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
            "cells_all_gates_ok": sg_ok,
            "cells_with_gate_failure": total - sg_ok,
            **_aggregate_safety_gate_summary(all_cells),
        },
        "context_length_summary": _aggregate_context_length_summary(all_cells),
        "model_results": model_results,
        "claim_scope_note": CLAIM_SCOPE_NOTE,
        "claim_note": CLAIM_SCOPE_NOTE,
        "forbidden_claims": list(SHADOW_FORBIDDEN_CLAIMS),
        "guarded_decode_time_shadow_cli_flag": PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        "blockers": blockers,
        "limitations": [
            "Longer-context validation for diagnostic guarded shadow only.",
            "Not general long-context support or benchmark suite.",
            "Target token lengths are approximate; actual counts recorded.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
    }


def validate_exp088_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "models_requested",
        "models_loaded",
        "models_blocked",
        "target_context_tokens",
        "prompt_families",
        "compressors_requested",
        "compressors_run",
        "max_new_tokens",
        "total_cells",
        "successful_cells",
        "failed_cells",
        "blocked_cells",
        "baseline_vs_guarded_token_match_cells",
        "baseline_vs_guarded_text_match_cells",
        "exactkv_failure_summary",
        "decode_time_shadow_callback_summary",
        "decode_time_vs_posthoc_shadow_match_summary",
        "safety_gate_summary",
        "context_length_summary",
        "claim_scope_note",
        "blockers",
        "limitations",
        "no_performance_claims_note",
        "model_results",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_088_ID:
        errors.append("experiment_id mismatch")

    if "context-length-scoped" not in (report.get("claim_scope_note") or "").lower():
        errors.append("claim_scope_note must include context-length-scoped wording")

    for idx, mr in enumerate(report.get("model_results", [])):
        for ck in ("model_id", "model_status", "cells", "model_level_summary"):
            if ck not in mr:
                errors.append(f"model_results[{idx}] missing {ck}")
        for cidx, cell in enumerate(mr.get("cells") or []):
            for ck in (
                "prompt_family",
                "target_context_tokens",
                "actual_prompt_token_count",
                "safety_gates",
            ):
                if ck not in cell:
                    errors.append(f"model_results[{idx}].cells[{cidx}] missing {ck}")

    return errors
