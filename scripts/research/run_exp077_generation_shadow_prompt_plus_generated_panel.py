#!/usr/bin/env python3
"""Experiment 077: prompt-plus-generated generation-shadow panel (Phase 16L).

Runs ExactKV generation unchanged once per prompt, then evaluates post-hoc shadow
diagnostics on:
  - prompt_prefix_only
  - prompt_plus_generated_tokens

This is an external observer panel, not generation integration.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.generation_shadow_observer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationOutput,
    GenerationShadowObserverConfig,
    GenerationShadowStatus,
    apply_tolerance_policy_to_shadow_cell,
    default_exactkv_generation,
    reconstruct_shadow_input_ids,
)
from exactkv.attention.generation_shadow_review import PROPOSED_SHADOW_CLI_FLAG  # noqa: E402

EXPERIMENT_077_ID = "exp077_generation_shadow_prompt_plus_generated_panel"
DEFAULT_EXP077_REPORT = Path("reports/experiment_077_generation_shadow_prompt_plus_generated_panel.json")


def _default_prompts(max_prompts: int) -> list[tuple[str, str]]:
    panel = [
        ("p0_capital_france", "The capital of France is"),
        ("p1_simple_math", "Two plus two equals"),
        ("p2_short_story", "Write one sentence about a cat:"),
        ("p3_coding", "Python code to add two numbers:"),
    ]
    return panel[:max_prompts]


def _split_csv(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def run_exp077_panel(
    *,
    model_id: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    shadow_modes: Sequence[str],
    compressors: Sequence[str],
    local_files_only: bool,
    allow_shadow_fail: bool,
    max_prompts: int,
    generation_fn: Callable[..., GenerationOutput] | None = None,
    shadow_replay_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable Exp 077 report dict."""
    # Compressor support (Phase 16L): noop only; others explicitly deferred.
    supported_compressors = [c for c in compressors if c == "noop"]
    deferred = [c for c in compressors if c != "noop"]
    blockers: list[str] = []
    if deferred:
        blockers.append(f"compressor expansion deferred (noop only): {deferred}")
    if not supported_compressors:
        supported_compressors = ["noop"]
        blockers.append("no supported compressors requested; defaulting to noop")

    # Load runtime once.
    runtime = None
    try:
        from exactkv.runtime.model_runtime import ModelRuntime

        runtime = ModelRuntime(model_id, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        runtime = None
        blockers.append(f"runtime load failed: {type(exc).__name__}: {exc}")

    prompt_results: list[dict[str, Any]] = []
    generation_success = 0
    shadow_success = 0
    blocked_shadow = 0
    ppg_success = 0
    ppg_blocked = 0
    total_cells = 0

    tol_summary: dict[str, int] = {}
    topk_top1_agree = 0

    for prompt_id, prompt_text in _default_prompts(max_prompts):
        prompt_preview = prompt_text if len(prompt_text) <= 80 else prompt_text[:77] + "..."
        shadow_cells: list[dict[str, Any]] = []

        if runtime is None and generation_fn is None:
            prompt_results.append({
                "prompt_id": prompt_id,
                "prompt_preview": prompt_preview,
                "generation_completed": False,
                "generation_output_preview": "",
                "generation_output_token_ids_available": False,
                "generation_output_token_count": 0,
                "shadow_cells": [],
            })
            continue

        # Generation runs once, unchanged (noop-only for now).
        comp = supported_compressors[0]
        if generation_fn is not None:
            gen_out = generation_fn(prompt=prompt_text, max_new_tokens=max_new_tokens, compressor_name=comp)
        else:
            gen_out = default_exactkv_generation(
                runtime=runtime,
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
                compressor_name=comp,
            )

        generation_success += 1 if gen_out.generation_completed else 0
        gen_preview = gen_out.generation_output_text if len(gen_out.generation_output_text) <= 80 else gen_out.generation_output_text[:77] + "..."
        token_ids_available = bool(gen_out.generation_output_token_ids)

        for mode in shadow_modes:
            total_cells += 1
            input_ids, seq_mode, recon_blockers = reconstruct_shadow_input_ids(
                gen_out,
                shadow_mode=mode,
                tokenizer_encode=None,
                prompt_text=prompt_text,
                generated_text_retokenize_ok=False,
            )
            if input_ids is None:
                blocked_shadow += 1
                if mode == "prompt_plus_generated_tokens":
                    ppg_blocked += 1
                shadow_cells.append({
                    "shadow_sequence_mode": seq_mode,
                    "shadow_sequence_length": 0,
                    "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                    "tolerance_policy_status": "blocked_missing_generated_token_ids"
                    if mode == "prompt_plus_generated_tokens" else "blocked",
                    "streaming_vs_materialized_metrics": None,
                    "full_vs_streaming_metrics": None,
                    "topk_agreement_metrics": None,
                    "interpretation_note": "Could not reconstruct shadow sequence.",
                    "blockers": recon_blockers,
                })
                continue

            # Shadow replay (Phase 16F-style).
            if shadow_replay_fn is not None:
                shadow_cell = shadow_replay_fn(model=getattr(runtime, "model", None), input_ids=input_ids, prompt_id=prompt_id)
            else:
                from exactkv.attention.generation_shadow_observer import default_offline_shadow_replay

                shadow_cell = default_offline_shadow_replay(
                    model=getattr(runtime, "model", None),
                    input_ids=input_ids,
                    prompt_id=prompt_id,
                    chunk_size=16,
                    accumulator_mode="float32",
                    allow_parity_fail=True,
                )

            if shadow_cell.get("blockers"):
                blocked_shadow += 1
                if mode == "prompt_plus_generated_tokens":
                    ppg_blocked += 1
                shadow_cells.append({
                    "shadow_sequence_mode": seq_mode,
                    "shadow_sequence_length": int(input_ids.shape[-1]),
                    "shadow_status": GenerationShadowStatus.SHADOW_BLOCKED.value,
                    "tolerance_policy_status": "blocked",
                    "streaming_vs_materialized_metrics": shadow_cell.get("streaming_vs_materialized_logit_metrics"),
                    "full_vs_streaming_metrics": shadow_cell.get("full_vs_streaming_logit_metrics"),
                    "topk_agreement_metrics": None,
                    "interpretation_note": "Shadow replay blocked.",
                    "blockers": list(shadow_cell.get("blockers") or []),
                })
                continue

            shadow_success += 1
            if mode == "prompt_plus_generated_tokens":
                ppg_success += 1

            num_layers = int(shadow_cell.get("num_layers_replayed") or 24)
            tol_status, interp = apply_tolerance_policy_to_shadow_cell(shadow_cell, num_layers=num_layers)
            tol_summary[tol_status] = tol_summary.get(tol_status, 0) + 1

            sm = shadow_cell.get("streaming_vs_materialized_logit_metrics") or {}
            fs = shadow_cell.get("full_vs_streaming_logit_metrics") or {}
            topk = {
                "top1_agreement": sm.get("top1_agreement"),
                "top5_overlap": sm.get("top5_overlap"),
                "top10_overlap": sm.get("top10_overlap"),
            }
            if topk.get("top1_agreement"):
                topk_top1_agree += 1

            shadow_cells.append({
                "shadow_sequence_mode": seq_mode,
                "shadow_sequence_length": int(input_ids.shape[-1]),
                "shadow_status": GenerationShadowStatus.SHADOW_COMPLETE.value,
                "tolerance_policy_status": tol_status,
                "streaming_vs_materialized_metrics": sm,
                "full_vs_streaming_metrics": fs,
                "topk_agreement_metrics": topk,
                "interpretation_note": interp,
                "blockers": [],
            })

        prompt_results.append({
            "prompt_id": prompt_id,
            "prompt_preview": prompt_preview,
            "generation_completed": bool(gen_out.generation_completed),
            "generation_output_preview": gen_preview,
            "generation_output_token_ids_available": token_ids_available,
            "generation_output_token_count": len(gen_out.generation_output_token_ids or []),
            "shadow_cells": shadow_cells,
        })

    status = "diagnostic_complete" if shadow_success == total_cells and total_cells else "diagnostic_partial"
    if generation_success == 0:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_077_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "shadow_modes": list(shadow_modes),
        "compressors": list(compressors),
        "total_prompts": len(prompt_results),
        "total_shadow_cells": total_cells,
        "generation_successful_prompts": generation_success,
        "shadow_successful_cells": shadow_success,
        "blocked_shadow_cells": blocked_shadow,
        "prompt_plus_generated_successful_cells": ppg_success,
        "prompt_plus_generated_blocked_cells": ppg_blocked,
        "generation_modified_by_shadow": False,
        "shadow_used_for_token_commit": False,
        "default_runtime_changed": False,
        "topk_agreement_summary": {
            "top1_agreement_cells": topk_top1_agree,
            "cell_count": shadow_success,
        },
        "tolerance_policy_summary": tol_summary,
        "prompt_results": prompt_results,
        "blockers": blockers,
        "limitations": [
            "External post-hoc observer panel; not generation integration.",
            "Prompt+generated replay is fixed-sequence analysis, not token generation.",
            "Top-k agreement is supplementary only.",
            "No per-round decode observer.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
    }


def validate_exp077_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "max_new_tokens",
        "shadow_modes",
        "compressors",
        "total_prompts",
        "total_shadow_cells",
        "generation_successful_prompts",
        "shadow_successful_cells",
        "blocked_shadow_cells",
        "prompt_plus_generated_successful_cells",
        "prompt_plus_generated_blocked_cells",
        "generation_modified_by_shadow",
        "shadow_used_for_token_commit",
        "default_runtime_changed",
        "topk_agreement_summary",
        "tolerance_policy_summary",
        "prompt_results",
        "blockers",
        "limitations",
        "no_performance_claims_note",
    )
    for k in required:
        if k not in report:
            errors.append(f"missing key: {k}")
    if report.get("experiment_id") != EXPERIMENT_077_ID:
        errors.append("experiment_id mismatch")
    if report.get("shadow_used_for_token_commit") is not False:
        errors.append("shadow_used_for_token_commit must be false")
    if report.get("generation_modified_by_shadow") is not False:
        errors.append("generation_modified_by_shadow must be false")
    if report.get("default_runtime_changed") is not False:
        errors.append("default_runtime_changed must be false")

    for i, pr in enumerate(report.get("prompt_results", [])):
        if not isinstance(pr, dict):
            errors.append(f"prompt_results[{i}] not dict")
            continue
        for ck in ("prompt_id", "generation_completed", "generation_output_token_ids_available", "shadow_cells"):
            if ck not in pr:
                errors.append(f"prompt_results[{i}] missing {ck}")
        for j, cell in enumerate(pr.get("shadow_cells", [])):
            if not isinstance(cell, dict):
                errors.append(f"prompt_results[{i}].shadow_cells[{j}] not dict")
                continue
            for ck in ("shadow_sequence_mode", "shadow_sequence_length", "shadow_status", "interpretation_note", "blockers"):
                if ck not in cell:
                    errors.append(f"prompt_results[{i}].shadow_cells[{j}] missing {ck}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 077 generation-shadow prompt+generated panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP077_REPORT)
    parser.add_argument(PROPOSED_SHADOW_CLI_FLAG, action="store_true", dest="shadow_observer")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--shadow-modes", default="prompt_prefix_only,prompt_plus_generated_tokens")
    parser.add_argument("--compressors", default="noop")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    parser.add_argument("--no-allow-shadow-fail", action="store_false", dest="allow_shadow_fail")
    args = parser.parse_args()

    if not args.shadow_observer:
        report = {
            "experiment_id": EXPERIMENT_077_ID,
            "status": "skipped",
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "shadow_modes": _split_csv(args.shadow_modes),
            "compressors": _split_csv(args.compressors),
            "total_prompts": 0,
            "total_shadow_cells": 0,
            "generation_successful_prompts": 0,
            "shadow_successful_cells": 0,
            "blocked_shadow_cells": 0,
            "prompt_plus_generated_successful_cells": 0,
            "prompt_plus_generated_blocked_cells": 0,
            "generation_modified_by_shadow": False,
            "shadow_used_for_token_commit": False,
            "default_runtime_changed": False,
            "topk_agreement_summary": {"top1_agreement_cells": 0, "cell_count": 0},
            "tolerance_policy_summary": {},
            "prompt_results": [],
            "blockers": [f"{PROPOSED_SHADOW_CLI_FLAG} not set"],
            "limitations": ["Observer disabled"],
            "no_performance_claims_note": (
                "No speed, throughput, latency, serving, measured active GPU memory, "
                "or production-memory claim is made."
            ),
        }
    else:
        report = run_exp077_panel(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            shadow_modes=_split_csv(args.shadow_modes),
            compressors=_split_csv(args.compressors),
            local_files_only=args.local_files_only,
            allow_shadow_fail=args.allow_shadow_fail,
            max_prompts=args.max_prompts,
        )

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    errors = validate_exp077_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 077: {report['status']} prompts={report['total_prompts']} "
        f"cells={report['total_shadow_cells']} shadow_ok={report['shadow_successful_cells']}"
    )
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

