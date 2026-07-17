"""Systems diagnostic panel: peak CUDA allocation + per-path wall-clock.

96-cell design (default): 2 models × 3 compressors × 2 ctx × 2 mnt × 4 prompts.

Claim boundary: diagnostic peak process allocation and harness path timing only —
NOT serving throughput, TTFT, RPS, or unqualified production VRAM savings.
Peak includes model weights + KV + temporaries.
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.attention.hf_single_layer_probe import generate_long_prompt_text
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.gpu_memory_pilot import collect_runpod_meta, cuda_available
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

SYSTEMS_DIAGNOSTIC_ID = "systems_diagnostic_panel"
DEFAULT_OUTPUT_DIR = Path("reports/external_panels/systems_diagnostic")
DEFAULT_PUBLIC_JSON = Path("reports/systems/systems_diagnostic.json")
DEFAULT_PUBLIC_MD = Path("reports/systems/systems_diagnostic.md")

CLAIM_BOUNDARY = (
    "Diagnostic peak CUDA allocation and per-path wall-clock on the "
    "systems_diagnostic panel (7B/8B). NOT serving throughput, TTFT, RPS, or "
    "unqualified production VRAM savings. Peak includes model weights + KV + temporaries."
)

FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "speedup",
    "active_gpu_kv_bytes",
    "active_gpu_memory_savings",
})

DEFAULT_MODELS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
SMOKE_MODELS: tuple[str, ...] = ("Qwen/Qwen2.5-0.5B",)
DEFAULT_COMPRESSORS: tuple[str, ...] = ("noop", "int8", "int4_sim")
DEFAULT_CONTEXT_BUCKETS: tuple[int, ...] = (2048, 4096)
DEFAULT_MAX_NEW_TOKENS: tuple[int, ...] = (64, 128)
DEFAULT_DRAFT_LEN = 4

_BASE_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "prompt_id": "sys_lb_qasper_001",
        "category": "longbench_reading",
        "prompt": (
            "Answer the question using only the passage. Passage: ExactKV measures "
            "the first token where compressed KV diverges from full-precision greedy "
            "decoding under a draft/verify/commit loop. Question: What does ExactKV "
            "measure first?"
        ),
    },
    {
        "prompt_id": "sys_lb_hotpot_001",
        "category": "longbench_reading",
        "prompt": (
            "Multi-hop reading. Fact A: Lossy KV can flip an argmax under greedy "
            "decoding. Fact B: Aggregate quality metrics can hide that flip. "
            "Question: Why can average scores look fine while decoding already branched?"
        ),
    },
    {
        "prompt_id": "sys_bfcl_simple_001",
        "category": "bfcl_tool",
        "prompt": (
            "Call a tool as JSON only. Schema: "
            '{"name":"get_weather","arguments":{"city":string}}. '
            "User: What is the weather in Boston?"
        ),
    },
    {
        "prompt_id": "sys_bfcl_parallel_001",
        "category": "bfcl_tool",
        "prompt": (
            "Emit a JSON array of two tool calls. "
            '[{"name":"get_weather","arguments":{"city":"Seattle"}},'
            '{"name":"get_time","arguments":{"tz":"PT"}}]. User: weather and time.'
        ),
    },
)

PROMPT_FILLER = (
    "ExactKV systems-diagnostic filler for context-bucket padding. "
    "Diagnostic peak CUDA and wall-clock only — not serving throughput. "
)


def assert_no_forbidden_fields(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        hits = FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {sorted(hits)} in {path}")
        for k, v in obj.items():
            assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_no_forbidden_fields(item, f"{path}[{i}]")


def _fit_prompt(
    runtime: ModelRuntime,
    base: Mapping[str, str],
    target_context_tokens: int,
) -> dict[str, Any]:
    tokenizer = runtime.tokenizer
    text = str(base["prompt"])
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > target_context_tokens:
        truncated = token_ids[:target_context_tokens]
        fitted = tokenizer.decode(truncated, skip_special_tokens=True)
        actual = len(tokenizer.encode(fitted, add_special_tokens=False))
    else:
        fitted, actual = generate_long_prompt_text(
            tokenizer,
            target_context_tokens,
            filler=PROMPT_FILLER + " " + text,
        )
    return {
        "prompt_id": f"{base['prompt_id']}_ctx{target_context_tokens}",
        "category": base["category"],
        "prompt": fitted,
        "context_bucket": target_context_tokens,
        "actual_context_tokens": actual,
        "base_prompt_id": base["prompt_id"],
    }


def _synthetic_arm(peak: int, wall_ms: float) -> dict[str, Any]:
    return {
        "output_ids": [1, 2, 3],
        "output_text": "synthetic",
        "wall_clock_ms": wall_ms,
        "gpu_peak_allocated_bytes": peak,
        "gpu_allocated_before_bytes": peak - 1000,
        "gpu_allocated_after_bytes": peak - 500,
    }


def _build_deterministic_cell(
    *,
    model_name: str,
    compressor_name: str,
    context_bucket: int,
    max_new_tokens: int,
    base: Mapping[str, str],
) -> dict[str, Any]:
    cell = {
        "prompt_id": f"{base['prompt_id']}_ctx{context_bucket}",
        "base_prompt_id": base["prompt_id"],
        "category": base["category"],
        "model_name": model_name,
        "compressor_name": compressor_name,
        "draft_len": DEFAULT_DRAFT_LEN,
        "max_new_tokens": max_new_tokens,
        "context_bucket": context_bucket,
        "actual_context_tokens": context_bucket,
        "full": _synthetic_arm(8_000_000_000, 100.0),
        "lossy": {
            **_synthetic_arm(7_500_000_000, 90.0),
            "token_exact_match": True,
            "first_divergence_idx": None,
        },
        "exactkv": {
            **_synthetic_arm(8_200_000_000, 150.0),
            "token_exact_match": True,
            "acceptance": {
                "total_drafted": 4,
                "total_accepted": 4,
                "total_rejected": 0,
                "total_corrections": 0,
                "acceptance_rate": 1.0,
            },
        },
        "memory": {
            "stored_kv_bytes": 1_000_000,
            "materialized_working_kv_bytes": 2_000_000,
        },
        "exactkv_failure": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "deterministic_synthetic": True,
    }
    assert_no_forbidden_fields(cell)
    return cell


def _cuda_sync_reset_peak() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _cuda_allocated() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated())


def _cuda_peak() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _best_effort_empty_cache() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _measure_arm(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    _cuda_sync_reset_peak()
    allocated_before = _cuda_allocated()
    t0 = time.perf_counter()
    result = fn()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    peak = _cuda_peak()
    allocated_after = _cuda_allocated()
    return result, {
        "wall_clock_ms": round(wall_ms, 3),
        "gpu_peak_allocated_bytes": peak,
        "gpu_allocated_before_bytes": allocated_before,
        "gpu_allocated_after_bytes": allocated_after,
    }


def run_systems_cell(
    runtime: ModelRuntime,
    prompt_entry: Mapping[str, Any],
    *,
    compressor_name: str,
    draft_len: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    compressor = get_compressor(compressor_name)
    prompt = prompt_entry["prompt"]

    full_res, full_diag = _measure_arm(
        lambda: generate_full_greedy(runtime, prompt, max_new_tokens)
    )
    _best_effort_empty_cache()

    lossy_res, lossy_diag = _measure_arm(
        lambda: generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    )
    _best_effort_empty_cache()

    ekv_res, ekv_diag = _measure_arm(
        lambda: ExactKVGenerator(runtime, compressor, draft_len=draft_len).generate(
            prompt, max_new_tokens
        )
    )
    _best_effort_empty_cache()

    full_ids = full_res.generated_ids.squeeze(0).tolist()
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    cell = {
        "prompt_id": prompt_entry["prompt_id"],
        "base_prompt_id": prompt_entry.get("base_prompt_id"),
        "category": prompt_entry.get("category", "unknown"),
        "model_name": runtime.model_name,
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "max_new_tokens": max_new_tokens,
        "context_bucket": prompt_entry.get("context_bucket"),
        "actual_context_tokens": prompt_entry.get("actual_context_tokens"),
        "full": {
            "output_ids": full_ids,
            "output_text": full_res.output_text,
            **full_diag,
        },
        "lossy": {
            "output_ids": lossy_ids,
            "output_text": lossy_res.output_text,
            "token_exact_match": lossy_exact,
            "first_divergence_idx": first_divergence_idx(
                full_res.generated_ids, lossy_res.generated_ids
            ),
            **lossy_diag,
        },
        "exactkv": {
            "output_ids": ekv_ids,
            "output_text": ekv_res.output_text,
            "token_exact_match": ekv_exact,
            "acceptance": summarize_acceptance(ekv_res.traces).to_dict(),
            **ekv_diag,
        },
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    assert_no_forbidden_fields(cell)
    return cell


def _cell_key(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        cell.get("model_name"),
        cell.get("compressor_name"),
        int(cell.get("context_bucket") or 0),
        int(cell.get("max_new_tokens") or 0),
        cell.get("prompt_id"),
    )


def run_systems_diagnostic_panel(
    *,
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
    context_buckets: Sequence[int] | None = None,
    max_new_tokens: Sequence[int] | None = None,
    draft_len: int = DEFAULT_DRAFT_LEN,
    device: str = "cuda",
    dtype: str = "float16",
    smoke: bool = False,
    deterministic_mode: bool = False,
    resume_cells: Sequence[Mapping[str, Any]] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    model_list = list(SMOKE_MODELS if smoke else (models or DEFAULT_MODELS))
    comp_list = list(compressors or DEFAULT_COMPRESSORS)
    buckets = list(context_buckets or DEFAULT_CONTEXT_BUCKETS)
    mnts = list(max_new_tokens or DEFAULT_MAX_NEW_TOKENS)
    if smoke:
        buckets = [min(512, buckets[0] if buckets else 512)]
        mnts = [min(16, mnts[0] if mnts else 16)]

    if deterministic_mode:
        cells = [
            _build_deterministic_cell(
                model_name=m,
                compressor_name=c,
                context_bucket=b,
                max_new_tokens=n,
                base=base,
            )
            for m in model_list
            for c in comp_list
            for b in buckets
            for n in mnts
            for base in _BASE_PROMPTS
        ]
        report = {
            "schema": "exactkv.systems_diagnostic.v1",
            "panel_id": SYSTEMS_DIAGNOSTIC_ID,
            "claim_boundary": CLAIM_BOUNDARY,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "design": {
                "models": model_list,
                "compressors": comp_list,
                "context_buckets": buckets,
                "max_new_tokens": mnts,
                "draft_len": draft_len,
                "n_prompts": len(_BASE_PROMPTS),
                "expected_cells": len(cells),
                "smoke": smoke,
                "deterministic_mode": True,
            },
            "hardware": {"cuda_available": False, "deterministic_synthetic": True},
            "cells": cells,
            "exactkv_failures": 0,
            "n_cells": len(cells),
        }
        assert_no_forbidden_fields(report)
        return report

    resume_index = {_cell_key(c): dict(c) for c in (resume_cells or [])}
    cells: list[dict[str, Any]] = []
    expected = (
        len(model_list) * len(comp_list) * len(buckets) * len(mnts) * len(_BASE_PROMPTS)
    )
    done = 0
    use_device = device
    if use_device == "cuda" and not cuda_available():
        use_device = "cpu"

    for model_name in model_list:
        if progress_callback:
            progress_callback(f"loading {model_name}")
        runtime = ModelRuntime(model_name, device=use_device, dtype=dtype)
        try:
            _ = generate_full_greedy(runtime, "Warmup.", 4)
            _best_effort_empty_cache()
        except Exception:
            pass

        fitted_cache: dict[int, list[dict[str, Any]]] = {
            bucket: [_fit_prompt(runtime, base, bucket) for base in _BASE_PROMPTS]
            for bucket in buckets
        }

        for compressor_name in comp_list:
            for bucket in buckets:
                for mnt in mnts:
                    for prompt_entry in fitted_cache[bucket]:
                        key = (
                            model_name,
                            compressor_name,
                            bucket,
                            mnt,
                            prompt_entry["prompt_id"],
                        )
                        done += 1
                        if key in resume_index:
                            cells.append(resume_index[key])
                            if progress_callback:
                                progress_callback(
                                    f"RESUME {done}/{expected} {key[1]} "
                                    f"ctx={bucket} mnt={mnt}"
                                )
                            continue
                        if progress_callback:
                            progress_callback(
                                f"cell {done}/{expected} model={model_name} "
                                f"comp={compressor_name} ctx={bucket} mnt={mnt} "
                                f"prompt={prompt_entry['prompt_id']}"
                            )
                        cells.append(
                            run_systems_cell(
                                runtime,
                                prompt_entry,
                                compressor_name=compressor_name,
                                draft_len=draft_len,
                                max_new_tokens=mnt,
                            )
                        )

        del runtime
        _best_effort_empty_cache()

    report = {
        "schema": "exactkv.systems_diagnostic.v1",
        "panel_id": SYSTEMS_DIAGNOSTIC_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "models": model_list,
            "compressors": comp_list,
            "context_buckets": buckets,
            "max_new_tokens": mnts,
            "draft_len": draft_len,
            "n_prompts": len(_BASE_PROMPTS),
            "expected_cells": expected,
            "smoke": smoke,
            "deterministic_mode": False,
        },
        "hardware": collect_runpod_meta(),
        "cells": cells,
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        "n_cells": len(cells),
    }
    assert_no_forbidden_fields(report)
    return report


def write_systems_diagnostic_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path | str,
    markdown_path: Path | str | None = None,
) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if markdown_path is None:
        return
    md = Path(markdown_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    n = int(report.get("n_cells") or 0)
    fails = int(report.get("exactkv_failures") or 0)
    md.write_text(
        "\n".join(
            [
                "# Systems diagnostic panel",
                "",
                f"**Cells:** {n}",
                f"**exactkv_failures:** {fails}",
                "",
                f"**Claim boundary:** {report.get('claim_boundary')}",
                "",
                f"Artifact: `{path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
