"""HF multi-request serving microbench: TTFT-like, RPS, peak CUDA.

Answers practical “real-world” questions (memory / latency / request rate) on the
ExactKV HF harness under a sustained serial request load.

Claim boundary
--------------
This is an **ExactKV HF multi-request diagnostic harness** — not vLLM integration,
not continuous batching, not production serving. Peak CUDA includes model weights
+ KV + temporaries. ``peak_delta_vs_full_bytes`` may be **positive** (ExactKV /
lossy higher than full) because dual state or compressor overhead dominates.
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from exactkv.compressors import get_compressor
from exactkv.metrics.exactness import token_exact_match
from exactkv.metrics.gpu_memory_pilot import collect_runpod_meta, cuda_available
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state

SERVING_MICROBENCH_ID = "serving_microbench_panel"
DEFAULT_OUTPUT_DIR = Path("reports/external_panels/serving_microbench")
DEFAULT_PUBLIC_JSON = Path("reports/systems/serving_microbench.json")
DEFAULT_PUBLIC_MD = Path("reports/systems/serving_microbench.md")

CLAIM_BOUNDARY = (
    "HF multi-request serving microbench (ExactKV harness): TTFT-like latency, "
    "completed-requests/sec under serial load, and peak CUDA allocation for "
    "full / lossy / ExactKV. NOT vLLM integration, NOT continuous batching, "
    "NOT production serving, NOT unqualified VRAM savings. Peak includes "
    "weights + KV + temporaries."
)

# Keep systems_diagnostic forbidden names out of public pack tops.
FORBIDDEN_FIELDS = frozenset({
    "speedup",
    "active_gpu_kv_bytes",
    "active_gpu_memory_savings",
    "throughput",  # use completed_requests_per_sec instead
})

DEFAULT_MODELS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
SMOKE_MODELS: tuple[str, ...] = ("Qwen/Qwen2.5-0.5B",)
DEFAULT_COMPRESSORS: tuple[str, ...] = ("int8", "int4_sim")
DEFAULT_CONTEXT_BUCKETS: tuple[int, ...] = (2048,)
DEFAULT_MAX_NEW_TOKENS: tuple[int, ...] = (64,)
DEFAULT_N_REQUESTS: tuple[int, ...] = (1, 4)
DEFAULT_DRAFT_LEN = 4

_BASE_PROMPT = {
    "prompt_id": "srv_lb_qasper_001",
    "category": "longbench_reading",
    "prompt": (
        "Answer the question using only the passage. Passage: ExactKV measures "
        "the first token where compressed KV diverges from full-precision greedy "
        "decoding under a draft/verify/commit loop. Question: What does ExactKV "
        "measure first?"
    ),
}

PROMPT_FILLER = (
    "ExactKV serving-microbench filler for context-bucket padding. "
    "HF multi-request diagnostic only — not vLLM / production serving. "
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
    body = str(base["prompt"])
    ids = tokenizer.encode(body, add_special_tokens=True)
    if len(ids) < target_context_tokens:
        filler_ids = tokenizer.encode(PROMPT_FILLER, add_special_tokens=False)
        while len(ids) < target_context_tokens and filler_ids:
            need = target_context_tokens - len(ids)
            ids.extend(filler_ids[:need])
            if len(filler_ids) < need:
                continue
            break
        body = tokenizer.decode(ids, skip_special_tokens=False)
        ids = tokenizer.encode(body, add_special_tokens=True)
    if len(ids) > target_context_tokens:
        ids = ids[:target_context_tokens]
        body = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "prompt_id": f"{base['prompt_id']}_ctx{target_context_tokens}",
        "base_prompt_id": base["prompt_id"],
        "category": base["category"],
        "prompt": body,
        "actual_context_tokens": len(ids),
        "context_bucket": target_context_tokens,
    }


def _cuda_sync_reset_peak() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


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


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _timed_full(
    runtime: ModelRuntime,
    prompt: str,
    max_new_tokens: int,
) -> tuple[Any, dict[str, float | int | None]]:
    """Full greedy with TTFT-like = prefill→first-token decision."""
    import torch

    _cuda_sync_reset_peak()
    t0 = time.perf_counter()
    full_state = prefill_to_full_state(runtime, prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    # Reuse generate_full_greedy for the full path (includes its own prefill).
    # Wall for the published e2e is the full generate; TTFT is the timed prefill.
    t1 = time.perf_counter()
    result = generate_full_greedy(runtime, prompt, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t1) * 1000.0
    return result, {
        "ttft_like_ms": round(ttft_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "gpu_peak_allocated_bytes": _cuda_peak(),
    }


def _timed_lossy(
    runtime: ModelRuntime,
    prompt: str,
    compressor: Any,
    max_new_tokens: int,
) -> tuple[Any, dict[str, float | int | None]]:
    import torch

    _cuda_sync_reset_peak()
    t0 = time.perf_counter()
    # Prefill+compress approximation for TTFT-like on lossy path
    full_state = prefill_to_full_state(runtime, prompt)
    _ = compressor.compress(full_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    result = generate_lossy_greedy(runtime, prompt, compressor, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t1) * 1000.0
    return result, {
        "ttft_like_ms": round(ttft_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "gpu_peak_allocated_bytes": _cuda_peak(),
    }


def _timed_exactkv(
    runtime: ModelRuntime,
    prompt: str,
    compressor: Any,
    draft_len: int,
    max_new_tokens: int,
) -> tuple[Any, dict[str, float | int | None]]:
    import torch

    _cuda_sync_reset_peak()
    ttft_box: dict[str, float] = {}

    def _on_round(_snapshot: Any = None) -> None:
        if "ttft_like_ms" not in ttft_box:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ttft_box["ttft_like_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gen = ExactKVGenerator(
        runtime, compressor, draft_len=draft_len, round_observer=_on_round
    )
    result = gen.generate(prompt, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    return result, {
        "ttft_like_ms": round(ttft_box.get("ttft_like_ms", e2e_ms), 3),
        "e2e_ms": round(e2e_ms, 3),
        "gpu_peak_allocated_bytes": _cuda_peak(),
    }


def _run_load(
    arm_fn: Callable[[], tuple[Any, dict[str, Any]]],
    n_requests: int,
) -> dict[str, Any]:
    ttfts: list[float] = []
    e2es: list[float] = []
    peaks: list[int] = []
    last_result: Any = None
    t_load0 = time.perf_counter()
    for _ in range(n_requests):
        result, diag = arm_fn()
        last_result = result
        ttfts.append(float(diag["ttft_like_ms"]))
        e2es.append(float(diag["e2e_ms"]))
        if diag.get("gpu_peak_allocated_bytes") is not None:
            peaks.append(int(diag["gpu_peak_allocated_bytes"]))
        _best_effort_empty_cache()
    load_s = time.perf_counter() - t_load0
    return {
        "n_requests": n_requests,
        "load_wall_s": round(load_s, 4),
        "completed_requests_per_sec": round(n_requests / load_s, 4) if load_s > 0 else None,
        "mean_ttft_like_ms": _mean(ttfts),
        "mean_e2e_ms": _mean(e2es),
        "gpu_peak_allocated_bytes": max(peaks) if peaks else None,
        "_last_result": last_result,
    }


def run_serving_cell(
    runtime: ModelRuntime,
    prompt_entry: Mapping[str, Any],
    *,
    compressor_name: str,
    draft_len: int,
    max_new_tokens: int,
    n_requests: int,
) -> dict[str, Any]:
    compressor = get_compressor(compressor_name)
    prompt = str(prompt_entry["prompt"])

    full_load = _run_load(
        lambda: _timed_full(runtime, prompt, max_new_tokens),
        n_requests,
    )
    full_res = full_load.pop("_last_result")

    lossy_load = _run_load(
        lambda: _timed_lossy(runtime, prompt, compressor, max_new_tokens),
        n_requests,
    )
    lossy_res = lossy_load.pop("_last_result")

    ekv_load = _run_load(
        lambda: _timed_exactkv(
            runtime, prompt, compressor, draft_len, max_new_tokens
        ),
        n_requests,
    )
    ekv_res = ekv_load.pop("_last_result")

    full_ids = full_res.generated_ids[0]
    lossy_ids = lossy_res.generated_ids[0]
    ekv_ids = ekv_res.output_ids
    if not isinstance(ekv_ids, torch.Tensor):
        ekv_ids = torch.tensor(list(ekv_ids), dtype=torch.long)

    full_peak = full_load.get("gpu_peak_allocated_bytes")

    def _delta(peak: int | None) -> int | None:
        if peak is None or full_peak is None:
            return None
        return int(peak) - int(full_peak)

    for arm_load, peak_key in (
        (full_load, "gpu_peak_allocated_bytes"),
        (lossy_load, "gpu_peak_allocated_bytes"),
        (ekv_load, "gpu_peak_allocated_bytes"),
    ):
        arm_load["peak_delta_vs_full_bytes"] = _delta(arm_load.get(peak_key))

    exact_match = token_exact_match(ekv_ids, full_ids)
    return {
        "prompt_id": prompt_entry["prompt_id"],
        "base_prompt_id": prompt_entry.get("base_prompt_id"),
        "category": prompt_entry.get("category"),
        "context_bucket": prompt_entry.get("context_bucket"),
        "actual_context_tokens": prompt_entry.get("actual_context_tokens"),
        "compressor_name": compressor_name,
        "max_new_tokens": max_new_tokens,
        "draft_len": draft_len,
        "n_requests": n_requests,
        "load_shape": f"serial_{n_requests}",
        "claim_boundary": CLAIM_BOUNDARY,
        "full": full_load,
        "lossy": lossy_load,
        "exactkv": ekv_load,
        "exact_match_exactkv_vs_full": exact_match,
        "lossy_diverged": not token_exact_match(lossy_ids, full_ids),
        "exactkv_failure": not exact_match,
    }


def run_serving_microbench_panel(
    *,
    device: str = "cpu",
    dtype: str = "float32",
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
    context_buckets: Sequence[int] | None = None,
    max_new_tokens_list: Sequence[int] | None = None,
    n_requests_list: Sequence[int] | None = None,
    draft_len: int = DEFAULT_DRAFT_LEN,
    smoke: bool = False,
    deterministic_mode: bool = False,
    resume_cells: Sequence[Mapping[str, Any]] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if smoke:
        models = SMOKE_MODELS
        compressors = ("int8",)
        context_buckets = (128,)
        max_new_tokens_list = (8,)
        n_requests_list = (1,)
    else:
        models = tuple(models or DEFAULT_MODELS)
        compressors = tuple(compressors or DEFAULT_COMPRESSORS)
        context_buckets = tuple(context_buckets or DEFAULT_CONTEXT_BUCKETS)
        max_new_tokens_list = tuple(max_new_tokens_list or DEFAULT_MAX_NEW_TOKENS)
        n_requests_list = tuple(n_requests_list or DEFAULT_N_REQUESTS)

    expected = (
        len(models)
        * len(compressors)
        * len(context_buckets)
        * len(max_new_tokens_list)
        * len(n_requests_list)
    )
    cells: list[dict[str, Any]] = [dict(c) for c in (resume_cells or [])]
    done_keys = {
        (
            c.get("model_name"),
            c.get("compressor_name"),
            c.get("context_bucket"),
            c.get("max_new_tokens"),
            c.get("n_requests"),
        )
        for c in cells
    }

    hardware = collect_runpod_meta() if cuda_available() else {
        "cuda_available": False,
        "device": device,
    }

    for model_name in models:
        if progress_callback:
            progress_callback(f"loading {model_name}")
        if deterministic_mode:
            # Tiny synthetic cell without HF weights.
            for comp in compressors:
                for ctx in context_buckets:
                    for mnt in max_new_tokens_list:
                        for nreq in n_requests_list:
                            key = (model_name, comp, ctx, mnt, nreq)
                            if key in done_keys:
                                continue
                            cells.append(
                                {
                                    "model_name": model_name,
                                    "compressor_name": comp,
                                    "context_bucket": ctx,
                                    "actual_context_tokens": ctx,
                                    "max_new_tokens": mnt,
                                    "n_requests": nreq,
                                    "load_shape": f"serial_{nreq}",
                                    "draft_len": draft_len,
                                    "prompt_id": f"det_ctx{ctx}",
                                    "claim_boundary": CLAIM_BOUNDARY,
                                    "full": {
                                        "n_requests": nreq,
                                        "load_wall_s": 1.0,
                                        "completed_requests_per_sec": float(nreq),
                                        "mean_ttft_like_ms": 10.0,
                                        "mean_e2e_ms": 100.0,
                                        "gpu_peak_allocated_bytes": 1_000_000,
                                        "peak_delta_vs_full_bytes": 0,
                                    },
                                    "lossy": {
                                        "n_requests": nreq,
                                        "load_wall_s": 1.1,
                                        "completed_requests_per_sec": round(nreq / 1.1, 4),
                                        "mean_ttft_like_ms": 11.0,
                                        "mean_e2e_ms": 110.0,
                                        "gpu_peak_allocated_bytes": 1_100_000,
                                        "peak_delta_vs_full_bytes": 100_000,
                                    },
                                    "exactkv": {
                                        "n_requests": nreq,
                                        "load_wall_s": 2.0,
                                        "completed_requests_per_sec": round(nreq / 2.0, 4),
                                        "mean_ttft_like_ms": 20.0,
                                        "mean_e2e_ms": 200.0,
                                        "gpu_peak_allocated_bytes": 1_200_000,
                                        "peak_delta_vs_full_bytes": 200_000,
                                    },
                                    "exact_match_exactkv_vs_full": True,
                                    "lossy_diverged": True,
                                    "exactkv_failure": False,
                                }
                            )
                            done_keys.add(key)
            continue

        runtime = ModelRuntime(model_name, device=device, dtype=dtype)
        try:
            for comp in compressors:
                for ctx in context_buckets:
                    prompt_entry = _fit_prompt(runtime, _BASE_PROMPT, int(ctx))
                    for mnt in max_new_tokens_list:
                        for nreq in n_requests_list:
                            key = (model_name, comp, ctx, mnt, nreq)
                            if key in done_keys:
                                continue
                            if progress_callback:
                                progress_callback(
                                    f"cell {len(cells)+1}/{expected} "
                                    f"model={model_name} comp={comp} "
                                    f"ctx={ctx} mnt={mnt} nreq={nreq}"
                                )
                            cell = run_serving_cell(
                                runtime,
                                prompt_entry,
                                compressor_name=str(comp),
                                draft_len=draft_len,
                                max_new_tokens=int(mnt),
                                n_requests=int(nreq),
                            )
                            cell["model_name"] = model_name
                            cells.append(cell)
                            done_keys.add(key)
                            if checkpoint_callback:
                                checkpoint_callback(
                                    {
                                        "schema": "exactkv.serving_microbench.v1",
                                        "panel_id": SERVING_MICROBENCH_ID,
                                        "claim_boundary": CLAIM_BOUNDARY,
                                        "generated_at": datetime.now(timezone.utc).isoformat(),
                                        "design": {
                                            "models": list(models),
                                            "compressors": list(compressors),
                                            "context_buckets": list(context_buckets),
                                            "max_new_tokens": list(max_new_tokens_list),
                                            "n_requests": list(n_requests_list),
                                            "draft_len": draft_len,
                                            "load_mode": "serial_requests",
                                            "expected_cells": expected,
                                            "smoke": smoke,
                                            "deterministic_mode": deterministic_mode,
                                            "partial": True,
                                        },
                                        "hardware": hardware,
                                        "cells": list(cells),
                                        "exactkv_failures": sum(
                                            1 for c in cells if c.get("exactkv_failure")
                                        ),
                                        "n_cells": len(cells),
                                    }
                                )
        finally:
            del runtime
            _best_effort_empty_cache()

    report = {
        "schema": "exactkv.serving_microbench.v1",
        "panel_id": SERVING_MICROBENCH_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "models": list(models),
            "compressors": list(compressors),
            "context_buckets": list(context_buckets),
            "max_new_tokens": list(max_new_tokens_list),
            "n_requests": list(n_requests_list),
            "draft_len": draft_len,
            "load_mode": "serial_requests",
            "expected_cells": expected,
            "smoke": smoke,
            "deterministic_mode": deterministic_mode,
        },
        "hardware": hardware,
        "cells": cells,
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        "n_cells": len(cells),
    }
    assert_no_forbidden_fields(report)
    return report


def write_serving_microbench_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if markdown_path is None:
        return
    lines = [
        f"# {report.get('panel_id')}",
        "",
        f"**Cells:** {report.get('n_cells')} · **exactkv_failures:** {report.get('exactkv_failures')}",
        "",
        f"**Claim boundary:** {report.get('claim_boundary')}",
        "",
        f"Artifact: `{json_path}`",
        "",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
