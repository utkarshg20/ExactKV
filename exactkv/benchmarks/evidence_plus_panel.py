"""Evidence-plus benchmark panel (8.5+ research upgrade).

Extends the Phase A scale methodology with:
  - context-length buckets (padded prefill tokens),
  - longer max_new_tokens,
  - optional real external compressors (KIVI offline, KVQuant sim, SnapKV experimental),
  - per-path wall-clock timing (full / lossy / exactkv).

Outputs: ``reports/evidence_plus/raw.json`` (+ optional markdown summary).

Claim boundaries: timing is diagnostic overhead only — not end-to-end serving speedup.
External adapters are labeled RESTRICTED_ADAPTER / experimental in cell metadata.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.attention.hf_single_layer_probe import generate_long_prompt_text
from exactkv.benchmarks.phase_a_scale_benchmark import (
    PHASE_A_BUILTIN_COMPRESSORS,
    CompressorResolution,
    _aggregate_compressor_metrics,
    _enrich_cell_metrics,
    _instantiate_compressor,
    build_deterministic_phase_a_cell,
    detect_model_availability,
    resolve_compressor,
    run_one_with_compressor,
)
from exactkv.benchmarks.reports import build_run_manifest
from exactkv.runtime.model_runtime import ModelRuntime

EVIDENCE_PLUS_ID = "evidence_plus_panel"
DEFAULT_OUTPUT = Path("reports/evidence_plus/raw.json")
DEFAULT_MARKDOWN = Path("reports/evidence_plus/summary.md")

# Context buckets (prefill target tokens) and generation lengths for 8.5+ panel.
DEFAULT_CONTEXT_BUCKETS: tuple[int, ...] = (512, 1024, 2048)
DEFAULT_MAX_NEW_TOKENS: tuple[int, ...] = (16, 32, 64)
DEFAULT_DRAFT_LEN = 4

BUILTIN_COMPRESSORS: tuple[str, ...] = ("noop", "int8", "int4_sim")
EXTERNAL_COMPRESSORS: tuple[str, ...] = (
    "kivi_offline",
    "kivi_offline_r32",
    "kvquant_sim",
    "snapkv_experimental",
)

DEFAULT_MODELS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
SMOKE_MODELS: tuple[str, ...] = ("Qwen/Qwen2.5-0.5B",)

PROMPT_FILLER = (
    "ExactKV evidence-plus panel: deterministic filler for long-context KV drift. "
    "Lossy compressed KV may diverge token-by-token from full-KV greedy decoding. "
)


def _kvpress_available() -> bool:
    return importlib.util.find_spec("kvpress") is not None


def _kvquant_available() -> bool:
    return importlib.util.find_spec("kvquant") is not None


def _kivi_utils_available() -> bool:
    from exactkv.compressors.kivi_adapter import _ensure_kivi_on_path  # noqa: PLC0415

    _ensure_kivi_on_path()
    try:
        import models.utils_quant  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def resolve_evidence_plus_compressor(
    name: str,
    *,
    runtime: ModelRuntime | None = None,
) -> CompressorResolution:
    """Resolve built-in + evidence-plus external compressors."""
    if name in BUILTIN_COMPRESSORS:
        return CompressorResolution(
            compressor_name=name,
            backend_tier="BUILTIN",
            adapter_available=True,
            probe_only=False,
        )
    if name in ("kivi_offline", "kivi_offline_r32"):
        ok = _kivi_utils_available()
        return CompressorResolution(
            compressor_name=name,
            backend_tier="RESTRICTED_ADAPTER" if ok else "UNAVAILABLE",
            adapter_available=ok,
            probe_only=False,
        )
    if name == "kvquant_sim":
        ok = _kvquant_available()
        return CompressorResolution(
            compressor_name=name,
            backend_tier="RESTRICTED_ADAPTER" if ok else "UNAVAILABLE",
            adapter_available=ok,
            probe_only=False,
        )
    if name == "snapkv_experimental":
        ok = _kvpress_available()
        return CompressorResolution(
            compressor_name=name,
            backend_tier="RESTRICTED_ADAPTER" if ok else "UNAVAILABLE",
            adapter_available=ok,
            probe_only=False,
        )
    # Fall back to Phase A resolution (noop/int8/int4_sim/spectralquant/shard/...)
    return resolve_compressor(name, runtime=runtime)


def _instantiate_evidence_plus_compressor(
    resolution: CompressorResolution,
    runtime: ModelRuntime,
) -> Any:
    if resolution.compressor_name == "kivi_offline":
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter  # noqa: PLC0415

        return create_kivi_offline_adapter(runtime)
    if resolution.compressor_name == "kivi_offline_r32":
        from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter  # noqa: PLC0415

        return create_kivi_offline_adapter(runtime, residual_length=32)
    if resolution.compressor_name == "kvquant_sim":
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter  # noqa: PLC0415

        return create_kvquant_sim_adapter(runtime)
    if resolution.compressor_name == "snapkv_experimental":
        from exactkv.compressors.kvpress_snapkv import (  # noqa: PLC0415
            create_snapkv_experimental_adapter,
        )

        # A5000 (24GB): deepcopy(model) for isolation OOMs on 7B fp16; panel uses shared model.
        isolate = os.environ.get("EXACTKV_SNAPKV_ISOLATE", "0") == "1"
        return create_snapkv_experimental_adapter(
            runtime,
            compression_ratio=0.5,
            isolate_compression_model=isolate,
        )
    return _instantiate_compressor(resolution, runtime)


def _load_long_context_rows(path: Path, *, limit: int) -> list[dict[str, str]]:
    """Load long_context.jsonl (uses ``id`` / ``primary_category`` schema)."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append(
                {
                    "prompt_id": entry.get("prompt_id") or entry.get("id", "lc_unknown"),
                    "category": entry.get("category")
                    or entry.get("primary_category", "long_context"),
                    "prompt": entry["prompt"],
                },
            )
            if len(rows) >= limit:
                break
    return rows


def load_base_prompts(*, max_prompts: int = 8) -> list[dict[str, str]]:
    """Mix long_context suite + stress-panel templates."""
    lc_path = Path("benchmarks/prompts/long_context.jsonl")
    prompts: list[dict[str, str]] = []
    if lc_path.is_file():
        prompts.extend(_load_long_context_rows(lc_path, limit=max(max_prompts // 2, 1)))
    from exactkv.safety.l4_runtime_coupling_stress_panel import STRESS_PANEL_PROMPTS

    for pid, text in STRESS_PANEL_PROMPTS:
        prompts.append({"prompt_id": pid, "category": pid.split("_", 1)[-1], "prompt": text})
        if len(prompts) >= max_prompts:
            break
    return prompts[:max_prompts]


def build_context_prompt(
    runtime: ModelRuntime,
    base: Mapping[str, str],
    target_context_tokens: int,
) -> dict[str, str]:
    """Pad base prompt to target prefill length; record actual token count."""
    text, actual = generate_long_prompt_text(
        runtime.tokenizer,
        target_context_tokens,
        filler=PROMPT_FILLER + base["prompt"],
    )
    return {
        "prompt_id": f"{base['prompt_id']}_ctx{target_context_tokens}",
        "category": base.get("category", "unknown"),
        "prompt": text,
        "context_bucket": target_context_tokens,
        "prefill_tokens": actual,
        "base_prompt_id": base["prompt_id"],
    }


def _timed_run_cell(
    runtime: ModelRuntime,
    prompt_entry: Mapping[str, Any],
    *,
    compressor_name: str,
    draft_len: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    resolution = resolve_evidence_plus_compressor(compressor_name, runtime=runtime)
    if not resolution.adapter_available and resolution.backend_tier == "UNAVAILABLE":
        return {
            "prompt_id": prompt_entry["prompt_id"],
            "model_name": runtime.model_name,
            "compressor_name": compressor_name,
            "status": "skipped",
            "skip_reason": f"{compressor_name} adapter unavailable in environment",
            "backend_tier": resolution.backend_tier,
        }

    t0 = time.perf_counter()
    if resolution.compressor_name in BUILTIN_COMPRESSORS:
        from exactkv.benchmarks.runner import RunConfig, run_one

        raw = run_one(
            runtime,
            dict(prompt_entry),
            RunConfig(
                compressor_name=compressor_name,
                draft_len=draft_len,
                max_new_tokens=max_new_tokens,
            ),
        )
        raw["backend_tier"] = "BUILTIN"
        raw["probe_only"] = False
        cell = _enrich_cell_metrics(raw)
    else:
        comp = _instantiate_evidence_plus_compressor(resolution, runtime)
        cell = run_one_with_compressor(
            runtime,
            prompt_entry,
            compressor=comp,
            compressor_name=compressor_name,
            draft_len=draft_len,
            max_new_tokens=max_new_tokens,
            backend_tier=resolution.backend_tier,
            adapter_available=resolution.adapter_available,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    cell["timing_ms"] = {"total_cell": round(elapsed_ms, 3)}
    cell["context_bucket"] = prompt_entry.get("context_bucket")
    cell["prefill_tokens"] = prompt_entry.get("prefill_tokens")
    cell["base_prompt_id"] = prompt_entry.get("base_prompt_id")
    cell["status"] = "ok"
    return cell


def run_evidence_plus_panel(
    *,
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
    context_buckets: Sequence[int] | None = None,
    max_new_tokens_values: Sequence[int] | None = None,
    max_prompts: int = 8,
    draft_len: int = DEFAULT_DRAFT_LEN,
    device: str = "cuda",
    dtype: str = "float16",
    deterministic_mode: bool = False,
    local_files_only: bool = False,
    try_external: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the evidence-plus panel and return a structured report dict."""
    if smoke:
        model_list = list(SMOKE_MODELS)
        bucket_list = [512]
        mnt_list = [16]
        compressor_list = list(BUILTIN_COMPRESSORS[:2])  # noop, int8
        prompt_count = 2
    else:
        model_list = list(models or DEFAULT_MODELS)
        bucket_list = list(context_buckets or DEFAULT_CONTEXT_BUCKETS)
        mnt_list = list(max_new_tokens_values or DEFAULT_MAX_NEW_TOKENS)
        compressor_list = list(compressors or BUILTIN_COMPRESSORS)
        if try_external:
            for ext in EXTERNAL_COMPRESSORS:
                if ext not in compressor_list:
                    compressor_list.append(ext)
        prompt_count = max_prompts

    if deterministic_mode:
        models_evaluated = model_list
        models_blocked: dict[str, str] = {}
    else:
        models_evaluated, models_blocked = detect_model_availability(
            model_list,
            local_files_only=local_files_only,
        )

    base_prompts = load_base_prompts(max_prompts=prompt_count)
    cells: list[dict[str, Any]] = []
    compressor_resolutions = {
        c: asdict(resolve_evidence_plus_compressor(c)) for c in compressor_list
    }
    cell_idx = 0
    expected_cells = (
        len(models_evaluated or [])
        * len(base_prompts)
        * len(bucket_list)
        * len(compressor_list)
        * len(mnt_list)
    )

    for model_name in models_evaluated:
        runtime: ModelRuntime | None = None
        if not deterministic_mode:
            runtime = ModelRuntime(model_name, device=device, dtype=dtype)

        for base in base_prompts:
            for bucket in bucket_list:
                if deterministic_mode:
                    prompt_entry = {
                        **base,
                        "prompt_id": f"{base['prompt_id']}_ctx{bucket}",
                        "context_bucket": bucket,
                        "prefill_tokens": bucket,
                        "base_prompt_id": base["prompt_id"],
                    }
                else:
                    assert runtime is not None
                    prompt_entry = build_context_prompt(runtime, base, bucket)

                for compressor_name in compressor_list:
                    res = resolve_evidence_plus_compressor(
                        compressor_name, runtime=runtime,
                    )
                    if res.backend_tier == "UNAVAILABLE":
                        cells.append(
                            {
                                "prompt_id": prompt_entry["prompt_id"],
                                "model_name": model_name,
                                "compressor_name": compressor_name,
                                "context_bucket": bucket,
                                "status": "skipped",
                                "skip_reason": "adapter unavailable",
                                "backend_tier": res.backend_tier,
                            },
                        )
                        continue

                    for mnt in mnt_list:
                        if deterministic_mode:
                            cell = build_deterministic_phase_a_cell(
                                model_name=model_name,
                                prompt_entry=prompt_entry,
                                compressor_name=compressor_name,
                                max_new_tokens=mnt,
                                draft_len=draft_len,
                                resolution=res,
                            )
                            cell["context_bucket"] = bucket
                            cell["prefill_tokens"] = prompt_entry.get("prefill_tokens")
                            cell["timing_ms"] = {"total_cell": 1.0}
                            cell["status"] = "ok"
                        else:
                            assert runtime is not None
                            cell = _timed_run_cell(
                                runtime,
                                prompt_entry,
                                compressor_name=compressor_name,
                                draft_len=draft_len,
                                max_new_tokens=mnt,
                            )
                        cells.append(cell)
                        cell_idx += 1
                        if not deterministic_mode:
                            print(
                                f"[evidence_plus] cell {cell_idx}/{expected_cells} "
                                f"model={model_name} ctx={bucket} "
                                f"comp={compressor_name} mnt={mnt} "
                                f"status={cell.get('status')} "
                                f"ms={(cell.get('timing_ms') or {}).get('total_cell')}",
                                flush=True,
                            )

    ok_cells = [c for c in cells if c.get("status") == "ok"]
    compressor_summary = _aggregate_compressor_metrics(ok_cells)

    # Per context-bucket divergence aggregates
    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for c in ok_cells:
        b = c.get("context_bucket")
        if b is not None:
            by_bucket.setdefault(int(b), []).append(c)

    bucket_summary: dict[str, Any] = {}
    for b, bucket_cells in sorted(by_bucket.items()):
        metrics = [c.get("metrics") or {} for c in bucket_cells]
        n = max(len(metrics), 1)
        div = sum(1 for m in metrics if m.get("token_level_divergence"))
        bucket_summary[str(b)] = {
            "num_cells": len(bucket_cells),
            "divergence_rate": div / n,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics) / n,
        }

    return {
        "phase_id": EVIDENCE_PLUS_ID,
        "status": "benchmark_complete",
        "deterministic_mode": deterministic_mode,
        "smoke": smoke,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": build_run_manifest(
            model_name=",".join(models_evaluated) if models_evaluated else "none",
            compressor_names=list(compressor_list),
            draft_lengths=[draft_len],
            prompt_suite="evidence_plus_long_context",
            max_new_tokens=max(mnt_list) if mnt_list else 0,
        ),
        "models_requested": list(model_list),
        "models_evaluated": models_evaluated,
        "models_blocked": models_blocked,
        "context_buckets": bucket_list,
        "max_new_tokens_values": mnt_list,
        "compressors": list(compressor_list),
        "compressor_resolutions": compressor_resolutions,
        "total_cells": len(cells),
        "cells_run": len(ok_cells),
        "cells_skipped": len(cells) - len(ok_cells),
        "exactkv_failures": sum(1 for c in ok_cells if c.get("exactkv_failure")),
        "compressor_summary": compressor_summary,
        "bucket_summary": bucket_summary,
        "cells": cells,
        "limitations_note": (
            "Evidence-plus panel: long-context prefill buckets + optional external "
            "adapters. timing_ms is diagnostic wall-clock per cell only — not "
            "end-to-end serving speedup. KIVI path uses offline simulate quant unless "
            "full KIVI repo is on PYTHONPATH."
        ),
        "reproducible_cli_command": (
            "python3 scripts/run_evidence_plus_panel.py --device cuda --dtype float16"
        ),
    }


def write_evidence_plus_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path = DEFAULT_OUTPUT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    summary_title: str = "Evidence-Plus Panel Summary",
) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {summary_title}",
        "",
        f"**Status:** {report.get('status')}",
        f"**Cells:** {report.get('total_cells')} total, {report.get('cells_run')} ok",
        f"**ExactKV failures:** {report.get('exactkv_failures')}",
        "",
        "## Per context bucket",
        "",
        "| Bucket | Cells | Divergence rate | Mean acceptance |",
        "|--------|------:|----------------:|----------------:|",
    ]
    for bucket, stats in sorted((report.get("bucket_summary") or {}).items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {bucket} | {stats.get('num_cells')} | "
            f"{stats.get('divergence_rate', 0):.3f} | {stats.get('mean_acceptance_rate', 0):.3f} |",
        )
    lines.extend(["", "## Compressor summary", ""])
    for comp, stats in sorted((report.get("compressor_summary") or {}).items()):
        lines.append(
            f"- `{comp}`: acceptance={stats.get('mean_acceptance_rate', 0):.3f}, "
            f"divergence_rate={stats.get('divergence_rate', 0):.3f}, "
            f"cells={stats.get('num_cells')}",
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
