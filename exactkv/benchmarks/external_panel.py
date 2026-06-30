"""External benchmark panels (LongBench, RULER, BFCL, HumanEval, MBPP).

Runs the same ExactKV cell methodology as evidence_plus_panel on prompts from
established dataset families. Reports drift metrics by task category and context
bucket. Does **not** compute official LongBench/RULER/BFCL scores.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.attention.hf_single_layer_probe import generate_long_prompt_text
from exactkv.benchmarks.evidence_plus_panel import (
    BUILTIN_COMPRESSORS,
    DEFAULT_DRAFT_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    PROMPT_FILLER,
    SMOKE_MODELS,
    _timed_run_cell,
    resolve_evidence_plus_compressor,
    write_evidence_plus_outputs,
)
from exactkv.benchmarks.external_dataset_loaders import load_external_prompts
from exactkv.benchmarks.phase_a_scale_benchmark import (
    _aggregate_compressor_metrics,
    build_deterministic_phase_a_cell,
    detect_model_availability,
)
from exactkv.benchmarks.reports import build_run_manifest
from exactkv.runtime.model_runtime import ModelRuntime

EXTERNAL_PANEL_ID = "external_benchmark_panel"
DEFAULT_OUTPUT = Path("reports/external_panels/raw.json")
DEFAULT_MARKDOWN = Path("reports/external_panels/summary.md")

FAMILY_DEFAULT_BUCKETS: dict[str, tuple[int, ...]] = {
    "longbench": (2048, 4096, 8192),
    "ruler": (4096, 8192, 16384, 32768),
    "bfcl": (512, 1024, 2048),
    "humaneval": (512, 1024),
    "mbpp": (512, 1024, 2048),
}

FAMILY_SMOKE_BUCKETS: dict[str, tuple[int, ...]] = {
    "longbench": (512,),
    "ruler": (512,),
    "bfcl": (512,),
    "humaneval": (512,),
    "mbpp": (512,),
}


_PROMPT_METADATA_KEYS = (
    "task_id",
    "test_list",
    "source",
    "source_dataset",
    "entry_point",
)


def _attach_prompt_metadata(
    cell: dict[str, Any],
    base: Mapping[str, Any],
    *,
    prompt_source: str,
) -> None:
    for key in _PROMPT_METADATA_KEYS:
        if key in base:
            cell[key] = base[key]
    cell["prompt_source"] = prompt_source


def build_external_context_prompt(
    runtime: ModelRuntime,
    base: Mapping[str, Any],
    target_context_tokens: int,
) -> dict[str, Any]:
    """Fit prompt to context bucket: truncate long contexts, pad short ones."""
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
            filler=PROMPT_FILLER + text,
        )
    return {
        "prompt_id": f"{base['prompt_id']}_ctx{target_context_tokens}",
        "category": base.get("category", "unknown"),
        "prompt": fitted,
        "context_bucket": target_context_tokens,
        "prefill_tokens": actual,
        "base_prompt_id": base["prompt_id"],
        "dataset_family": base.get("dataset_family"),
        "task_type": base.get("task_type") or base.get("category"),
        "source_dataset": base.get("source_dataset"),
    }


def _aggregate_by_category(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if cell.get("status") != "ok":
            continue
        key = str(cell.get("task_type") or cell.get("category") or "unknown")
        buckets[key].append(dict(cell.get("metrics") or {}))

    summary: dict[str, Any] = {}
    for key, metrics in sorted(buckets.items()):
        n = max(len(metrics), 1)
        div = sum(1 for m in metrics if m.get("token_level_divergence"))
        summary[key] = {
            "num_cells": len(metrics),
            "divergence_rate": div / n,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics) / n,
        }
    return summary


def run_external_panel(
    family: str,
    *,
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
    context_buckets: Sequence[int] | None = None,
    max_new_tokens_values: Sequence[int] | None = None,
    max_prompts: int = 12,
    prompt_source: str = "pilot",
    longbench_subsets: Sequence[str] | None = None,
    draft_len: int = DEFAULT_DRAFT_LEN,
    device: str = "cuda",
    dtype: str = "float16",
    deterministic_mode: bool = False,
    local_files_only: bool = False,
    smoke: bool = False,
    store_top_k_logits: bool = False,
) -> dict[str, Any]:
    family = family.lower()
    if smoke:
        model_list = list(SMOKE_MODELS)
        bucket_list = list(FAMILY_SMOKE_BUCKETS.get(family, (512,)))
        mnt_list = [16]
        compressor_list = list(BUILTIN_COMPRESSORS[:2])
        prompt_count = 2
    else:
        from exactkv.benchmarks.evidence_plus_panel import DEFAULT_MODELS

        model_list = list(models or DEFAULT_MODELS)
        bucket_list = list(context_buckets or FAMILY_DEFAULT_BUCKETS.get(family, (2048, 4096)))
        mnt_list = list(max_new_tokens_values or DEFAULT_MAX_NEW_TOKENS[:2])
        compressor_list = list(compressors or BUILTIN_COMPRESSORS)
        prompt_count = max_prompts

    if deterministic_mode:
        models_evaluated = model_list
        models_blocked: dict[str, str] = {}
        base_prompts = load_external_prompts(family, source="pilot", max_prompts=prompt_count)
    else:
        models_evaluated, models_blocked = detect_model_availability(
            model_list,
            local_files_only=local_files_only,
        )
        base_prompts = load_external_prompts(
            family,
            source=prompt_source,
            max_prompts=prompt_count,
            subsets=longbench_subsets,
        )

    cells: list[dict[str, Any]] = []
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
                        "task_type": base.get("category"),
                    }
                else:
                    assert runtime is not None
                    prompt_entry = build_external_context_prompt(runtime, base, bucket)

                for compressor_name in compressor_list:
                    res = resolve_evidence_plus_compressor(compressor_name, runtime=runtime)
                    if res.backend_tier == "UNAVAILABLE":
                        cells.append(
                            {
                                "prompt_id": prompt_entry["prompt_id"],
                                "model_name": model_name,
                                "compressor_name": compressor_name,
                                "dataset_family": family,
                                "context_bucket": bucket,
                                "status": "skipped",
                                "skip_reason": "adapter unavailable",
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
                        cell["dataset_family"] = family
                        cell["task_type"] = prompt_entry.get("task_type")
                        cell["task_category"] = base.get("category")
                        _attach_prompt_metadata(cell, base, prompt_source=prompt_source)
                        # Optionally capture top-k logits at divergence for forensic analysis
                        if store_top_k_logits and not deterministic_mode and runtime is not None:
                            if cell.get("status") == "ok" and (cell.get("metrics") or {}).get("token_level_divergence"):
                                from exactkv.benchmarks.runner import capture_divergence_topk
                                from exactkv.compressors import get_compressor
                                comp_obj = get_compressor(compressor_name)
                                full_ids = (cell.get("full") or {}).get("output_ids", [])
                                lossy_ids = (cell.get("lossy") or {}).get("output_ids", [])
                                if full_ids and lossy_ids:
                                    topk = capture_divergence_topk(
                                        runtime,
                                        prompt_entry.get("prompt", ""),
                                        comp_obj,
                                        full_ids,
                                        lossy_ids,
                                    )
                                    if topk is not None:
                                        cell["divergence_topk_logits"] = topk
                        cells.append(cell)
                        cell_idx += 1
                        if not deterministic_mode:
                            print(
                                f"[{family}] cell {cell_idx}/{expected_cells} "
                                f"model={model_name} task={base.get('category')} "
                                f"ctx={bucket} comp={compressor_name} mnt={mnt}",
                                flush=True,
                            )

    ok_cells = [c for c in cells if c.get("status") == "ok"]
    by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for c in ok_cells:
        b = c.get("context_bucket")
        if b is not None:
            by_bucket[str(int(b))].append(c)

    bucket_summary: dict[str, Any] = {}
    for b, bucket_cells in sorted(by_bucket.items(), key=lambda x: int(x[0])):
        metrics = [c.get("metrics") or {} for c in bucket_cells]
        n = max(len(metrics), 1)
        div = sum(1 for m in metrics if m.get("token_level_divergence"))
        bucket_summary[b] = {
            "num_cells": len(bucket_cells),
            "divergence_rate": div / n,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics) / n,
        }

    return {
        "phase_id": EXTERNAL_PANEL_ID,
        "dataset_family": family,
        "status": "benchmark_complete",
        "deterministic_mode": deterministic_mode,
        "smoke": smoke,
        "prompt_source": prompt_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": build_run_manifest(
            model_name=",".join(models_evaluated) if models_evaluated else "none",
            compressor_names=list(compressor_list),
            draft_lengths=[draft_len],
            prompt_suite=f"{family}_{prompt_source}",
            max_new_tokens=max(mnt_list) if mnt_list else 0,
        ),
        "models_evaluated": models_evaluated,
        "models_blocked": models_blocked,
        "context_buckets": bucket_list,
        "max_new_tokens_values": mnt_list,
        "compressors": list(compressor_list),
        "total_cells": len(cells),
        "cells_run": len(ok_cells),
        "cells_skipped": len(cells) - len(ok_cells),
        "exactkv_failures": sum(1 for c in ok_cells if c.get("exactkv_failure")),
        "compressor_summary": _aggregate_compressor_metrics(ok_cells),
        "bucket_summary": bucket_summary,
        "category_summary": _aggregate_by_category(ok_cells),
        "cells": cells,
        "limitations_note": (
            f"External {family} panel measures ExactKV drift metrics only. "
            "This is not an official LongBench/RULER/BFCL/HumanEval/MBPP score reproduction."
        ),
        "reproducible_cli_command": (
            f"python3 scripts/run_external_panel.py --family {family} --device cuda --dtype float16"
        ),
    }


def write_external_panel_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path, Path]:
    family = str(report.get("dataset_family") or "external")
    json_path = json_path or Path(f"reports/external_panels/{family}_raw.json")
    # Always derive markdown sibling from json_path.parent so absolute paths work
    # regardless of the process cwd (e.g. running from /workspace on RunPod).
    if markdown_path is None:
        markdown_path = json_path.parent / f"{family}_summary.md"
    family_title = family.replace("_", " ").title()
    return write_evidence_plus_outputs(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        summary_title=f"External Panel: {family_title}",
    )
