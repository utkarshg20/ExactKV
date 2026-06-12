#!/usr/bin/env python3
"""Experiment 031: active GPU memory isolation (V13 Phase 4).

Diagnostic GPU memory isolation only — not production serving evidence.
Prerequisite: Experiment 030 exactness gate passed (phase4_memory_allowed=True).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.exactness import token_exact_match
from exactkv.metrics.gpu_memory_isolation import (
    assert_memory_artifact_safe,
    collect_gpu_memory_environment,
    cuda_available,
    measure_cuda_region,
    measure_model_loaded_baseline,
    require_cuda,
    summarize_memory_trials,
)
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 32
DRAFT_LENS = (4, 8)
EXPERIMENT_CLASS = "v13_gpu_memory_isolation"
NUM_TRIALS_DEFAULT = 2

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

STRATIFIED_SUITES: list[tuple[str, int]] = [
    ("core_v2", 3),
    ("long_context", 3),
    ("retrieval_copy", 3),
    ("tool_json", 3),
]

ARMS = (
    "model_loaded_baseline",
    "full_greedy",
    "lossy_only",
    "exactkv_sequential",
    "exactkv_span",
)


def load_exp031_prompt_panel(*, smoke: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    suites = STRATIFIED_SUITES if not smoke else STRATIFIED_SUITES[:2]
    count = 1 if smoke else None
    for suite_name, n in suites:
        take = count if count is not None else n
        suite = load_v10_suite(suite_name)
        if len(suite) < take:
            raise ValueError(
                f"Suite {suite_name!r} has {len(suite)} prompts; need {take}"
            )
        out.extend(suite[:take])
    return out


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _mib(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) / (1024 * 1024):.2f}"


def _run_trials(
    device: str,
    fn: Callable[[], Any],
    *,
    model_loaded_allocated: int,
    model_loaded_reserved: int,
    num_trials: int,
) -> tuple[Any, list[dict[str, Any]], dict[str, float | int]]:
    trials: list[dict[str, Any]] = []
    last_result: Any = None
    readings = []
    for _ in range(num_trials):
        result, reading = measure_cuda_region(
            device,
            fn,
            model_loaded_allocated=model_loaded_allocated,
            model_loaded_reserved=model_loaded_reserved,
        )
        last_result = result
        readings.append(reading)
        trials.append(reading.to_dict())
    return last_result, trials, summarize_memory_trials(readings)


def _generated_token_count(result: Any) -> int:
    if hasattr(result, "generated_ids"):
        return int(result.generated_ids.shape[1])
    return int(result.output_ids.shape[1])


def _v5_from_prompt(
    runtime: ModelRuntime, prompt: str, compressor_name: str | None
) -> dict[str, Any] | None:
    if not compressor_name:
        return None
    comp = get_compressor(compressor_name)
    return estimate_kv_memory(runtime, prompt, comp).to_dict()


def aggregate(cells: list[dict[str, Any]], baseline: dict[str, int]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_draft: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_map: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    parity_failures = 0

    for cell in cells:
        by_arm[cell["arm"]].append(cell)
        if cell["arm"] in ("exactkv_sequential", "exactkv_span"):
            key = (cell["prompt_id"], cell["compressor_name"], cell["draft_len"])
            pair_map.setdefault(key, {})[cell["arm"]] = cell
            by_draft[cell["draft_len"]].append(cell)
        if cell.get("compressor_name") and cell["arm"] in (
            "lossy_only",
            "exactkv_sequential",
            "exactkv_span",
        ):
            by_comp[cell["compressor_name"]].append(cell)

    for pair in pair_map.values():
        seq = pair.get("exactkv_sequential")
        span = pair.get("exactkv_span")
        if seq and span and seq.get("output_ids") != span.get("output_ids"):
            parity_failures += 1

    def _arm_memory_stats(arm_cells: list[dict[str, Any]]) -> dict[str, Any]:
        peaks_a = [
            c["trial_stats"]["mean_peak_allocated"] for c in arm_cells if c.get("trial_stats")
        ]
        peaks_r = [
            c["trial_stats"]["mean_peak_reserved"] for c in arm_cells if c.get("trial_stats")
        ]
        max_peaks_a = [
            c["trial_stats"]["max_peak_allocated"] for c in arm_cells if c.get("trial_stats")
        ]
        max_peaks_r = [
            c["trial_stats"]["max_peak_reserved"] for c in arm_cells if c.get("trial_stats")
        ]
        deltas_a = [
            c["trial_stats"]["mean_delta_peak_allocated_vs_model_loaded"]
            for c in arm_cells
            if c.get("trial_stats")
        ]
        exactkv_failures = sum(1 for c in arm_cells if c.get("exactkv_failure"))
        return {
            "cells": len(arm_cells),
            "exactkv_failures": exactkv_failures,
            "mean_peak_allocated": mean(peaks_a) if peaks_a else None,
            "max_peak_allocated": max(max_peaks_a) if max_peaks_a else None,
            "mean_peak_reserved": mean(peaks_r) if peaks_r else None,
            "max_peak_reserved": max(max_peaks_r) if max_peaks_r else None,
            "mean_delta_peak_allocated_vs_model_loaded": mean(deltas_a) if deltas_a else None,
        }

    arm_stats = {arm: _arm_memory_stats(cs) for arm, cs in by_arm.items()}
    seq_stats = arm_stats.get("exactkv_sequential", {})
    span_stats = arm_stats.get("exactkv_span", {})
    fg_stats = arm_stats.get("full_greedy", {})

    exactness_gate_passed = (
        seq_stats.get("exactkv_failures", 0) == 0
        and span_stats.get("exactkv_failures", 0) == 0
        and parity_failures == 0
    )

    def _ratio(left: float | None, right: float | None) -> float | None:
        if left is None or right is None or right == 0:
            return None
        return left / right

    fg_peak = fg_stats.get("mean_peak_allocated")
    seq_peak = seq_stats.get("mean_peak_allocated")
    span_peak = span_stats.get("mean_peak_allocated")
    fg_delta = fg_stats.get("mean_delta_peak_allocated_vs_model_loaded")
    seq_delta = seq_stats.get("mean_delta_peak_allocated_vs_model_loaded")
    span_delta = span_stats.get("mean_delta_peak_allocated_vs_model_loaded")

    # Savings claim: only if ExactKV peak is robustly lower than full greedy.
    noise_mib = 2.0 * 1024 * 1024  # 2 MiB allocator slack threshold
    savings_candidate = (
        seq_peak is not None
        and fg_peak is not None
        and seq_peak < fg_peak - noise_mib
    )
    active_gpu_memory_savings_claim_allowed = False  # default forbidden

    if savings_candidate and exactness_gate_passed:
        # Still forbidden unless attributable — Exp 031 on 0.5B likely dominated by weights.
        active_gpu_memory_savings_claim_allowed = False

    memory_verdict = "indistinguishable"
    if seq_peak is not None and fg_peak is not None:
        if seq_peak > fg_peak + noise_mib:
            memory_verdict = "exactkv_uses_more_peak_allocated"
        elif seq_peak < fg_peak - noise_mib:
            memory_verdict = "exactkv_uses_less_peak_allocated"
        else:
            memory_verdict = "indistinguishable_within_allocator_noise"

    v5_cells = [c for c in cells if c.get("v5_accounting")]
    mean_v5_footprint = (
        mean(c["v5_accounting"]["total_kv_footprint_bytes"] for c in v5_cells)
        if v5_cells
        else None
    )

    by_draft_stats: dict[str, Any] = {}
    for dl, dl_cells in sorted(by_draft.items()):
        by_draft_stats[str(dl)] = {
            "exactkv_sequential": _arm_memory_stats(
                [c for c in dl_cells if c["arm"] == "exactkv_sequential"]
            ),
            "exactkv_span": _arm_memory_stats(
                [c for c in dl_cells if c["arm"] == "exactkv_span"]
            ),
        }

    by_comp_stats: dict[str, Any] = {}
    for comp, comp_cells in sorted(by_comp.items()):
        by_comp_stats[comp] = {
            "exactkv_sequential": _arm_memory_stats(
                [c for c in comp_cells if c["arm"] == "exactkv_sequential"]
            ),
            "exactkv_span": _arm_memory_stats(
                [c for c in comp_cells if c["arm"] == "exactkv_span"]
            ),
            "lossy_only": _arm_memory_stats(
                [c for c in comp_cells if c["arm"] == "lossy_only"]
            ),
        }

    return {
        "model_loaded_baseline": baseline,
        "exactness_gate_passed": exactness_gate_passed,
        "span_sequential_parity_failures": parity_failures,
        "by_arm": arm_stats,
        "comparisons": {
            "exactkv_sequential_vs_full_greedy": {
                "ratio_mean_peak_allocated": _ratio(seq_peak, fg_peak),
                "ratio_mean_delta_peak_allocated": _ratio(seq_delta, fg_delta),
            },
            "exactkv_span_vs_full_greedy": {
                "ratio_mean_peak_allocated": _ratio(span_peak, fg_peak),
                "ratio_mean_delta_peak_allocated": _ratio(span_delta, fg_delta),
            },
            "exactkv_span_vs_exactkv_sequential": {
                "ratio_mean_peak_allocated": _ratio(span_peak, seq_peak),
                "ratio_mean_delta_peak_allocated": _ratio(span_delta, seq_delta),
            },
        },
        "v5_vs_cuda": {
            "mean_v5_total_kv_footprint_bytes": mean_v5_footprint,
            "note": (
                "V5 total_kv_footprint_bytes is an accounting sum (MiB-scale); "
                "CUDA peak includes model weights and allocator reservation (GiB-scale)."
            ),
        },
        "memory_verdict": memory_verdict,
        "active_gpu_memory_savings_claim_allowed": active_gpu_memory_savings_claim_allowed,
        "by_draft_len": by_draft_stats,
        "by_compressor": by_comp_stats,
        "phase5_feasibility_allowed": exactness_gate_passed,
    }


def run_experiment(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    num_trials: int,
) -> dict[str, Any]:
    baseline_snap = measure_model_loaded_baseline(device)
    model_loaded_allocated = baseline_snap["allocated_bytes"]
    model_loaded_reserved = baseline_snap["reserved_bytes"]

    cells: list[dict[str, Any]] = []
    cells.append({
        "arm": "model_loaded_baseline",
        "prompt_id": None,
        "v10_suite": None,
        "compressor_name": None,
        "draft_len": None,
        "verification_method": None,
        "baseline_snapshot": baseline_snap,
        "trial_stats": {
            "mean_peak_allocated": baseline_snap["allocated_bytes"],
            "max_peak_allocated": baseline_snap["allocated_bytes"],
            "mean_peak_reserved": baseline_snap["reserved_bytes"],
            "max_peak_reserved": baseline_snap["reserved_bytes"],
            "mean_delta_peak_allocated_vs_model_loaded": 0,
            "max_delta_peak_allocated_vs_model_loaded": 0,
        },
        "exactkv_failure": False,
        "token_exact_match_full": None,
        "v5_accounting": None,
    })

    full_cache: dict[str, list[int]] = {}
    total_steps = (
        len(prompts)
        + len(prompts) * len(COMPRESSORS)
        + len(prompts) * len(COMPRESSORS) * len(DRAFT_LENS) * 2
    )
    step = 0

    for pe in prompts:
        pid = pe["prompt_id"]
        prompt = pe["prompt"]
        if pid not in full_cache:
            full_res = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
            full_cache[pid] = full_res.generated_ids.squeeze(0).tolist()

        step += 1
        print(f"  [{step}/{total_steps}] full_greedy × {pid}", flush=True)

        def fg_fn() -> Any:
            return generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)

        fg_result, fg_trials, fg_stats = _run_trials(
            device,
            fg_fn,
            model_loaded_allocated=model_loaded_allocated,
            model_loaded_reserved=model_loaded_reserved,
            num_trials=num_trials,
        )
        fg_ids = fg_result.generated_ids.squeeze(0).tolist()
        cells.append({
            "arm": "full_greedy",
            "prompt_id": pid,
            "v10_suite": pe.get("v10_suite", ""),
            "compressor_name": None,
            "draft_len": None,
            "verification_method": None,
            "trials": fg_trials,
            "trial_stats": fg_stats,
            "generated_tokens": _generated_token_count(fg_result),
            "output_ids": fg_ids,
            "exactkv_failure": False,
            "token_exact_match_full": True,
            "v5_accounting": None,
        })

        for comp in COMPRESSORS:
            step += 1
            print(f"  [{step}/{total_steps}] lossy_only × {pid} × {comp}", flush=True)

            def lossy_fn(c=comp) -> Any:
                return generate_lossy_greedy(
                    runtime, prompt, get_compressor(c), MAX_NEW_TOKENS
                )

            lossy_result, lossy_trials, lossy_stats = _run_trials(
                device,
                lossy_fn,
                model_loaded_allocated=model_loaded_allocated,
                model_loaded_reserved=model_loaded_reserved,
                num_trials=num_trials,
            )
            lossy_ids = lossy_result.generated_ids.squeeze(0).tolist()
            lossy_match = token_exact_match(
                torch.tensor([full_cache[pid]]),
                torch.tensor([lossy_ids]),
            )
            cells.append({
                "arm": "lossy_only",
                "prompt_id": pid,
                "v10_suite": pe.get("v10_suite", ""),
                "compressor_name": comp,
                "draft_len": None,
                "verification_method": None,
                "trials": lossy_trials,
                "trial_stats": lossy_stats,
                "generated_tokens": _generated_token_count(lossy_result),
                "output_ids": lossy_ids,
                "exactkv_failure": False,
                "token_exact_match_full": lossy_match,
                "v5_accounting": _v5_from_prompt(runtime, prompt, comp),
            })

            for draft_len in DRAFT_LENS:
                for vm in ("sequential", "span"):
                    step += 1
                    arm = f"exactkv_{vm}"
                    print(
                        f"  [{step}/{total_steps}] {arm} × {pid} × {comp} "
                        f"× draft_len={draft_len}",
                        flush=True,
                    )

                    def exactkv_fn(
                        c=comp, dl=draft_len, method=vm
                    ) -> Any:
                        gen = ExactKVGenerator(
                            runtime,
                            get_compressor(c),
                            draft_len=dl,
                            verification_method=method,  # type: ignore[arg-type]
                        )
                        return gen.generate(prompt, MAX_NEW_TOKENS)

                    ek_result, ek_trials, ek_stats = _run_trials(
                        device,
                        exactkv_fn,
                        model_loaded_allocated=model_loaded_allocated,
                        model_loaded_reserved=model_loaded_reserved,
                        num_trials=num_trials,
                    )
                    ek_ids = ek_result.output_ids.squeeze(0).tolist()
                    ek_match = token_exact_match(
                        torch.tensor([full_cache[pid]]),
                        torch.tensor([ek_ids]),
                    )
                    cells.append({
                        "arm": arm,
                        "prompt_id": pid,
                        "v10_suite": pe.get("v10_suite", ""),
                        "compressor_name": comp,
                        "draft_len": draft_len,
                        "verification_method": vm,
                        "trials": ek_trials,
                        "trial_stats": ek_stats,
                        "generated_tokens": _generated_token_count(ek_result),
                        "output_ids": ek_ids,
                        "exactkv_failure": not ek_match,
                        "token_exact_match_full": ek_match,
                        "v5_accounting": _v5_from_prompt(runtime, prompt, comp),
                    })

    agg = aggregate(cells, baseline_snap)
    env = collect_gpu_memory_environment(device)

    return {
        "experiment": "031",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "environment": env,
        "diagnostic_disclaimer": {
            "diagnostic_gpu_memory_isolation": True,
            "not_production_serving_evidence": True,
            "hardware_model_prompt_specific": True,
            "v5_accounting_distinct_from_active_cuda": True,
            "no_speedup_claim": True,
            "no_throughput_claim": True,
            "no_latency_claim": True,
            "no_runtime_improvement_claim": True,
            "no_tokens_per_second_claim": True,
            "no_production_serving_claim": True,
            "no_model_accuracy_improvement_claim": True,
            "active_gpu_memory_savings_claim_allowed": agg[
                "active_gpu_memory_savings_claim_allowed"
            ],
        },
        "methodology": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "draft_lens": list(DRAFT_LENS),
            "compressors": COMPRESSORS,
            "arms": list(ARMS),
            "num_trials": num_trials,
            "cuda_empty_cache_before_trial": True,
            "cuda_reset_peak_stats_before_trial": True,
            "cuda_synchronize": True,
            "prerequisite_exp030_exactness_gate": True,
        },
        "prompt_panel": "v10_stratified_12" if len(prompts) >= 12 else "smoke",
        "prompt_count": len(prompts),
        "cells": cells,
        "aggregate": agg,
        "interpretation": {
            "exactness_gate_passed": agg["exactness_gate_passed"],
            "memory_verdict": agg["memory_verdict"],
            "active_gpu_memory_savings_claim_allowed": agg[
                "active_gpu_memory_savings_claim_allowed"
            ],
            "general_memory_savings_claim_allowed": False,
            "claim_notes": (
                "Active GPU memory savings claims remain forbidden unless measured "
                "active allocation robustly decreases vs full greedy and the cause "
                "is attributable to KV compression — not weights or temporaries."
            ),
        },
    }


def write_csv_report(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for c in report["cells"]:
        if c["arm"] == "model_loaded_baseline":
            continue
        stats = c.get("trial_stats") or {}
        v5 = c.get("v5_accounting") or {}
        rows.append({
            "arm": c["arm"],
            "prompt_id": c.get("prompt_id") or "",
            "v10_suite": c.get("v10_suite") or "",
            "compressor_name": c.get("compressor_name") or "",
            "draft_len": c.get("draft_len") or "",
            "verification_method": c.get("verification_method") or "",
            "exactkv_failure": c.get("exactkv_failure", False),
            "token_exact_match_full": c.get("token_exact_match_full", True),
            "generated_tokens": c.get("generated_tokens") or "",
            "mean_peak_allocated": stats.get("mean_peak_allocated", ""),
            "max_peak_allocated": stats.get("max_peak_allocated", ""),
            "mean_peak_reserved": stats.get("mean_peak_reserved", ""),
            "max_peak_reserved": stats.get("max_peak_reserved", ""),
            "mean_delta_peak_allocated_vs_model_loaded": stats.get(
                "mean_delta_peak_allocated_vs_model_loaded", ""
            ),
            "v5_total_kv_footprint_bytes": v5.get("total_kv_footprint_bytes", ""),
            "v5_stored_kv_bytes": v5.get("stored_kv_bytes", ""),
            "v5_materialized_working_kv_bytes": v5.get(
                "materialized_working_kv_bytes", ""
            ),
        })
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    env = report["environment"]
    interp = report["interpretation"]
    baseline = agg["model_loaded_baseline"]

    verdict_text = {
        "exactkv_uses_more_peak_allocated": (
            "ExactKV appears to use **more** peak allocated memory than full greedy "
            "in this diagnostic setup (above allocator noise threshold)."
        ),
        "exactkv_uses_less_peak_allocated": (
            "ExactKV peak allocated is **lower** than full greedy in this setup, "
            "but savings claims remain **forbidden** until attribution is robust."
        ),
        "indistinguishable_within_allocator_noise": (
            "ExactKV and full greedy peak allocated are **indistinguishable** within "
            "allocator noise in this diagnostic setup."
        ),
    }.get(
        agg["memory_verdict"],
        "Insufficient data for memory verdict.",
    )

    span_text = ""
    seq_s = agg["by_arm"].get("exactkv_sequential", {})
    span_s = agg["by_arm"].get("exactkv_span", {})
    if seq_s.get("mean_peak_allocated") and span_s.get("mean_peak_allocated"):
        if span_s["mean_peak_allocated"] > seq_s["mean_peak_allocated"]:
            span_text = (
                "Span verify appears to use **more** peak allocated than sequential "
                "in this setup (batched teacher-forced forwards)."
            )
        elif span_s["mean_peak_allocated"] < seq_s["mean_peak_allocated"]:
            span_text = (
                "Span verify appears to use **less** peak allocated than sequential "
                "in this setup."
            )
        else:
            span_text = (
                "Span vs sequential peak allocated are **indistinguishable** in this setup."
            )

    lines = [
        "# Experiment 031: Active GPU Memory Isolation",
        "",
        "_Generated by `scripts/run_experiment_031_gpu_memory_isolation.py`. "
        "V13 Phase 4 — diagnostic GPU memory isolation only._",
        "",
        "> This is a **diagnostic GPU memory isolation experiment**, not production serving evidence.",
        "> Results are **hardware/model/prompt specific**.",
        "> Active CUDA allocation is **distinct** from V5 KV accounting.",
        "> **No active GPU memory savings claim** is allowed unless measured active allocation "
        "robustly decreases vs full greedy and the cause is attributable.",
        "> **No speedup, throughput, latency, runtime, tokens/sec, production serving, or "
        "model accuracy improvement claim** is made.",
        "> Exactness gate must pass before ExactKV memory comparisons are interpreted.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Determine whether ExactKV reduces, increases, or leaves unchanged active GPU "
        "memory allocation in a controlled diagnostic setup, separating model weights, "
        "prefill/decode activity, KV accounting, temporary tensors, and CUDA allocator effects.",
        "",
        "## 2. Why this follows Exp 030/030b",
        "",
        "Experiment 030 passed the exactness gate on RunPod fp16 (`phase4_memory_allowed=True`). "
        "Exp 030b restored batched span parity. Phase 4 memory isolation proceeds without "
        "new timing benchmarks.",
        "",
        "## 3. Environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{report['model_name']}` |",
        f"| dtype | `{report['dtype']}` |",
        f"| device | `{report['device']}` |",
        f"| GPU | `{env.get('gpu_device_name', 'N/A')}` |",
        f"| Total GPU memory | {_mib(env.get('gpu_total_memory_bytes'))} MiB |",
        f"| torch | `{env.get('torch_version', 'N/A')}` |",
        f"| transformers | `{env.get('transformers_version', 'N/A')}` |",
        f"| CUDA | `{env.get('cuda_version', 'N/A')}` |",
        f"| PYTORCH_CUDA_ALLOC_CONF | `{env.get('pytorch_cuda_alloc_conf') or 'default'}` |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_031_gpu_memory_isolation.py "
        "--device cuda --dtype float16",
        "```",
        "",
        "## 4. Prompt subset",
        "",
        f"| Panel | `{report['prompt_panel']}` |",
        f"| Prompt count | **{report['prompt_count']}** |",
        "",
        "3× `core_v2`, 3× `long_context`, 3× `retrieval_copy`, 3× `tool_json`.",
        "",
        "## 5. Arms compared",
        "",
        "| Arm | Description |",
        "|---|---|",
        "| `model_loaded_baseline` | Model on GPU, no generation |",
        "| `full_greedy` | Full-KV greedy generation |",
        "| `lossy_only` | `generate_lossy_greedy` (not exactness-preserving) |",
        "| `exactkv_sequential` | ExactKV + `verification_method=\"sequential\"` |",
        "| `exactkv_span` | ExactKV + `verification_method=\"span\"` (opt-in) |",
        "",
        "## 6. Memory measurement methodology",
        "",
        f"| Trials per cell | {report['methodology']['num_trials']} |",
        "| Pre-trial | `torch.cuda.empty_cache()`, `reset_peak_memory_stats()`, synchronize |",
        "| Metrics | `memory_allocated`, `memory_reserved`, peak allocated/reserved |",
        "| Delta baseline | peak vs model-loaded allocated/reserved |",
        "| V5 accounting | `estimate_kv_memory` after prefill per compressor |",
        "",
        "`torch.cuda.max_memory_allocated()` is not end-to-end process VRAM. "
        "`max_memory_reserved()` reflects allocator reservation and can obscure small KV differences.",
        "",
        "## 7. Exactness gate",
        "",
        f"| Exactness gate passed | **{agg['exactness_gate_passed']}** |",
        f"| Sequential ExactKV failures | **{agg['by_arm'].get('exactkv_sequential', {}).get('exactkv_failures', 'N/A')}** |",
        f"| Span ExactKV failures | **{agg['by_arm'].get('exactkv_span', {}).get('exactkv_failures', 'N/A')}** |",
        f"| Span vs sequential parity failures | **{agg['span_sequential_parity_failures']}** |",
        "",
        "## 8. Model-loaded baseline",
        "",
        f"| Allocated | {_mib(baseline['allocated_bytes'])} MiB |",
        f"| Reserved | {_mib(baseline['reserved_bytes'])} MiB |",
        "",
        "## 9. Peak allocated results by arm",
        "",
        "| Arm | cells | mean peak allocated (MiB) | max peak allocated (MiB) |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = agg["by_arm"].get(arm, {})
        lines.append(
            f"| `{arm}` | {s.get('cells', 0)} | "
            f"{_mib(s.get('mean_peak_allocated'))} | "
            f"{_mib(s.get('max_peak_allocated'))} |"
        )

    lines.extend([
        "",
        "## 10. Peak reserved results by arm",
        "",
        "| Arm | cells | mean peak reserved (MiB) | max peak reserved (MiB) |",
        "|---|---:|---:|---:|",
    ])
    for arm in ARMS:
        s = agg["by_arm"].get(arm, {})
        lines.append(
            f"| `{arm}` | {s.get('cells', 0)} | "
            f"{_mib(s.get('mean_peak_reserved'))} | "
            f"{_mib(s.get('max_peak_reserved'))} |"
        )

    cmp = agg["comparisons"]
    lines.extend([
        "",
        "## 11. Delta vs model-loaded baseline",
        "",
        "| Arm | mean Δ peak allocated (MiB) |",
        "|---|---:|",
    ])
    for arm in ARMS:
        s = agg["by_arm"].get(arm, {})
        lines.append(
            f"| `{arm}` | {_mib(s.get('mean_delta_peak_allocated_vs_model_loaded'))} |"
        )

    fg = agg["by_arm"].get("full_greedy", {})
    seq = agg["by_arm"].get("exactkv_sequential", {})
    span = agg["by_arm"].get("exactkv_span", {})
    lines.extend([
        "",
        "## 12. ExactKV vs full greedy memory result",
        "",
        f"| Full greedy mean peak allocated | {_mib(fg.get('mean_peak_allocated'))} MiB |",
        f"| ExactKV sequential mean peak allocated | {_mib(seq.get('mean_peak_allocated'))} MiB |",
        f"| ExactKV span mean peak allocated | {_mib(span.get('mean_peak_allocated'))} MiB |",
        f"| Seq / full greedy peak ratio | {_fmt(cmp['exactkv_sequential_vs_full_greedy'].get('ratio_mean_peak_allocated'))} |",
        "",
        verdict_text,
        "",
        "## 13. Span vs sequential memory result",
        "",
        f"| Sequential mean peak allocated | {_mib(seq.get('mean_peak_allocated'))} MiB |",
        f"| Span mean peak allocated | {_mib(span.get('mean_peak_allocated'))} MiB |",
        f"| Span / sequential peak ratio | {_fmt(cmp['exactkv_span_vs_exactkv_sequential'].get('ratio_mean_peak_allocated'))} |",
        "",
        span_text,
        "",
        "## 14. Results by draft_len",
        "",
        "| draft_len | arm | cells | mean peak allocated (MiB) |",
        "|---:|---|---:|---:|",
    ])
    for dl, bucket in agg["by_draft_len"].items():
        for arm in ("exactkv_sequential", "exactkv_span"):
            s = bucket[arm]
            lines.append(
                f"| {dl} | `{arm}` | {s.get('cells', 0)} | "
                f"{_mib(s.get('mean_peak_allocated'))} |"
            )

    lines.extend([
        "",
        "## 15. Results by compressor",
        "",
        "| compressor | arm | cells | mean peak allocated (MiB) |",
        "|---|---|---:|---:|",
    ])
    for comp, bucket in agg["by_compressor"].items():
        for arm in ("lossy_only", "exactkv_sequential", "exactkv_span"):
            s = bucket.get(arm, {})
            if s.get("cells"):
                lines.append(
                    f"| `{comp}` | `{arm}` | {s.get('cells', 0)} | "
                    f"{_mib(s.get('mean_peak_allocated'))} |"
                )

    v5 = agg["v5_vs_cuda"]
    lines.extend([
        "",
        "## 16. V5 accounting vs active CUDA memory",
        "",
        f"| Mean V5 `total_kv_footprint_bytes` | {_mib(v5.get('mean_v5_total_kv_footprint_bytes'))} MiB |",
        "",
        v5["note"],
        "",
        "If V5 accounting improves but active GPU memory does not, that gap is expected when "
        "model weights dominate peak allocation and materialized working KV is counted separately.",
        "",
        "## 17. Whether ExactKV currently needs more memory or less memory in this setup",
        "",
        verdict_text,
        "",
        "## 18. Whether active GPU memory savings claim is allowed",
        "",
        f"| Active GPU memory savings claim | **{'Allowed (caveated)' if interp['active_gpu_memory_savings_claim_allowed'] else 'Forbidden'}** |",
        f"| General memory savings claim | **Forbidden** |",
        "",
        interp["claim_notes"],
        "",
        "## 19. What this proves",
        "",
        "- Controlled diagnostic GPU memory observations for five arms on a 12-prompt panel.",
        "- Exactness gate status for ExactKV arms during memory measurements.",
        "- V5 KV accounting figures alongside CUDA peak observations.",
        "- Whether allocator-dominated measurements obscure compressor KV differences.",
        "",
        "## 20. What this does not prove",
        "",
        "- Production serving VRAM footprint or multi-request memory behaviour.",
        "- General GPU memory savings on other models, hardware, or prompt distributions.",
        "- Speedup, throughput, latency, or tokens/sec improvement.",
        "- Model accuracy improvement.",
        "- Isolated KV-cache VRAM without model weights (not achieved at 0.5B scale).",
        "",
        "## 21. Limitations",
        "",
        "- Single model (Qwen2.5-0.5B) and 12-prompt panel.",
        "- PyTorch allocator reservation can dominate small KV deltas.",
        "- Model weights (~GiB) dwarf V5 KV accounting (~MiB) on this panel.",
        "- `lossy_only` is not exactness-preserving; included for diagnostic comparison only.",
        "- Batched span verify may allocate larger temporary tensors than sequential verify.",
        "",
        "## 22. Next steps",
        "",
        f"- Phase 5 SnapKV/ShardKV feasibility (Exp 032): "
        f"**{'allowed' if agg['phase5_feasibility_allowed'] else 'blocked until exactness gate passes'}**.",
        "- Longer context / larger model (Exp 033) if KV deltas become measurable above noise.",
        "- Phase 9 headline audit before any public memory or performance wording.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 031 GPU memory isolation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS_DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="2 prompts, 1 trial")
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_031_gpu_memory_isolation.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_031_gpu_memory_isolation.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_031_GPU_MEMORY_ISOLATION.md"),
    )
    args = parser.parse_args()

    if args.smoke:
        args.num_trials = 1

    if not cuda_available():
        print(
            "ERROR: CUDA unavailable. Experiment 031 requires RunPod GPU. "
            "Stopping without final memory conclusions.",
            file=sys.stderr,
        )
        return 2

    require_cuda(args.device)
    dtype = args.dtype or "float16"

    prompts = load_exp031_prompt_panel(smoke=args.smoke)
    n_gen = (
        len(prompts)
        + len(prompts) * len(COMPRESSORS)
        + len(prompts) * len(COMPRESSORS) * len(DRAFT_LENS) * 2
    )
    print(
        f"Experiment 031 — {len(prompts)} prompts; "
        f"generation cells={n_gen}; trials={args.num_trials}"
    )

    runtime = ModelRuntime(model_name=MODEL_NAME, device=args.device, dtype=dtype)
    report = run_experiment(
        runtime,
        prompts,
        device=args.device,
        num_trials=args.num_trials,
    )

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    assert_memory_artifact_safe(report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(report, md_path)

    agg = report["aggregate"]
    print(
        f"Done: exactness_gate={agg['exactness_gate_passed']} "
        f"parity_fail={agg['span_sequential_parity_failures']} "
        f"memory_verdict={agg['memory_verdict']} "
        f"phase5_allowed={agg['phase5_feasibility_allowed']}"
    )
    return 0 if agg["exactness_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
