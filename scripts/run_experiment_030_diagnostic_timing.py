#!/usr/bin/env python3
"""Experiment 030: diagnostic timing harness (V13 Phase 3).

First approved timing experiment. Diagnostic only — not production benchmarking.
Prerequisite: Experiment 029 exactness grid passed (phase3_timing_allowed=True).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.exactness import token_exact_match
from exactkv.metrics.timing import (
    collect_timing_environment,
    estimate_sequential_verifier_forwards,
    estimate_span_verifier_forwards,
    summarize_trials,
    timed_call,
    tokens_per_second,
)
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 32
DRAFT_LENS = (4, 8)
EXPERIMENT_CLASS = "v13_diagnostic_timing"

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

STRATIFIED_SUITES: list[tuple[str, int]] = [
    ("core_v2", 5),
    ("long_context", 5),
    ("retrieval_copy", 5),
    ("tool_json", 5),
]

ARMS = (
    "full_greedy",
    "lossy_only",
    "exactkv_sequential",
    "exactkv_span",
)


def load_exp030_prompt_panel(*, smoke: bool = False) -> list[dict[str, Any]]:
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


def _count_generated(result: Any) -> int:
    return int(result.generated_ids.shape[1])


def _exactkv_counts(result: Any) -> dict[str, int]:
    return {
        "generated_tokens": int(result.output_ids.shape[1]),
        "draft_rounds": result.num_rounds,
        "total_accepted": result.total_accepted,
        "total_rejected": result.total_rejected,
        "total_corrections": result.total_corrections,
    }


def _run_full_greedy(runtime: ModelRuntime, prompt: str) -> Any:
    return generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)


def _run_lossy(
    runtime: ModelRuntime, prompt: str, compressor_name: str
) -> Any:
    comp = get_compressor(compressor_name)
    return generate_lossy_greedy(runtime, prompt, comp, MAX_NEW_TOKENS)


def _run_exactkv(
    runtime: ModelRuntime,
    prompt: str,
    compressor_name: str,
    draft_len: int,
    verification_method: str,
) -> Any:
    comp = get_compressor(compressor_name)
    gen = ExactKVGenerator(
        runtime,
        comp,
        draft_len=draft_len,
        verification_method=verification_method,  # type: ignore[arg-type]
    )
    return gen.generate(prompt, MAX_NEW_TOKENS)


def _timed_arm(
    runtime: ModelRuntime,
    device: str,
    fn: Callable[[], Any],
    *,
    num_warmup: int,
    num_trials: int,
) -> tuple[Any, list[dict[str, float]], dict[str, float]]:
    for _ in range(num_warmup):
        timed_call(device, fn)

    trials: list[dict[str, float]] = []
    last_result: Any = None
    for _ in range(num_trials):
        result, wall = timed_call(device, fn)
        last_result = result
        gen_tok = (
            _count_generated(result)
            if hasattr(result, "generated_ids")
            else int(result.output_ids.shape[1])
        )
        trials.append({
            "wall_time_seconds": wall,
            "generated_tokens": gen_tok,
            "tokens_per_second": tokens_per_second(gen_tok, wall),
        })

    stats = summarize_trials(
        [t["wall_time_seconds"] for t in trials],
        [int(t["generated_tokens"]) for t in trials],
    )
    return last_result, trials, stats


def _check_exactness(
    output_ids: list[int],
    full_ids: list[int],
) -> tuple[bool, bool]:
    out_tensor = torch.tensor([output_ids], dtype=torch.long)
    full_tensor = torch.tensor([full_ids], dtype=torch.long)
    match = token_exact_match(full_tensor, out_tensor)
    return match, not match


def run_full_greedy_cell(
    runtime: ModelRuntime,
    device: str,
    prompt_entry: dict[str, Any],
    *,
    num_warmup: int,
    num_trials: int,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]

    def fn() -> Any:
        return _run_full_greedy(runtime, prompt)

    result, trials, stats = _timed_arm(
        runtime, device, fn, num_warmup=num_warmup, num_trials=num_trials
    )
    output_ids = result.generated_ids.squeeze(0).tolist()
    return {
        "arm": "full_greedy",
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "compressor_name": None,
        "draft_len": None,
        "verification_method": None,
        "output_ids": output_ids,
        "exactkv_failure": False,
        "token_exact_match_full": True,
        "timing_valid": True,
        "trials": trials,
        "trial_stats": stats,
        "draft_rounds": None,
        "total_accepted": None,
        "total_rejected": None,
        "total_corrections": None,
        "verifier_forwards_estimated": None,
    }


def run_lossy_cell(
    runtime: ModelRuntime,
    device: str,
    prompt_entry: dict[str, Any],
    compressor_name: str,
    full_ids: list[int],
    *,
    num_warmup: int,
    num_trials: int,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]

    def fn() -> Any:
        return _run_lossy(runtime, prompt, compressor_name)

    result, trials, stats = _timed_arm(
        runtime, device, fn, num_warmup=num_warmup, num_trials=num_trials
    )
    output_ids = result.generated_ids.squeeze(0).tolist()
    match, failure = _check_exactness(output_ids, full_ids)
    return {
        "arm": "lossy_only",
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "compressor_name": compressor_name,
        "draft_len": None,
        "verification_method": None,
        "output_ids": output_ids,
        "exactkv_failure": failure,
        "token_exact_match_full": match,
        "timing_valid": True,
        "trials": trials,
        "trial_stats": stats,
        "draft_rounds": None,
        "total_accepted": None,
        "total_rejected": None,
        "total_corrections": None,
        "verifier_forwards_estimated": None,
    }


def run_exactkv_cell(
    runtime: ModelRuntime,
    device: str,
    prompt_entry: dict[str, Any],
    compressor_name: str,
    draft_len: int,
    verification_method: str,
    full_ids: list[int],
    *,
    num_warmup: int,
    num_trials: int,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    arm = (
        "exactkv_sequential"
        if verification_method == "sequential"
        else "exactkv_span"
    )

    def fn() -> Any:
        return _run_exactkv(
            runtime, prompt, compressor_name, draft_len, verification_method
        )

    result, trials, stats = _timed_arm(
        runtime, device, fn, num_warmup=num_warmup, num_trials=num_trials
    )
    output_ids = result.output_ids.squeeze(0).tolist()
    match, failure = _check_exactness(output_ids, full_ids)
    if verification_method == "sequential":
        verifier_forwards = estimate_sequential_verifier_forwards(result.traces)
    else:
        verifier_forwards = estimate_span_verifier_forwards(result.traces)
    counts = _exactkv_counts(result)
    timing_valid = not failure
    return {
        "arm": arm,
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "verification_method": verification_method,
        "output_ids": output_ids,
        "exactkv_failure": failure,
        "token_exact_match_full": match,
        "timing_valid": timing_valid,
        "trials": trials,
        "trial_stats": stats,
        "draft_rounds": counts["draft_rounds"],
        "total_accepted": counts["total_accepted"],
        "total_rejected": counts["total_rejected"],
        "total_corrections": counts["total_corrections"],
        "verifier_forwards_estimated": verifier_forwards,
    }


def _arm_stats(cells: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [c for c in cells if c.get("timing_valid", True)]
    invalid = [c for c in cells if not c.get("timing_valid", True)]
    walls = [c["trial_stats"]["mean_wall_time_seconds"] for c in valid]
    tps = [c["trial_stats"]["mean_tokens_per_second"] for c in valid]
    return {
        "cells": len(cells),
        "valid_timing_cells": len(valid),
        "invalid_timing_cells": len(invalid),
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        "mean_wall_time_seconds": mean(walls) if walls else None,
        "median_wall_time_seconds": median(walls) if walls else None,
        "mean_tokens_per_second": mean(tps) if tps else None,
        "median_tokens_per_second": median(tps) if tps else None,
    }


def _compare_arms(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if not left or not right:
        return {"label": label, "ratio_mean_wall_time": None, "ratio_mean_tps": None}
    lw = left.get("mean_wall_time_seconds")
    rw = right.get("mean_wall_time_seconds")
    lt = left.get("mean_tokens_per_second")
    rt = right.get("mean_tokens_per_second")
    return {
        "label": label,
        "left_mean_wall_time_seconds": lw,
        "right_mean_wall_time_seconds": rw,
        "ratio_mean_wall_time": (lw / rw) if lw and rw else None,
        "left_mean_tokens_per_second": lt,
        "right_mean_tokens_per_second": rt,
        "ratio_mean_tokens_per_second": (lt / rt) if lt and rt else None,
    }


def aggregate(report_cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_draft: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)

    exactkv_seq: list[dict[str, Any]] = []
    exactkv_span: list[dict[str, Any]] = []
    parity_failures = 0

    pair_map: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}

    for cell in report_cells:
        by_arm[cell["arm"]].append(cell)
        if cell["arm"] in ("exactkv_sequential", "exactkv_span"):
            key = (
                cell["prompt_id"],
                cell["compressor_name"],
                cell["draft_len"],
            )
            pair_map.setdefault(key, {})[cell["arm"]] = cell
            if cell["arm"] == "exactkv_sequential":
                exactkv_seq.append(cell)
            else:
                exactkv_span.append(cell)
            by_draft[cell["draft_len"]].append(cell)
            by_comp[cell["compressor_name"]].append(cell)

    for key, pair in pair_map.items():
        seq = pair.get("exactkv_sequential")
        span = pair.get("exactkv_span")
        if seq and span and seq["output_ids"] != span["output_ids"]:
            parity_failures += 1

    arm_stats = {arm: _arm_stats(cells) for arm, cells in by_arm.items()}
    seq_only = _arm_stats(exactkv_seq)
    span_only = _arm_stats(exactkv_span)

    def _bucket_arm(cells: list[dict[str, Any]], arm: str) -> dict[str, Any]:
        return _arm_stats([c for c in cells if c["arm"] == arm])

    by_draft_stats = {
        str(dl): {
            "exactkv_sequential": _bucket_arm(cells, "exactkv_sequential"),
            "exactkv_span": _bucket_arm(cells, "exactkv_span"),
        }
        for dl, cells in sorted(by_draft.items())
    }
    by_comp_stats = {
        comp: {
            "exactkv_sequential": _bucket_arm(cells, "exactkv_sequential"),
            "exactkv_span": _bucket_arm(cells, "exactkv_span"),
        }
        for comp, cells in sorted(by_comp.items())
    }

    seq_valid = [c for c in exactkv_seq if c["timing_valid"]]
    span_valid = [c for c in exactkv_span if c["timing_valid"]]
    mean_seq_vf = (
        mean(c["verifier_forwards_estimated"] for c in seq_valid)
        if seq_valid
        else None
    )
    mean_span_vf = (
        mean(c["verifier_forwards_estimated"] for c in span_valid)
        if span_valid
        else None
    )

    exactness_gate_passed = (
        seq_only["exactkv_failures"] == 0
        and span_only["exactkv_failures"] == 0
        and parity_failures == 0
    )

    return {
        "exactness_gate_passed": exactness_gate_passed,
        "span_sequential_parity_failures": parity_failures,
        "by_arm": arm_stats,
        "exactkv_sequential_summary": seq_only,
        "exactkv_span_summary": span_only,
        "comparisons": {
            "span_vs_sequential": _compare_arms(
                span_only, seq_only, label="exactkv_span / exactkv_sequential"
            ),
            "span_vs_full_greedy": _compare_arms(
                span_only,
                arm_stats.get("full_greedy"),
                label="exactkv_span / full_greedy",
            ),
            "sequential_vs_full_greedy": _compare_arms(
                seq_only,
                arm_stats.get("full_greedy"),
                label="exactkv_sequential / full_greedy",
            ),
        },
        "verifier_forwards": {
            "mean_sequential_estimated": mean_seq_vf,
            "mean_span_estimated": mean_span_vf,
            "reduction_ratio_span_over_sequential": (
                mean_span_vf / mean_seq_vf
                if mean_seq_vf and mean_span_vf is not None and mean_seq_vf > 0
                else None
            ),
        },
        "by_draft_len": by_draft_stats,
        "by_compressor": by_comp_stats,
        "phase4_memory_allowed": exactness_gate_passed,
    }


def run_experiment(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    num_warmup: int,
    num_trials: int,
) -> dict[str, Any]:
    full_cache: dict[str, list[int]] = {}
    cells: list[dict[str, Any]] = []
    total_steps = (
        len(prompts)
        + len(prompts) * len(COMPRESSORS)
        + len(prompts) * len(COMPRESSORS) * len(DRAFT_LENS) * 2
    )
    step = 0

    for pe in prompts:
        pid = pe["prompt_id"]
        if pid not in full_cache:
            full_res = generate_full_greedy(runtime, pe["prompt"], MAX_NEW_TOKENS)
            full_cache[pid] = full_res.generated_ids.squeeze(0).tolist()

        step += 1
        print(f"  [{step}/{total_steps}] full_greedy × {pid}", flush=True)
        cells.append(
            run_full_greedy_cell(
                runtime,
                device,
                pe,
                num_warmup=num_warmup,
                num_trials=num_trials,
            )
        )

        for comp in COMPRESSORS:
            step += 1
            print(
                f"  [{step}/{total_steps}] lossy_only × {pid} × {comp}",
                flush=True,
            )
            cells.append(
                run_lossy_cell(
                    runtime,
                    device,
                    pe,
                    comp,
                    full_cache[pid],
                    num_warmup=num_warmup,
                    num_trials=num_trials,
                )
            )

            for draft_len in DRAFT_LENS:
                for vm in ("sequential", "span"):
                    step += 1
                    arm = f"exactkv_{vm}"
                    print(
                        f"  [{step}/{total_steps}] {arm} × {pid} × {comp} "
                        f"× draft_len={draft_len}",
                        flush=True,
                    )
                    cells.append(
                        run_exactkv_cell(
                            runtime,
                            device,
                            pe,
                            comp,
                            draft_len,
                            vm,
                            full_cache[pid],
                            num_warmup=num_warmup,
                            num_trials=num_trials,
                        )
                    )

    agg = aggregate(cells)
    env = collect_timing_environment(device)
    cuda_ok = env.get("cuda_available", False)
    diagnostic_only_cpu = device == "cpu" or not cuda_ok

    slower_than_full = None
    fg = agg["by_arm"].get("full_greedy", {})
    seq = agg.get("exactkv_sequential_summary", {})
    span = agg.get("exactkv_span_summary", {})
    if fg.get("mean_tokens_per_second") and seq.get("mean_tokens_per_second"):
        slower_than_full = seq["mean_tokens_per_second"] < fg["mean_tokens_per_second"]

    return {
        "experiment": "030",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "environment": env,
        "diagnostic_disclaimer": {
            "diagnostic_timing_harness": True,
            "not_production_benchmark": True,
            "hardware_model_prompt_specific": True,
            "cpu_timing_not_for_interpretation": diagnostic_only_cpu,
            "no_production_serving_claim": True,
            "no_active_gpu_memory_savings_claim": True,
            "no_model_accuracy_improvement_claim": True,
        },
        "methodology": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "draft_lens": list(DRAFT_LENS),
            "compressors": COMPRESSORS,
            "arms": list(ARMS),
            "num_warmup": num_warmup,
            "num_trials": num_trials,
            "timer": "time.perf_counter",
            "cuda_synchronize": cuda_ok,
            "prerequisite_exp029_phase3_timing_allowed": True,
        },
        "prompt_panel": "v10_stratified_20_smoke" if len(prompts) < 20 else "v10_stratified_20",
        "prompt_count": len(prompts),
        "cells": cells,
        "aggregate": agg,
        "interpretation": {
            "exactness_gate_passed": agg["exactness_gate_passed"],
            "exactkv_slower_than_full_greedy_diagnostic": slower_than_full,
            "span_faster_than_sequential_diagnostic": (
                span.get("mean_tokens_per_second", 0)
                > seq.get("mean_tokens_per_second", 0)
                if span.get("mean_tokens_per_second") and seq.get("mean_tokens_per_second")
                else None
            ),
            "general_speed_claim_allowed": False,
            "diagnostic_speed_claim_allowed": (
                agg["exactness_gate_passed"]
                and not diagnostic_only_cpu
                and agg["exactness_gate_passed"]
            ),
            "speed_claim_notes": (
                "Diagnostic throughput numbers may be reported only for this "
                "controlled setup with full methodology caveats. General speedup, "
                "production throughput, or latency improvement claims remain "
                "forbidden until V13 Phase 9 approval."
            ),
        },
    }


def write_csv_report(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for c in report["cells"]:
        rows.append({
            "arm": c["arm"],
            "prompt_id": c["prompt_id"],
            "v10_suite": c["v10_suite"],
            "compressor_name": c.get("compressor_name") or "",
            "draft_len": c.get("draft_len") or "",
            "verification_method": c.get("verification_method") or "",
            "exactkv_failure": c.get("exactkv_failure", False),
            "token_exact_match_full": c.get("token_exact_match_full", True),
            "timing_valid": c.get("timing_valid", True),
            "mean_wall_time_seconds": c["trial_stats"]["mean_wall_time_seconds"],
            "median_wall_time_seconds": c["trial_stats"]["median_wall_time_seconds"],
            "mean_tokens_per_second": c["trial_stats"]["mean_tokens_per_second"],
            "draft_rounds": c.get("draft_rounds") or "",
            "verifier_forwards_estimated": c.get("verifier_forwards_estimated") or "",
            "total_accepted": c.get("total_accepted") or "",
            "total_rejected": c.get("total_rejected") or "",
            "total_corrections": c.get("total_corrections") or "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    env = report["environment"]
    interp = report["interpretation"]
    lines = [
        "# Experiment 030: Diagnostic Timing Harness",
        "",
        "_Generated by `scripts/run_experiment_030_diagnostic_timing.py`. "
        "V13 Phase 3 — diagnostic timing only._",
        "",
        "> This is a **diagnostic timing harness**, not a production benchmark.",
        "> Results are **hardware/model/prompt specific**.",
        "> Throughput/timing numbers are valid **only for this controlled setup**.",
        "> **No production serving claim.**",
        "> **No active GPU memory savings claim.**",
        "> **No model accuracy improvement claim.**",
        "> Exactness gate must pass before timing is interpreted.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Measure whether ExactKV slows down or speeds up generation under a "
        "controlled diagnostic harness, and whether span verification reduces "
        "verifier overhead compared to sequential verification.",
        "",
        "## 2. Why timing is allowed only after Exp 029",
        "",
        "Experiment 029 span exactness grid passed with zero failures "
        "(`phase3_timing_allowed=True`). Phase 3 timing is gated on that result.",
        "",
        "## 3. Environment",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | `{report['model_name']}` |",
        f"| dtype | `{report['dtype']}` |",
        f"| device | `{report['device']}` |",
        f"| GPU | `{env.get('gpu_device_name', 'N/A')}` |",
        f"| torch | `{env.get('torch_version', 'N/A')}` |",
        f"| transformers | `{env.get('transformers_version', 'N/A')}` |",
        f"| CUDA | `{env.get('cuda_version', 'N/A')}` |",
        f"| CPU timing (not for interpretation) | "
        f"**{report['diagnostic_disclaimer']['cpu_timing_not_for_interpretation']}** |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_030_diagnostic_timing.py "
        "--device cuda --dtype float16",
        "```",
        "",
        "## 4. Prompt subset",
        "",
        f"| Panel | `{report['prompt_panel']}` |",
        f"| Prompt count | **{report['prompt_count']}** |",
        "",
        "5× `core_v2`, 5× `long_context`, 5× `retrieval_copy`, 5× `tool_json`.",
        "",
        "## 5. Arms compared",
        "",
        "| Arm | Description |",
        "|---|---|",
        "| `full_greedy` | `generate_full_greedy` baseline |",
        "| `lossy_only` | `generate_lossy_greedy` (no verification) |",
        "| `exactkv_sequential` | ExactKV + `verification_method=\"sequential\"` |",
        "| `exactkv_span` | ExactKV + `verification_method=\"span\"` (opt-in) |",
        "",
        "## 6. Timing methodology",
        "",
        f"| Warmup runs | {report['methodology']['num_warmup']} |",
        f"| Measured trials | {report['methodology']['num_trials']} |",
        f"| Timer | `{report['methodology']['timer']}` |",
        f"| CUDA synchronize | {report['methodology']['cuda_synchronize']} |",
        f"| max_new_tokens | {report['methodology']['max_new_tokens']} |",
        f"| draft_len | {', '.join(str(d) for d in report['methodology']['draft_lens'])} |",
        "",
        "## 7. Exactness gate",
        "",
        f"| Exactness gate passed | **{agg['exactness_gate_passed']}** |",
        f"| Sequential ExactKV failures | **{agg['exactkv_sequential_summary']['exactkv_failures']}** |",
        f"| Span ExactKV failures | **{agg['exactkv_span_summary']['exactkv_failures']}** |",
        f"| Span vs sequential parity failures | **{agg['span_sequential_parity_failures']}** |",
        "",
        "## 8. Timing results by arm",
        "",
        "| Arm | cells | valid | mean wall (s) | median wall (s) | mean tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = agg["by_arm"].get(arm, {})
        lines.append(
            f"| `{arm}` | {stats.get('cells', 0)} | {stats.get('valid_timing_cells', 0)} | "
            f"{_fmt(stats.get('mean_wall_time_seconds'))} | "
            f"{_fmt(stats.get('median_wall_time_seconds'))} | "
            f"{_fmt(stats.get('mean_tokens_per_second'))} |"
        )

    cmp = agg["comparisons"]
    lines.extend([
        "",
        "## 9. Span vs sequential timing",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Sequential mean tok/s | {_fmt(cmp['span_vs_sequential'].get('right_mean_tokens_per_second'))} |",
        f"| Span mean tok/s | {_fmt(cmp['span_vs_sequential'].get('left_mean_tokens_per_second'))} |",
        f"| Span/sequential mean tok/s ratio | {_fmt(cmp['span_vs_sequential'].get('ratio_mean_tokens_per_second'))} |",
        f"| Span/sequential mean wall ratio | {_fmt(cmp['span_vs_sequential'].get('ratio_mean_wall_time'))} |",
        "",
        "## 10. ExactKV vs full greedy timing",
        "",
        f"| Sequential / full greedy mean tok/s ratio | "
        f"{_fmt(cmp['sequential_vs_full_greedy'].get('ratio_mean_tokens_per_second'))} |",
        f"| Span / full greedy mean tok/s ratio | "
        f"{_fmt(cmp['span_vs_full_greedy'].get('ratio_mean_tokens_per_second'))} |",
        "",
        "## 11. Verifier-call / round-count analysis",
        "",
        f"| Mean sequential verifier forwards (est.) | "
        f"{_fmt(agg['verifier_forwards'].get('mean_sequential_estimated'), 2)} |",
        f"| Mean span verifier forwards (est.) | "
        f"{_fmt(agg['verifier_forwards'].get('mean_span_estimated'), 2)} |",
        f"| Span/sequential forwards ratio | "
        f"{_fmt(agg['verifier_forwards'].get('reduction_ratio_span_over_sequential'), 3)} |",
        "",
        "## 12. Results by draft_len",
        "",
        "| draft_len | arm | cells | mean wall (s) | mean tok/s |",
        "|---:|---|---:|---:|---:|",
    ])
    for dl, bucket in agg["by_draft_len"].items():
        for arm in ("exactkv_sequential", "exactkv_span"):
            s = bucket[arm]
            lines.append(
                f"| {dl} | `{arm}` | {s.get('cells', 0)} | "
                f"{_fmt(s.get('mean_wall_time_seconds'))} | "
                f"{_fmt(s.get('mean_tokens_per_second'))} |"
            )

    lines.extend([
        "",
        "## 13. Results by compressor",
        "",
        "| compressor | arm | cells | mean wall (s) | mean tok/s |",
        "|---|---|---:|---:|---:|",
    ])
    for comp, bucket in agg["by_compressor"].items():
        for arm in ("exactkv_sequential", "exactkv_span"):
            s = bucket[arm]
            lines.append(
                f"| `{comp}` | `{arm}` | {s.get('cells', 0)} | "
                f"{_fmt(s.get('mean_wall_time_seconds'))} | "
                f"{_fmt(s.get('mean_tokens_per_second'))} |"
            )

    slower = interp.get("exactkv_slower_than_full_greedy_diagnostic")
    span_faster = interp.get("span_faster_than_sequential_diagnostic")
    slower_text = (
        "ExactKV sequential appears **slower** than full greedy in this diagnostic setup."
        if slower is True
        else (
            "ExactKV sequential appears **faster** than full greedy in this diagnostic setup "
            "(caveats apply)."
            if slower is False
            else "Insufficient valid timing data."
        )
    )
    span_text = (
        "Span appears **faster** than sequential in this diagnostic setup."
        if span_faster is True
        else (
            "Span appears **slower or equal** to sequential in this diagnostic setup."
            if span_faster is False
            else "Insufficient valid timing data."
        )
    )

    lines.extend([
        "",
        "## 14. Whether ExactKV currently slows down or speeds up",
        "",
        slower_text,
        "",
        span_text,
        "",
        "## 15. Whether a speed/throughput claim is allowed",
        "",
        f"| General speed/throughput claim | **Forbidden** (`general_speed_claim_allowed={interp['general_speed_claim_allowed']}`) |",
        f"| Diagnostic claim (this setup only) | "
        f"**{'Allowed with caveats' if interp['diagnostic_speed_claim_allowed'] else 'Not yet — need CUDA run + exactness gate'}** |",
        "",
        interp["speed_claim_notes"],
        "",
        "## 16. What this proves",
        "",
        "- Controlled diagnostic timing for four arms on a fixed 20-prompt panel.",
        "- Exactness gate status for ExactKV arms during timing runs.",
        "- Estimated verifier forward-pass counts for sequential vs span.",
        "",
        "## 17. What this does not prove",
        "",
        "- Production serving throughput or latency.",
        "- General speedup on other models, hardware, or prompt distributions.",
        "- Active GPU memory savings.",
        "- Model accuracy improvement.",
        "",
        "## 18. Limitations",
        "",
        "- Single model (Qwen2.5-0.5B) and small prompt panel.",
        "- Diagnostic harness overhead (Python loop, deep-copy) included.",
        "- CPU runs are for debugging only — not for timing interpretation.",
        "",
        "## 19. Next steps",
        "",
        f"- Phase 4 active GPU memory isolation (Exp 031): "
        f"**{'allowed' if agg['phase4_memory_allowed'] else 'blocked until exactness gate passes'}**.",
        "- Re-run on documented RunPod GPU if this report used CPU/debug settings.",
        "- Phase 9 headline audit before any public performance wording.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_dtype(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 030 diagnostic timing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 prompts × subset; 1 warmup/trial for script validation",
    )
    parser.add_argument(
        "--allow-cpu-timing",
        action="store_true",
        help="Allow CPU timing (marks report as not for interpretation)",
    )
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_030_diagnostic_timing.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_030_diagnostic_timing.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_030_DIAGNOSTIC_TIMING.md"),
    )
    args = parser.parse_args()

    if args.device == "cpu" and not args.allow_cpu_timing and not args.smoke:
        print(
            "ERROR: CPU timing requires --allow-cpu-timing or --smoke. "
            "Use --device cuda on RunPod for interpretable results.",
            file=sys.stderr,
        )
        return 2

    if args.smoke:
        args.num_warmup = 1
        args.num_trials = 1

    dtype = args.dtype or _default_dtype(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available; falling back to CPU smoke/debug mode.",
            file=sys.stderr,
        )
        args.device = "cpu"
        dtype = "float32"
        args.allow_cpu_timing = True

    prompts = load_exp030_prompt_panel(smoke=args.smoke)
    n_exactkv = len(prompts) * len(COMPRESSORS) * len(DRAFT_LENS) * 2
    print(
        f"Experiment 030 — {len(prompts)} prompts; "
        f"ExactKV cells={n_exactkv}; warmup={args.num_warmup}; trials={args.num_trials}"
    )

    runtime = ModelRuntime(model_name=MODEL_NAME, device=args.device, dtype=dtype)
    report = run_experiment(
        runtime,
        prompts,
        device=args.device,
        num_warmup=args.num_warmup,
        num_trials=args.num_trials,
    )

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(report, md_path)

    agg = report["aggregate"]
    print(
        f"Done: exactness_gate={agg['exactness_gate_passed']} "
        f"parity_fail={agg['span_sequential_parity_failures']} "
        f"phase4_allowed={agg['phase4_memory_allowed']}"
    )
    if not agg["exactness_gate_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
