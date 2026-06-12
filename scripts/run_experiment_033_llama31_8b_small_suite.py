#!/usr/bin/env python3
"""Experiment 033: Llama-3.1-8B small-suite exactness validation (V13 Phase 6).

Larger-model exactness validation only — not a timing or GPU memory benchmark.
No speedup, throughput, latency, runtime_seconds, or active GPU memory savings claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 16
DRAFT_LEN_DEFAULT = 4
EXPERIMENT_CLASS = "v13_llama31_8b_small_suite"

REQUIRED_COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
]

SNAPKV_NAME = "snapkv_experimental"

STRATIFIED_SUITES: list[tuple[str, int]] = [
    ("core_v2", 3),
    ("long_context", 3),
    ("retrieval_copy", 3),
    ("tool_json", 3),
]

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


def _require_cuda(device: str) -> None:
    if device != "cuda":
        raise SystemExit("Experiment 033 requires --device cuda (RunPod GPU).")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; Experiment 033 requires a RunPod GPU.")


def _default_dtype(device: str) -> str:
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if device == "cuda":
        return "float16"
    return "float32"


def collect_environment(device: str) -> dict[str, Any]:
    import transformers

    env: dict[str, Any] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "hf_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        first = out.split("\n")[0]
        parts = [p.strip() for p in first.split(",")]
        env["gpu_name"] = parts[0] if parts else "unknown"
        env["gpu_memory_total"] = parts[1] if len(parts) > 1 else "unknown"
        env["gpu_driver"] = parts[2] if len(parts) > 2 else "unknown"
    except (OSError, subprocess.CalledProcessError):
        env["gpu_name"] = "unknown"
        env["gpu_memory_total"] = "unknown"
        env["gpu_driver"] = "unknown"
    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env["gpu_index"] = idx
        env["gpu_total_bytes"] = props.total_memory
    return env


def check_model_access(model_name: str) -> dict[str, Any]:
    """Verify gated Llama model access before loading full weights."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    status: dict[str, Any] = {
        "model_name": model_name,
        "token_present": bool(token),
        "accessible": False,
        "gated": None,
        "local_cache_hit": False,
        "error": None,
    }
    try:
        from transformers import AutoConfig

        try:
            AutoConfig.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
            status["local_cache_hit"] = True
            status["accessible"] = True
            return status
        except OSError:
            pass

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.model_info(model_name)
        status["gated"] = info.gated
        if not token:
            status["error"] = (
                f"{model_name} is gated; HF_TOKEN or HUGGING_FACE_HUB_TOKEN required"
            )
            return status
        AutoConfig.from_pretrained(model_name, token=token, trust_remote_code=True)
        status["accessible"] = True
    except Exception as exc:  # noqa: BLE001
        status["error"] = str(exc)
    return status


def load_exp033_prompt_panel() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for suite_name, count in STRATIFIED_SUITES:
        suite = load_v10_suite(suite_name)
        if len(suite) < count:
            raise ValueError(
                f"Suite {suite_name!r} has {len(suite)} prompts; need {count}"
            )
        out.extend(suite[:count])
    return out


def _check_cache_alignment(res: Any) -> bool:
    return all(
        t.full_seq_len_after == t.compressed_seq_len_after for t in res.traces
    )


def _run_exactkv(
    runtime: ModelRuntime,
    prompt: str,
    compressor: Any,
    draft_len: int,
    verification_method: str,
) -> dict[str, Any]:
    gen = ExactKVGenerator(
        runtime,
        compressor,
        draft_len=draft_len,
        verification_method=verification_method,  # type: ignore[arg-type]
    )
    res = gen.generate(prompt, MAX_NEW_TOKENS)
    acc = summarize_acceptance(res.traces)
    return {
        "output_ids": res.output_ids.squeeze(0).tolist(),
        "total_accepted": res.total_accepted,
        "total_rejected": res.total_rejected,
        "total_corrections": res.total_corrections,
        "acceptance_rate": res.acceptance_rate,
        "num_rounds": res.num_rounds,
        "acceptance": acc.to_dict(),
        "cache_alignment_ok": _check_cache_alignment(res),
    }


def _probe_snapkv(runtime: ModelRuntime) -> dict[str, Any]:
    """Attempt optional snapkv_experimental; skip honestly on VRAM/runtime blockers."""
    probe: dict[str, Any] = {
        "compressor": SNAPKV_NAME,
        "attempted": False,
        "included": False,
        "skip_reason": None,
    }
    try:
        import kvpress  # noqa: F401
    except ImportError:
        probe["skip_reason"] = "kvpress not installed in active environment"
        return probe

    if runtime.device.type != "cuda":
        probe["skip_reason"] = "snapkv_experimental requires CUDA for this probe"
        return probe

    total = torch.cuda.get_device_properties(runtime.device).total_memory
    allocated = torch.cuda.memory_allocated(runtime.device)
    # Isolated deepcopy duplicates ~full model weights — unsafe on 8B when tight.
    if allocated > total * 0.40:
        probe["skip_reason"] = (
            "insufficient VRAM headroom for isolated compression model copy "
            f"(allocated={allocated}, total={total})"
        )
        return probe

    from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

    probe["attempted"] = True
    try:
        adapter = create_snapkv_experimental_adapter(
            runtime, compression_ratio=0.5, isolate_compression_model=True
        )
        del adapter
        torch.cuda.empty_cache()
        probe["included"] = True
    except torch.cuda.OutOfMemoryError:
        probe["skip_reason"] = "OOM creating snapkv_experimental isolated compression model"
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        probe["skip_reason"] = f"snapkv init failed: {exc}"
        torch.cuda.empty_cache()
    return probe


def _resolve_compressors(
    runtime: ModelRuntime,
    *,
    try_snapkv: bool,
) -> tuple[list[str], dict[str, Any]]:
    names = list(REQUIRED_COMPRESSORS)
    snapkv_status: dict[str, Any] = {"included": False, "skip_reason": "not requested"}
    if try_snapkv:
        snapkv_status = _probe_snapkv(runtime)
        if snapkv_status.get("included"):
            names.append(SNAPKV_NAME)
    return names, snapkv_status


def _get_compressor(runtime: ModelRuntime, name: str, cache: dict[str, Any]) -> Any:
    if name in cache:
        return cache[name]
    if name == SNAPKV_NAME:
        from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter

        comp = create_snapkv_experimental_adapter(runtime, compression_ratio=0.5)
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    compressor_name: str,
    compressor: Any,
    draft_len: int,
    full_ids: list[int],
    *,
    run_span: bool,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=runtime.device)

    sequential = _run_exactkv(runtime, prompt, compressor, draft_len, "sequential")
    seq_ids = sequential["output_ids"]
    seq_exact = token_exact_match(full_tensor, torch.tensor([seq_ids]))

    span: dict[str, Any] | None = None
    span_exact = None
    span_seq_parity = None
    counters_match = None
    span_blocker: str | None = None

    if run_span:
        try:
            span = _run_exactkv(runtime, prompt, compressor, draft_len, "span")
            span_ids = span["output_ids"]
            span_exact = token_exact_match(full_tensor, torch.tensor([span_ids]))
            span_seq_parity = seq_ids == span_ids
            counters_match = (
                sequential["total_accepted"] == span["total_accepted"]
                and sequential["total_rejected"] == span["total_rejected"]
                and sequential["total_corrections"] == span["total_corrections"]
            )
        except Exception as exc:  # noqa: BLE001
            span_blocker = str(exc)
            run_span = False

    alignment_ok = sequential["cache_alignment_ok"]
    if span is not None:
        alignment_ok = alignment_ok and span["cache_alignment_ok"]

    suite = prompt_entry.get("v10_suite", prompt_entry.get("category", "unknown"))

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": suite,
        "category": prompt_entry.get("category", suite),
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_name": runtime.model_name,
        "full_output_ids": full_ids,
        "sequential": {
            **sequential,
            "exactkv_failure": not seq_exact,
            "token_exact_match_full": seq_exact,
        },
        "span": span,
        "span_attempted": run_span or span is not None,
        "span_blocker": span_blocker,
        "span_matches_sequential": span_seq_parity,
        "counters_match": counters_match,
        "cache_alignment_ok": alignment_ok,
        "exactkv_failure_sequential": not seq_exact,
        "exactkv_failure_span": (not span_exact) if span_exact is not None else None,
        "rejected_tokens_never_committed": seq_exact
        and (span_exact is None or span_exact),
    }


def _mean_accept(results: list[dict[str, Any]], key: str = "sequential") -> float:
    if not results:
        return 0.0
    return sum(r[key]["acceptance_rate"] for r in results) / len(results)


def _aggregate(results: list[dict[str, Any]], *, span_enabled: bool) -> dict[str, Any]:
    seq_fail = sum(1 for r in results if r["exactkv_failure_sequential"])
    span_results = [r for r in results if r.get("span") is not None]
    span_fail = sum(1 for r in span_results if r["exactkv_failure_span"])
    parity_fail = sum(
        1 for r in span_results if not r.get("span_matches_sequential", True)
    )
    counter_mismatch = sum(
        1 for r in span_results if not r.get("counters_match", True)
    )
    alignment_fail = sum(1 for r in results if not r["cache_alignment_ok"])
    span_blockers = [r for r in results if r.get("span_blocker")]

    by_draft: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_draft[r["draft_len"]].append(r)
        by_comp[r["compressor_name"]].append(r)
        by_suite[r["v10_suite"]].append(r)

    def _bucket(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        span_bucket = [r for r in bucket if r.get("span") is not None]
        return {
            "cells": len(bucket),
            "sequential_exactkv_failures": sum(
                1 for r in bucket if r["exactkv_failure_sequential"]
            ),
            "span_exactkv_failures": sum(
                1 for r in span_bucket if r["exactkv_failure_span"]
            ),
            "span_sequential_parity_failures": sum(
                1 for r in span_bucket if not r.get("span_matches_sequential", True)
            ),
            "mean_sequential_acceptance": _mean_accept(bucket, "sequential"),
            "mean_span_acceptance": _mean_accept(span_bucket, "span")
            if span_bucket
            else None,
            "cache_alignment_failures": sum(
                1 for r in bucket if not r["cache_alignment_ok"]
            ),
        }

    exactness_gate_passed = (
        seq_fail == 0
        and (span_fail == 0 if span_enabled else True)
        and (parity_fail == 0 if span_enabled else True)
        and alignment_fail == 0
    )

    return {
        "total_cells": len(results),
        "sequential_exactkv_failures": seq_fail,
        "span_exactkv_failures": span_fail,
        "span_sequential_parity_failures": parity_fail,
        "counter_mismatch_cells": counter_mismatch,
        "cache_alignment_failures": alignment_fail,
        "span_blocker_cells": len(span_blockers),
        "all_span_match_sequential": parity_fail == 0,
        "all_sequential_match_full": seq_fail == 0,
        "all_span_match_full": span_fail == 0 if span_enabled else None,
        "all_cache_alignment_ok": alignment_fail == 0,
        "exactness_gate_passed": exactness_gate_passed,
        "mean_sequential_acceptance_rate": _mean_accept(results, "sequential"),
        "by_draft_len": {str(k): _bucket(v) for k, v in sorted(by_draft.items())},
        "by_compressor": {k: _bucket(v) for k, v in sorted(by_comp.items())},
        "by_v10_suite": {k: _bucket(v) for k, v in sorted(by_suite.items())},
    }


def run_experiment(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    draft_lens: list[int],
    *,
    span_enabled: bool,
) -> dict[str, Any]:
    compressor_cache: dict[str, Any] = {}
    full_cache: dict[str, list[int]] = {}
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(compressors) * len(draft_lens)
    idx = 0

    for pe in prompts:
        pid = pe["prompt_id"]
        if pid not in full_cache:
            print(f"  full greedy baseline: {pid}", flush=True)
            full_res = generate_full_greedy(runtime, pe["prompt"], MAX_NEW_TOKENS)
            full_cache[pid] = full_res.generated_ids.squeeze(0).tolist()

        for comp_name in compressors:
            compressor = _get_compressor(runtime, comp_name, compressor_cache)
            for draft_len in draft_lens:
                idx += 1
                print(
                    f"  [{idx}/{total}] {pid} × {comp_name} × draft_len={draft_len}",
                    flush=True,
                )
                results.append(
                    run_one_cell(
                        runtime,
                        pe,
                        comp_name,
                        compressor,
                        draft_len,
                        full_cache[pid],
                        run_span=span_enabled,
                    )
                )

    agg = _aggregate(results, span_enabled=span_enabled)
    return {
        "experiment": "033",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": runtime.dtype_str,
        "prompt_count": len(prompts),
        "prompt_panel": STRATIFIED_SUITES,
        "compressors": compressors,
        "draft_lens": draft_lens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "verification_methods": ["sequential"] + (["span"] if span_enabled else []),
        "results": results,
        "aggregate": agg,
    }


def write_csv_report(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for r in report.get("results", []):
        row = {
            "prompt_id": r["prompt_id"],
            "v10_suite": r["v10_suite"],
            "compressor_name": r["compressor_name"],
            "draft_len": r["draft_len"],
            "sequential_exactkv_failure": r["exactkv_failure_sequential"],
            "span_exactkv_failure": r.get("exactkv_failure_span"),
            "span_matches_sequential": r.get("span_matches_sequential"),
            "cache_alignment_ok": r["cache_alignment_ok"],
            "sequential_accepted": r["sequential"]["total_accepted"],
            "sequential_rejected": r["sequential"]["total_rejected"],
            "sequential_corrections": r["sequential"]["total_corrections"],
        }
        if r.get("span"):
            row["span_accepted"] = r["span"]["total_accepted"]
            row["span_rejected"] = r["span"]["total_rejected"]
            row["span_corrections"] = r["span"]["total_corrections"]
        rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    report: dict[str, Any],
    path: Path,
    *,
    environment: dict[str, Any],
    model_access: dict[str, Any],
    snapkv_status: dict[str, Any],
    blocked: bool = False,
) -> None:
    agg = report.get("aggregate", {})
    compressors = report.get("compressors", REQUIRED_COMPRESSORS)
    passed = bool(agg.get("exactness_gate_passed")) and not blocked

    lines = [
        "# Experiment 033: Llama-3.1-8B Small-Suite Validation",
        "",
        "_Generated by `scripts/run_experiment_033_llama31_8b_small_suite.py`. "
        "V13 Phase 6 — larger-model exactness validation only._",
        "",
        f"**Status:** {'PASS' if passed else 'BLOCKED / FAIL'}",
        "",
        "> This is a **larger-model exactness validation**, not a timing benchmark.",
        "> This does **not** claim speedup, throughput, latency, runtime, tokens/sec, "
        "active GPU memory savings, production serving, or model accuracy improvement.",
        "> ExactKV's full-KV verifier remains authoritative.",
        "> External Llama, Shard, SnapKV, SpectralQuant, or kvpress results are "
        "not ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate that ExactKV preserves exact greedy output on a public-legible "
        "larger model (`meta-llama/Llama-3.1-8B-Instruct`) under a small stratified "
        "V10 prompt panel.",
        "",
        "## 2. Why Phase 6 follows SnapKV Phase 5b",
        "",
        "Phase 5b proved factory-only `snapkv_experimental` on Qwen2.5-0.5B. "
        "Phase 6 tests whether the exactness invariant holds on Llama-3.1-8B for "
        "public legibility — without timing or VRAM claims.",
        "",
        "## 3. Model and environment",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | `{MODEL_NAME}` |",
        f"| Model access | {'OK' if model_access.get('accessible') else 'BLOCKED'} |",
        f"| Gated | {model_access.get('gated')} |",
        f"| HF token present | {model_access.get('token_present')} |",
        f"| Local cache hit | {model_access.get('local_cache_hit')} |",
        f"| dtype | `{report.get('dtype', '—')}` |",
        f"| device | `{report.get('device', environment.get('device', 'cuda'))}` |",
        f"| GPU | {environment.get('gpu_name', 'unknown')} |",
        f"| VRAM | {environment.get('gpu_memory_total', 'unknown')} |",
        f"| torch | {environment.get('torch')} |",
        f"| transformers | {environment.get('transformers')} |",
        f"| CUDA | {environment.get('cuda_version')} |",
        "",
    ]
    if model_access.get("error"):
        lines.extend([
            f"**Access error:** {model_access['error']}",
            "",
        ])

    lines.extend([
        "## 4. Prompt panel",
        "",
        "12 prompts: 3× `core_v2`, 3× `long_context`, 3× `retrieval_copy`, "
        "3× `tool_json`.",
        "",
        f"| Prompt count | **{report.get('prompt_count', 12)}** |",
        "",
        "## 5. Compressor panel",
        "",
        ", ".join(f"`{c}`" for c in compressors),
        "",
        "## 6. Verification methods",
        "",
        f"{', '.join(report.get('verification_methods', ['sequential']))} "
        f"(draft_lens={report.get('draft_lens', [DRAFT_LEN_DEFAULT])})",
        "",
        "## 7. Exactness gate",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Sequential ExactKV failures | **{agg.get('sequential_exactkv_failures', 'N/A')}** |",
        f"| Span ExactKV failures | **{agg.get('span_exactkv_failures', 'N/A')}** |",
        f"| exactness_gate_passed | **{agg.get('exactness_gate_passed', False)}** |",
        "",
        "## 8. Sequential result",
        "",
        f"all_sequential_match_full: **{agg.get('all_sequential_match_full', False)}**",
        "",
        "## 9. Span result",
        "",
    ])
    if report.get("verification_methods", []) == ["sequential"]:
        lines.append("Span not run (blocked or disabled).")
    else:
        lines.append(
            f"all_span_match_full: **{agg.get('all_span_match_full')}**; "
            f"span_blocker_cells: **{agg.get('span_blocker_cells', 0)}**"
        )

    lines.extend([
        "",
        "## 10. Span vs sequential parity",
        "",
        f"| Parity failures | **{agg.get('span_sequential_parity_failures', 'N/A')}** |",
        f"| All span match sequential | **{agg.get('all_span_match_sequential', 'N/A')}** |",
        "",
        "## 11. Acceptance/rejection/correction summary",
        "",
        f"Mean sequential acceptance: **{agg.get('mean_sequential_acceptance_rate', 0):.4f}**",
        "",
        "## 12. Results by prompt category",
        "",
        "| suite | cells | seq fail | span fail | parity fail |",
        "|---|---:|---:|---:|---:|",
    ])
    for suite, stats in agg.get("by_v10_suite", {}).items():
        lines.append(
            f"| `{suite}` | {stats['cells']} | {stats['sequential_exactkv_failures']} | "
            f"{stats['span_exactkv_failures']} | {stats['span_sequential_parity_failures']} |"
        )

    lines.extend([
        "",
        "## 13. Results by compressor",
        "",
        "| compressor | cells | seq fail | span fail | parity fail | mean accept |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for comp, stats in agg.get("by_compressor", {}).items():
        lines.append(
            f"| `{comp}` | {stats['cells']} | {stats['sequential_exactkv_failures']} | "
            f"{stats['span_exactkv_failures']} | {stats['span_sequential_parity_failures']} | "
            f"{stats['mean_sequential_acceptance']:.4f} |"
        )

    lines.extend([
        "",
        "## 14. SnapKV experimental status",
        "",
    ])
    if snapkv_status.get("included"):
        lines.append("`snapkv_experimental` included in compressor panel.")
    else:
        reason = snapkv_status.get("skip_reason") or "not attempted"
        lines.append(
            f"`snapkv_experimental` **skipped**: {reason}"
        )

    lines.extend([
        "",
        "## 15. Any blockers",
        "",
    ])
    if blocked:
        lines.append(
            f"- **BLOCKER:** {model_access.get('error') or 'Llama model access unavailable'}"
        )
    elif not passed:
        lines.append("- Exactness or span parity gate failed — see aggregate.")
    else:
        lines.append("- None on this panel.")

    lines.extend([
        "",
        "## 16. What this proves",
        "",
    ])
    if passed:
        lines.extend([
            "- ExactKV preserves exact greedy output on Llama-3.1-8B for this panel.",
            "- Span verification matches sequential when both are run.",
        ])
    else:
        lines.append("- Experiment did not complete the exactness gate.")

    lines.extend([
        "",
        "## 17. What this does not prove",
        "",
        "- No speedup, throughput, latency, runtime, tokens/sec, or active GPU memory savings.",
        "- No production serving readiness or model accuracy improvement.",
        "- No Shard or SpectralQuant integration.",
        "- No production SnapKV claim.",
        "",
        "## 18. Recommendation for Phase 6b Shard probe or Phase 7 killer demo",
        "",
    ])
    if passed:
        lines.append(
            "**Proceed to Phase 7 killer correction demo (Exp 034).** Optional "
            "Phase 6b Shard Llama external-drafter probe may run in parallel on "
            "this Llama panel."
        )
    elif blocked:
        lines.append(
            "**Resolve Llama gated access** (HF token + license acceptance), re-run "
            "Exp 033, then proceed to Phase 7 or Phase 6b Shard probe."
        )
    else:
        lines.append(
            "Fix exactness failures before Phase 7 or Phase 6b Shard probe."
        )

    lines.extend([
        "",
        "---",
        "",
        "Restricted larger-model validation only. Full-KV verifier remains authoritative. "
        "`snapkv_experimental` remains factory-only if used.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_blocker_artifacts(
    *,
    environment: dict[str, Any],
    model_access: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    md_path: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "experiment": "033",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "status": "blocked",
        "blocker": model_access,
        "environment": environment,
        "prompt_panel": STRATIFIED_SUITES,
        "compressors": REQUIRED_COMPRESSORS,
        "results": [],
        "aggregate": {
            "exactness_gate_passed": False,
            "sequential_exactkv_failures": None,
            "blocked": True,
        },
    }
    _assert_no_forbidden(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(
        report,
        md_path,
        environment=environment,
        model_access=model_access,
        snapkv_status={"included": False, "skip_reason": "experiment blocked before run"},
        blocked=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 033 Llama-3.1-8B small suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--include-draft-len-8",
        action="store_true",
        help="Also run draft_len=8 if runtime is stable",
    )
    parser.add_argument(
        "--try-snapkv",
        action="store_true",
        help="Attempt optional snapkv_experimental (skipped on VRAM blockers)",
    )
    parser.add_argument(
        "--no-span",
        action="store_true",
        help="Sequential only (use only when span is incompatible)",
    )
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_033_llama31_8b_small_suite.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_033_llama31_8b_small_suite.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_033_LLAMA31_8B_SMALL_SUITE.md"),
    )
    args = parser.parse_args()

    _require_cuda(args.device)
    environment = collect_environment(args.device)
    model_access = check_model_access(MODEL_NAME)

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)

    if not model_access["accessible"]:
        print(f"BLOCKER: cannot access {MODEL_NAME}: {model_access.get('error')}")
        _write_blocker_artifacts(
            environment=environment,
            model_access=model_access,
            json_path=json_path,
            csv_path=csv_path,
            md_path=md_path,
        )
        print(f"Wrote blocker report {md_path}")
        return 2

    dtype = args.dtype or _default_dtype(args.device)
    draft_lens = [DRAFT_LEN_DEFAULT]
    if args.include_draft_len_8:
        draft_lens.append(8)

    prompts = load_exp033_prompt_panel()
    print(
        f"Experiment 033 — {MODEL_NAME} — {len(prompts)} prompts on {args.device} ({dtype})"
    )
    runtime = ModelRuntime(model_name=MODEL_NAME, device=args.device, dtype=dtype)

    compressors, snapkv_status = _resolve_compressors(runtime, try_snapkv=args.try_snapkv)
    if snapkv_status.get("skip_reason") and args.try_snapkv:
        print(f"snapkv_experimental skipped: {snapkv_status['skip_reason']}")

    span_enabled = not args.no_span
    report = run_experiment(
        runtime,
        prompts,
        compressors,
        draft_lens,
        span_enabled=span_enabled,
    )
    report["environment"] = environment
    report["model_access"] = model_access
    report["snapkv_status"] = snapkv_status
    report["status"] = "complete"

    _assert_no_forbidden(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(
        report,
        md_path,
        environment=environment,
        model_access=model_access,
        snapkv_status=snapkv_status,
        blocked=False,
    )

    agg = report["aggregate"]
    print(
        f"Done: seq_fail={agg['sequential_exactkv_failures']} "
        f"span_fail={agg['span_exactkv_failures']} "
        f"parity_fail={agg['span_sequential_parity_failures']} "
        f"gate={agg['exactness_gate_passed']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if agg["exactness_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
