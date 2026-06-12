#!/usr/bin/env python3
"""Experiment 029: span verification exactness grid (V13).

Exactness/parity grid — sequential vs span vs full greedy.  No timing,
throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_all_v10_prompts, load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 16
DRAFT_LENS = (2, 4, 8)
EXPERIMENT_CLASS = "v13_span_verification_grid"

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "backend_passthrough",
]

STRATIFIED_SUITES: list[tuple[str, int]] = [
    ("core_v2", 10),
    ("long_context", 10),
    ("retrieval_copy", 10),
    ("tool_json", 10),
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


def load_exp029_prompt_panel(*, full_suite: bool = False) -> list[dict[str, Any]]:
    if full_suite:
        return load_all_v10_prompts()
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
    compressor_name: str,
    draft_len: int,
    verification_method: str,
) -> dict[str, Any]:
    comp = get_compressor(compressor_name)
    gen = ExactKVGenerator(
        runtime,
        comp,
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


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
    compressor_name: str,
    draft_len: int,
    full_ids: list[int],
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=runtime.device)

    sequential = _run_exactkv(
        runtime, prompt, compressor_name, draft_len, "sequential"
    )
    span = _run_exactkv(
        runtime, prompt, compressor_name, draft_len, "span"
    )

    seq_ids = sequential["output_ids"]
    span_ids = span["output_ids"]

    seq_exact = token_exact_match(full_tensor, torch.tensor([seq_ids]))
    span_exact = token_exact_match(full_tensor, torch.tensor([span_ids]))
    span_seq_parity = seq_ids == span_ids
    counters_match = (
        sequential["total_accepted"] == span["total_accepted"]
        and sequential["total_rejected"] == span["total_rejected"]
        and sequential["total_corrections"] == span["total_corrections"]
    )
    alignment_ok = (
        sequential["cache_alignment_ok"]
        and span["cache_alignment_ok"]
    )

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "compressor_name": compressor_name,
        "draft_len": draft_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_name": runtime.model_name,
        "sequential": {
            **sequential,
            "exactkv_failure": not seq_exact,
            "token_exact_match_full": seq_exact,
        },
        "span": {
            **span,
            "exactkv_failure": not span_exact,
            "token_exact_match_full": span_exact,
        },
        "span_matches_sequential": span_seq_parity,
        "counters_match": counters_match,
        "cache_alignment_ok": alignment_ok,
        "exactkv_failure_sequential": not seq_exact,
        "exactkv_failure_span": not span_exact,
    }


def _mean_accept(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(r[key]["acceptance_rate"] for r in results) / len(results)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    seq_fail = sum(1 for r in results if r["exactkv_failure_sequential"])
    span_fail = sum(1 for r in results if r["exactkv_failure_span"])
    parity_fail = sum(1 for r in results if not r["span_matches_sequential"])
    counter_mismatch = sum(1 for r in results if not r["counters_match"])
    alignment_fail = sum(1 for r in results if not r["cache_alignment_ok"])

    by_draft: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_draft[r["draft_len"]].append(r)
        by_comp[r["compressor_name"]].append(r)

    def _bucket_stats(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cells": len(bucket),
            "sequential_exactkv_failures": sum(
                1 for r in bucket if r["exactkv_failure_sequential"]
            ),
            "span_exactkv_failures": sum(
                1 for r in bucket if r["exactkv_failure_span"]
            ),
            "span_sequential_parity_failures": sum(
                1 for r in bucket if not r["span_matches_sequential"]
            ),
            "mean_sequential_acceptance": _mean_accept(bucket, "sequential"),
            "mean_span_acceptance": _mean_accept(bucket, "span"),
            "cache_alignment_failures": sum(
                1 for r in bucket if not r["cache_alignment_ok"]
            ),
        }

    return {
        "total_cells": len(results),
        "generator_runs": len(results) * 2,
        "sequential_exactkv_failures": seq_fail,
        "span_exactkv_failures": span_fail,
        "span_sequential_parity_failures": parity_fail,
        "counter_mismatch_cells": counter_mismatch,
        "cache_alignment_failures": alignment_fail,
        "all_span_match_sequential": parity_fail == 0,
        "all_span_match_full": span_fail == 0,
        "all_sequential_match_full": seq_fail == 0,
        "all_cache_alignment_ok": alignment_fail == 0,
        "mean_sequential_acceptance_rate": _mean_accept(results, "sequential"),
        "mean_span_acceptance_rate": _mean_accept(results, "span"),
        "by_draft_len": {str(k): _bucket_stats(v) for k, v in sorted(by_draft.items())},
        "by_compressor": {k: _bucket_stats(v) for k, v in sorted(by_comp.items())},
        "phase3_timing_allowed": (
            seq_fail == 0
            and span_fail == 0
            and parity_fail == 0
            and counter_mismatch == 0
            and alignment_fail == 0
        ),
    }


def run_grid(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    compressors: list[str],
) -> dict[str, Any]:
    full_cache: dict[str, list[int]] = {}
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(compressors) * len(DRAFT_LENS)
    idx = 0

    for pe in prompts:
        pid = pe["prompt_id"]
        if pid not in full_cache:
            full_res = generate_full_greedy(runtime, pe["prompt"], MAX_NEW_TOKENS)
            full_cache[pid] = full_res.generated_ids.squeeze(0).tolist()

        for comp in compressors:
            for draft_len in DRAFT_LENS:
                idx += 1
                print(
                    f"  [{idx}/{total}] {pid} × {comp} × draft_len={draft_len}",
                    flush=True,
                )
                results.append(
                    run_one_cell(
                        runtime,
                        pe,
                        comp,
                        draft_len,
                        full_cache[pid],
                    )
                )

    agg = _aggregate(results)
    return {
        "experiment": "029",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": runtime.model_name,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "prompt_panel": (
            "v10_full_128" if len(prompts) == 128 else "v10_stratified_40"
        ),
        "prompt_count": len(prompts),
        "compressors": compressors,
        "draft_lens": list(DRAFT_LENS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "verification_methods": ["sequential", "span"],
        "results": results,
        "aggregate": agg,
    }


def write_csv_report(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for r in report["results"]:
        rows.append({
            "prompt_id": r["prompt_id"],
            "v10_suite": r["v10_suite"],
            "compressor_name": r["compressor_name"],
            "draft_len": r["draft_len"],
            "sequential_exactkv_failure": r["exactkv_failure_sequential"],
            "span_exactkv_failure": r["exactkv_failure_span"],
            "span_matches_sequential": r["span_matches_sequential"],
            "counters_match": r["counters_match"],
            "cache_alignment_ok": r["cache_alignment_ok"],
            "sequential_accepted": r["sequential"]["total_accepted"],
            "sequential_rejected": r["sequential"]["total_rejected"],
            "sequential_corrections": r["sequential"]["total_corrections"],
            "span_accepted": r["span"]["total_accepted"],
            "span_rejected": r["span"]["total_rejected"],
            "span_corrections": r["span"]["total_corrections"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    compressors = report["compressors"]
    lines = [
        "# Experiment 029: Span Verification Exactness Grid",
        "",
        "_Generated by `scripts/run_experiment_029_span_verification_grid.py`. "
        "V13 — exactness/parity grid only._",
        "",
        "> This is an **exactness/parity grid**, not a timing benchmark.",
        "> This does **not** prove speedup.",
        "> This does **not** measure throughput, latency, runtime, tokens/sec, "
        "active GPU memory, or production serving.",
        "> Span verification remains **opt-in**, not default.",
        "> ExactKV does **not** claim model accuracy improvement.",
        "> Phase 3 timing is allowed **only if** this grid has zero exactness/parity failures.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate span verification beyond Experiment 028 smoke on a larger panel "
        "with multiple draft lengths, compressors, and verification modes.",
        "",
        "## 2. Why this follows Experiment 028",
        "",
        "Experiment 028 proved span ≡ sequential on 32 smoke cells. Experiment 029 "
        "extends to stratified V10 prompts × draft_len {2,4,8} before Phase 3 timing.",
        "",
        "## 3. Model/environment",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | `{report['model_name']}` |",
        f"| dtype | `{report['dtype']}` |",
        f"| device | `{report['device']}` |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_029_span_verification_grid.py",
        "```",
        "",
        "Full 128-prompt panel:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_029_span_verification_grid.py --full-suite",
        "```",
        "",
        "## 4. Prompt panel",
        "",
        f"| Panel | `{report['prompt_panel']}` |",
        f"| Prompt count | **{report['prompt_count']}** |",
        "",
        "Stratified default: 10× `core_v2`, 10× `long_context`, 10× `retrieval_copy`, "
        "10× `tool_json`.",
        "",
        "## 5. Compressor panel",
        "",
        ", ".join(f"`{c}`" for c in compressors),
        "",
        "## 6. Grid configuration",
        "",
        f"| draft_len | {', '.join(str(d) for d in report['draft_lens'])} |",
        f"| max_new_tokens | {report['max_new_tokens']} |",
        f"| Generator runs (seq + span) | **{agg['generator_runs']}** |",
        f"| Grid cells | **{agg['total_cells']}** |",
        "",
        "## 7. Exactness result",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Sequential ExactKV failures | **{agg['sequential_exactkv_failures']}** |",
        f"| Span ExactKV failures | **{agg['span_exactkv_failures']}** |",
        "",
        "## 8. Sequential vs span output parity",
        "",
        f"| Parity failures | **{agg['span_sequential_parity_failures']}** |",
        f"| All span match sequential | **{agg['all_span_match_sequential']}** |",
        "",
        "## 9. Span vs full greedy result",
        "",
        f"| Span failures vs full greedy | **{agg['span_exactkv_failures']}** |",
        "",
        "## 10. Sequential vs full greedy result",
        "",
        f"| Sequential failures vs full greedy | **{agg['sequential_exactkv_failures']}** |",
        "",
        "## 11. Acceptance/rejection/correction comparison",
        "",
        f"| Mode | Mean acceptance |",
        f"|---|---:|",
        f"| Sequential | {agg['mean_sequential_acceptance_rate']:.4f} |",
        f"| Span | {agg['mean_span_acceptance_rate']:.4f} |",
        f"| Counter mismatch cells | **{agg['counter_mismatch_cells']}** |",
        "",
        "## 12. Results by draft_len",
        "",
        "| draft_len | cells | seq fail | span fail | parity fail | mean accept (seq) | mean accept (span) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dl, stats in agg["by_draft_len"].items():
        lines.append(
            f"| {dl} | {stats['cells']} | {stats['sequential_exactkv_failures']} | "
            f"{stats['span_exactkv_failures']} | {stats['span_sequential_parity_failures']} | "
            f"{stats['mean_sequential_acceptance']:.4f} | {stats['mean_span_acceptance']:.4f} |"
        )
    lines.extend([
        "",
        "## 13. Results by compressor",
        "",
        "| compressor | cells | seq fail | span fail | parity fail | mean accept (seq) | mean accept (span) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for comp, stats in agg["by_compressor"].items():
        lines.append(
            f"| `{comp}` | {stats['cells']} | {stats['sequential_exactkv_failures']} | "
            f"{stats['span_exactkv_failures']} | {stats['span_sequential_parity_failures']} | "
            f"{stats['mean_sequential_acceptance']:.4f} | {stats['mean_span_acceptance']:.4f} |"
        )
    lines.extend([
        "",
        "## 14. Cache alignment result",
        "",
        f"| Cache alignment failures | **{agg['cache_alignment_failures']}** |",
        f"| All cells aligned | **{agg['all_cache_alignment_ok']}** |",
        "",
        "## 15. Any mismatches/bugs found and fixes",
        "",
        "None — grid passed with zero exactness/parity/alignment failures."
        if agg["phase3_timing_allowed"]
        else "Failures recorded in aggregate; Phase 3 timing **not** allowed until resolved.",
        "",
        "## 16. What this proves",
        "",
        "- Span verification preserves exact greedy output on the grid panel.",
        "- Span outputs match sequential outputs and acceptance counters cell-for-cell.",
        "- Cache alignment invariant holds for both verification modes.",
        "",
        "## 17. What this does not prove",
        "",
        "- Speedup or reduced verifier overhead (Phase 3 diagnostic timing only).",
        "- Universal benchmark coverage beyond this panel.",
        "- Production serving readiness.",
        "",
        "## 18. Whether Phase 3 timing harness is now allowed",
        "",
        f"**{'Yes' if agg['phase3_timing_allowed'] else 'No'}** — "
        f"`phase3_timing_allowed={agg['phase3_timing_allowed']}`. "
        "Phase 3 may proceed only with zero exactness/parity failures on this grid.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_dtype(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 029 span exactness grid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Use all 128 V10 prompts instead of stratified 40",
    )
    parser.add_argument(
        "--report-json",
        default=str(_ROOT / "reports" / "experiment_029_span_verification_grid.json"),
    )
    parser.add_argument(
        "--report-csv",
        default=str(_ROOT / "reports" / "experiment_029_span_verification_grid.csv"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_029_SPAN_VERIFICATION_GRID.md"),
    )
    args = parser.parse_args()

    dtype = args.dtype or _default_dtype(args.device)
    prompts = load_exp029_prompt_panel(full_suite=args.full_suite)
    n_cells = len(prompts) * len(COMPRESSORS) * len(DRAFT_LENS)
    print(
        f"Experiment 029 — {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"× {len(DRAFT_LENS)} draft_lens = {n_cells} cells "
        f"({n_cells * 2} generator runs)"
    )
    runtime = ModelRuntime(model_name=MODEL_NAME, device=args.device, dtype=dtype)
    report = run_grid(runtime, prompts, COMPRESSORS)
    _assert_no_forbidden(report)

    json_path = Path(args.report_json)
    csv_path = Path(args.report_csv)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv_report(report, csv_path)
    write_markdown(report, md_path)

    agg = report["aggregate"]
    print(
        f"Done: seq_fail={agg['sequential_exactkv_failures']} "
        f"span_fail={agg['span_exactkv_failures']} "
        f"parity_fail={agg['span_sequential_parity_failures']} "
        f"phase3_allowed={agg['phase3_timing_allowed']}"
    )
    return 0 if agg["phase3_timing_allowed"] else 1


if __name__ == "__main__":
    sys.exit(main())
