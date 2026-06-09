#!/usr/bin/env python3
"""Experiment 007: harness-based serving-context compatibility evaluation (V8 Phase D).

Mode B only — restricted local ServingCacheLifecycleHarness.  No vLLM, LMCache,
or PagedAttention integration.  No timing, throughput, latency, speedup, or
runtime_seconds fields.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.prompts import load_core_prompts
from exactkv.benchmarks.reports import (
    build_run_manifest,
    load_json_report,
    validate_report,
    write_csv_report,
    write_json_report,
)
from exactkv.benchmarks.runner import RunConfig
from exactkv.benchmarks.sweeps import _compute_aggregate
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.serving.cache_lifecycle import AUTHORITATIVE_FULL, ServingCacheLifecycleHarness

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
PROMPT_SUITE = "core"
EXPERIMENT_CLASS = "harness_sim"
HARNESS_BLOCK_SIZE = 16

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "k_full_v8",
    "k8_v_full",
    "backend_passthrough",
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden_fields(obj: Any, path: str = "report") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict,
    config: RunConfig,
) -> dict[str, Any]:
    """One benchmark cell with standard metrics plus serving-harness metadata."""
    compressor = get_compressor(config.compressor_name)
    caps_dict: dict = {}
    if hasattr(compressor, "capabilities"):
        caps_dict = asdict(compressor.capabilities)

    prompt = prompt_entry["prompt"]
    max_new = config.max_new_tokens

    full_res = generate_full_greedy(runtime, prompt, max_new)
    full_ids = full_res.generated_ids.squeeze(0).tolist()

    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()
    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    harness = ServingCacheLifecycleHarness(block_size=HARNESS_BLOCK_SIZE)
    full_state = prefill_to_full_state(runtime, prompt)
    harness.register_authoritative_full(full_state)
    compressed = compressor.compress(full_state)
    harness.register_compressed_cache(compressed, compressor=compressor)
    harness.validate_invariants()
    initial_logical = full_state.seq_len

    ekv_res = ExactKVGenerator(
        runtime, compressor, draft_len=config.draft_len
    ).generate(prompt, max_new)
    for trace in ekv_res.traces:
        committed = trace.acceptance.num_accepted
        if trace.acceptance.correction_token is not None:
            committed += 1
        harness.append_committed_tokens(committed)
        harness.validate_invariants()

    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    summary = harness.summarize()
    auth_entry = next(
        e for e in summary["entries"] if e["owner"] == AUTHORITATIVE_FULL
    )
    comp_entry = next(
        e for e in summary["entries"] if e["owner"] == "compressed_draft"
    )

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
        "model_name": runtime.model_name,
        "compressor_name": config.compressor_name,
        "compressor_capabilities": caps_dict,
        "draft_len": config.draft_len,
        "max_new_tokens": max_new,
        "full": {
            "output_ids": full_ids,
            "output_text": full_res.output_text,
        },
        "lossy": {
            "output_ids": lossy_ids,
            "output_text": lossy_res.output_text,
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
        },
        "exactkv": {
            "output_ids": ekv_ids,
            "output_text": ekv_res.output_text,
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
        "serving_harness": {
            "experiment_class": EXPERIMENT_CLASS,
            "invariants_valid": summary["invariants_valid"],
            "verification_uses": summary["verification_uses"],
            "authoritative_cache_id": summary["authoritative_cache_id"],
            "compressed_cache_id": summary["compressed_cache_id"],
            "owners_separate": summary["authoritative_cache_id"]
            != summary["compressed_cache_id"],
            "initial_logical_seq_len": initial_logical,
            "final_logical_seq_len": summary["authoritative_logical_seq_len"],
            "authoritative_physical_seq_len": auth_entry["physical_seq_len"],
            "compressed_physical_seq_len": comp_entry["physical_seq_len"],
            "identity_mapping": (
                auth_entry["physical_seq_len"] == auth_entry["logical_seq_len"]
                and comp_entry["physical_seq_len"] == comp_entry["logical_seq_len"]
            ),
            "block_count_authoritative": len(auth_entry["blocks"]),
            "block_count_compressed": len(comp_entry["blocks"]),
            "commit_rounds": len(ekv_res.traces),
            "stored_kv_bytes": comp_entry.get("stored_kv_bytes"),
            "materialized_working_kv_bytes": comp_entry.get(
                "materialized_working_kv_bytes"
            ),
            "total_kv_footprint_bytes": comp_entry.get("total_kv_footprint_bytes"),
            "supports_real_bytes_claim": comp_entry.get("supports_real_bytes_claim"),
            "is_simulated": comp_entry.get("is_simulated"),
            "exactkv_token_match": ekv_exact,
        },
    }


def _compute_harness_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    harness_rows = [r["serving_harness"] for r in results]
    return {
        "experiment_class": EXPERIMENT_CLASS,
        "all_invariants_valid": all(h["invariants_valid"] for h in harness_rows),
        "all_verification_uses_authoritative_full": all(
            h["verification_uses"] == AUTHORITATIVE_FULL for h in harness_rows
        ),
        "all_owners_separate": all(h["owners_separate"] for h in harness_rows),
        "all_identity_mapping": all(h["identity_mapping"] for h in harness_rows),
        "all_exactkv_token_match": all(h["exactkv_token_match"] for h in harness_rows),
        "mean_final_logical_seq_len": sum(
            h["final_logical_seq_len"] for h in harness_rows
        )
        / max(len(harness_rows), 1),
        "cells_with_commit_rounds": sum(1 for h in harness_rows if h["commit_rounds"] > 0),
        "total_commit_rounds": sum(h["commit_rounds"] for h in harness_rows),
    }


def run_experiment_007(
    runtime: ModelRuntime,
    prompts: list[dict],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total_cells = len(prompts) * len(COMPRESSORS)

    for i, prompt_entry in enumerate(prompts):
        for compressor_name in COMPRESSORS:
            cell_idx = i * len(COMPRESSORS) + COMPRESSORS.index(compressor_name) + 1
            print(
                f"  [{cell_idx}/{total_cells}] {prompt_entry['prompt_id']} × "
                f"{compressor_name}",
                flush=True,
            )
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results.append(run_one_cell(runtime, prompt_entry, config))

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite=PROMPT_SUITE,
        compressor_names=COMPRESSORS,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=DTYPE,
    )
    manifest["experiment"] = "007_serving_context"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["serving_mode"] = "harness_sim"
    manifest["harness_block_size"] = HARNESS_BLOCK_SIZE
    manifest["phase_c_status"] = "no-go_deferred"

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["serving_harness_gates"] = _compute_harness_aggregate(results)

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    harness_gates = agg["serving_harness_gates"]
    by_comp = agg["acceptance_by_compressor"]

    lines = [
        "# Experiment 007: Serving-Context Compatibility (Harness Mode)",
        "",
        "_Generated by `scripts/run_experiment_007_serving_context.py`. "
        "V8 Phase D — Mode B harness evaluation only._",
        "",
        "> This is a **restricted local serving/cache-lifecycle harness**.",
        "> This is **not** vLLM integration.",
        "> This is **not** LMCache integration.",
        "> This is **not** PagedAttention integration.",
        "> It does **not** claim production serving behavior.",
        "> It does **not** measure throughput, latency, speedup, runtime, or tokens/sec.",
        "> It does **not** report active GPU memory.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, not measured "
        "peak GPU memory.",
        "> ExactKV preserves the exactness gate: `exactkv_output_ids == full_output_ids`.",
        "> External serving-stack claims are not ExactKV results.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Evaluate whether ExactKV's verified compressed-KV workflow remains correct and",
        "measurable when wrapped in the Phase B **ServingCacheLifecycleHarness** —",
        "modelling serving-style cache ownership, logical/physical sequence mapping,",
        "block tables, and append-after-commit lifecycle without external serving stacks.",
        "",
        "## 2. Why harness mode was chosen after Phase A no-go",
        "",
        "Phase A ([`docs/SERVING_CONTEXT_FEASIBILITY.md`](SERVING_CONTEXT_FEASIBILITY.md))",
        "concluded **no-go** for vLLM and LMCache Phase C integration: authoritative",
        "full-precision KV is not safely exportable for ExactKV verification, and tiering",
        "semantics worsen ownership clarity. Experiment 007 uses **Mode B** (harness-only)",
        "as the approved V8 evaluation path.",
        "",
        "## 3. Model and prompt suite",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{MODEL_NAME}`, {DTYPE}, CPU-first |",
        f"| Prompt suite | `{PROMPT_SUITE}` (34 prompts) |",
        f"| `draft_len` | {DRAFT_LEN} |",
        f"| `max_new_tokens` | {MAX_NEW_TOKENS} |",
        f"| Experiment class | `{EXPERIMENT_CLASS}` |",
        f"| Total cells | **{agg['total_runs']}** |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_007_serving_context.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_007_serving_context.json`,",
        "`reports/experiment_007_serving_context.csv`.",
        "",
        "## 4. Compressor set",
        "",
        "| Compressor | Role in panel |",
        "|---|---|",
        "| `noop` | Lossless identity baseline |",
        "| `int8` | Real symmetric INT8 |",
        "| `k8_v4_sim` | Simulated asymmetric K8/V4 |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) |",
        "| `k_full_v8` | Real INT8 V, full K |",
        "| `k8_v_full` | Real INT8 K, full V |",
        "| `backend_passthrough` | V6 BackendAdapter PoC |",
        "",
        "## 5. Exactness result",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        f"| Harness exactness (`exactkv_token_match`) | "
        f"{sum(1 for r in report['results'] if r['serving_harness']['exactkv_token_match'])} / "
        f"{agg['total_runs']} |",
        "",
        "## 6. Acceptance by compressor",
        "",
        "| Compressor | Accept rate | Avg accept/round | Drafted | Accepted | Rejected | Corrections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(by_comp, key=lambda r: r["compressor_name"]):
        lines.append(
            f"| `{row['compressor_name']}` | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{_fmt_rate(row['mean_average_accepted_length'])} | "
            f"{row['total_drafted']} | {row['total_accepted']} | "
            f"{row['total_rejected']} | {row['total_corrections']} |"
        )

    lines.extend([
        "",
        "## 7. Divergence/rejection/correction summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Lossy divergence cells | {agg['lossy_divergence_count']} / {agg['total_runs']} |",
        f"| Total rejected (ExactKV) | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Mean acceptance (all cells) | {_fmt_rate(agg['mean_acceptance_rate'])} |",
        "",
        "## 8. Serving harness invariant summary",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| All `invariants_valid` | **{harness_gates['all_invariants_valid']}** |",
        f"| All `verification_uses == authoritative_full` | "
        f"**{harness_gates['all_verification_uses_authoritative_full']}** |",
        f"| All owners separate | **{harness_gates['all_owners_separate']}** |",
        f"| All identity logical/physical mapping | **{harness_gates['all_identity_mapping']}** |",
        f"| Total commit rounds tracked | {harness_gates['total_commit_rounds']} |",
        "",
        "## 9. Cache ownership result",
        "",
        "Every cell registered **separate** `authoritative_full` and `compressed_draft`",
        "entries. The harness summary records `verification_uses: authoritative_full`",
        "for all cells. Compressed draft never replaced authoritative storage.",
        "",
        "## 10. Logical vs physical sequence result",
        "",
        "All seven compressors in this panel use **identity mapping**",
        "(`physical_seq_len == logical_seq_len`) at prefill and after tracked commit",
        "appends. No pruned-cache cells were included (kvpress remains isolated).",
        "",
        "## 11. Block/page mapping result",
        "",
        f"Default `block_size={HARNESS_BLOCK_SIZE}`. Block tables were built for every",
        "cell; `validate_invariants()` confirmed non-empty, non-reversed block ranges.",
        "",
        "## 12. Append lifecycle result",
        "",
        f"The harness `append_committed_tokens` tracked **{harness_gates['total_commit_rounds']}**",
        "commit rounds across all cells. Logical sequence length advanced consistently",
        "after each ExactKV commit round.",
        "",
        "## 13. Workspace-memory accounting table",
        "",
        "Representative per-compressor means from harness metadata (compressed draft entry):",
        "",
        "| Compressor | Stored KV | Materialized KV | Total footprint † | Real bytes? | Simulated? |",
        "|---|---|---|---|---|---|",
    ])

    # Mean memory by compressor from harness metadata
    mem_by_comp: dict[str, list[dict]] = {}
    for r in report["results"]:
        mem_by_comp.setdefault(r["compressor_name"], []).append(r["serving_harness"])

    for name in COMPRESSORS:
        rows = mem_by_comp[name]
        stored = [h["stored_kv_bytes"] for h in rows if h["stored_kv_bytes"] is not None]
        mat = [
            h["materialized_working_kv_bytes"]
            for h in rows
            if h["materialized_working_kv_bytes"] is not None
        ]
        tot = [
            h["total_kv_footprint_bytes"]
            for h in rows
            if h["total_kv_footprint_bytes"] is not None
        ]
        real = rows[0]["supports_real_bytes_claim"]
        sim = rows[0]["is_simulated"]
        lines.append(
            f"| `{name}` | {_fmt_bytes(int(sum(stored)/len(stored)) if stored else None)} | "
            f"{_fmt_bytes(int(sum(mat)/len(mat)) if mat else None)} | "
            f"{_fmt_bytes(int(sum(tot)/len(tot)) if tot else None)} | "
            f"{'yes' if real else 'no'} | {'yes ⚠️' if sim else 'no'} |"
        )

    lines.extend([
        "",
        "† `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured",
        "peak GPU memory. **Active GPU memory is not reported.**",
        "",
        "## 14. What this proves",
        "",
        "- ExactKV's exactness gate holds (`exactkv_failures == 0`) while serving-harness",
        "  lifecycle invariants are enforced on every cell.",
        "- Authoritative full KV and compressed draft KV can be **modelled as separate",
        "  owners** alongside the existing HF runtime.",
        "- Logical sequence length, physical length (identity here), and block mapping",
        "  can be tracked through commit appends without mutating ExactKV states.",
        "- V5 workspace-memory honesty (`supports_real_bytes_claim`, `is_simulated`)",
        "  remains correct in harness summaries.",
        "",
        "## 15. What this does not prove",
        "",
        "- Compatibility with vLLM, LMCache, or real PagedAttention serving.",
        "- Production serving behaviour, throughput, or latency improvements.",
        "- Multi-request batching, GPU block pools, or cross-request cache sharing.",
        "- That simulated `_sim` compressors achieve real packed-bit memory savings.",
        "",
        "## 16. Relation to vLLM/LMCache/PagedAttention feasibility",
        "",
        "Phase A documented why direct stack integration is deferred. Experiment 007",
        "validates the **local harness fallback** — the evaluation path recommended when",
        "Phase C remains no-go. PagedAttention concepts appear only as block-table",
        "analogues in `exactkv/serving/cache_lifecycle.py`.",
        "",
        "## 17. Relation to V8 final launch",
        "",
        "Experiment 007 completes V8 Phase D. Phase E (release notes, audit, tag) may",
        "proceed if this report and exit criteria are satisfied. Phase C remains",
        "**no-go/deferred**.",
        "",
        "## 18. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al.,",
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic evaluation",
        "harness; Experiment 007 does not claim novel serving-stack integration.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 007 (harness mode)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_007_serving_context.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_007_serving_context.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_007_SERVING_CONTEXT.md",
    )
    args = parser.parse_args()

    prompts = load_core_prompts()
    expected = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 007: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"× 1 draft_len = {expected} cells"
    )

    print(f"Loading model {MODEL_NAME} ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)

    report = run_experiment_007(runtime, prompts)
    _assert_no_forbidden_fields(report)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    write_json_report(report, json_path, manifest=report["manifest"])
    write_csv_report(report, csv_path)

    warnings = validate_report(load_json_report(json_path))
    if warnings:
        print("validate_report warnings:", warnings, file=sys.stderr)
        return 1

    md_path.write_text(generate_markdown_report(report), encoding="utf-8")

    agg = report["aggregate"]
    gates = agg["serving_harness_gates"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"serving_harness_gates: {gates}")

    if agg["exactkv_failures"] != 0:
        return 1
    if not all([
        gates["all_invariants_valid"],
        gates["all_verification_uses_authoritative_full"],
        gates["all_owners_separate"],
        gates["all_exactkv_token_match"],
    ]):
        print("Serving harness gate failure", gates, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
