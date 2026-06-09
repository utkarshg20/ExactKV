#!/usr/bin/env python3
"""Experiment 005: restricted KVPress KnormPress vs ExactKV baselines.

Run ONLY in the isolated ``[kvpress]`` environment (``.venv-kvpress``).
``kvpress_knorm_restricted`` is NOT registered in the default compressor registry.

No timing, throughput, latency, speedup, or runtime_seconds fields are produced.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Repo root on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import kvpress  # noqa: F401 — gate: must run in [kvpress] env
except ImportError as exc:
    raise SystemExit(
        "Experiment 005 requires the [kvpress] optional extra. "
        "Use: .venv-kvpress/bin/python scripts/run_experiment_005_kvpress_knorm.py"
    ) from exc

import torch

from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.prompts import load_core_prompts
from exactkv.benchmarks.reports import (
    build_run_manifest,
    write_csv_report,
    write_json_report,
)
from exactkv.benchmarks.runner import RunConfig
from exactkv.benchmarks.sweeps import _compute_aggregate
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors import get_compressor
from exactkv.compressors.kvpress_knorm import (
    count_attention_forward_hooks,
    create_kvpress_knorm_adapter,
)
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
COMPRESSION_RATIO = 0.5
PROMPT_SUITE = "core"
KVPRESS_NAME = "kvpress_knorm_restricted"

COMPRESSORS = [
    "noop",
    "int8",
    "int4_sim",
    "k8_v4_sim",
    "k_full_v8",
    "k8_v_full",
    "backend_passthrough",
    KVPRESS_NAME,
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

_KVPRESS_MEMORY_NOTE = (
    "KVPress KnormPress (token-dropping): stored_kv_bytes and "
    "materialized_working_kv_bytes reflect real pruned DynamicCache tensor "
    "bytes, not packed low-bit quantization. metadata_bytes=0. "
    "total_kv_footprint_bytes is a conservative accounting sum, not measured "
    "peak GPU memory."
)


def _assert_no_forbidden_fields(obj: Any, path: str = "report") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden performance fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def _resolve_compressor(runtime: ModelRuntime, name: str, cache: dict[str, Any]):
    if name in cache:
        return cache[name]
    if name == KVPRESS_NAME:
        comp = create_kvpress_knorm_adapter(runtime, compression_ratio=COMPRESSION_RATIO)
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def _kvpress_gate_snapshot(
    runtime: ModelRuntime,
    adapter: Any,
    prompt: str,
) -> dict[str, Any]:
    state = prefill_to_full_state(runtime, prompt)
    verify_hooks_before = count_attention_forward_hooks(runtime.model)
    compressed = adapter.compress(state)
    verify_hooks_after_compress = count_attention_forward_hooks(runtime.model)
    cache = adapter.materialize_for_draft(compressed)
    physical = kv_seq_len(cache)
    logical = state.seq_len
    stats = adapter.stats(compressed)
    return {
        "verify_hooks_before": verify_hooks_before,
        "verify_hooks_after_compress": verify_hooks_after_compress,
        "compress_hooks_before": compressed.data["__hook_count_before__"],
        "compress_hooks_during": compressed.data["__hook_count_during__"],
        "compress_hooks_after": compressed.data["__hook_count_after__"],
        "logical_seq_len": logical,
        "physical_seq_len": physical,
        "pruning_occurred": physical < logical,
        "stored_kv_bytes": stats.stored_kv_bytes,
        "materialized_working_kv_bytes": stats.materialized_working_kv_bytes,
        "metadata_bytes": stats.metadata_bytes,
        "total_kv_footprint_bytes": stats.total_kv_footprint_bytes,
        "supports_real_bytes_claim": adapter.capabilities.supports_real_bytes_claim,
    }


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict,
    config: RunConfig,
    compressor: Any,
) -> dict[str, Any]:
    """Mirror ``runner.run_one`` with an explicit compressor instance."""
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

    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=config.draft_len).generate(
        prompt, max_new
    )
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()
    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)

    mem = estimate_kv_memory(runtime, prompt, compressor)
    if config.compressor_name == KVPRESS_NAME:
        mem.memory_claim_note = _KVPRESS_MEMORY_NOTE

    result = {
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
    }

    if config.compressor_name == KVPRESS_NAME:
        result["kvpress_gates"] = _kvpress_gate_snapshot(runtime, compressor, prompt)

    return result


def run_experiment_005(
    runtime: ModelRuntime,
    prompts: list[dict],
) -> dict[str, Any]:
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    for prompt_entry in prompts:
        for compressor_name in COMPRESSORS:
            compressor = _resolve_compressor(runtime, compressor_name, compressor_cache)
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results.append(run_one_cell(runtime, prompt_entry, config, compressor))

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite=PROMPT_SUITE,
        compressor_names=COMPRESSORS,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=DTYPE,
    )
    try:
        manifest["kvpress_version"] = importlib.metadata.version("kvpress")
    except importlib.metadata.PackageNotFoundError:
        manifest["kvpress_version"] = "unknown"
    manifest["experiment"] = "005_kvpress_knorm"
    manifest["kvpress_compressor_label"] = KVPRESS_NAME
    manifest["kvpress_compression_ratio"] = COMPRESSION_RATIO
    manifest["environment"] = "[kvpress] optional extra / .venv-kvpress"

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )

    kvpress_results = [r for r in results if r["compressor_name"] == KVPRESS_NAME]
    gates = [r["kvpress_gates"] for r in kvpress_results]
    aggregate["kvpress_gates"] = {
        "verify_hooks_always_zero": all(
            g["verify_hooks_before"] == 0 and g["verify_hooks_after_compress"] == 0
            for g in gates
        ),
        "compress_hooks_return_to_zero": all(
            g["compress_hooks_after"] == g["compress_hooks_before"] for g in gates
        ),
        "pruning_on_all_prompts": all(g["pruning_occurred"] for g in gates),
        "logical_equals_prefill_len": all(
            g["logical_seq_len"] > 0 and g["physical_seq_len"] < g["logical_seq_len"]
            for g in gates
        ),
        "workspace_materialized_equals_stored": all(
            g["materialized_working_kv_bytes"] == g["stored_kv_bytes"] for g in gates
        ),
        "supports_real_bytes_claim": all(g["supports_real_bytes_claim"] for g in gates),
    }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 005 (kvpress Knorm)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_005_kvpress_knorm.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_005_kvpress_knorm.csv",
        help="CSV report path",
    )
    args = parser.parse_args()

    print(f"Loading model {MODEL_NAME} ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)
    prompts = load_core_prompts()
    print(
        f"Running Experiment 005: {len(prompts)} prompts × {len(COMPRESSORS)} "
        f"compressors = {len(prompts) * len(COMPRESSORS)} cells"
    )

    report = run_experiment_005(runtime, prompts)
    _assert_no_forbidden_fields(report)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    write_json_report(report, json_path, manifest=report["manifest"])
    write_csv_report(report, csv_path)

    agg = report["aggregate"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"lossy_divergence_count: {agg['lossy_divergence_count']}")
    print(f"kvpress_gates: {agg['kvpress_gates']}")

    if agg["exactkv_failures"] != 0:
        return 1
    gates = agg["kvpress_gates"]
    if not all(gates.values()):
        print("KVPress gate failure", gates, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
