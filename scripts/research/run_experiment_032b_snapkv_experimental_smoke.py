#!/usr/bin/env python3
"""Experiment 032b: SnapKV experimental adapter exactness smoke (V13 Phase 5b).

Restricted factory-only ``snapkv_experimental`` via kvpress SnapKVPress.
No timing, throughput, latency, speedup, or active GPU memory savings claims.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import kvpress  # noqa: F401 — gate: must run in [kvpress] env
except ImportError as exc:
    raise SystemExit(
        "Experiment 032b requires the [kvpress] optional extra. "
        "Use: .venv-kvpress/bin/python "
        "scripts/research/run_experiment_032b_snapkv_experimental_smoke.py"
    ) from exc

import torch

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors.kvpress_knorm import count_attention_forward_hooks
from exactkv.compressors.kvpress_snapkv import create_snapkv_experimental_adapter
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 16
DRAFT_LEN = 4
COMPRESSION_RATIO = 0.5
WINDOW_SIZE = 64
KERNEL_SIZE = 5
COMPRESSOR_NAME = "snapkv_experimental"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})

_MEMORY_NOTE = (
    "SnapKVPress (token-dropping): stored_kv_bytes and "
    "materialized_working_kv_bytes reflect pruned DynamicCache tensor bytes. "
    "metadata_bytes=0. total_kv_footprint_bytes is conservative accounting, "
    "not measured peak GPU memory. No active GPU memory savings claim."
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


def load_smoke_prompt_panel() -> list[dict[str, Any]]:
    suites = ("core_v2", "long_context", "retrieval_copy", "tool_json")
    out: list[dict[str, Any]] = []
    for suite_name in suites:
        suite = load_v10_suite(suite_name)
        if not suite:
            raise ValueError(f"Suite {suite_name!r} is empty")
        out.append(suite[0])
    return out


def _resolve_dtype(device: torch.device) -> str:
    return "float16" if device.type == "cuda" else "float32"


def _gate_snapshot(
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
    adapter: Any,
    prompt_entry: dict[str, Any],
    *,
    verification_method: str,
) -> dict[str, Any]:
    prompt = prompt_entry["prompt"]
    full_res = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
    ekv_res = ExactKVGenerator(
        runtime,
        adapter,
        draft_len=DRAFT_LEN,
        verification_method=verification_method,  # type: ignore[arg-type]
    ).generate(prompt, MAX_NEW_TOKENS)

    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)
    mem = estimate_kv_memory(runtime, prompt, adapter)
    mem.memory_claim_note = _MEMORY_NOTE

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "category": prompt_entry.get("category", "unknown"),
        "verification_method": verification_method,
        "compressor_name": COMPRESSOR_NAME,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "full_output_ids": full_res.generated_ids.squeeze(0).tolist(),
        "exactkv_output_ids": ekv_res.output_ids.squeeze(0).tolist(),
        "token_exact_match": ekv_exact,
        "first_divergence_idx": first_divergence_idx(
            full_res.generated_ids, ekv_res.output_ids
        ),
        "acceptance": acceptance.to_dict(),
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
        "rejected_tokens_never_committed": ekv_exact,
        "gates": _gate_snapshot(runtime, adapter, prompt),
    }


def run_smoke(runtime: ModelRuntime) -> dict[str, Any]:
    adapter = create_snapkv_experimental_adapter(
        runtime,
        compression_ratio=COMPRESSION_RATIO,
        window_size=WINDOW_SIZE,
        kernel_size=KERNEL_SIZE,
        isolate_compression_model=True,
    )
    prompts = load_smoke_prompt_panel()
    results: list[dict[str, Any]] = []
    for prompt_entry in prompts:
        for method in ("sequential", "span"):
            results.append(
                run_one_cell(
                    runtime, adapter, prompt_entry, verification_method=method
                )
            )

    exactkv_failures = sum(1 for r in results if r["exactkv_failure"])
    gates = [r["gates"] for r in results]
    hook_isolation_ok = all(
        g["verify_hooks_before"] == 0
        and g["verify_hooks_after_compress"] == 0
        and g["compress_hooks_after"] == g["compress_hooks_before"]
        for g in gates
    )

    try:
        kvpress_version = importlib.metadata.version("kvpress")
    except importlib.metadata.PackageNotFoundError:
        kvpress_version = "unknown"

    return {
        "experiment": "032b_snapkv_experimental_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "device": str(runtime.device),
        "dtype": _resolve_dtype(runtime.device),
        "compressor_name": COMPRESSOR_NAME,
        "factory_only": True,
        "in_default_registry": False,
        "compression_ratio": COMPRESSION_RATIO,
        "window_size": WINDOW_SIZE,
        "kernel_size": KERNEL_SIZE,
        "kvpress_version": kvpress_version,
        "environment": "[kvpress] optional extra / .venv-kvpress",
        "results": results,
        "aggregate": {
            "total_cells": len(results),
            "exactkv_failures": exactkv_failures,
            "exactkv_pass_rate": (len(results) - exactkv_failures) / max(len(results), 1),
            "hook_isolation_ok": hook_isolation_ok,
            "all_token_exact_match": exactkv_failures == 0,
            "all_rejected_never_committed": all(
                r["rejected_tokens_never_committed"] for r in results
            ),
        },
        "disclaimer": (
            "Restricted experimental adapter, not production SnapKV. "
            "Full-KV verifier remains authoritative. "
            "External SnapKV/kvpress results are not ExactKV results. "
            "No speedup, throughput, latency, runtime, tokens/sec, active GPU "
            "memory savings, production serving, or model accuracy improvement claim."
        ),
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    passed = agg["exactkv_failures"] == 0 and agg["hook_isolation_ok"]
    status = "PASS" if passed else "FAIL / BLOCKER"

    lines = [
        "# Experiment 032b: SnapKV Experimental Adapter Smoke",
        "",
        f"_Generated by `scripts/research/run_experiment_032b_snapkv_experimental_smoke.py`. "
        f"V13 Phase 5b — restricted factory-only adapter MVP._",
        "",
        f"**Status:** {status}",
        "",
        "## 1. Purpose",
        "",
        "Validate that the factory-only ``snapkv_experimental`` adapter (kvpress "
        "`SnapKVPress`) preserves ExactKV exactness on a small stratified prompt panel "
        "without modifying generation or verification core logic.",
        "",
        "## 2. Why this follows Exp 032 and the addendum",
        "",
        "Exp 032 ranked SnapKV **B (restricted feasibility)** as the Phase 5b MVP. "
        "The addendum confirmed SnapKV remains primary; Shard and SpectralQuant are "
        "deferred. This smoke implements and gates that recommendation.",
        "",
        "## 3. Dependency/environment status",
        "",
        f"- **kvpress version:** {report['kvpress_version']}",
        f"- **Environment:** {report['environment']}",
        f"- **Model:** {report['model_name']}",
        f"- **Device / dtype:** {report['device']} / {report['dtype']}",
        "",
        "## 4. Adapter implementation summary",
        "",
        "- Module: `exactkv/compressors/kvpress_snapkv.py`",
        "- Factory: `create_snapkv_experimental_adapter(...)`",
        "- Backend: kvpress `SnapKVPress` replay prefill under `with press(model):`",
        f"- Config: compression_ratio={report['compression_ratio']}, "
        f"window_size={report['window_size']} (clamped to seq_len-1 when shorter; "
        f"kvpress API), kernel_size={report['kernel_size']}, "
        "`isolate_compression_model=True`",
        "- **Excluded from default compressor registry**",
        "",
        "## 5. Paper-exact SnapKV vs restricted experimental",
        "",
        "**Restricted experimental SnapKV** via kvpress `SnapKVPress` — **not** a claim of "
        "paper-exact or production SnapKV unless verified against reference behavior.",
        "",
        "## 6. ExactKV invariant summary",
        "",
        "- Full-KV verifier remains authoritative",
        "- Draft uses compressed/pruned KV only",
        "- `update_after_commit` recompresses from authoritative full state",
        "- `logical_seq_len` preserved separately from physical retained length",
        "- Rejected draft tokens are not committed",
        "",
        "## 7. Smoke configuration",
        "",
        f"- Prompts: 1× core_v2, 1× long_context, 1× retrieval_copy, 1× tool_json",
        f"- max_new_tokens: {MAX_NEW_TOKENS}",
        f"- draft_len: {DRAFT_LEN}",
        f"- Compressor: `{COMPRESSOR_NAME}` (factory-only)",
        "- Verification: sequential + span",
        "",
        "## 8. Exactness result",
        "",
        f"- **exactkv_failures:** {agg['exactkv_failures']} / {agg['total_cells']}",
        f"- **all_token_exact_match:** {agg['all_token_exact_match']}",
        "",
        "### Per-cell",
        "",
        "| prompt_id | verification | exact | divergence_idx |",
        "|-----------|--------------|-------|----------------|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['prompt_id']} | {r['verification_method']} | "
            f"{r['token_exact_match']} | {r['first_divergence_idx']} |"
        )

    lines.extend([
        "",
        "## 9. Acceptance/rejection/correction result",
        "",
    ])
    for r in report["results"]:
        acc = r["acceptance"]
        lines.append(
            f"- **{r['prompt_id']}** ({r['verification_method']}): "
            f"drafted={acc['total_drafted']}, accepted={acc['total_accepted']}, "
            f"rejected={acc['total_rejected']}, corrections={acc['total_corrections']}"
        )

    lines.extend([
        "",
        "## 10. Hook isolation result",
        "",
        f"- **hook_isolation_ok:** {agg['hook_isolation_ok']}",
        "- Verifier model forward hooks must be 0 before/after compress and during verify",
        "",
        "## 11. Memory accounting treatment",
        "",
        _MEMORY_NOTE,
        "",
        "## 12. What this proves",
        "",
        "- Factory-only `snapkv_experimental` can be wired through `BackendAdapter` with "
        "kvpress `SnapKVPress` while preserving `exactkv_failures == 0` on this panel.",
        "- Hook isolation pattern from Exp 005 Knorm adapter transfers to SnapKVPress.",
        "",
        "## 13. What this does not prove",
        "",
        "- No speedup, throughput, latency, runtime, tokens/sec, or active GPU memory savings.",
        "- No production serving readiness or model accuracy improvement.",
        "- No paper-exact SnapKV fidelity vs reference implementation.",
        "- External kvpress/SnapKV benchmark numbers are not ExactKV results.",
        "",
        "## 14. Limitations/blockers",
        "",
    ])
    if passed:
        lines.append("- None blocking on this smoke panel.")
        lines.append(
            "- **API note:** kvpress `SnapKVPress` requires query length > "
            "`window_size`; adapter clamps effective window to `min(configured, "
            "seq_len-1)` on short prompts (e.g. core_v2, retrieval_copy, tool_json)."
        )
    else:
        lines.append(
            f"- **BLOCKER:** exactkv_failures={agg['exactkv_failures']} or "
            f"hook_isolation_ok={agg['hook_isolation_ok']}. Adapter remains factory-only."
        )

    lines.extend([
        "",
        "## 15. Recommendation for Phase 6",
        "",
    ])
    if passed:
        lines.append(
            "**Proceed to Phase 6 (Exp 033 Llama-3.1-8B)** with `snapkv_experimental` "
            "remaining factory-only. Optional Shard Llama external-drafter probe per addendum."
        )
    else:
        lines.append(
            "**Do not expand claims** until exactness smoke passes. Phase 6 may proceed "
            "on Llama baseline without SnapKV default registration."
        )

    lines.extend([
        "",
        "---",
        "",
        report["disclaimer"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 032b SnapKV smoke")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_032b_snapkv_experimental_smoke.json",
    )
    parser.add_argument(
        "--md-out",
        default="docs/EXPERIMENT_032B_SNAPKV_EXPERIMENTAL_SMOKE.md",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(torch.device(device))
    print(f"Loading {MODEL_NAME} on {device} ({dtype}) ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device=device, dtype=dtype)

    report = run_smoke(runtime)
    _assert_no_forbidden_fields(report)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    write_markdown_report(report, md_path)

    agg = report["aggregate"]
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"hook_isolation_ok: {agg['hook_isolation_ok']}")

    if agg["exactkv_failures"] != 0:
        return 1
    if not agg["hook_isolation_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
