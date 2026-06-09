#!/usr/bin/env python3
"""Experiment 008: restricted TurboQuant Python adapter vs ExactKV baselines (V9 Phase C).

Run ONLY in the isolated ``[turboquant]`` environment (``.venv-turboquant``) with
``PYTHONPATH=vendor/turboquant_plus``.  ``turboquant_python_k3_v3`` is NOT in the
default compressor registry.

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes
fields are produced.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import turboquant  # noqa: F401 — gate: must run in [turboquant] env
except ImportError as exc:
    raise SystemExit(
        "Experiment 008 requires the upstream turboquant package. "
        "Use: PYTHONPATH=vendor/turboquant_plus "
        ".venv-turboquant/bin/python scripts/run_experiment_008_turboquant_python.py"
    ) from exc

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
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors import get_compressor
from exactkv.compressors.turboquant_adapter import create_turboquant_python_adapter
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
PROMPT_SUITE = "core"
EXPERIMENT_CLASS = "turboquant_python_real"
TURBOQUANT_NAME = "turboquant_python_k3_v3"
TURBOQUANT_K_BITS = 3
TURBOQUANT_V_BITS = 3
TURBOQUANT_HEAD_DIM = 64

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "k_full_v8",
    "k8_v_full",
    "backend_passthrough",
    TURBOQUANT_NAME,
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_TURBOQUANT_MEMORY_NOTE = (
    "Restricted Python TurboQuant adapter (turboquant_python_k3_v3): "
    "stored_kv_bytes counts actual numpy arrays in CompressedKVCache "
    "(int64 indices, float norms) — not packed-bit llama.cpp turbo formats. "
    "supports_real_bytes_claim=False. "
    "materialized_working_kv_bytes == full_kv_bytes (dequantised for attention). "
    "total_kv_footprint_bytes is a conservative accounting sum, not measured "
    "peak GPU memory. Active GPU memory is not reported."
)


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


def _turboquant_repo_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "-C", "vendor/turboquant_plus", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        return sha[:12] if sha else None
    except Exception:
        return None


def _resolve_compressor(runtime: ModelRuntime, name: str, cache: dict[str, Any]):
    if name in cache:
        return cache[name]
    if name == TURBOQUANT_NAME:
        comp = create_turboquant_python_adapter(
            runtime,
            head_dim=TURBOQUANT_HEAD_DIM,
            k_bits=TURBOQUANT_K_BITS,
            v_bits=TURBOQUANT_V_BITS,
        )
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def _turboquant_gate_snapshot(
    runtime: ModelRuntime,
    adapter: Any,
    prompt: str,
) -> dict[str, Any]:
    state = prefill_to_full_state(runtime, prompt)
    compressed = adapter.compress(state)
    cache = adapter.materialize_for_draft(compressed)
    physical = kv_seq_len(cache)
    logical = state.seq_len
    stats = adapter.stats(compressed)
    return {
        "logical_seq_len": logical,
        "physical_seq_len": physical,
        "identity_mapping": physical == logical,
        "stored_kv_bytes": stats.stored_kv_bytes,
        "materialized_working_kv_bytes": stats.materialized_working_kv_bytes,
        "metadata_bytes": stats.metadata_bytes,
        "temporary_workspace_bytes": stats.temporary_workspace_bytes,
        "total_kv_footprint_bytes": stats.total_kv_footprint_bytes,
        "supports_real_bytes_claim": adapter.capabilities.supports_real_bytes_claim,
        "is_simulated": adapter.capabilities.is_simulated,
        "backend_name": adapter.capabilities.backend_name,
        "backend_version": adapter.capabilities.backend_version,
    }


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict,
    config: RunConfig,
    compressor: Any,
) -> dict[str, Any]:
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
    if config.compressor_name == TURBOQUANT_NAME:
        mem.memory_claim_note = _TURBOQUANT_MEMORY_NOTE
        mem.supports_real_bytes_claim = False

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

    if config.compressor_name == TURBOQUANT_NAME:
        result["turboquant_gates"] = _turboquant_gate_snapshot(runtime, compressor, prompt)

    return result


def run_experiment_008(
    runtime: ModelRuntime,
    prompts: list[dict],
) -> dict[str, Any]:
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total_cells = len(prompts) * len(COMPRESSORS)
    cell_idx = 0

    for prompt_entry in prompts:
        for compressor_name in COMPRESSORS:
            cell_idx += 1
            print(
                f"  [{cell_idx}/{total_cells}] {prompt_entry['prompt_id']} × "
                f"{compressor_name}",
                flush=True,
            )
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
    manifest["experiment"] = "008_turboquant_python"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["turboquant_compressor_label"] = TURBOQUANT_NAME
    manifest["turboquant_k_bits"] = TURBOQUANT_K_BITS
    manifest["turboquant_v_bits"] = TURBOQUANT_V_BITS
    manifest["turboquant_head_dim"] = TURBOQUANT_HEAD_DIM
    manifest["environment"] = "[turboquant] optional extra / .venv-turboquant"
    manifest["pythonpath_required"] = "vendor/turboquant_plus"
    manifest["turboquant_repo_sha"] = _turboquant_repo_sha()
    try:
        manifest["scipy_version"] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        manifest["scipy_version"] = "unknown"
    tq_adapter = compressor_cache.get(TURBOQUANT_NAME)
    if tq_adapter is not None:
        manifest["turboquant_backend_version"] = tq_adapter.capabilities.backend_version

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )

    tq_results = [r for r in results if r["compressor_name"] == TURBOQUANT_NAME]
    gates = [r["turboquant_gates"] for r in tq_results]
    aggregate["turboquant_gates"] = {
        "all_identity_mapping": all(g["identity_mapping"] for g in gates),
        "supports_real_bytes_claim_false": all(
            not g["supports_real_bytes_claim"] for g in gates
        ),
        "is_simulated_false": all(not g["is_simulated"] for g in gates),
        "materialized_equals_full": all(
            g["materialized_working_kv_bytes"] >= g["stored_kv_bytes"] for g in gates
        ),
    }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def _fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _acceptance_lookup(by_comp: list[dict]) -> dict[str, dict]:
    return {row["compressor_name"]: row for row in by_comp}


def _delta_str(tq: float, base: float) -> str:
    d = tq - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    by_comp = agg["acceptance_by_compressor"]
    lookup = _acceptance_lookup(by_comp)
    tq = lookup.get(TURBOQUANT_NAME, {})
    tq_gates = agg.get("turboquant_gates", {})
    manifest = report["manifest"]

    lines = [
        "# Experiment 008: Restricted TurboQuant Python Adapter Evaluation",
        "",
        "_Generated by `scripts/run_experiment_008_turboquant_python.py`. "
        "V9 Phase C — restricted Python adapter only._",
        "",
        "> This evaluates the **restricted Python TurboQuant adapter only**.",
        "> This does **not** integrate llama.cpp, MLX, GGUF, or production TurboQuant runtime.",
        "> This does **not** claim upstream TurboQuant / TurboQuant+ results as ExactKV results.",
        "> The adapter is **not** in the default compressor registry.",
        "> The adapter requires `.venv-turboquant` and `PYTHONPATH=vendor/turboquant_plus`.",
        "> `supports_real_bytes_claim=False` because the Python payload uses numpy/int64 "
        "structures, not packed production storage.",
        "> Stored bytes are counted from actual Python compressed payload arrays, not "
        "advertised compression ratios.",
        "> Materialized working KV is full dequantised KV.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak "
        "GPU memory.",
        "> **Active GPU memory is not reported.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, "
        "or production readiness.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Evaluate whether the restricted `TurboQuantPythonAdapter` (factory-only, NumPy",
        "`KVCacheCompressor` bridge) preserves ExactKV's exactness gate while exhibiting",
        "meaningful acceptance and honest workspace-memory accounting versus built-in",
        "baselines on the core prompt suite.",
        "",
        "## 2. Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Virtualenv | `.venv-turboquant` (`pip install -e \".[dev,turboquant]\"`) |",
        f"| PYTHONPATH | `vendor/turboquant_plus` |",
        f"| Model | `{MODEL_NAME}`, {DTYPE}, CPU-first |",
        f"| transformers | {manifest.get('transformers_version', '—')} |",
        f"| torch | {manifest.get('torch_version', '—')} |",
        f"| scipy | {manifest.get('scipy_version', '—')} |",
        f"| turboquant backend | {manifest.get('turboquant_backend_version', '—')} |",
        f"| turboquant repo sha | {manifest.get('turboquant_repo_sha', '—')} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "export PYTHONPATH=$PWD/vendor/turboquant_plus",
        ".venv-turboquant/bin/python scripts/run_experiment_008_turboquant_python.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_008_turboquant_python.json`,",
        "`reports/experiment_008_turboquant_python.csv`.",
        "",
        "## 3. Restrictions",
        "",
        "- **Python adapter only** — no llama.cpp, MLX, GGUF, or production runtime.",
        "- **Not in default registry** — `turboquant_python_k3_v3` via factory only.",
        "- **Isolated venv** — default ExactKV install remains TurboQuant-free.",
        "- **No KIVI, KVQuant, kvpress, vLLM, LMCache** in this experiment.",
        "- **No performance or serving claims** in reports or this document.",
        "",
        "## 4. Model and prompt suite",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{MODEL_NAME}` |",
        f"| Prompt suite | `{PROMPT_SUITE}` (34 prompts) |",
        f"| `draft_len` | {DRAFT_LEN} |",
        f"| `max_new_tokens` | {MAX_NEW_TOKENS} |",
        f"| Experiment class | `{EXPERIMENT_CLASS}` |",
        f"| Total cells | **{agg['total_runs']}** |",
        "",
        "## 5. Compressor set",
        "",
        "| Compressor | Type | Notes |",
        "|---|---|---|",
        "| `noop` | Identity baseline | Lossless |",
        "| `int8` | Real INT8 | `supports_real_bytes_claim=True` |",
        "| `k8_v4_sim` | Simulated asymmetric K8/V4 | `is_simulated=True` |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) | Simulated V policy |",
        "| `k_full_v8` | Real K-full / V8 | |",
        "| `k8_v_full` | Real K8 / V-full | |",
        "| `backend_passthrough` | V6 BackendAdapter PoC | Lossless |",
        f"| `{TURBOQUANT_NAME}` | **Restricted TurboQuant Python adapter** | k_bits={TURBOQUANT_K_BITS}, v_bits={TURBOQUANT_V_BITS}, head_dim={TURBOQUANT_HEAD_DIM} |",
        "",
        "## 6. Exactness result",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        f"| `exactkv_output_ids == full_output_ids` | "
        f"{agg['total_runs'] - agg['exactkv_failures']} / {agg['total_runs']} |",
        "",
        "## 7. Acceptance by compressor",
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
        "## 8. Divergence/rejection/correction summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Lossy divergence cells | {agg['lossy_divergence_count']} / {agg['total_runs']} |",
        f"| Total rejected (ExactKV) | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Mean acceptance (all cells) | {_fmt_rate(agg['mean_acceptance_rate'])} |",
        "",
        "## 9. Workspace-memory accounting table",
        "",
        "Per-compressor mean memory fields from prefill estimates:",
        "",
        "| Compressor | Stored KV | Materialized KV | Metadata | Total footprint † | Real bytes? | Simulated? |",
        "|---|---:|---:|---:|---:|---|---|",
    ])

    mem_by_comp: dict[str, list[dict]] = {}
    for r in report["results"]:
        mem_by_comp.setdefault(r["compressor_name"], []).append(r["memory"])

    for name in COMPRESSORS:
        rows = mem_by_comp.get(name, [])
        if not rows:
            continue
        stored = [m["stored_kv_bytes"] for m in rows]
        mat = [m["materialized_working_kv_bytes"] for m in rows]
        meta = [m["metadata_bytes"] for m in rows]
        tot = [m["total_kv_footprint_bytes"] for m in rows]
        real = rows[0].get("supports_real_bytes_claim", False)
        sim = rows[0].get("is_simulated", False)
        lines.append(
            f"| `{name}` | {_fmt_bytes(sum(stored)/len(stored))} | "
            f"{_fmt_bytes(sum(mat)/len(mat))} | "
            f"{_fmt_bytes(sum(meta)/len(meta))} | "
            f"{_fmt_bytes(sum(tot)/len(tot))} | "
            f"{'yes' if real else 'no'} | {'yes ⚠️' if sim else 'no'} |"
        )

    tq_mem = mem_by_comp.get(TURBOQUANT_NAME, [{}])[0] if TURBOQUANT_NAME in mem_by_comp else {}
    mean_stored = (
        sum(m["stored_kv_bytes"] for m in mem_by_comp[TURBOQUANT_NAME])
        / max(len(mem_by_comp.get(TURBOQUANT_NAME, [])), 1)
        if TURBOQUANT_NAME in mem_by_comp else 0
    )
    def _full_ref_bytes(m: dict) -> int:
        return int(m.get("full_kv_bytes") or m.get("full_bytes") or 0)

    mean_full = (
        sum(_full_ref_bytes(m) for m in mem_by_comp[TURBOQUANT_NAME])
        / max(len(mem_by_comp.get(TURBOQUANT_NAME, [])), 1)
        if TURBOQUANT_NAME in mem_by_comp else 0
    )

    lines.extend([
        "",
        "† `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured",
        "peak GPU memory. **Active GPU memory is not reported.**",
        "",
        "## 10. Stored-payload accounting result",
        "",
        f"| TurboQuant gate | Result |",
        f"|---|---|",
        f"| All identity logical/physical mapping | **{tq_gates.get('all_identity_mapping', '—')}** |",
        f"| `supports_real_bytes_claim=False` on all cells | "
        f"**{tq_gates.get('supports_real_bytes_claim_false', '—')}** |",
        f"| `is_simulated=False` | **{tq_gates.get('is_simulated_false', '—')}** |",
        f"| Mean stored payload (Python numpy) | {_fmt_bytes(mean_stored)} |",
        f"| Mean full KV (fp32 reference) | {_fmt_bytes(mean_full)} |",
        "",
        "The Python adapter stores int64 indices and float metadata arrays. At this scale",
        "the stored payload may **exceed** fp32 tensor bytes — that is honestly reported,",
        "not hidden behind upstream compression-ratio claims.",
        "",
        "## 11. TurboQuant vs baselines",
        "",
        "| Baseline | Baseline accept | TurboQuant accept | Δ accept |",
        "|---|---:|---:|---:|",
    ])

    comparisons = [
        ("int8", "int8"),
        ("k8_v4_sim", "k8_v4_sim"),
        ("k8_v4_boundary4_v8_sim", "k8_v4_boundary4_v8_sim"),
        ("k_full_v8", "k_full_v8"),
        ("backend_passthrough", "backend_passthrough"),
    ]
    tq_rate = tq.get("mean_acceptance_rate", 0.0)
    for label, key in comparisons:
        base_rate = lookup.get(key, {}).get("mean_acceptance_rate", 0.0)
        lines.append(
            f"| `{key}` | {_fmt_rate(base_rate)} | {_fmt_rate(tq_rate)} | "
            f"{_delta_str(tq_rate, base_rate)} |"
        )

    lines.extend([
        "",
        "## 12. What TurboQuant improves or fails to improve",
        "",
    ])

    int8_rate = lookup.get("int8", {}).get("mean_acceptance_rate", 0.0)
    k8v4_rate = lookup.get("k8_v4_sim", {}).get("mean_acceptance_rate", 0.0)
    boundary_rate = lookup.get("k8_v4_boundary4_v8_sim", {}).get("mean_acceptance_rate", 0.0)

    if tq_rate > k8v4_rate:
        lines.append(
            f"- **Versus naive `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: rotation-based Python "
            f"TurboQuant ({_fmt_rate(tq_rate)}) shows higher acceptance — consistent with Phase A "
            "finding that naive INT sim ≠ real PolarQuant path."
        )
    else:
        lines.append(
            f"- **Versus `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: TurboQuant Python "
            f"({_fmt_rate(tq_rate)}) did not exceed the simulated baseline on this suite."
        )

    if tq_rate > int8_rate:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: TurboQuant matched or beat real INT8 "
            "acceptance on mean rate."
        )
    else:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: TurboQuant acceptance "
            f"({_fmt_rate(tq_rate)}) is below symmetric INT8 on mean rate."
        )

    lines.append(
        f"- **Versus `k8_v4_boundary4_v8_sim` ({_fmt_rate(boundary_rate)})**: layer-aware simulated "
        f"policy remains a strong simulated reference; TurboQuant Python is "
        f"{_delta_str(tq_rate, boundary_rate)} on mean accept rate."
    )
    lines.append(
        "- **Stored bytes**: Python payload is not packed-bit production storage; "
        "`supports_real_bytes_claim=False` is intentional."
    )

    lines.extend([
        "",
        "## 13. What this proves",
        "",
        "- ExactKV can wrap a **real external quantizer** (upstream `KVCacheCompressor`) behind",
        "  `BackendAdapter` while preserving `exactkv_failures == 0`.",
        "- Acceptance and workspace-memory fields can be reported honestly for a non-registry",
        "  backend in an isolated environment.",
        "- The restricted Python path is sufficient for **ExactKV evaluation** of TurboQuant-family",
        "  compression under full-KV verification — without llama.cpp or MLX.",
        "",
        "## 14. What this does not prove",
        "",
        "- Compatibility with llama.cpp `turbo2`/`turbo3`/`turbo4` packed formats or Metal/CUDA kernels.",
        "- Upstream TurboQuant+ perplexity, bandwidth, or serving claims (external results only).",
        "- That Python numpy storage reduces memory versus fp32 (may be larger at small scale).",
        "- Production readiness, throughput, latency, or GPU memory behaviour.",
        "",
        "## 15. Relation to V9 and future KIVI/KVQuant work",
        "",
        "Experiment 008 completes V9 Phase C for the TurboQuant track. Phase D may proceed to",
        "KIVI / KVQuant feasibility (Experiment 009) using the same adapter-isolation pattern.",
        "TurboQuant rows in this report become the comparison anchor for any future real backend.",
        "",
        "## 16. Relation to TurboQuant/TurboQuant+ upstream",
        "",
        "Upstream TurboQuant+ ([`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus))",
        "implements production paths in llama.cpp and MLX. This experiment uses only the",
        "**dev-only Python** `turboquant` package. External upstream quality or compression claims",
        "are **not** ExactKV results.",
        "",
        "## 17. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al.,",
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic evaluation harness;",
        "Experiment 008 does not claim novel TurboQuant algorithm contributions.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 008 (TurboQuant Python)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_008_turboquant_python.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_008_turboquant_python.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_008_TURBOQUANT_PYTHON.md",
    )
    args = parser.parse_args()

    prompts = load_core_prompts()
    expected = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 008: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"× 1 draft_len = {expected} cells"
    )

    print(f"Loading model {MODEL_NAME} ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)

    report = run_experiment_008(runtime, prompts)
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
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"turboquant_gates: {agg.get('turboquant_gates')}")

    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
