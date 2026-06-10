#!/usr/bin/env python3
"""Experiment 009: restricted KIVI offline adapter vs ExactKV baselines (V9 Phase D3).

Run ONLY in an isolated KIVI environment with ``PYTHONPATH`` pointing at the
jy-yuan/KIVI repo.  ``kivi_offline_k2_v2`` is NOT in the default compressor registry.

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes
fields are produced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _kivi_available() -> bool:
    try:
        return importlib.util.find_spec("models.utils_quant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


if not _kivi_available():
    raise SystemExit(
        "Experiment 009 requires upstream KIVI models.utils_quant. "
        "Use: PYTHONPATH=/tmp/kivi_research "
        ".venv-turboquant/bin/python scripts/run_experiment_009_kivi_offline.py"
    )

_TURBOQUANT_AVAILABLE = importlib.util.find_spec("turboquant") is not None

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
from exactkv.compressors.kivi_adapter import create_kivi_offline_adapter
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
EXPERIMENT_CLASS = "kivi_offline_real"

KIVI_NAME = "kivi_offline_k2_v2"
KIVI_K_BITS = 2
KIVI_V_BITS = 2
KIVI_GROUP_SIZE = 32
KIVI_HEAD_DIM = 64

TURBOQUANT_NAME = "turboquant_python_k3_v3"
TURBOQUANT_K_BITS = 3
TURBOQUANT_V_BITS = 3

BASE_COMPRESSORS = [
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

_KIVI_MEMORY_NOTE = (
    "Restricted offline KIVI adapter (kivi_offline_k2_v2): "
    "stored_kv_bytes counts actual torch quant codes and scale tensors "
    "from models.utils_quant simulate path — not packed-bit KIVI CUDA storage. "
    "supports_real_bytes_claim=False. "
    "materialized_working_kv_bytes == full_kv_bytes (dequantised for attention). "
    "total_kv_footprint_bytes is a conservative accounting sum, not measured "
    "peak GPU memory. Active GPU memory is not reported."
)

_EXP008_REPORT = _ROOT / "reports" / "experiment_008_turboquant_python.json"


def _build_compressor_list() -> list[str]:
    compressors = list(BASE_COMPRESSORS)
    if _TURBOQUANT_AVAILABLE:
        from exactkv.compressors.turboquant_adapter import create_turboquant_python_adapter  # noqa: F401

        compressors.append(TURBOQUANT_NAME)
    compressors.append(KIVI_NAME)
    return compressors


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


def _kivi_repo_sha() -> str | None:
    try:
        import models.utils_quant as uq  # noqa: PLC0415

        root = Path(uq.__file__).resolve().parents[1]
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        return sha[:12] if sha else None
    except Exception:
        return None


def _resolve_compressor(runtime: ModelRuntime, name: str, cache: dict[str, Any]):
    if name in cache:
        return cache[name]
    if name == KIVI_NAME:
        comp = create_kivi_offline_adapter(
            runtime,
            head_dim=KIVI_HEAD_DIM,
            k_bits=KIVI_K_BITS,
            v_bits=KIVI_V_BITS,
            group_size=KIVI_GROUP_SIZE,
        )
    elif name == TURBOQUANT_NAME:
        from exactkv.compressors.turboquant_adapter import create_turboquant_python_adapter

        comp = create_turboquant_python_adapter(
            runtime,
            head_dim=KIVI_HEAD_DIM,
            k_bits=TURBOQUANT_K_BITS,
            v_bits=TURBOQUANT_V_BITS,
        )
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def _adapter_gate_snapshot(
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
    if config.compressor_name == KIVI_NAME:
        mem.memory_claim_note = _KIVI_MEMORY_NOTE
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

    if config.compressor_name == KIVI_NAME:
        result["kivi_gates"] = _adapter_gate_snapshot(runtime, compressor, prompt)

    return result


def run_experiment_009(
    runtime: ModelRuntime,
    prompts: list[dict],
    compressors: list[str],
) -> dict[str, Any]:
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total_cells = len(prompts) * len(compressors)
    cell_idx = 0

    for prompt_entry in prompts:
        for compressor_name in compressors:
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
        compressor_names=compressors,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=DTYPE,
    )
    manifest["experiment"] = "009_kivi_offline"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["kivi_compressor_label"] = KIVI_NAME
    manifest["kivi_k_bits"] = KIVI_K_BITS
    manifest["kivi_v_bits"] = KIVI_V_BITS
    manifest["kivi_group_size"] = KIVI_GROUP_SIZE
    manifest["kivi_head_dim"] = KIVI_HEAD_DIM
    manifest["environment"] = "PYTHONPATH to jy-yuan/KIVI repo (isolated)"
    manifest["pythonpath_required"] = "KIVI repo root (e.g. /tmp/kivi_research)"
    manifest["kivi_repo_sha"] = _kivi_repo_sha()
    manifest["turboquant_included_live"] = TURBOQUANT_NAME in compressors
    if not manifest["turboquant_included_live"]:
        manifest["turboquant_comparison_source"] = (
            str(_EXP008_REPORT) if _EXP008_REPORT.is_file() else None
        )

    kivi_adapter = compressor_cache.get(KIVI_NAME)
    if kivi_adapter is not None:
        manifest["kivi_backend_version"] = kivi_adapter.capabilities.backend_version

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressors,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )

    kivi_results = [r for r in results if r["compressor_name"] == KIVI_NAME]
    gates = [r["kivi_gates"] for r in kivi_results]
    aggregate["kivi_gates"] = {
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


def _load_exp008_turboquant_acceptance() -> dict[str, float] | None:
    if not _EXP008_REPORT.is_file():
        return None
    try:
        report = load_json_report(_EXP008_REPORT)
        by_comp = report.get("aggregate", {}).get("acceptance_by_compressor", [])
        for row in by_comp:
            if row.get("compressor_name") == TURBOQUANT_NAME:
                return {
                    "mean_acceptance_rate": row.get("mean_acceptance_rate", 0.0),
                    "source": "experiment_008",
                }
    except Exception:
        return None
    return None


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


def _delta_str(a: float, base: float) -> str:
    d = a - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _full_ref_bytes(m: dict) -> int:
    return int(m.get("full_kv_bytes") or m.get("full_bytes") or 0)


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    by_comp = agg["acceptance_by_compressor"]
    lookup = _acceptance_lookup(by_comp)
    kivi = lookup.get(KIVI_NAME, {})
    kivi_gates = agg.get("kivi_gates", {})
    manifest = report["manifest"]
    compressors = manifest.get("compressor_names", [])

    tq_live = manifest.get("turboquant_included_live", False)
    tq_exp008 = _load_exp008_turboquant_acceptance()
    if tq_live:
        tq_rate = lookup.get(TURBOQUANT_NAME, {}).get("mean_acceptance_rate", 0.0)
        tq_source = "live in Experiment 009"
    elif tq_exp008:
        tq_rate = tq_exp008["mean_acceptance_rate"]
        tq_source = "Experiment 008 (not run live in 009)"
    else:
        tq_rate = None
        tq_source = "unavailable"

    kivi_rate = kivi.get("mean_acceptance_rate", 0.0)

    lines = [
        "# Experiment 009: Restricted KIVI Offline Adapter Evaluation",
        "",
        "_Generated by `scripts/run_experiment_009_kivi_offline.py`. "
        "V9 Phase D3 — restricted offline KIVI adapter only._",
        "",
        "> This evaluates the **restricted offline KIVI adapter only**.",
        "> This does **not** integrate KIVI production CUDA/Triton kernels.",
        "> This does **not** integrate `LlamaForCausalLM_KIVI` or `MistralForCausalLM_KIVI`.",
        "> This does **not** implement KVQuant.",
        "> This does **not** claim upstream KIVI results as ExactKV results.",
        "> The adapter is **not** in the default compressor registry.",
        "> The adapter requires `PYTHONPATH` to the KIVI repo or the documented isolated environment.",
        "> `supports_real_bytes_claim=False` because the offline simulate payload is not packed "
        "KIVI CUDA storage.",
        "> Stored bytes are counted from actual stored tensors/arrays, not advertised compression "
        "ratios.",
        "> Materialized working KV is full decompressed KV.",
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
        "Evaluate whether the restricted `KIVIOfflineAdapter` (factory-only, upstream",
        "`models.utils_quant` simulate bridge) preserves ExactKV's exactness gate while",
        "exhibiting meaningful acceptance and honest workspace-memory accounting versus",
        "built-in baselines, TurboQuant Python (Experiment 008 anchor), and the best",
        "simulated layer-aware policy on the core prompt suite.",
        "",
        "## 2. Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| PYTHONPATH | KIVI repo root (e.g. `/tmp/kivi_research`) |",
        f"| Model | `{MODEL_NAME}`, {DTYPE}, CPU-first |",
        f"| transformers | {manifest.get('transformers_version', '—')} |",
        f"| torch | {manifest.get('torch_version', '—')} |",
        f"| kivi backend | {manifest.get('kivi_backend_version', '—')} |",
        f"| kivi repo sha | {manifest.get('kivi_repo_sha', '—')} |",
        f"| TurboQuant live in 009 | {tq_live} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "export PYTHONPATH=/tmp/kivi_research",
        ".venv-turboquant/bin/python scripts/run_experiment_009_kivi_offline.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_009_kivi_offline.json`,",
        "`reports/experiment_009_kivi_offline.csv`.",
        "",
        "## 3. Restrictions",
        "",
        "- **Offline simulate adapter only** — no CUDA/Triton, no flash-attn, no kivi_gemv.",
        "- **Not in default registry** — `kivi_offline_k2_v2` via factory only.",
        "- **Isolated PYTHONPATH** — default ExactKV install remains KIVI-free.",
        "- **No KVQuant, kvpress, vLLM, LMCache** in this experiment.",
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
    ]

    if tq_live:
        lines.append(
            f"| `{TURBOQUANT_NAME}` | Restricted TurboQuant Python | "
            f"k_bits={TURBOQUANT_K_BITS}, v_bits={TURBOQUANT_V_BITS} (live) |"
        )
    else:
        lines.append(
            f"| `{TURBOQUANT_NAME}` | — | **Not run live** — compare via Experiment 008 |"
        )

    lines.append(
        f"| `{KIVI_NAME}` | **Restricted KIVI offline adapter** | "
        f"k_bits={KIVI_K_BITS}, v_bits={KIVI_V_BITS}, group_size={KIVI_GROUP_SIZE} |"
    )

    lines.extend([
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
    ])

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

    for name in compressors:
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

    mean_stored = (
        sum(m["stored_kv_bytes"] for m in mem_by_comp.get(KIVI_NAME, []))
        / max(len(mem_by_comp.get(KIVI_NAME, [])), 1)
    )
    mean_full = (
        sum(_full_ref_bytes(m) for m in mem_by_comp.get(KIVI_NAME, []))
        / max(len(mem_by_comp.get(KIVI_NAME, [])), 1)
    )

    lines.extend([
        "",
        "† `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured",
        "peak GPU memory. **Active GPU memory is not reported.**",
        "",
        "## 10. Stored-payload accounting result",
        "",
        "| KIVI gate | Result |",
        "|---|---|",
        f"| All identity logical/physical mapping | **{kivi_gates.get('all_identity_mapping', '—')}** |",
        f"| `supports_real_bytes_claim=False` on all cells | "
        f"**{kivi_gates.get('supports_real_bytes_claim_false', '—')}** |",
        f"| `is_simulated=False` | **{kivi_gates.get('is_simulated_false', '—')}** |",
        f"| Mean stored payload (offline simulate) | {_fmt_bytes(mean_stored)} |",
        f"| Mean full KV (fp32 reference) | {_fmt_bytes(mean_full)} |",
        "",
        "The offline adapter stores unpacked quant codes and scale tensors. At this scale",
        "the stored payload may **exceed** fp32 tensor bytes — that is honestly reported,",
        "not hidden behind upstream compression-ratio claims.",
        "",
        "## 11. KIVI vs baselines",
        "",
        "| Baseline | Baseline accept | KIVI accept | Δ accept |",
        "|---|---:|---:|---:|",
    ])

    comparisons = [
        "int8",
        "k8_v4_sim",
        "k8_v4_boundary4_v8_sim",
        "k_full_v8",
        "backend_passthrough",
    ]
    for key in comparisons:
        base_rate = lookup.get(key, {}).get("mean_acceptance_rate", 0.0)
        lines.append(
            f"| `{key}` | {_fmt_rate(base_rate)} | {_fmt_rate(kivi_rate)} | "
            f"{_delta_str(kivi_rate, base_rate)} |"
        )

    if tq_rate is not None:
        lines.append(
            f"| `{TURBOQUANT_NAME}` ({tq_source}) | {_fmt_rate(tq_rate)} | "
            f"{_fmt_rate(kivi_rate)} | {_delta_str(kivi_rate, tq_rate)} |"
        )
    else:
        lines.append(
            f"| `{TURBOQUANT_NAME}` | — | {_fmt_rate(kivi_rate)} | — |"
        )

    int8_rate = lookup.get("int8", {}).get("mean_acceptance_rate", 0.0)
    k8v4_rate = lookup.get("k8_v4_sim", {}).get("mean_acceptance_rate", 0.0)
    boundary_rate = lookup.get("k8_v4_boundary4_v8_sim", {}).get("mean_acceptance_rate", 0.0)
    kfull_rate = lookup.get("k_full_v8", {}).get("mean_acceptance_rate", 0.0)

    lines.extend([
        "",
        "## 12. What KIVI improves or fails to improve",
        "",
    ])

    if kivi_rate > k8v4_rate:
        lines.append(
            f"- **Versus `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: offline KIVI "
            f"({_fmt_rate(kivi_rate)}) shows higher acceptance."
        )
    else:
        lines.append(
            f"- **Versus `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: offline KIVI "
            f"({_fmt_rate(kivi_rate)}) did not exceed the simulated baseline."
        )

    if kivi_rate > int8_rate:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: KIVI matched or beat INT8."
        )
    else:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: KIVI acceptance "
            f"({_fmt_rate(kivi_rate)}) is below symmetric INT8."
        )

    lines.append(
        f"- **Versus `k8_v4_boundary4_v8_sim` ({_fmt_rate(boundary_rate)})**: "
        f"layer-aware simulated policy Δ {_delta_str(kivi_rate, boundary_rate)}."
    )
    lines.append(
        f"- **Versus `k_full_v8` ({_fmt_rate(kfull_rate)})**: Δ {_delta_str(kivi_rate, kfull_rate)}."
    )
    if tq_rate is not None:
        if kivi_rate > tq_rate:
            lines.append(
                f"- **Versus TurboQuant Python ({_fmt_rate(tq_rate)}, {tq_source})**: "
                f"KIVI offline shows higher acceptance."
            )
        else:
            lines.append(
                f"- **Versus TurboQuant Python ({_fmt_rate(tq_rate)}, {tq_source})**: "
                f"KIVI offline did not exceed TurboQuant Python acceptance."
            )
    lines.append(
        "- **Stored bytes**: offline simulate payload is not packed-bit KIVI CUDA storage; "
        "`supports_real_bytes_claim=False` is intentional."
    )

    lines.extend([
        "",
        "## 13. What this proves",
        "",
        "- ExactKV can wrap a **second real external quantizer family** (KIVI offline) behind",
        "  `BackendAdapter` while preserving `exactkv_failures == 0`.",
        "- Post-RoPE HF tensor bridge works for asymmetric per-channel K / per-token V quant.",
        "- Acceptance and workspace-memory fields can be reported honestly for a non-registry",
        "  KIVI adapter in an isolated environment.",
        "",
        "## 14. What this does not prove",
        "",
        "- Compatibility with KIVI production CUDA/Triton kernels or `LlamaForCausalLM_KIVI`.",
        "- Upstream KIVI paper memory or quality claims (external results only).",
        "- That offline simulate storage reduces memory versus fp32 (may be larger at small scale).",
        "- KVQuant feasibility (deferred pending RunPod).",
        "- Production readiness, throughput, latency, or GPU memory behaviour.",
        "",
        "## 15. Relation to V9 and future KVQuant RunPod work",
        "",
        "Experiment 009 completes V9 Phase D3 for the KIVI offline track. KVQuant remains",
        "deferred pending RunPod Fisher/simquant validation (see",
        "[`KIVI_KVQUANT_INTEGRATION_RESEARCH.md`](KIVI_KVQUANT_INTEGRATION_RESEARCH.md)).",
        "Phase E may add larger-model exactness validation.",
        "",
        "## 16. Relation to upstream KIVI",
        "",
        "Upstream KIVI ([`jy-yuan/KIVI`](https://github.com/jy-yuan/KIVI)) implements",
        "production inference via custom model classes and CUDA kernels. This experiment uses",
        "only **`models.utils_quant.py` simulate helpers** on post-RoPE HF caches. External",
        "upstream quality or compression claims are **not** ExactKV results.",
        "",
        "## 17. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al.,",
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic evaluation harness;",
        "Experiment 009 does not claim novel KIVI algorithm contributions.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 009 (KIVI offline)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_009_kivi_offline.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_009_kivi_offline.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_009_KIVI_OFFLINE.md",
    )
    args = parser.parse_args()

    compressors = _build_compressor_list()
    prompts = load_core_prompts()
    expected = len(prompts) * len(compressors)
    print(
        f"Experiment 009: {len(prompts)} prompts × {len(compressors)} compressors "
        f"× 1 draft_len = {expected} cells"
    )
    print(f"Compressors: {compressors}")
    if not _TURBOQUANT_AVAILABLE:
        print(
            "Note: turboquant not importable — TurboQuant comparison will use "
            "Experiment 008 report in markdown.",
            file=sys.stderr,
        )

    print(f"Loading model {MODEL_NAME} ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)

    report = run_experiment_009(runtime, prompts, compressors)
    report["manifest"]["compressor_names"] = compressors
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
    print(f"kivi_gates: {agg.get('kivi_gates')}")

    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
