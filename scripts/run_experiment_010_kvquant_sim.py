#!/usr/bin/env python3
"""Experiment 010: restricted KVQuant simquant adapter vs ExactKV baselines (V9 Phase D6).

Run ONLY in the isolated KVQuant RunPod environment with
``EXACTKV_KVQUANT_QUANTIZERS`` pointing at a validated quantizers pickle.
``kvquant_sim_qwen05b`` is NOT in the default compressor registry.

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes
fields are produced.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _kvquant_available() -> bool:
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_QUANTIZERS_PATH = os.environ.get("EXACTKV_KVQUANT_QUANTIZERS", "")


def _require_kvquant_env() -> None:
    """Gate live experiment execution to the isolated KVQuant RunPod environment."""
    if not _kvquant_available():
        raise SystemExit(
            "Experiment 010 requires the upstream kvquant package. "
            "Use the isolated KVQuant venv from D4b/D5 with transformers~=4.44."
        )
    if not _QUANTIZERS_PATH or not os.path.isfile(_QUANTIZERS_PATH):
        raise SystemExit(
            "Experiment 010 requires EXACTKV_KVQUANT_QUANTIZERS pointing at a "
            "quantizers.pickle file (e.g. /workspace/kvquant_d4/quantizers_qwen05b.pickle)."
        )
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("Experiment 010 requires a CUDA GPU (RunPod L40S or equivalent).")


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
from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
PROMPT_SUITE = "core"
EXPERIMENT_CLASS = "kvquant_sim_real"

KVQUANT_NAME = "kvquant_sim_qwen05b"
KVQUANT_ABITS = 4

TURBOQUANT_NAME = "turboquant_python_k3_v3"
KIVI_NAME = "kivi_offline_k2_v2"

# Anchor acceptance rates from published Experiment 008/009 docs (fallback).
_ANCHOR_TURBOQUANT_ACCEPT = 0.435
_ANCHOR_KIVI_ACCEPT = 0.012

COMPRESSORS = [
    "noop",
    "backend_passthrough",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "k_full_v8",
    "k8_v_full",
    KVQUANT_NAME,
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_KVQUANT_MEMORY_NOTE = (
    "Restricted KVQuant simquant adapter (kvquant_sim_qwen05b): "
    "stored_kv_bytes counts the external quantizers pickle file size and adapter "
    "metadata — not packed-bit KVQuant CUDA storage. "
    "supports_real_bytes_claim=False. "
    "materialized_working_kv_bytes reflects full KV working cache when materialized. "
    "total_kv_footprint_bytes is a conservative accounting sum, not measured "
    "peak GPU memory. Active GPU memory is not reported."
)

_EXP008_REPORT = _ROOT / "reports" / "experiment_008_turboquant_python.json"
_EXP009_REPORT = _ROOT / "reports" / "experiment_009_kivi_offline.json"


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


def _kvquant_repo_sha() -> str | None:
    try:
        import kvquant.simquant_module_quantizer as smq  # noqa: PLC0415

        root = Path(smq.__file__).resolve().parents[2]
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
    if name == KVQUANT_NAME:
        comp = create_kvquant_sim_adapter(
            runtime,
            quantizers_path=_QUANTIZERS_PATH,
            abits=KVQUANT_ABITS,
        )
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


def _kvquant_gate_snapshot(
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
        "quantizers_path": _QUANTIZERS_PATH,
        "quantizers_pickle_bytes": os.path.getsize(_QUANTIZERS_PATH),
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
    if config.compressor_name == KVQUANT_NAME:
        mem.memory_claim_note = _KVQUANT_MEMORY_NOTE
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

    if config.compressor_name == KVQUANT_NAME:
        result["kvquant_gates"] = _kvquant_gate_snapshot(runtime, compressor, prompt)

    return result


def run_experiment_010(
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
    manifest["experiment"] = "010_kvquant_sim"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["kvquant_compressor_label"] = KVQUANT_NAME
    manifest["kvquant_abits"] = KVQUANT_ABITS
    manifest["kvquant_quantizers_path"] = _QUANTIZERS_PATH
    manifest["kvquant_quantizers_pickle_bytes"] = os.path.getsize(_QUANTIZERS_PATH)
    manifest["environment"] = "KVQuant isolated venv (RunPod L40S); transformers~=4.44"
    manifest["turboquant_included_live"] = False
    manifest["kivi_included_live"] = False
    manifest["turboquant_comparison_source"] = (
        str(_EXP008_REPORT) if _EXP008_REPORT.is_file() else "EXPERIMENT_008_TURBOQUANT_PYTHON.md"
    )
    manifest["kivi_comparison_source"] = (
        str(_EXP009_REPORT) if _EXP009_REPORT.is_file() else "EXPERIMENT_009_KIVI_OFFLINE.md"
    )
    manifest["kvquant_repo_sha"] = _kvquant_repo_sha()

    kv_adapter = compressor_cache.get(KVQUANT_NAME)
    if kv_adapter is not None:
        manifest["kvquant_backend_version"] = kv_adapter.capabilities.backend_version

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )

    kv_results = [r for r in results if r["compressor_name"] == KVQUANT_NAME]
    gates = [r["kvquant_gates"] for r in kv_results]
    aggregate["kvquant_gates"] = {
        "all_identity_mapping": all(g["identity_mapping"] for g in gates),
        "supports_real_bytes_claim_false": all(
            not g["supports_real_bytes_claim"] for g in gates
        ),
        "is_simulated_false": all(not g["is_simulated"] for g in gates),
        "quantizers_pickle_bytes_constant": len(
            {g["quantizers_pickle_bytes"] for g in gates}
        )
        == 1,
        "stored_equals_pickle": all(
            g["stored_kv_bytes"] == g["quantizers_pickle_bytes"] for g in gates
        ),
    }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _load_anchor_acceptance(
    report_path: Path,
    compressor_name: str,
    fallback: float,
    source_label: str,
) -> dict[str, Any]:
    if report_path.is_file():
        try:
            report = load_json_report(report_path)
            for row in report.get("aggregate", {}).get("acceptance_by_compressor", []):
                if row.get("compressor_name") == compressor_name:
                    return {
                        "mean_acceptance_rate": row.get("mean_acceptance_rate", fallback),
                        "source": report_path.name,
                    }
        except Exception:
            pass
    return {
        "mean_acceptance_rate": fallback,
        "source": source_label,
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
    kv = lookup.get(KVQUANT_NAME, {})
    kv_gates = agg.get("kvquant_gates", {})
    manifest = report["manifest"]

    tq_anchor = _load_anchor_acceptance(
        _EXP008_REPORT,
        TURBOQUANT_NAME,
        _ANCHOR_TURBOQUANT_ACCEPT,
        "EXPERIMENT_008_TURBOQUANT_PYTHON.md",
    )
    kivi_anchor = _load_anchor_acceptance(
        _EXP009_REPORT,
        KIVI_NAME,
        _ANCHOR_KIVI_ACCEPT,
        "EXPERIMENT_009_KIVI_OFFLINE.md",
    )

    kv_rate = kv.get("mean_acceptance_rate", 0.0)
    tq_rate = tq_anchor["mean_acceptance_rate"]
    kivi_rate = kivi_anchor["mean_acceptance_rate"]

    lines = [
        "# Experiment 010: Restricted KVQuant Simquant Adapter Evaluation",
        "",
        "_Generated by `scripts/run_experiment_010_kvquant_sim.py`. "
        "V9 Phase D6 — restricted KVQuant simquant adapter only._",
        "",
        "> This evaluates the **restricted KVQuant simquant adapter only**.",
        "> This uses KVQuant's **pre-RoPE** `k_proj`/`v_proj` simquant path.",
        "> This is **not** post-RoPE tensor approximation.",
        "> This does **not** integrate KVQuant deployment CUDA.",
        "> This does **not** integrate forked transformers deployment.",
        "> This does **not** claim upstream KVQuant results as ExactKV results.",
        "> The adapter is **not** in the default compressor registry.",
        "> The adapter requires a RunPod-style KVQuant environment and an **external "
        "quantizers pickle** (not committed).",
        "> `supports_real_bytes_claim=False` because no packed-bit KV storage is measured.",
        "> Stored bytes are counted from the quantizer artifact / adapter-owned metadata, "
        "not advertised compression ratios.",
        "> Materialized working KV is full KV or draft cache materialization as implemented "
        "by the adapter.",
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
        "Evaluate whether the restricted `KVQuantSimAdapter` (factory-only, pre-RoPE",
        "`QuantLinearSim` on a deep-copied draft model) preserves ExactKV's exactness gate",
        "while exhibiting meaningful acceptance and honest workspace-memory accounting versus",
        "built-in baselines on the core prompt suite. Cross-experiment anchors compare against",
        "TurboQuant Python (Experiment 008) and KIVI offline (Experiment 009) without",
        "running those adapters live in the KVQuant environment.",
        "",
        "## 2. Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | RunPod L40S (CUDA required) |",
        f"| KVQuant venv | Isolated; `pip install -e KVQuant/quant` |",
        f"| transformers | {manifest.get('transformers_version', '—')} (pin ~=4.44 in KVQuant venv) |",
        f"| torch | {manifest.get('torch_version', '—')} |",
        f"| Model | `{MODEL_NAME}`, {DTYPE}, `{DEVICE}` |",
        f"| Quantizers | `{manifest.get('kvquant_quantizers_path', '—')}` |",
        f"| Quantizers size | {_fmt_bytes(manifest.get('kvquant_quantizers_pickle_bytes'))} |",
        f"| kvquant backend | {manifest.get('kvquant_backend_version', '—')} |",
        f"| kvquant repo sha | {manifest.get('kvquant_repo_sha', '—')} |",
        f"| TurboQuant live in 010 | False |",
        f"| KIVI live in 010 | False |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "export EXACTKV_KVQUANT_QUANTIZERS=/workspace/kvquant_d4/quantizers_qwen05b.pickle",
        "source /workspace/kvquant_d4/.venv-kvquant/bin/activate",
        "cd /workspace/ExactKV",
        "python scripts/run_experiment_010_kvquant_sim.py",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_010_kvquant_sim.json`,",
        "`reports/experiment_010_kvquant_sim.csv`.",
        "",
        "## 3. Restrictions",
        "",
        "- **Simquant adapter only** — no deployment CUDA, no forked transformers.",
        "- **Not in default registry** — `kvquant_sim_qwen05b` via factory only.",
        "- **Isolated KVQuant venv** — default ExactKV install remains KVQuant-free.",
        "- **TurboQuant and KIVI not run live** — anchor comparisons from Experiments 008/009.",
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
        "| `backend_passthrough` | V6 BackendAdapter PoC | Lossless |",
        "| `int8` | Real INT8 | `supports_real_bytes_claim=True` |",
        "| `k8_v4_sim` | Simulated asymmetric K8/V4 | `is_simulated=True` |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) | Simulated V policy |",
        "| `k_full_v8` | Real K-full / V8 | |",
        "| `k8_v_full` | Real K8 / V-full | |",
        f"| `{KVQUANT_NAME}` | **Restricted KVQuant simquant adapter** | "
        f"abits={KVQUANT_ABITS}; pre-RoPE k_proj/v_proj |",
        f"| `{TURBOQUANT_NAME}` | — | **Not run live** — cross-experiment anchor (Exp 008) |",
        f"| `{KIVI_NAME}` | — | **Not run live** — cross-experiment anchor (Exp 009) |",
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

    mean_stored = (
        sum(m["stored_kv_bytes"] for m in mem_by_comp.get(KVQUANT_NAME, []))
        / max(len(mem_by_comp.get(KVQUANT_NAME, [])), 1)
    )
    mean_full = (
        sum(_full_ref_bytes(m) for m in mem_by_comp.get(KVQUANT_NAME, []))
        / max(len(mem_by_comp.get(KVQUANT_NAME, [])), 1)
    )

    lines.extend([
        "",
        "† `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured",
        "peak GPU memory. **Active GPU memory is not reported.**",
        "",
        "## 10. Stored-payload / quantizer-artifact accounting result",
        "",
        "| KVQuant gate | Result |",
        "|---|---|",
        f"| All identity logical/physical mapping | **{kv_gates.get('all_identity_mapping', '—')}** |",
        f"| `supports_real_bytes_claim=False` on all cells | "
        f"**{kv_gates.get('supports_real_bytes_claim_false', '—')}** |",
        f"| `is_simulated=False` | **{kv_gates.get('is_simulated_false', '—')}** |",
        f"| Stored bytes == quantizers pickle size | "
        f"**{kv_gates.get('stored_equals_pickle', '—')}** |",
        f"| Mean stored (quantizer artifact) | {_fmt_bytes(mean_stored)} |",
        f"| Mean full KV (fp16 reference) | {_fmt_bytes(mean_full)} |",
        "",
        "Stored bytes count the external quantizers pickle and adapter metadata — not",
        "packed-bit KVQuant CUDA storage. No compression-ratio claims are made.",
        "",
        "## 11. KVQuant vs live baselines",
        "",
        "| Baseline | Baseline accept | KVQuant accept | Δ accept |",
        "|---|---:|---:|---:|",
    ])

    live_comparisons = [
        "int8",
        "k8_v4_sim",
        "k8_v4_boundary4_v8_sim",
        "k_full_v8",
        "backend_passthrough",
    ]
    for key in live_comparisons:
        base_rate = lookup.get(key, {}).get("mean_acceptance_rate", 0.0)
        lines.append(
            f"| `{key}` | {_fmt_rate(base_rate)} | {_fmt_rate(kv_rate)} | "
            f"{_delta_str(kv_rate, base_rate)} |"
        )

    lines.extend([
        "",
        "## 12. KVQuant vs real-backend anchors (cross-experiment)",
        "",
        "_TurboQuant and KIVI were **not** run live in the KVQuant environment. "
        "Acceptance rates below are **cross-experiment anchors**, not same-run rows._",
        "",
        "| Anchor | Source | Anchor accept | KVQuant accept | Δ accept |",
        "|---|---|---:|---:|---:|",
        f"| `{TURBOQUANT_NAME}` | {tq_anchor['source']} | {_fmt_rate(tq_rate)} | "
        f"{_fmt_rate(kv_rate)} | {_delta_str(kv_rate, tq_rate)} |",
        f"| `{KIVI_NAME}` | {kivi_anchor['source']} | {_fmt_rate(kivi_rate)} | "
        f"{_fmt_rate(kv_rate)} | {_delta_str(kv_rate, kivi_rate)} |",
        "",
        "## 13. What KVQuant improves or fails to improve",
        "",
    ])

    int8_rate = lookup.get("int8", {}).get("mean_acceptance_rate", 0.0)
    k8v4_rate = lookup.get("k8_v4_sim", {}).get("mean_acceptance_rate", 0.0)
    boundary_rate = lookup.get("k8_v4_boundary4_v8_sim", {}).get("mean_acceptance_rate", 0.0)
    kfull_rate = lookup.get("k_full_v8", {}).get("mean_acceptance_rate", 0.0)
    passthrough_rate = lookup.get("backend_passthrough", {}).get("mean_acceptance_rate", 1.0)

    if kv_rate > k8v4_rate:
        lines.append(
            f"- **Versus `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: KVQuant simquant "
            f"({_fmt_rate(kv_rate)}) shows higher acceptance."
        )
    else:
        lines.append(
            f"- **Versus `k8_v4_sim` ({_fmt_rate(k8v4_rate)})**: KVQuant simquant "
            f"({_fmt_rate(kv_rate)}) did not exceed the simulated baseline."
        )

    if kv_rate > int8_rate:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: KVQuant matched or beat INT8."
        )
    else:
        lines.append(
            f"- **Versus `int8` ({_fmt_rate(int8_rate)})**: KVQuant acceptance "
            f"({_fmt_rate(kv_rate)}) is below symmetric INT8."
        )

    lines.append(
        f"- **Versus `k8_v4_boundary4_v8_sim` ({_fmt_rate(boundary_rate)})**: "
        f"layer-aware simulated policy Δ {_delta_str(kv_rate, boundary_rate)}."
    )
    lines.append(
        f"- **Versus `k_full_v8` ({_fmt_rate(kfull_rate)})**: Δ {_delta_str(kv_rate, kfull_rate)}."
    )
    lines.append(
        f"- **Versus `backend_passthrough` ({_fmt_rate(passthrough_rate)})**: "
        f"lossless passthrough remains the acceptance ceiling."
    )

    if kv_rate > tq_rate:
        lines.append(
            f"- **Versus TurboQuant Python anchor ({_fmt_rate(tq_rate)}, {tq_anchor['source']})**: "
            f"KVQuant simquant shows higher acceptance."
        )
    else:
        lines.append(
            f"- **Versus TurboQuant Python anchor ({_fmt_rate(tq_rate)}, {tq_anchor['source']})**: "
            f"KVQuant simquant did not exceed TurboQuant Python acceptance."
        )

    if kv_rate > kivi_rate:
        lines.append(
            f"- **Versus KIVI offline anchor ({_fmt_rate(kivi_rate)}, {kivi_anchor['source']})**: "
            f"KVQuant simquant shows higher acceptance."
        )
    else:
        lines.append(
            f"- **Versus KIVI offline anchor ({_fmt_rate(kivi_rate)}, {kivi_anchor['source']})**: "
            f"KVQuant simquant did not exceed KIVI offline acceptance."
        )

    lines.append(
        "- **Stored bytes**: quantizer pickle artifact — not packed-bit KV storage; "
        "`supports_real_bytes_claim=False` is intentional."
    )

    lines.extend([
        "",
        "## 14. What this proves",
        "",
        "- ExactKV can wrap a **third real external quantizer family** (KVQuant simquant) behind",
        "  `BackendAdapter` while preserving `exactkv_failures == 0`.",
        "- Pre-RoPE `k_proj`/`v_proj` quantization via draft-model clone works under full-KV",
        "  verification without post-RoPE tensor approximation.",
        "- Acceptance and workspace-memory fields can be reported honestly for a non-registry",
        "  KVQuant adapter in an isolated RunPod environment.",
        "",
        "## 15. What this does not prove",
        "",
        "- Compatibility with KVQuant deployment CUDA kernels or forked transformers deployment.",
        "- Upstream KVQuant paper memory, bandwidth, or quality claims (external results only).",
        "- That quantizer-pickle accounting reflects runtime GPU KV footprint.",
        "- Production readiness, throughput, latency, or GPU memory behaviour.",
        "",
        "## 16. Relation to V9 and future larger-model validation",
        "",
        "Experiment 010 completes V9 Phase D6 for the KVQuant simquant track. Phase E may add",
        "larger-model (≥1.5B) exactness validation on RunPod using the same adapter-isolation",
        "pattern.",
        "",
        "## 17. Relation to upstream KVQuant",
        "",
        "Upstream KVQuant ([`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant))",
        "implements deployment CUDA paths and forked transformers. This experiment uses only the",
        "**simquant** `QuantLinearSim` path on pre-RoPE projectors with an external quantizers",
        "pickle. External upstream quality or compression claims are **not** ExactKV results.",
        "",
        "## 18. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al.,",
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic evaluation harness;",
        "Experiment 010 does not claim novel KVQuant algorithm contributions.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    _require_kvquant_env()

    parser = argparse.ArgumentParser(description="Run Experiment 010 (KVQuant simquant)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_010_kvquant_sim.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_010_kvquant_sim.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_010_KVQUANT_SIM.md",
    )
    args = parser.parse_args()

    prompts = load_core_prompts()
    expected = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 010: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"× 1 draft_len = {expected} cells"
    )
    print(f"Compressors: {COMPRESSORS}")
    print(f"Quantizers: {_QUANTIZERS_PATH}")
    print(
        "Note: TurboQuant and KIVI are cross-experiment anchors only — not run live.",
        file=sys.stderr,
    )

    print(f"Loading model {MODEL_NAME} on {DEVICE} ({DTYPE}) ...")
    runtime = ModelRuntime(model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE)

    report = run_experiment_010(runtime, prompts)
    _assert_no_forbidden_fields(report)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    json_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"kvquant_gates: {agg.get('kvquant_gates')}")
    for row in agg.get("acceptance_by_compressor", []):
        print(
            f"  {row['compressor_name']}: accept={row['mean_acceptance_rate']:.3f}",
            flush=True,
        )

    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
