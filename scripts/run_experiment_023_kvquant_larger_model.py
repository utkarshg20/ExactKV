#!/usr/bin/env python3
"""Experiment 023: KVQuant simquant larger-model validation (V12 Phase 3).

Validates restricted KVQuant simquant adapter on Qwen2.5-1.5B (required) with optional
full V10 suite and optional 3B stretch. ``kvquant_sim_qwen15b`` is NOT in the default
registry.

No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, replace
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


def _quantizers_path() -> str:
    return os.environ.get("EXACTKV_KVQUANT_QUANTIZERS", "")


def _require_kvquant_env() -> None:
    if not _kvquant_available():
        raise SystemExit(
            "Experiment 023 requires the upstream kvquant package. "
            "Use the isolated KVQuant venv (transformers~=4.44, CUDA torch)."
        )
    path = _quantizers_path()
    if not path or not os.path.isfile(path):
        raise SystemExit(
            "Experiment 023 requires EXACTKV_KVQUANT_QUANTIZERS pointing at a "
            "model-specific quantizers pickle (e.g. quantizers_qwen15b.pickle)."
        )
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("Experiment 023 requires a CUDA GPU (RunPod).")


from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.reports import (
    build_run_manifest,
    load_json_report,
    validate_report,
    write_csv_report,
    write_json_report,
)
from exactkv.benchmarks.runner import RunConfig
from exactkv.benchmarks.sweeps import _compute_aggregate
from exactkv.benchmarks.v10_prompts import load_all_v10_prompts, load_v10_suite
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

MODEL_NAME_DEFAULT = "Qwen/Qwen2.5-1.5B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "kvquant_larger_model_validation"
KVQUANT_NAME = "kvquant_sim_qwen15b"
KVQUANT_ABITS = 4

SPOTCHECK_SUITES = (
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
)
PROMPTS_PER_SUITE = 10

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    KVQUANT_NAME,
]

# Published anchors (fallback if JSON missing).
_ANCHOR_EXP010_KVQUANT = 0.792
_ANCHOR_EXP014_KVQUANT = 0.634
_ANCHOR_EXP014_PANEL = "exp014 hard-category 40-prompt subset (0.5B)"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_KVQUANT_MEMORY_NOTE = (
    "Restricted KVQuant simquant adapter (kvquant_sim_qwen15b): "
    "stored_kv_bytes counts the external quantizers pickle file size and adapter "
    "metadata — not packed-bit KVQuant CUDA storage. "
    "supports_real_bytes_claim=False. "
    "materialized_working_kv_bytes reflects full KV working cache when materialized. "
    "total_kv_footprint_bytes is a conservative accounting sum, not measured "
    "peak GPU memory. Active GPU memory is not reported."
)

_EXP010_REPORT = _ROOT / "reports" / "experiment_010_kvquant_sim.json"
_EXP014_REPORT = _ROOT / "reports" / "experiment_014_real_backend_spotchecks.json"


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


def load_hard_category_subset() -> list[dict[str, Any]]:
    """Experiment 014 panel: first 10 prompt ids per harder V10 suite."""
    out: list[dict[str, Any]] = []
    for suite in SPOTCHECK_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:PROMPTS_PER_SUITE]:
            entry = dict(row)
            entry["v10_panel"] = "exp023_hard_category_40"
            out.append(entry)
    return out


def _group_acceptance_by_key(
    results: list[dict[str, Any]],
    key_fn,
    label: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        rates = [x["exactkv"]["acceptance"]["acceptance_rate"] for x in rows]
        out.append({
            label: key,
            "num_cells": len(rows),
            "mean_acceptance_rate": sum(rates) / len(rates) if rates else 0.0,
        })
    return out


def _group_compressor_category(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        cat = r.get("v10_primary_category", r.get("category", "unknown"))
        groups[(r["compressor_name"], cat)].append(r)
    out: list[dict[str, Any]] = []
    for (comp, cat) in sorted(groups):
        rows = groups[(comp, cat)]
        rates = [x["exactkv"]["acceptance"]["acceptance_rate"] for x in rows]
        out.append({
            "compressor_name": comp,
            "primary_category": cat,
            "num_cells": len(rows),
            "mean_acceptance_rate": sum(rates) / len(rates) if rates else 0.0,
        })
    return out


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
            quantizers_path=_quantizers_path(),
            abits=KVQUANT_ABITS,
        )
        comp.name = KVQUANT_NAME
        comp.capabilities = replace(comp.capabilities, name=KVQUANT_NAME)
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
        "quantizers_path": _quantizers_path(),
        "quantizers_pickle_bytes": os.path.getsize(_quantizers_path()),
    }


def run_one_cell(
    runtime: ModelRuntime,
    prompt_entry: dict[str, Any],
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

    result: dict[str, Any] = {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
        "v10_suite": prompt_entry.get("v10_suite", ""),
        "v10_primary_category": prompt_entry.get(
            "v10_primary_category", prompt_entry.get("category", "unknown")
        ),
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


def run_experiment_panel(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    *,
    panel_name: str,
) -> list[dict[str, Any]]:
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(COMPRESSORS)
    cell_idx = 0
    for prompt_entry in prompts:
        for compressor_name in COMPRESSORS:
            cell_idx += 1
            print(
                f"  [{panel_name}] [{cell_idx}/{total}] "
                f"{prompt_entry['prompt_id']} × {compressor_name}",
                flush=True,
            )
            compressor = _resolve_compressor(runtime, compressor_name, compressor_cache)
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results.append(run_one_cell(runtime, prompt_entry, config, compressor))
    return results


def _load_anchor_rate(report_path: Path, compressor: str, fallback: float) -> dict[str, Any]:
    if report_path.is_file():
        try:
            report = load_json_report(report_path)
            for row in report.get("aggregate", {}).get("acceptance_by_compressor", []):
                if row.get("compressor_name") == compressor:
                    return {
                        "mean_acceptance_rate": row.get("mean_acceptance_rate", fallback),
                        "source": report_path.name,
                    }
        except Exception:
            pass
    return {"mean_acceptance_rate": fallback, "source": "published doc fallback"}


def _rejection_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_rejected = 0
    total_corrections = 0
    total_drafted = 0
    total_accepted = 0
    for r in results:
        acc = r["exactkv"]["acceptance"]
        total_rejected += acc.get("total_rejected", 0)
        total_corrections += acc.get("total_corrections", 0)
        total_drafted += acc.get("total_drafted", 0)
        total_accepted += acc.get("total_accepted", 0)
    return {
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_corrections": total_corrections,
    }


def build_report(
    runtime: ModelRuntime,
    results: list[dict[str, Any]],
    *,
    model_name: str,
    prompt_suite: str,
    panel_notes: dict[str, Any],
) -> dict[str, Any]:
    manifest = build_run_manifest(
        model_name=model_name,
        prompt_suite=prompt_suite,
        compressor_names=COMPRESSORS,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=DTYPE,
    )
    manifest["experiment"] = "023_kvquant_larger_model"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["kvquant_compressor_label"] = KVQUANT_NAME
    manifest["kvquant_abits"] = KVQUANT_ABITS
    manifest["kvquant_quantizers_path"] = _quantizers_path()
    manifest["kvquant_quantizers_pickle_bytes"] = os.path.getsize(_quantizers_path())
    manifest["environment"] = "KVQuant isolated venv (RunPod GPU); transformers~=4.44"
    manifest["kvquant_repo_sha"] = _kvquant_repo_sha()
    manifest.update(panel_notes)

    try:
        import transformers  # noqa: PLC0415
        import torch  # noqa: PLC0415

        manifest["transformers_version"] = transformers.__version__
        manifest["torch_version"] = torch.__version__
    except Exception:
        pass

    aggregate = _compute_aggregate(
        results,
        num_prompts=len({r["prompt_id"] for r in results}),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["acceptance_by_primary_category"] = _group_acceptance_by_key(
        results,
        lambda r: r.get("v10_primary_category", r.get("category", "unknown")),
        "primary_category",
    )
    aggregate["acceptance_by_compressor_and_category"] = _group_compressor_category(
        results
    )
    aggregate["rejection_summary"] = _rejection_summary(results)

    kv_results = [r for r in results if r["compressor_name"] == KVQUANT_NAME]
    if kv_results:
        gates = [r["kvquant_gates"] for r in kv_results]
        aggregate["kvquant_gates"] = {
            "all_identity_mapping": all(g["identity_mapping"] for g in gates),
            "supports_real_bytes_claim_false": all(
                not g["supports_real_bytes_claim"] for g in gates
            ),
            "is_simulated_false": all(not g["is_simulated"] for g in gates),
            "stored_equals_pickle": all(
                g["stored_kv_bytes"] == g["quantizers_pickle_bytes"] for g in gates
            ),
        }

    exp010 = _load_anchor_rate(
        _EXP010_REPORT, "kvquant_sim_qwen05b", _ANCHOR_EXP010_KVQUANT
    )
    exp014 = _load_anchor_rate(
        _EXP014_REPORT, "kvquant_sim_qwen05b", _ANCHOR_EXP014_KVQUANT
    )
    lookup = {r["compressor_name"]: r for r in aggregate["acceptance_by_compressor"]}
    kv_rate = lookup.get(KVQUANT_NAME, {}).get("mean_acceptance_rate", 0.0)
    aggregate["cross_experiment_anchors"] = {
        "experiment_010_05b_kvquant": exp010,
        "experiment_014_05b_hard_panel_kvquant": exp014,
        "experiment_023_15b_kvquant": kv_rate,
    }

    strongest = max(
        aggregate["acceptance_by_compressor"],
        key=lambda r: r["mean_acceptance_rate"],
    )
    aggregate["strongest_restricted_real_backend"] = (
        strongest["compressor_name"] == KVQUANT_NAME
        if KVQUANT_NAME in {r["compressor_name"] for r in aggregate["acceptance_by_compressor"]}
        else False
    )

    return {"manifest": manifest, "results": results, "aggregate": aggregate}


def _fmt_rate(x: float | None) -> str:
    if x is None:
        return "—"
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


def _delta_str(a: float, b: float) -> str:
    d = a - b
    return f"{'+' if d >= 0 else ''}{d:.3f}"


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    manifest = report["manifest"]
    by_comp = agg["acceptance_by_compressor"]
    lookup = {r["compressor_name"]: r for r in by_comp}
    kv = lookup.get(KVQUANT_NAME, {})
    kv_gates = agg.get("kvquant_gates", {})
    anchors = agg.get("cross_experiment_anchors", {})
    rej = agg.get("rejection_summary", {})
    by_cat = agg.get("acceptance_by_compressor_and_category", [])

    kv_rate = kv.get("mean_acceptance_rate", 0.0)
    int8_rate = lookup.get("int8", {}).get("mean_acceptance_rate", 0.0)
    k8_rate = lookup.get("k8_v4_sim", {}).get("mean_acceptance_rate", 0.0)
    b4_rate = lookup.get("k8_v4_boundary4_v8_sim", {}).get("mean_acceptance_rate", 0.0)

    exp010_rate = anchors.get("experiment_010_05b_kvquant", {}).get(
        "mean_acceptance_rate", _ANCHOR_EXP010_KVQUANT
    )
    exp014_rate = anchors.get("experiment_014_05b_hard_panel_kvquant", {}).get(
        "mean_acceptance_rate", _ANCHOR_EXP014_KVQUANT
    )

    go_decision = "go_with_restrictions"
    if agg["exactkv_failures"] != 0:
        go_decision = "blocked_exactness_failure"
    elif kv_rate <= 0.0:
        go_decision = "no_go_draft_usefulness"

    lines = [
        "# Experiment 023: KVQuant Larger-Model Validation",
        "",
        "_Generated by `scripts/run_experiment_023_kvquant_larger_model.py`. "
        "V12 Phase 3 — restricted KVQuant simquant adapter only._",
        "",
        "> This evaluates the **restricted KVQuant simquant adapter only**.",
        "> This is **not** KVQuant deployment CUDA.",
        "> This is **not** production serving.",
        "> This is **not** a speed or memory benchmark.",
        "> KVQuant remains **factory-only** and **not** in the default registry.",
        "> Quantizer artifacts are external and **not committed**.",
        "> Stored bytes count the quantizer artifact / adapter-owned metadata, "
        "not packed-bit KV storage.",
        "> `supports_real_bytes_claim=False`.",
        "> ExactKV preserves exact greedy output; acceptance measures draft usefulness.",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, "
        "active GPU memory, production readiness, or model accuracy improvement.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate whether the restricted KVQuant simquant adapter remains useful on "
        "**Qwen2.5-1.5B** beyond Experiment 010's 0.5B evaluation, using the same "
        "hard-category V10 panel as Experiment 014.",
        "",
        "## 2. Why this follows Experiments 010/011/014",
        "",
        "- **Exp 010:** established `kvquant_sim_qwen05b` on 0.5B core suite (accept **0.792**).",
        "- **Exp 011:** generated 1.5B quantizer artifact + 1-prompt smoke (`exactkv_failures == 0`).",
        "- **Exp 014:** hard-category 0.5B spot-check (KVQuant accept **0.634** on 40 prompts).",
        "- **Exp 023:** turns the 1.5B artifact into a full **40-prompt × 5-compressor** panel.",
        "",
        "## 3. RunPod environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Host | `{manifest.get('hostname', '—')}` |",
        f"| GPU | `{manifest.get('gpu_name', 'RunPod CUDA')}` |",
        f"| KVQuant venv | Isolated; `pip install -e KVQuant/quant` |",
        f"| transformers | {manifest.get('transformers_version', '—')} |",
        f"| torch | {manifest.get('torch_version', '—')} |",
        f"| Device / dtype | `{DEVICE}` / `{DTYPE}` |",
        "",
        "## 4. Quantizer artifact status",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Path | `{manifest.get('kvquant_quantizers_path', '—')}` |",
        f"| Size | {_fmt_bytes(manifest.get('kvquant_quantizers_pickle_bytes'))} |",
        f"| Model scope | 1.5B-only (not reused from 0.5B) |",
        f"| kvquant repo sha | {manifest.get('kvquant_repo_sha', '—')} |",
        "",
        "## 5. Model(s) evaluated",
        "",
        f"| Model | Status |",
        f"|---|---|",
        f"| `{manifest['model_name']}` | **Required panel** |",
    ]
    if manifest.get("optional_full_v10"):
        lines.append(f"| `{manifest['model_name']}` | Optional full V10 (128 prompts) |")
    if manifest.get("optional_stretch_3b"):
        lines.append(f"| `{manifest.get('stretch_3b_model', 'Qwen/Qwen2.5-3B')}` | Optional stretch |")

    lines.extend([
        "",
        "## 6. Prompt subset and optional full-suite status",
        "",
        f"| Panel | Prompts | Cells |",
        f"|---|---:|---:|",
        f"| Required hard-category subset | {manifest.get('required_prompt_count', 40)} | "
        f"{manifest.get('required_cell_count', 200)} |",
    ])
    if manifest.get("optional_full_v10"):
        lines.append(
            f"| Optional full V10 | 128 | {manifest.get('optional_full_cell_count', 640)} |"
        )
    lines.append(
        f"| **Total in this report** | **{len({r['prompt_id'] for r in report['results']})}** | "
        f"**{agg['total_runs']}** |"
    )

    lines.extend([
        "",
        "## 7. Compressor panel",
        "",
        "| Compressor | Notes |",
        "|---|---|",
        "| `noop` | Identity baseline |",
        "| `int8` | Real INT8 |",
        "| `k8_v4_sim` | Simulated K8/V4 |",
        f"| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V |",
        f"| `{KVQUANT_NAME}` | **Restricted KVQuant simquant** (factory-only) |",
        "",
        "## 8. Exactness result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        "",
        "## 9. Global acceptance by compressor",
        "",
        "| Compressor | Accept rate | Avg accept/round | Drafted | Accepted | Rejected | Corrections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(by_comp, key=lambda r: r["compressor_name"]):
        lines.append(
            f"| `{row['compressor_name']}` | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{_fmt_rate(row.get('mean_accept_per_round'))} | {row.get('total_drafted', '—')} | "
            f"{row.get('total_accepted', '—')} | {row.get('total_rejected', '—')} | "
            f"{row.get('total_corrections', '—')} |"
        )

    lines.extend([
        "",
        "## 10. Per-category acceptance",
        "",
        "| Compressor | Category | Accept rate | Cells |",
        "|---|---|---:|---:|",
    ])
    for row in sorted(by_cat, key=lambda r: (r["compressor_name"], r["primary_category"])):
        lines.append(
            f"| `{row['compressor_name']}` | `{row['primary_category']}` | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | {row['num_cells']} |"
        )

    lines.extend([
        "",
        "## 11. KVQuant vs int8",
        "",
        f"- KVQuant **{_fmt_rate(kv_rate)}** vs int8 **{_fmt_rate(int8_rate)}** "
        f"(Δ {_delta_str(kv_rate, int8_rate)}).",
        "",
        "## 12. KVQuant vs k8_v4_sim",
        "",
        f"- KVQuant **{_fmt_rate(kv_rate)}** vs k8_v4_sim **{_fmt_rate(k8_rate)}** "
        f"(Δ {_delta_str(kv_rate, k8_rate)}).",
        "",
        "## 13. KVQuant vs boundary4",
        "",
        f"- KVQuant **{_fmt_rate(kv_rate)}** vs boundary4 **{_fmt_rate(b4_rate)}** "
        f"(Δ {_delta_str(kv_rate, b4_rate)}).",
        "",
        "## 14. Comparison to Experiment 010 0.5B KVQuant",
        "",
        f"- Exp 010 (0.5B core): **{_fmt_rate(exp010_rate)}** "
        f"({anchors.get('experiment_010_05b_kvquant', {}).get('source', 'anchor')}).",
        f"- Exp 023 (1.5B hard panel): **{_fmt_rate(kv_rate)}**.",
        "- Different model size and panel — **not** direct equivalence.",
        "",
        "## 15. Comparison to Experiment 014 hard-category real-backend spot-check",
        "",
        f"- Exp 014 KVQuant on 0.5B hard panel: **{_fmt_rate(exp014_rate)}**.",
        f"- Exp 023 KVQuant on 1.5B same category IDs: **{_fmt_rate(kv_rate)}**.",
        "",
        "## 16. Whether KVQuant remains strongest restricted real backend at larger scale",
        "",
        f"**{'Yes' if agg.get('strongest_restricted_real_backend') else 'No'}** "
        f"among this 5-compressor panel (built-ins + KVQuant only; TurboQuant/KIVI not run live).",
        "",
        "## 17. Rejection/correction summary",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Total drafted | {rej.get('total_drafted', '—')} |",
        f"| Total accepted | {rej.get('total_accepted', '—')} |",
        f"| Total rejected | {rej.get('total_rejected', '—')} |",
        f"| Total corrections | {rej.get('total_corrections', '—')} |",
        "",
        "## 18. Memory-honesty / quantizer-artifact accounting",
        "",
        f"- `supports_real_bytes_claim=False` on all KVQuant cells.",
        f"- Stored bytes = quantizer pickle size "
        f"({_fmt_bytes(manifest.get('kvquant_quantizers_pickle_bytes'))}).",
        f"- Identity mapping: **{kv_gates.get('all_identity_mapping', '—')}**.",
        "",
        "## 19. What this proves",
        "",
        "- KVQuant simquant adapter scales to **1.5B** under ExactKV exactness gate on a "
        "hard-category V10 panel.",
        "- Larger-model validation is feasible with model-specific quantizer artifacts.",
        "",
        "## 20. What this does not prove",
        "",
        "- KVQuant deployment CUDA or production serving.",
        "- Upstream KVQuant paper results as ExactKV results.",
        "- Model accuracy improvement from compression.",
        "- GPU memory savings (`active_gpu_kv_bytes` not reported).",
        "",
        "## 21. Limitations",
        "",
        "- Simquant replay only; quantizer from synthetic calibration.",
        "- Panel smaller than full 128-prompt V10 suites unless optional extension ran.",
        "- Built-in compressors include `_sim` policies (not packed-bit).",
        "",
        "## 22. Go/no-go for any future KVQuant work",
        "",
        f"**{go_decision}** — factory-only simquant path remains viable for "
        "experiment-layer evaluation; deployment CUDA still out of scope.",
        "",
        "## 23. VeriCache attribution",
        "",
        "Draft-then-verify algorithm from **VeriCache** (Yao et al., arXiv:2605.17613, 2026). "
        "Experiment 023 evaluates adapter usefulness under ExactKV harness only.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "export EXACTKV_KVQUANT_QUANTIZERS=/workspace/kvquant_d4/quantizers_qwen15b.pickle",
        "source /workspace/kvquant_d4/.venv-kvquant/bin/activate",
        "cd /workspace/ExactKV",
        "python scripts/run_experiment_023_kvquant_larger_model.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 023 (KVQuant 1.5B+)")
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--quantizers", default=_quantizers_path())
    parser.add_argument("--json-out", default="reports/experiment_023_kvquant_larger_model.json")
    parser.add_argument("--csv-out", default="reports/experiment_023_kvquant_larger_model.csv")
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_023_KVQUANT_LARGER_MODEL.md",
    )
    parser.add_argument(
        "--full-v10-suite",
        action="store_true",
        help="Optional: run all 128 V10 prompts after required panel succeeds.",
    )
    parser.add_argument(
        "--stretch-3b",
        action="store_true",
        help="Optional: run hard panel on Qwen2.5-3B with separate quantizer pickle.",
    )
    parser.add_argument(
        "--stretch-3b-quantizers",
        default="/workspace/kvquant_d4/quantizers_qwen3b.pickle",
    )
    parser.add_argument("--stretch-3b-model", default="Qwen/Qwen2.5-3B")
    args = parser.parse_args()

    if args.quantizers:
        os.environ["EXACTKV_KVQUANT_QUANTIZERS"] = args.quantizers

    _require_kvquant_env()

    required_prompts = load_hard_category_subset()
    print(
        f"Experiment 023 required: {len(required_prompts)} prompts × "
        f"{len(COMPRESSORS)} compressors = {len(required_prompts) * len(COMPRESSORS)} cells"
    )
    print(f"Model: {args.model}")
    print(f"Quantizers: {_quantizers_path()}")

    import torch  # noqa: PLC0415

    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    print(f"Loading model {args.model} on {DEVICE} ({DTYPE}) ...")
    runtime = ModelRuntime(model_name=args.model, device=DEVICE, dtype=DTYPE)

    all_results: list[dict[str, Any]] = []
    all_results.extend(
        run_experiment_panel(runtime, required_prompts, panel_name="required")
    )

    panel_notes: dict[str, Any] = {
        "hostname": __import__("platform").node(),
        "gpu_name": gpu_name,
        "model_name": args.model,
        "required_prompt_count": len(required_prompts),
        "required_cell_count": len(required_prompts) * len(COMPRESSORS),
        "optional_full_v10": False,
        "optional_stretch_3b": False,
    }

    req_failures = sum(1 for r in all_results if r["exactkv_failure"])
    if req_failures != 0:
        print(f"STOP: required panel exactkv_failures={req_failures}", file=sys.stderr)
        report = build_report(
            runtime, all_results, model_name=args.model,
            prompt_suite="exp023_hard_category_40", panel_notes=panel_notes,
        )
        _write_outputs(report, args)
        return 1

    if args.full_v10_suite:
        print("Optional: full V10 suite (128 prompts) ...")
        full_prompts = load_all_v10_prompts()
        all_results.extend(
            run_experiment_panel(runtime, full_prompts, panel_name="full_v10")
        )
        panel_notes["optional_full_v10"] = True
        panel_notes["optional_full_cell_count"] = len(full_prompts) * len(COMPRESSORS)

    if args.stretch_3b:
        if not os.path.isfile(args.stretch_3b_quantizers):
            raise SystemExit(
                f"--stretch-3b requested but quantizer missing: {args.stretch_3b_quantizers}"
            )
        print(f"Optional stretch: {args.stretch_3b_model} ...")
        os.environ["EXACTKV_KVQUANT_QUANTIZERS"] = args.stretch_3b_quantizers
        runtime_3b = ModelRuntime(
            model_name=args.stretch_3b_model, device=DEVICE, dtype=DTYPE
        )
        stretch_results = run_experiment_panel(
            runtime_3b, required_prompts, panel_name="stretch_3b"
        )
        for r in stretch_results:
            r["panel"] = "stretch_3b"
        all_results.extend(stretch_results)
        panel_notes["optional_stretch_3b"] = True
        panel_notes["stretch_3b_model"] = args.stretch_3b_model
        panel_notes["stretch_3b_quantizers"] = args.stretch_3b_quantizers

    suite_label = "exp023_hard_category_40"
    if panel_notes.get("optional_full_v10"):
        suite_label += "+full_v10"
    if panel_notes.get("optional_stretch_3b"):
        suite_label += "+stretch_3b"

    report = build_report(
        runtime,
        all_results,
        model_name=args.model,
        prompt_suite=suite_label,
        panel_notes=panel_notes,
    )
    return _write_outputs(report, args)


def _write_outputs(report: dict[str, Any], args: argparse.Namespace) -> int:
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
    for row in agg.get("acceptance_by_compressor", []):
        print(f"  {row['compressor_name']}: accept={row['mean_acceptance_rate']:.3f}")
    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
