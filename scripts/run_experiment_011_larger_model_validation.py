#!/usr/bin/env python3
"""Experiment 011: larger-model RunPod validation (V9 Phase E).

Validates whether ExactKV findings hold on Qwen/Qwen2.5-1.5B (optional 3B stretch)
using the core compressor panel. Real backends (KVQuant/TurboQuant/KIVI) are
optional and never required for the Phase E gate.

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

MODEL_NAME_DEFAULT = "Qwen/Qwen2.5-1.5B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
PROMPT_SUITE = "core"
EXPERIMENT_CLASS = "larger_model_validation"

BASE_COMPRESSORS = [
    "noop",
    "backend_passthrough",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "k_full_v8",
    "k8_v_full",
]

KVQUANT_NAME = "kvquant_sim_qwen05b"  # adapter label; 1.5B uses separate pickle
KVQUANT_15B_PICKLE_DEFAULT = "/workspace/kvquant_d4/quantizers_qwen15b.pickle"
KVQUANT_CALIB_SCRIPT = _ROOT / "scripts" / "research" / "kvquant_runpod_synthetic_calib_15b.py"

# 0.5B anchors from Experiment 010 (cuda fp16, same suite/config).
_ANCHOR_05B_SOURCE = "EXPERIMENT_010_KVQUANT_SIM.md"
_ANCHOR_05B_ACCEPT = {
    "int8": 0.966,
    "k8_v4_sim": 0.898,
    "k8_v4_boundary4_v8_sim": 0.950,
    "k_full_v8": 0.990,
    "k8_v_full": 0.968,
}

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_EXP010_REPORT = _ROOT / "reports" / "experiment_010_kvquant_sim.json"


def _require_cuda_env() -> None:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit(
            "Experiment 011 requires a CUDA GPU (RunPod L40S or equivalent)."
        )


def _kvquant_available() -> bool:
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


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


def _load_05b_anchors() -> dict[str, float]:
    if _EXP010_REPORT.is_file():
        try:
            report = load_json_report(_EXP010_REPORT)
            lookup = {
                row["compressor_name"]: row["mean_acceptance_rate"]
                for row in report.get("aggregate", {}).get("acceptance_by_compressor", [])
            }
            out = {}
            for key in _ANCHOR_05B_ACCEPT:
                if key in lookup:
                    out[key] = lookup[key]
            if out:
                return out
        except Exception:
            pass
    return dict(_ANCHOR_05B_ACCEPT)


def _try_kvquant_15b_artifact(pickle_path: str, *, attempt_calib: bool) -> dict[str, Any]:
    """Return metadata about optional 1.5B KVQuant quantizer artifact."""
    meta: dict[str, Any] = {
        "attempted": attempt_calib,
        "pickle_path": pickle_path,
        "available": False,
        "generated": False,
        "error": None,
        "pickle_bytes": None,
        "key_count": None,
    }
    if os.path.isfile(pickle_path):
        meta["available"] = True
        meta["pickle_bytes"] = os.path.getsize(pickle_path)
        return meta

    if not attempt_calib:
        meta["error"] = "artifact missing; calibration not requested"
        return meta

    if not _kvquant_available():
        meta["error"] = "kvquant package not importable"
        return meta

    if not KVQUANT_CALIB_SCRIPT.is_file():
        meta["error"] = f"calibration script missing: {KVQUANT_CALIB_SCRIPT}"
        return meta

    print(f"Attempting KVQuant 1.5B synthetic calibration → {pickle_path}", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, str(KVQUANT_CALIB_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode != 0:
            meta["error"] = (proc.stderr or proc.stdout or "calibration failed")[:500]
            return meta
        if os.path.isfile(pickle_path):
            meta["available"] = True
            meta["generated"] = True
            meta["pickle_bytes"] = os.path.getsize(pickle_path)
            try:
                import pickle  # noqa: PLC0415

                with open(pickle_path, "rb") as handle:
                    q = pickle.load(handle)
                meta["key_count"] = len(q)
            except Exception as exc:
                meta["error"] = f"pickle load check failed: {exc}"
        else:
            meta["error"] = "calibration exited 0 but pickle not found"
    except subprocess.TimeoutExpired:
        meta["error"] = "calibration timed out (1800s)"
    except Exception as exc:
        meta["error"] = str(exc)[:500]
    return meta


def _build_compressor_list(
    *,
    try_kvquant: bool,
    kvquant_pickle: str,
) -> tuple[list[str], dict[str, Any]]:
    compressors = list(BASE_COMPRESSORS)
    kv_meta = _try_kvquant_15b_artifact(kvquant_pickle, attempt_calib=try_kvquant)
    if kv_meta.get("available"):
        compressors.append(KVQUANT_NAME)
        kv_meta["included_in_sweep"] = True
    else:
        kv_meta["included_in_sweep"] = False
    return compressors, kv_meta


def _resolve_compressor(
    runtime: ModelRuntime,
    name: str,
    cache: dict[str, Any],
    *,
    kvquant_pickle: str | None,
):
    if name in cache:
        return cache[name]
    if name == KVQUANT_NAME:
        if not kvquant_pickle or not os.path.isfile(kvquant_pickle):
            raise RuntimeError(f"KVQuant row requested but pickle missing: {kvquant_pickle}")
        from exactkv.compressors.kvquant_adapter import create_kvquant_sim_adapter

        comp = create_kvquant_sim_adapter(runtime, quantizers_path=kvquant_pickle)
    else:
        comp = get_compressor(name)
    cache[name] = comp
    return comp


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
    }


def run_experiment_011(
    runtime: ModelRuntime,
    prompts: list[dict],
    compressors: list[str],
    *,
    kvquant_meta: dict[str, Any],
) -> dict[str, Any]:
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total_cells = len(prompts) * len(compressors)
    cell_idx = 0
    kv_pickle = kvquant_meta.get("pickle_path") if kvquant_meta.get("included_in_sweep") else None

    for prompt_entry in prompts:
        for compressor_name in compressors:
            cell_idx += 1
            print(
                f"  [{cell_idx}/{total_cells}] {prompt_entry['prompt_id']} × "
                f"{compressor_name}",
                flush=True,
            )
            compressor = _resolve_compressor(
                runtime,
                compressor_name,
                compressor_cache,
                kvquant_pickle=kv_pickle,
            )
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
    manifest["experiment"] = "011_larger_model_validation"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["gpu_note"] = "RunPod L40S 46GB; larger-model validation not a performance benchmark"
    manifest["anchor_05b_source"] = (
        str(_EXP010_REPORT) if _EXP010_REPORT.is_file() else _ANCHOR_05B_SOURCE
    )
    manifest["kvquant_optional"] = kvquant_meta
    manifest["turboquant_included_live"] = False
    manifest["kivi_included_live"] = False

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=compressors,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["anchor_05b_acceptance"] = _load_05b_anchors()

    boundary = next(
        (r for r in aggregate["acceptance_by_compressor"]
         if r["compressor_name"] == "k8_v4_boundary4_v8_sim"),
        {},
    )
    k8v4 = next(
        (r for r in aggregate["acceptance_by_compressor"]
         if r["compressor_name"] == "k8_v4_sim"),
        {},
    )
    aggregate["layer_aware_transfer"] = {
        "boundary4_accept_15b": boundary.get("mean_acceptance_rate"),
        "k8_v4_sim_accept_15b": k8v4.get("mean_acceptance_rate"),
        "boundary4_minus_k8v4_15b": (
            (boundary.get("mean_acceptance_rate", 0) - k8v4.get("mean_acceptance_rate", 0))
            if boundary and k8v4 else None
        ),
        "boundary4_accept_05b_anchor": _load_05b_anchors().get("k8_v4_boundary4_v8_sim"),
        "k8_v4_sim_accept_05b_anchor": _load_05b_anchors().get("k8_v4_sim"),
    }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


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
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def generate_markdown_report(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    by_comp = agg["acceptance_by_compressor"]
    lookup = {row["compressor_name"]: row for row in by_comp}
    manifest = report["manifest"]
    anchors = agg.get("anchor_05b_acceptance", _ANCHOR_05B_ACCEPT)
    layer_xfer = agg.get("layer_aware_transfer", {})
    kv_meta = manifest.get("kvquant_optional", {})
    compressors = manifest.get("compressor_names", BASE_COMPRESSORS)

    lines = [
        "# Experiment 011: Larger-Model RunPod Validation (Qwen2.5-1.5B)",
        "",
        "_Generated by `scripts/run_experiment_011_larger_model_validation.py`. "
        "V9 Phase E — larger-model validation only._",
        "",
        "> This is **larger-model validation**, not a performance benchmark.",
        "> This does **not** claim throughput, latency, speedup, runtime, tokens/sec, "
        "active GPU memory, or production readiness.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, not measured peak "
        "GPU memory.",
        "> **Active GPU memory is not reported.**",
        "> Simulated compressors are **not** real packed-bit backends.",
        "> External backend paper results are **not** ExactKV results.",
        "> ExactKV preserves the exactness gate: `exactkv_output_ids == full_output_ids`.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate whether ExactKV's main findings — exactness preservation, acceptance "
        "ordering, and layer-aware simulated V advantage — hold on **Qwen/Qwen2.5-1.5B** "
        "beyond the 0.5B RunPod/CPU sweeps, using honest workspace-memory accounting.",
        "",
        "## 2. RunPod environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | RunPod L40S (~46 GB) |",
        f"| Model | `{manifest.get('model_name', MODEL_NAME_DEFAULT)}` |",
        f"| dtype | {DTYPE} |",
        f"| device | {DEVICE} |",
        f"| transformers | {manifest.get('transformers_version', '—')} |",
        f"| torch | {manifest.get('torch_version', '—')} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "cd /workspace/ExactKV",
        "python scripts/run_experiment_011_larger_model_validation.py",
        "# optional KVQuant 1.5B row:",
        "python scripts/run_experiment_011_larger_model_validation.py --try-kvquant",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_011_qwen15b_validation.json`,",
        "`reports/experiment_011_qwen15b_validation.csv`.",
        "",
        "## 3. Model and prompt suite",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{manifest.get('model_name', MODEL_NAME_DEFAULT)}` |",
        f"| Prompt suite | `{PROMPT_SUITE}` (34 prompts) |",
        f"| `draft_len` | {DRAFT_LEN} |",
        f"| `max_new_tokens` | {MAX_NEW_TOKENS} |",
        f"| Experiment class | `{EXPERIMENT_CLASS}` |",
        f"| Total cells | **{agg['total_runs']}** |",
        "",
        "## 4. Compressor set",
        "",
        "| Compressor | Type | Notes |",
        "|---|---|---|",
        "| `noop` | Identity baseline | Lossless |",
        "| `backend_passthrough` | V6 BackendAdapter PoC | Lossless |",
        "| `int8` | Real INT8 | `supports_real_bytes_claim=True` |",
        "| `k8_v4_sim` | Simulated asymmetric K8/V4 | `is_simulated=True` |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) | Best simulated policy (V7/006C) |",
        "| `k_full_v8` | Real K-full / V8 | |",
        "| `k8_v_full` | Real K8 / V-full | |",
    ]

    if KVQUANT_NAME in compressors:
        lines.append(
            f"| `{KVQUANT_NAME}` | Optional KVQuant simquant | 1.5B quantizer pickle |"
        )
    else:
        lines.append("| `kvquant_sim_*` | — | **Not included** (see §11) |")

    lines.extend([
        "",
        "## 5. Exactness result",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        f"| `exactkv_output_ids == full_output_ids` | "
        f"{agg['total_runs'] - agg['exactkv_failures']} / {agg['total_runs']} |",
        "",
        "## 6. Acceptance by compressor",
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
        "## 7. Divergence/rejection/correction summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Lossy divergence cells | {agg['lossy_divergence_count']} / {agg['total_runs']} |",
        f"| Total rejected (ExactKV) | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Mean acceptance (all cells) | {_fmt_rate(agg['mean_acceptance_rate'])} |",
        "",
        "## 8. Workspace-memory accounting table",
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
        stored = sum(m["stored_kv_bytes"] for m in rows) / len(rows)
        mat = sum(m["materialized_working_kv_bytes"] for m in rows) / len(rows)
        meta_b = sum(m["metadata_bytes"] for m in rows) / len(rows)
        tot = sum(m["total_kv_footprint_bytes"] for m in rows) / len(rows)
        real = rows[0].get("supports_real_bytes_claim", False)
        sim = rows[0].get("is_simulated", False)
        lines.append(
            f"| `{name}` | {_fmt_bytes(stored)} | {_fmt_bytes(mat)} | "
            f"{_fmt_bytes(meta_b)} | {_fmt_bytes(tot)} | "
            f"{'yes' if real else 'no'} | {'yes ⚠️' if sim else 'no'} |"
        )

    lines.extend([
        "",
        "† `total_kv_footprint_bytes` is a **conservative accounting sum**, not measured",
        "peak GPU memory. **Active GPU memory is not reported.**",
        "",
        "## 9. Comparison to Qwen2.5-0.5B results",
        "",
        f"_0.5B anchors from {manifest.get('anchor_05b_source', _ANCHOR_05B_SOURCE)} "
        "(cross-experiment; same core suite, draft_len=4, max_new_tokens=16)._",
        "",
        "| Compressor | 0.5B accept (anchor) | 1.5B accept | Δ (1.5B − 0.5B) |",
        "|---|---:|---:|---:|",
    ])

    compare_keys = [
        "int8",
        "k8_v4_sim",
        "k8_v4_boundary4_v8_sim",
        "k_full_v8",
        "k8_v_full",
    ]
    for key in compare_keys:
        a05 = anchors.get(key, 0.0)
        a15 = lookup.get(key, {}).get("mean_acceptance_rate", 0.0)
        lines.append(
            f"| `{key}` | {_fmt_rate(a05)} | {_fmt_rate(a15)} | {_delta_str(a15, a05)} |"
        )

    b15 = layer_xfer.get("boundary4_accept_15b")
    k15 = layer_xfer.get("k8_v4_sim_accept_15b")
    b05 = layer_xfer.get("boundary4_accept_05b_anchor")
    k05 = layer_xfer.get("k8_v4_sim_accept_05b_anchor")
    delta15 = layer_xfer.get("boundary4_minus_k8v4_15b")
    delta05 = (b05 - k05) if b05 is not None and k05 is not None else None

    lines.extend([
        "",
        "## 10. Whether V7/V9 findings transfer to 1.5B",
        "",
        f"| Finding | 0.5B | 1.5B |",
        f"|---|---:|---:|",
        f"| `k8_v4_boundary4_v8_sim` accept | {_fmt_rate(b05)} | {_fmt_rate(b15)} |",
        f"| `k8_v4_sim` accept | {_fmt_rate(k05)} | {_fmt_rate(k15)} |",
        f"| boundary4 − k8_v4_sim Δ | {_fmt_rate(delta05)} | {_fmt_rate(delta15)} |",
        "",
    ])

    if delta15 is not None and delta15 > 0:
        lines.append(
            f"On 1.5B, layer-aware boundary V (**{_fmt_rate(b15)}**) still exceeds uniform "
            f"`k8_v4_sim` (**{_fmt_rate(k15)}**) by **{_fmt_rate(delta15)}** accept rate — "
            "consistent with V7/006C on 0.5B."
        )
    elif delta15 is not None:
        lines.append(
            f"On 1.5B, boundary4 did not exceed `k8_v4_sim` (Δ {_fmt_rate(delta15)}). "
            "Investigate before claiming V7 transfer."
        )

    int8_15 = lookup.get("int8", {}).get("mean_acceptance_rate", 0.0)
    int8_05 = anchors.get("int8", 0.0)
    if agg["exactkv_failures"] == 0:
        lines.append(
            "- **Exactness gate**: `exactkv_failures == 0` on 1.5B — draft-verify-commit "
            "holds at larger scale."
        )
    if int8_15 >= int8_05 - 0.05:
        lines.append(
            f"- **INT8 reference**: 1.5B accept ({_fmt_rate(int8_15)}) remains in family "
            f"of 0.5B ({_fmt_rate(int8_05)})."
        )

    lines.extend([
        "",
        "## 11. Optional KVQuant 1.5B artifact result",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Attempted | {kv_meta.get('attempted', False)} |",
        f"| Included in sweep | {kv_meta.get('included_in_sweep', False)} |",
        f"| Pickle path | `{kv_meta.get('pickle_path', '—')}` |",
        f"| Generated this run | {kv_meta.get('generated', False)} |",
        f"| Pickle bytes | {_fmt_bytes(kv_meta.get('pickle_bytes'))} |",
        f"| Error (if any) | {kv_meta.get('error') or '—'} |",
        "",
    ])

    if kv_meta.get("included_in_sweep"):
        kv_row = lookup.get(KVQUANT_NAME, {})
        lines.append(
            f"KVQuant simquant row accept rate on 1.5B: **{_fmt_rate(kv_row.get('mean_acceptance_rate'))}**."
        )
    else:
        lines.append(
            "KVQuant 1.5B row was **not** included. Phase E gate does not require it."
        )

    lines.extend([
        "",
        "## 12. What this proves",
        "",
        "- ExactKV's **exactness gate** holds on Qwen2.5-1.5B for the core compressor panel.",
        "- Acceptance and workspace-memory reporting extend to a larger model on RunPod GPU.",
        "- Layer-aware simulated V and INT8 ordering can be compared cross-model using "
        "honest accounting (not throughput).",
        "",
        "## 13. What this does not prove",
        "",
        "- Production serving readiness, GPU memory peaks, or external-paper speedup claims.",
        "- That simulated `_sim` compressors represent packed-bit storage.",
        "- That optional KVQuant/TurboQuant/KIVI backends scale without per-model artifacts.",
        "- Results on 3B or beyond (optional stretch not run unless documented).",
        "",
        "## 14. Relation to V9 release readiness",
        "",
        "Experiment 011 completes V9 **Phase E** when `exactkv_failures == 0` on ≥1.5B. "
        "Phase F (`RELEASE_NOTES_V0.9.0.md`, experiment index, v1.0.0 readiness) may proceed.",
        "",
        "## 15. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al.,",
        "arXiv:2605.17613, 2026). Experiment 011 evaluates the harness at larger model scale;",
        "it does not claim novel compression algorithm contributions.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    _require_cuda_env()

    parser = argparse.ArgumentParser(description="Run Experiment 011 (larger-model validation)")
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT)
    parser.add_argument(
        "--try-kvquant",
        action="store_true",
        help="Attempt KVQuant 1.5B quantizer artifact; add row only if pickle exists",
    )
    parser.add_argument(
        "--kvquant-pickle",
        default=KVQUANT_15B_PICKLE_DEFAULT,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run first prompt only (debug)",
    )
    parser.add_argument(
        "--json-out",
        default="reports/experiment_011_qwen15b_validation.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_011_qwen15b_validation.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_011_LARGER_MODEL_VALIDATION.md",
    )
    args = parser.parse_args()

    prompts = load_core_prompts()
    if args.smoke:
        prompts = prompts[:1]
        args.json_out = "reports/experiment_011_qwen15b_smoke.json"
        args.csv_out = "reports/experiment_011_qwen15b_smoke.csv"

    compressors, kv_meta = _build_compressor_list(
        try_kvquant=args.try_kvquant,
        kvquant_pickle=args.kvquant_pickle,
    )
    expected = len(prompts) * len(compressors)
    print(
        f"Experiment 011: {len(prompts)} prompts × {len(compressors)} compressors "
        f"× 1 draft_len = {expected} cells",
        flush=True,
    )
    print(f"Model: {args.model}", flush=True)
    print(f"Compressors: {compressors}", flush=True)
    if args.try_kvquant:
        print(f"KVQuant optional meta: {kv_meta}", flush=True)

    print(f"Loading model {args.model} on {DEVICE} ({DTYPE}) ...", flush=True)
    runtime = ModelRuntime(model_name=args.model, device=DEVICE, dtype=DTYPE)

    report = run_experiment_011(runtime, prompts, compressors, kvquant_meta=kv_meta)
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
