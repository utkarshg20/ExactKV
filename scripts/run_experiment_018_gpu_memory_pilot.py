#!/usr/bin/env python3
"""Experiment 018: active GPU memory methodology pilot (V11 Phase 4).

Records PyTorch CUDA allocation observations in an isolated pilot artifact.
Does NOT modify standard report schema.  No timing, throughput, latency,
speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_all_v10_prompts
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.gpu_memory_pilot import (
    assert_pilot_artifact_safe,
    collect_runpod_meta,
    cuda_available,
    measure_exactkv_cell_gpu_memory,
)
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.model_runtime import ModelRuntime

PRIMARY_MODEL = "Qwen/Qwen2.5-0.5B"
OPTIONAL_MODEL = "Qwen/Qwen2.5-1.5B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "v11_active_gpu_memory"
PROMPT_SUITE = "v10_subset_10"

COMPRESSORS = [
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "backend_passthrough",
]

# 10 prompts: ≥1 per required suite (core_v2, long_context, retrieval_copy, tool_json).
V10_SUBSET_IDS = [
    "cv2_nat_001",   # core_v2
    "cv2_qa_001",    # core_v2
    "cs_py_001",     # code_structured
    "lc_001",        # long_context
    "lc_002",        # long_context
    "rm_001",        # reasoning_math
    "ml_fr_001",     # multilingual
    "rc_001",        # retrieval_copy
    "rc_002",        # retrieval_copy
    "tj_001",        # tool_json
]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def load_v10_subset_prompts(ids: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = ids or V10_SUBSET_IDS
    all_prompts = load_all_v10_prompts()
    by_id = {p["prompt_id"]: p for p in all_prompts}
    missing = sorted(set(wanted) - set(by_id))
    if missing:
        raise ValueError(f"V10 subset ids not found: {missing}")
    return [by_id[i] for i in wanted]


def _assert_no_forbidden_fields(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def run_pilot_for_model(
    model_name: str,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    print(f"Loading model {model_name} ({dtype}, {device}) ...")
    runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)

    cells: list[dict[str, Any]] = []
    total = len(prompts) * len(COMPRESSORS)
    idx = 0

    for prompt_entry in prompts:
        for compressor_name in COMPRESSORS:
            idx += 1
            print(
                f"  [{idx}/{total}] {model_name} {prompt_entry['prompt_id']} × "
                f"{compressor_name}",
                flush=True,
            )
            compressor = get_compressor(compressor_name)
            caps_dict: dict = {}
            if hasattr(compressor, "capabilities"):
                caps_dict = asdict(compressor.capabilities)

            prompt = prompt_entry["prompt"]
            v5 = estimate_kv_memory(runtime, prompt, compressor)

            result, gpu_snap, exact = measure_exactkv_cell_gpu_memory(
                runtime,
                prompt,
                compressor,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            acceptance = summarize_acceptance(result.traces)
            gpu = gpu_snap.to_dict()

            # Consistency check: peak should not be below post-prefill baseline.
            measurement_valid = (
                gpu["gpu_peak_allocated_during_run_bytes"]
                >= gpu["gpu_allocated_after_prefill_bytes"]
                >= gpu["gpu_baseline_model_loaded_bytes"]
            )

            cells.append({
                "model_name": model_name,
                "prompt_id": prompt_entry["prompt_id"],
                "v10_suite": prompt_entry.get("v10_suite", ""),
                "v10_primary_category": prompt_entry.get("v10_primary_category", ""),
                "compressor_name": compressor_name,
                "compressor_capabilities": caps_dict,
                "draft_len": DRAFT_LEN,
                "max_new_tokens": MAX_NEW_TOKENS,
                "exactkv_failure": not exact,
                "exactkv_token_match": exact,
                "acceptance_rate": acceptance.acceptance_rate,
                "v5_accounting": v5.to_dict(),
                "gpu_pilot_observations": gpu,
                "measurement_valid": measurement_valid,
                "pilot_delta_prefill_to_peak_bytes": (
                    gpu["gpu_peak_allocated_during_run_bytes"]
                    - gpu["gpu_allocated_after_prefill_bytes"]
                ),
                "pilot_delta_v5_footprint_bytes": v5.total_kv_footprint_bytes,
            })

    exactkv_failures = sum(1 for c in cells if c["exactkv_failure"])
    invalid_measurements = sum(1 for c in cells if not c["measurement_valid"])

    return {
        "model_name": model_name,
        "cells": cells,
        "summary": {
            "total_cells": len(cells),
            "exactkv_failures": exactkv_failures,
            "invalid_measurements": invalid_measurements,
            "mean_peak_bytes": mean(
                c["gpu_pilot_observations"]["gpu_peak_allocated_during_run_bytes"]
                for c in cells
            ),
            "mean_v5_footprint_bytes": mean(
                c["v5_accounting"]["total_kv_footprint_bytes"] for c in cells
            ),
        },
    }


def _aggregate_by_compressor(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault(c["compressor_name"], []).append(c)
    rows = []
    for name, group in sorted(groups.items()):
        peaks = [
            g["gpu_pilot_observations"]["gpu_peak_allocated_during_run_bytes"]
            for g in group
        ]
        v5s = [g["v5_accounting"]["total_kv_footprint_bytes"] for g in group]
        rows.append({
            "compressor_name": name,
            "num_cells": len(group),
            "mean_peak_allocated_bytes": mean(peaks),
            "mean_v5_total_kv_footprint_bytes": mean(v5s),
            "mean_delta_prefill_to_peak_bytes": mean(
                g["pilot_delta_prefill_to_peak_bytes"] for g in group
            ),
        })
    return rows


def build_pilot_artifact(
    model_runs: list[dict[str, Any]],
    *,
    runpod_meta: dict[str, Any],
    include_15b: bool,
) -> dict[str, Any]:
    all_cells = []
    for run in model_runs:
        all_cells.extend(run["cells"])

    exactkv_failures = sum(1 for c in all_cells if c["exactkv_failure"])
    invalid = sum(1 for c in all_cells if not c["measurement_valid"])
    total_cells = len(all_cells)

    # Stability: coefficient of variation on peak across noop cells per model.
    stability_notes: list[str] = []
    for run in model_runs:
        noop_peaks = [
            c["gpu_pilot_observations"]["gpu_peak_allocated_during_run_bytes"]
            for c in run["cells"]
            if c["compressor_name"] == "noop"
        ]
        if len(noop_peaks) >= 2:
            cv = pstdev(noop_peaks) / mean(noop_peaks) if mean(noop_peaks) else 0.0
            stability_notes.append(
                f"{run['model_name']} noop peak CV={cv:.4f} "
                f"(n={len(noop_peaks)})"
            )

    invalid_rate = invalid / max(total_cells, 1)
    if exactkv_failures > 0:
        decision = "deferral_exactness_failure"
    elif invalid_rate > 0.2:
        decision = "deferral_measurement_unstable"
    elif total_cells == 0:
        decision = "deferral_no_data"
    else:
        decision = "pilot_success"

    return {
        "manifest": {
            "experiment": "018_gpu_memory_pilot",
            "experiment_class": EXPERIMENT_CLASS,
            "methodology_ref": "docs/GPU_MEMORY_METHODOLOGY.md",
            "artifact_type": "isolated_pilot_only",
            "standard_report_schema_modified": False,
            "active_gpu_kv_bytes_in_schema": False,
            "device": DEVICE,
            "dtype": DTYPE,
            "draft_len": DRAFT_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_suite": PROMPT_SUITE,
            "prompt_ids": V10_SUBSET_IDS,
            "compressors": COMPRESSORS,
            "models_run": [r["model_name"] for r in model_runs],
            "include_15b": include_15b,
            "runpod_meta": runpod_meta,
        },
        "model_runs": model_runs,
        "aggregate": {
            "total_cells": total_cells,
            "exactkv_failures": exactkv_failures,
            "invalid_measurements": invalid,
            "invalid_measurement_rate": invalid_rate,
            "by_compressor": _aggregate_by_compressor(all_cells),
            "stability_notes": stability_notes,
            "decision": decision,
        },
        "note": (
            "PyTorch CUDA allocation pilot only. Not a performance benchmark. "
            "gpu_* fields are device-level observations, not KV-only peaks. "
            "total_kv_footprint_bytes remains V5 accounting."
        ),
    }


def write_pilot_csv(artifact: dict[str, Any], path: Path) -> None:
    rows = []
    for run in artifact["model_runs"]:
        for c in run["cells"]:
            gpu = c["gpu_pilot_observations"]
            v5 = c["v5_accounting"]
            rows.append({
                "model_name": c["model_name"],
                "prompt_id": c["prompt_id"],
                "v10_suite": c["v10_suite"],
                "compressor_name": c["compressor_name"],
                "exactkv_failure": c["exactkv_failure"],
                "gpu_baseline_model_loaded_bytes": gpu["gpu_baseline_model_loaded_bytes"],
                "gpu_allocated_after_prefill_bytes": gpu["gpu_allocated_after_prefill_bytes"],
                "gpu_peak_allocated_during_run_bytes": gpu["gpu_peak_allocated_during_run_bytes"],
                "gpu_allocated_after_run_bytes": gpu["gpu_allocated_after_run_bytes"],
                "gpu_allocated_after_cleanup_bytes": gpu.get("gpu_allocated_after_cleanup_bytes"),
                "v5_total_kv_footprint_bytes": v5["total_kv_footprint_bytes"],
                "v5_stored_kv_bytes": v5["stored_kv_bytes"],
                "measurement_valid": c["measurement_valid"],
            })
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_bytes(n: float | int | None) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def generate_markdown_report(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    manifest = artifact["manifest"]
    rp = manifest.get("runpod_meta", {})
    decision = agg["decision"]
    pilot_success = decision == "pilot_success"

    lines = [
        "# Experiment 018: Active GPU Memory Methodology Pilot",
        "",
        "_Generated by `scripts/run_experiment_018_gpu_memory_pilot.py`. "
        "V11 Phase 4 — GPU memory methodology pilot only._",
        "",
        "> This is a **GPU memory methodology pilot**, not a performance benchmark.",
        "> Measurements are **PyTorch CUDA allocation observations**, not universal "
        "hardware-independent memory claims.",
        "> `total_kv_footprint_bytes` remains a **conservative accounting sum**, "
        "not measured peak GPU memory.",
        "> Measured GPU allocation includes **model weights, framework allocator "
        "behaviour, temporary tensors, and other non-KV allocations** unless "
        "carefully isolated.",
        "> ExactKV does **not** claim throughput, latency, speedup, runtime, "
        "tokens/sec, or production readiness.",
        "> ExactKV does **not** claim active GPU memory as a stable standard metric "
        "unless methodology is approved later.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Define and pilot a cautious **active GPU memory methodology** clearly "
        "separate from V5 `total_kv_footprint_bytes`, per V11 Phase 4 / D14.",
        "",
        "## 2. Methodology",
        "",
        "See [`GPU_MEMORY_METHODOLOGY.md`](GPU_MEMORY_METHODOLOGY.md). Pilot uses "
        "`torch.cuda.memory_allocated`, `torch.cuda.max_memory_allocated`, and "
        "`torch.cuda.reset_peak_memory_stats` at defined lifecycle points. "
        "Fields live in the **isolated pilot artifact only** — not standard reports.",
        "",
        "## 3. RunPod environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {rp.get('gpu_device_name', '—')} |",
        f"| Host | {rp.get('hostname', '—')} |",
        f"| torch | {rp.get('torch_version', '—')} |",
        f"| CUDA | {rp.get('cuda_version', '—')} |",
        f"| dtype | {DTYPE} |",
        f"| device | cuda |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_018_gpu_memory_pilot.py \\",
        "  --device cuda --dtype float16",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_018_gpu_memory_pilot.json`,",
        "`reports/experiment_018_gpu_memory_pilot.csv`.",
        "",
        "## 4. Prompt subset",
        "",
        f"**{PROMPT_SUITE}** — {len(V10_SUBSET_IDS)} deterministic V10 prompts "
        "(includes `core_v2`, `long_context`, `retrieval_copy`, `tool_json`).",
        "",
        "| Prompt ID | Suite |",
        "|---|---|",
    ]
    prompts = load_v10_subset_prompts()
    for p in prompts:
        lines.append(f"| `{p['prompt_id']}` | `{p.get('v10_suite', '')}` |")

    lines.extend([
        "",
        "## 5. Compressor panel",
        "",
        "| Compressor | Role |",
        "|---|---|",
        "| `noop` | Lossless identity baseline |",
        "| `int8` | Real symmetric INT8 |",
        "| `k8_v4_sim` | Simulated uniform K8/V4 |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) |",
        "| `backend_passthrough` | V6 BackendAdapter PoC |",
        "",
        f"Models: {', '.join(f'`{m}`' for m in manifest['models_run'])}.",
        "",
        "## 6. Exactness result",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total cells | {agg['total_cells']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        "",
        "## 7. V5 accounting recap",
        "",
        "Mean `total_kv_footprint_bytes` by compressor (accounting sum, not GPU peak):",
        "",
        "| Compressor | Mean V5 footprint |",
        "|---|---:|",
    ])
    for row in agg["by_compressor"]:
        lines.append(
            f"| `{row['compressor_name']}` | "
            f"{_fmt_bytes(row['mean_v5_total_kv_footprint_bytes'])} |"
        )

    lines.extend([
        "",
        "## 8. GPU allocation observations",
        "",
        "Mean `gpu_peak_allocated_during_run_bytes` by compressor "
        "(device-level PyTorch allocation, includes weights + KV + temporaries):",
        "",
        "| Compressor | Mean peak allocated | Mean Δ prefill→peak |",
        "|---|---:|---:|",
    ])
    for row in agg["by_compressor"]:
        lines.append(
            f"| `{row['compressor_name']}` | "
            f"{_fmt_bytes(row['mean_peak_allocated_bytes'])} | "
            f"{_fmt_bytes(row['mean_delta_prefill_to_peak_bytes'])} |"
        )

    if agg.get("stability_notes"):
        lines.append("")
        lines.append("Stability notes:")
        for note in agg["stability_notes"]:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "## 9. How measured GPU allocation differs from `total_kv_footprint_bytes`",
        "",
        "- V5 footprint counts **KV-related tensor shapes** on CPU/GPU — typically "
        "**MiB-scale** for these prompts.",
        "- Pilot peak counts **entire GPU tensor arena** including **~1–3 GiB model "
        "weights** plus KV growth and temporaries — typically **GiB-scale**.",
        "- Prefill→peak delta is a **heuristic** for generation-time growth, not an "
        "isolated KV-cache measurement.",
        "- Compressor ordering on V5 footprint **does not** match GPU peak ordering "
        "when model weights dominate.",
        "",
        "## 10. Whether measurements are stable enough for v1.0.0",
        "",
    ])
    if pilot_success:
        lines.extend([
            "**Cautiously yes for methodology documentation** — measurements are "
            "internally consistent (`exactkv_failures == 0`, invalid rate "
            f"{agg['invalid_measurement_rate']:.1%}).",
            "They are **not** stable enough to become a **standard published metric** "
            "without continued caveats and hardware-specific labelling.",
        ])
    else:
        lines.extend([
            f"**Not yet** — decision `{decision}`. See §14.",
        ])

    lines.extend([
        "",
        "## 11. What this proves",
        "",
        "- A reproducible **pilot protocol** for CUDA allocation snapshots at defined "
        "ExactKV lifecycle points.",
        "- V5 accounting and device-level allocation measure **different quantities**.",
        "- Exactness gate holds during instrumented GPU runs.",
        "",
        "## 12. What this does not prove",
        "",
        "- Universal or hardware-independent GPU memory requirements.",
        "- KV-cache-only footprint (weights and temporaries are included).",
        "- Production serving memory behaviour.",
        "- Throughput, latency, or speedup.",
        "- That external paper GPU claims apply to ExactKV.",
        "",
        "## 13. Limitations",
        "",
        "- Single-GPU RunPod pilot; allocator caching affects repeatability.",
        "- `float16` only; no 3B or real-backend panels in this phase.",
        "- No vLLM/LMCache/paged serving semantics.",
        "- Post-cleanup bytes are best-effort (allocator may retain caches).",
        "",
        "## 14. Decision: pilot-success or deferral",
        "",
        f"**Decision:** `{decision}`",
        "",
    ])
    if pilot_success:
        lines.append(
            "Methodology published; pilot observations recorded with caveats. "
            "`active_gpu_kv_bytes` **not** added to standard schema."
        )
    else:
        lines.append(
            "Methodology published; active GPU memory reporting **deferred** for "
            "standard metrics pending further isolation work."
        )

    lines.extend([
        "",
        "## 15. Relation to v1.0.0 readiness",
        "",
        "Experiment 018 closes the **D14 methodology gate** for V11 Phase 4. "
        "v1.0.0 still requires Phase 5–6 (optional attention logging, launch package). "
        "V5 `total_kv_footprint_bytes` remains the **stable memory story** for launch docs.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Experiment 018 GPU memory pilot"
    )
    parser.add_argument(
        "--json-out",
        default="reports/experiment_018_gpu_memory_pilot.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_018_gpu_memory_pilot.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_018_GPU_MEMORY_PILOT.md",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE, choices=["float16", "bfloat16"])
    parser.add_argument(
        "--include-15b",
        action="store_true",
        help="Also run Qwen2.5-1.5B panel if 0.5B succeeds",
    )
    parser.add_argument(
        "--skip-15b",
        action="store_true",
        help="Do not run 1.5B even if --include-15b",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not cuda_available():
        raise SystemExit(
            "Experiment 018 requires CUDA (RunPod GPU). "
            "Methodology doc can still be published; pilot deferred."
        )

    prompts = load_v10_subset_prompts()
    expected_05 = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 018: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"= {expected_05} cells per model"
    )

    model_runs: list[dict[str, Any]] = []
    run_05 = run_pilot_for_model(
        PRIMARY_MODEL, prompts, device=args.device, dtype=args.dtype
    )
    model_runs.append(run_05)

    if run_05["summary"]["exactkv_failures"] != 0:
        print("ERROR: 0.5B exactkv_failures > 0; skipping 1.5B", file=sys.stderr)
    elif args.include_15b and not args.skip_15b:
        try:
            run_15 = run_pilot_for_model(
                OPTIONAL_MODEL, prompts, device=args.device, dtype=args.dtype
            )
            model_runs.append(run_15)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"OOM on 1.5B — continuing with 0.5B only: {exc}", file=sys.stderr)
            else:
                raise

    runpod_meta = collect_runpod_meta()
    artifact = build_pilot_artifact(
        model_runs, runpod_meta=runpod_meta, include_15b=len(model_runs) > 1
    )
    _assert_no_forbidden_fields(artifact)
    assert_pilot_artifact_safe(artifact)

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_pilot_csv(artifact, csv_path)
    md_path.write_text(generate_markdown_report(artifact), encoding="utf-8")

    agg = artifact["aggregate"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")
    print(f"decision: {agg['decision']}")

    if agg["exactkv_failures"] != 0:
        return 1
    if agg["decision"].startswith("deferral"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
