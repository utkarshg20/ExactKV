#!/usr/bin/env python3
"""Experiment 020: autopsy-guided repair policy pilot (V11 Phase 5b).

Policy selection is experiment-layer only; core ExactKV generator unchanged.
No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.repair_policy import (
    ALL_POLICIES,
    POLICY_BASELINE_BOUNDARY4,
    POLICY_BASELINE_K8,
    POLICY_CATEGORY_ADAPTIVE,
    aggregate_policy_results,
    assert_policy_artifact_safe,
    load_pilot_prompts,
    run_policy_cell,
)
from exactkv.metrics.gpu_memory_pilot import collect_runpod_meta

MODEL_PRIMARY = "Qwen/Qwen2.5-0.5B"
MODEL_OPTIONAL = "Qwen/Qwen2.5-1.5B"
DTYPE = "float16"
DEVICE = "cuda"
MAX_NEW_TOKENS = 32
EXPERIMENT_CLASS = "v11_repair_policy_pilot"

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


def run_pilot_for_model(
    model_name: str,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    from exactkv.runtime.model_runtime import ModelRuntime

    print(f"Loading model {model_name} ({dtype}, {device}) ...", flush=True)
    runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(ALL_POLICIES)
    idx = 0

    for prompt_entry in prompts:
        for policy_name in ALL_POLICIES:
            idx += 1
            spec_hint = policy_name
            print(
                f"  [{idx}/{total}] {model_name} {prompt_entry['prompt_id']} × "
                f"{spec_hint}",
                flush=True,
            )
            results.append(
                run_policy_cell(
                    runtime,
                    prompt_entry,
                    policy_name,
                    max_new_tokens=MAX_NEW_TOKENS,
                    compressor_cache=compressor_cache,
                )
            )

    return {
        "model_name": model_name,
        "results": results,
        "aggregate": aggregate_policy_results(results),
    }


def build_artifact(
    model_runs: list[dict[str, Any]],
    *,
    runpod_meta: dict[str, Any],
    panel: str,
    prompt_count: int,
) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    for run in model_runs:
        all_results.extend(run["results"])
    agg = aggregate_policy_results(all_results)
    return {
        "manifest": {
            "experiment": "020_repair_policy_pilot",
            "experiment_class": EXPERIMENT_CLASS,
            "artifact_type": "isolated_policy_pilot",
            "standard_report_schema_modified": False,
            "core_generator_modified": False,
            "policies_enabled_by_default": False,
            "device": DEVICE,
            "dtype": DTYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_panel": panel,
            "prompt_count": prompt_count,
            "policies": list(ALL_POLICIES),
            "models_run": [r["model_name"] for r in model_runs],
            "runpod_meta": runpod_meta,
        },
        "model_runs": model_runs,
        "aggregate": agg,
        "note": (
            "Repair-policy pilot only. Policies select compressors/draft_len in the "
            "experiment layer; ExactKV verification unchanged."
        ),
    }


def write_pilot_csv(artifact: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in artifact["model_runs"]:
        for r in run["results"]:
            acc = r["exactkv"]["acceptance"]
            rows.append({
                "model_name": r["model_name"],
                "prompt_id": r["prompt_id"],
                "v10_suite": r.get("v10_suite", ""),
                "policy_name": r["policy_name"],
                "compressor_name": r["compressor_name"],
                "draft_len": r["draft_len"],
                "exactkv_failure": r["exactkv_failure"],
                "acceptance_rate": acc["acceptance_rate"],
                "total_rejected": acc.get("total_rejected", 0),
                "total_corrections": acc.get("total_corrections", 0),
                "lossy_diverged": r["lossy"].get("lossy_diverged", False),
                "first_divergence_idx": r["lossy"].get("first_divergence_idx"),
            })
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_rate(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def generate_markdown_report(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    manifest = artifact["manifest"]
    rp = manifest.get("runpod_meta", {})
    hyp = agg.get("hypothesis_evaluation", {})
    global_p = agg.get("global_by_policy", {})
    best_k8 = agg.get("best_policy_vs_baseline_k8_v4") or {}
    best_b4 = agg.get("best_policy_vs_baseline_boundary4") or {}

    lines = [
        "# Experiment 020: Autopsy-Guided Repair Policy Pilot",
        "",
        "_Generated by `scripts/run_experiment_020_repair_policy_pilot.py`. "
        "V11 Phase 5b — repair-policy pilot only._",
        "",
        "> This is a **repair-policy pilot**, not a production policy.",
        "> Policies select existing compressors and draft lengths; they do **not** "
        "change ExactKV verification.",
        "> ExactKV preserves exact greedy output; acceptance changes only affect "
        "draft usefulness.",
        "> **No claim of final model accuracy improvement.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> **Repair policies are not enabled by default** unless separately approved.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Test whether simple repair policies derived from Experiment 019 improve draft "
        "acceptance while preserving exact greedy output.",
        "",
        "## 2. Why this follows Experiment 019",
        "",
        "Experiment 019 proposed fallback-int8, structured-safe, category-adaptive, and "
        "draft-len hypotheses. Experiment 020 implements them as **pilot selectors** "
        "without modifying core ExactKV.",
        "",
        "## 3. Model and environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Models | `{', '.join(manifest['models_run'])}` |",
        f"| dtype / device | `{manifest['dtype']}` / `{manifest['device']}` |",
        f"| GPU | {rp.get('gpu_device_name', rp.get('gpu_name', '—'))} |",
        f"| torch | {rp.get('torch_version', '—')} |",
        f"| CUDA | {rp.get('cuda_version', '—')} |",
        f"| Total cells | **{agg['total_cells']}** |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_020_repair_policy_pilot.py "
        "--device cuda --dtype float16",
        "```",
        "",
        "## 4. Prompt subset",
        "",
        f"- **Panel:** `{manifest['prompt_panel']}`",
        f"- **Prompt count:** {manifest['prompt_count']}",
        "- Same prompts used for every policy (no per-policy cherry-picking).",
        "",
        "## 5. Policy definitions",
        "",
        "| Policy | Compressor / draft_len rule |",
        "|---|---|",
        "| `baseline_k8_v4` | always `k8_v4_sim`, draft_len=4 |",
        "| `baseline_boundary4` | always `k8_v4_boundary4_v8_sim`, draft_len=4 |",
        "| `fallback_int8_for_hard_categories` | int8 on long_context/retrieval_copy; "
        "else boundary4; draft_len=4 |",
        "| `structured_safe_mode` | int8 on tool_json/code_structured; else boundary4; "
        "draft_len=4 |",
        "| `category_adaptive_policy` | int8 on hard+structured suites; boundary4 on "
        "core_v2; draft_len=4 |",
        "| `draft_len_adaptive_policy` | category_adaptive compressors; draft_len=4 on "
        "hard/tool_json; draft_len=8 on core_v2/code_structured |",
        "",
        f"- **max_new_tokens:** {manifest['max_new_tokens']}",
        "",
        "## 6. Exactness result",
        "",
        f"| ExactKV failures | **{agg['exactkv_failures']}** |",
        "",
        "## 7. Global acceptance by policy",
        "",
        "| Policy | Mean accept | Rejected | Corrections | Lossy div |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ALL_POLICIES:
        row = global_p.get(policy, {})
        lines.append(
            f"| `{policy}` | {_fmt_rate(row.get('mean_acceptance_rate'))} | "
            f"{row.get('total_rejected', 0)} | {row.get('total_corrections', 0)} | "
            f"{row.get('lossy_divergence_count', 0)} |"
        )

    lines.extend([
        "",
        "## 8. Per-category acceptance by policy",
        "",
        "| Policy | Suite | Mean accept | Rejected | Corrections |",
        "|---|---|---:|---:|---:|",
    ])
    for row in agg.get("per_category_by_policy", []):
        lines.append(
            f"| `{row['policy_name']}` | `{row['v10_suite']}` | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | {row['total_rejected']} | "
            f"{row['total_corrections']} |"
        )

    lines.extend([
        "",
        "## 9. Rejection/correction summary",
        "",
        f"| Total cells | {agg['total_cells']} |",
    ])
    total_rej = sum(r.get("total_rejected", 0) for r in global_p.values())
    total_corr = sum(r.get("total_corrections", 0) for r in global_p.values())
    lines.append(f"| Total rejected (all policies) | {total_rej} |")
    lines.append(f"| Total corrections (all policies) | {total_corr} |")

    lines.extend([
        "",
        "## 10. Prompt-level win/loss table",
        "",
        "### vs `baseline_k8_v4`",
        "",
        "| Policy | Wins | Losses | Ties |",
        "|---|---:|---:|---:|",
    ])
    for comp in agg.get("comparisons_vs_baseline_k8_v4", []):
        lines.append(
            f"| `{comp['policy_name']}` | {comp['wins']} | {comp['losses']} | "
            f"{comp['ties']} |"
        )

    lines.extend([
        "",
        "### vs `baseline_boundary4`",
        "",
        "| Policy | Wins | Losses | Ties |",
        "|---|---:|---:|---:|",
    ])
    for comp in agg.get("comparisons_vs_baseline_boundary4", []):
        lines.append(
            f"| `{comp['policy_name']}` | {comp['wins']} | {comp['losses']} | "
            f"{comp['ties']} |"
        )

    lines.extend([
        "",
        "## 11. Best policy vs baseline_k8_v4",
        "",
    ])
    if best_k8:
        lines.append(
            f"**`{best_k8.get('policy_name', '—')}`** — wins {best_k8.get('wins', 0)}, "
            f"losses {best_k8.get('losses', 0)}, ties {best_k8.get('ties', 0)}."
        )
    else:
        lines.append("_No comparison available._")

    lines.extend([
        "",
        "## 12. Best policy vs baseline_boundary4",
        "",
    ])
    if best_b4:
        lines.append(
            f"**`{best_b4.get('policy_name', '—')}`** — wins {best_b4.get('wins', 0)}, "
            f"losses {best_b4.get('losses', 0)}, ties {best_b4.get('ties', 0)}."
        )
    else:
        lines.append("_No comparison available._")

    lines.extend([
        "",
        "## 13. Which repair hypotheses were supported",
        "",
    ])
    for item in hyp.get("supported", []):
        lines.append(f"- {item}")
    if not hyp.get("supported"):
        lines.append("_None met pilot thresholds._")

    lines.extend([
        "",
        "## 14. Which repair hypotheses failed",
        "",
    ])
    for item in hyp.get("rejected", []):
        lines.append(f"- {item}")
    if not hyp.get("rejected"):
        lines.append("_None explicitly rejected._")

    if hyp.get("inconclusive"):
        lines.extend([
            "",
            "### Inconclusive",
            "",
        ])
        for item in hyp["inconclusive"]:
            lines.append(f"- {item}")

    lines.extend([
        "",
        "## 15. What this proves",
        "",
        "- Policy selection in the experiment layer can change acceptance without "
        "touching verification.",
        "- Category-aware compressor/draft_len rules can be compared fairly on the "
        "same prompt panel.",
        "",
        "## 16. What this does not prove",
        "",
        "- No production policy deployment; not enabled in core ExactKV.",
        "- No claim of speed, memory, or final accuracy improvement.",
        "- Pilot panel is small; not a universal benchmark.",
        "",
        "## 17. Limitations",
        "",
        "- Policies use fixed rules from Exp 019/014; no online margin gating.",
        "- draft_len_adaptive uses draft_len=8 on core/code — higher rejection risk.",
        "- Optional 1.5B repeat may show different rankings.",
        "",
        "## 18. Whether to implement any policy in core ExactKV later",
        "",
    ])
    cat_acc = global_p.get(POLICY_CATEGORY_ADAPTIVE, {}).get("mean_acceptance_rate")
    b4_acc = global_p.get(POLICY_BASELINE_BOUNDARY4, {}).get("mean_acceptance_rate")
    if cat_acc is not None and b4_acc is not None and cat_acc > b4_acc + 1e-9:
        lines.append(
            "**Deferred recommendation:** `category_adaptive_policy` merits a future "
            "optional experiment-layer hook — **not** default core behavior without "
            "separate approval."
        )
    else:
        lines.append(
            "**No default core integration recommended** from this pilot; keep policies "
            "in experiment/analysis layer until a larger panel confirms gains."
        )

    lines.extend([
        "",
        "## 19. Relation to v1.0.0 readiness",
        "",
        "Experiment 020 validates Exp 019 hypotheses at pilot scale. Phase 6 launch "
        "package readiness does not require enabling repair policies in core ExactKV.",
        "",
        "## 20. VeriCache attribution",
        "",
        "ExactKV draft-verify-commit loop inspired by "
        "[VeriCache](https://arxiv.org/abs/2605.17613); policies are ExactKV pilot "
        "selectors on built-in compressors only.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 020 repair policy pilot")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--panel", choices=("exp019", "exp014_hard"), default="exp019")
    parser.add_argument("--include-15b", action="store_true")
    parser.add_argument("--skip-15b", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=_ROOT / "reports")
    parser.add_argument("--docs-dir", type=Path, default=_ROOT / "docs")
    args = parser.parse_args()

    prompts = load_pilot_prompts(panel=args.panel)
    try:
        runpod_meta = collect_runpod_meta()
    except Exception:
        runpod_meta = {}

    model_runs = [
        run_pilot_for_model(
            MODEL_PRIMARY,
            prompts,
            device=args.device,
            dtype=args.dtype,
        )
    ]

    if args.include_15b and not args.skip_15b:
        if model_runs[0]["aggregate"]["exactkv_failures"] == 0:
            try:
                model_runs.append(
                    run_pilot_for_model(
                        MODEL_OPTIONAL,
                        prompts,
                        device=args.device,
                        dtype=args.dtype,
                    )
                )
            except Exception as exc:
                print(f"Optional 1.5B run skipped: {exc}", flush=True)

    artifact = build_artifact(
        model_runs,
        runpod_meta=runpod_meta,
        panel=args.panel,
        prompt_count=len(prompts),
    )
    assert_policy_artifact_safe(artifact)
    _assert_no_forbidden(artifact)

    json_path = args.reports_dir / "experiment_020_repair_policy_pilot.json"
    csv_path = args.reports_dir / "experiment_020_repair_policy_pilot.csv"
    md_path = args.docs_dir / "EXPERIMENT_020_REPAIR_POLICY_PILOT.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    write_pilot_csv(artifact, csv_path)
    md_path.write_text(generate_markdown_report(artifact), encoding="utf-8")

    agg = artifact["aggregate"]
    print(f"\nExperiment 020 complete: {agg['total_cells']} cells", flush=True)
    print(f"  exactkv_failures: {agg['exactkv_failures']}", flush=True)
    g = agg["global_by_policy"]
    for p in ALL_POLICIES:
        print(f"  {p}: accept={g[p]['mean_acceptance_rate']:.3f}", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
