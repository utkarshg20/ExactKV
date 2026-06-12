#!/usr/bin/env python3
"""Experiment 025: full-suite repair-policy validation (V12 Phase 5).

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
    EXP025_POLICIES,
    POLICY_BASELINE_BOUNDARY4,
    POLICY_BASELINE_K8,
    POLICY_CATEGORY_ADAPTIVE,
    POLICY_FALLBACK_INT8_HARD,
    POLICY_INT8_ALL,
    aggregate_policy_results,
    assert_policy_artifact_safe,
    load_full_v10_prompts,
    run_policy_cell,
)
from exactkv.benchmarks.v10_prompts import list_v10_suites
from exactkv.metrics.gpu_memory_pilot import collect_runpod_meta

MODEL_PRIMARY = "Qwen/Qwen2.5-0.5B"
MODEL_OPTIONAL_15B = "Qwen/Qwen2.5-1.5B"
MODEL_OPTIONAL_3B = "Qwen/Qwen2.5-3B"
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "v12_full_suite_repair_policy"

# Experiment 020 pilot anchors (0.5B+1.5B, 25 prompts, max_new_tokens=32).
EXP020_PILOT_ACCEPT = {
    POLICY_BASELINE_K8: 0.940,
    POLICY_BASELINE_BOUNDARY4: 0.932,
    POLICY_FALLBACK_INT8_HARD: 0.979,
    POLICY_CATEGORY_ADAPTIVE: 0.973,
    "draft_len_adaptive_policy": 0.965,
}

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


def _default_device_dtype() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "float32"


def run_full_suite_for_model(
    model_name: str,
    prompts: list[dict[str, Any]],
    *,
    device: str,
    dtype: str,
    policies: tuple[str, ...] = EXP025_POLICIES,
) -> dict[str, Any]:
    from exactkv.runtime.model_runtime import ModelRuntime

    print(f"Loading model {model_name} ({dtype}, {device}) ...", flush=True)
    runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    total = len(prompts) * len(policies)
    idx = 0

    for prompt_entry in prompts:
        for policy_name in policies:
            idx += 1
            print(
                f"  [{idx}/{total}] {model_name} {prompt_entry['prompt_id']} × "
                f"{policy_name}",
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
        "aggregate": aggregate_policy_results(
            results,
            include_int8_all_comparison=True,
            include_primary_category=True,
        ),
    }


def build_artifact(
    model_runs: list[dict[str, Any]],
    *,
    runpod_meta: dict[str, Any],
    prompt_count: int,
    skipped_models: list[str],
) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    for run in model_runs:
        all_results.extend(run["results"])
    agg = aggregate_policy_results(
        all_results,
        include_int8_all_comparison=True,
        include_primary_category=True,
    )
    return {
        "manifest": {
            "experiment": "025_full_suite_repair_policy",
            "experiment_class": EXPERIMENT_CLASS,
            "artifact_type": "isolated_policy_validation",
            "standard_report_schema_modified": False,
            "core_generator_modified": False,
            "policies_enabled_by_default": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_panel": "exp025_full_v10",
            "prompt_count": prompt_count,
            "v10_suites": list_v10_suites(),
            "policies": list(EXP025_POLICIES),
            "models_run": [r["model_name"] for r in model_runs],
            "models_skipped": skipped_models,
            "runpod_meta": runpod_meta,
        },
        "model_runs": model_runs,
        "aggregate": agg,
        "exp020_pilot_anchors": EXP020_PILOT_ACCEPT,
        "note": (
            "Full-suite repair-policy validation only. Policies select "
            "compressors/draft_len in the experiment layer; ExactKV verification "
            "unchanged."
        ),
    }


def write_csv(artifact: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in artifact["model_runs"]:
        for r in run["results"]:
            acc = r["exactkv"]["acceptance"]
            rows.append({
                "model_name": r["model_name"],
                "prompt_id": r["prompt_id"],
                "v10_suite": r.get("v10_suite", ""),
                "v10_primary_category": r.get("v10_primary_category", ""),
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


def _suite_summary_table(agg: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | Suite | Mean accept | Rejected | Corrections |",
        "|---|---|---:|---:|---:|",
    ]
    for row in agg.get("per_category_by_policy", []):
        lines.append(
            f"| `{row['policy_name']}` | `{row['v10_suite']}` | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | {row['total_rejected']} | "
            f"{row['total_corrections']} |"
        )
    return lines


def _primary_category_table(agg: dict[str, Any]) -> list[str]:
    lines = [
        "| Policy | Primary category | Mean accept | Rejected | Corrections |",
        "|---|---|---:|---:|---:|",
    ]
    for row in agg.get("per_primary_category_by_policy", []):
        lines.append(
            f"| `{row['policy_name']}` | `{row['v10_primary_category']}` | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | {row['total_rejected']} | "
            f"{row['total_corrections']} |"
        )
    return lines


def _win_loss_section(comparisons: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Policy | Wins | Losses | Ties |",
        "|---|---:|---:|---:|",
    ]
    for comp in comparisons:
        lines.append(
            f"| `{comp['policy_name']}` | {comp['wins']} | {comp['losses']} | "
            f"{comp['ties']} |"
        )
    return lines


def _generalization_paragraph(agg: dict[str, Any]) -> str:
    global_p = agg.get("global_by_policy", {})
    parts: list[str] = []
    for policy, pilot in EXP020_PILOT_ACCEPT.items():
        full = global_p.get(policy, {}).get("mean_acceptance_rate")
        if full is None:
            continue
        delta = full - pilot
        direction = "held" if abs(delta) <= 0.02 else ("shrank" if delta < 0 else "grew")
        parts.append(f"`{policy}` pilot {pilot:.3f} → full {_fmt_rate(full)} ({direction})")
    int8_full = global_p.get(POLICY_INT8_ALL, {}).get("mean_acceptance_rate")
    if int8_full is not None:
        parts.append(f"`int8_all` full-suite anchor {_fmt_rate(int8_full)}")
    beaten = agg.get("int8_all_beaten_globally", False)
    parts.append(
        "No policy beats `int8_all` globally."
        if not beaten
        else "At least one policy beats `int8_all` globally."
    )
    return " ".join(parts) if parts else "_No pilot comparison available._"


def generate_markdown_report(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    manifest = artifact["manifest"]
    rp = manifest.get("runpod_meta", {})
    global_p = agg.get("global_by_policy", {})
    best_k8 = agg.get("best_policy_vs_baseline_k8_v4") or {}
    best_b4 = agg.get("best_policy_vs_baseline_boundary4") or {}
    best_int8 = agg.get("best_policy_vs_int8_all") or {}

    lines = [
        "# Experiment 025: Full-Suite Repair-Policy Validation",
        "",
        "_Generated by `scripts/run_experiment_025_full_suite_repair_policy.py`. "
        "V12 Phase 5 — repair-policy validation only._",
        "",
        "> This is a **repair-policy validation experiment**, not a production policy.",
        "> Policies select existing compressors and draft lengths; they do **not** "
        "change ExactKV verification.",
        "> **Repair policies are not enabled by default.**",
        "> ExactKV preserves exact greedy output; acceptance changes only measure "
        "draft usefulness.",
        "> **No claim of final model accuracy improvement.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> Simulated compressors (`_sim`) are **not** real packed-bit backends.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate whether autopsy-guided repair policies from Experiment 020 survive "
        "the full 128-prompt V10 suite at anchor generation settings.",
        "",
        "## 2. Why this follows Experiment 020",
        "",
        "Experiment 020 showed strong pilot gains on a 25-prompt subset "
        "(`fallback_int8_for_hard_categories` 0.979, `category_adaptive_policy` 0.973). "
        "Experiment 025 tests whether those gains generalize on all seven V10 suites.",
        "",
        "## 3. Model and environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Models run | `{', '.join(manifest['models_run'])}` |",
    ]
    if manifest.get("models_skipped"):
        lines.append(
            f"| Models skipped | `{', '.join(manifest['models_skipped'])}` |"
        )
    lines.extend([
        f"| dtype / device | See runpod_meta / CLI |",
        f"| GPU | {rp.get('gpu_device_name', rp.get('gpu_name', '—'))} |",
        f"| torch | {rp.get('torch_version', '—')} |",
        f"| CUDA | {rp.get('cuda_version', '— (CPU run)')} |",
        f"| Total cells | **{agg['total_cells']}** |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_025_full_suite_repair_policy.py "
        "--device cuda --dtype float16",
        "```",
        "",
        "## 4. Prompt suite summary",
        "",
        f"| V10 suites | {len(manifest['v10_suites'])} |",
        f"| Total prompts | **{manifest['prompt_count']}** |",
        f"| Policies | {len(manifest['policies'])} |",
        f"| Cells per model | {manifest['prompt_count']} × {len(manifest['policies'])} "
        f"= **{manifest['prompt_count'] * len(manifest['policies'])}** |",
        "",
        "Suites: "
        + ", ".join(f"`{s}`" for s in manifest["v10_suites"])
        + ".",
        "",
        "## 5. Policy definitions",
        "",
        "| Policy | Compressor / draft_len rule |",
        "|---|---|",
        "| `baseline_k8_v4` | always `k8_v4_sim`, draft_len=4 |",
        "| `baseline_boundary4` | always `k8_v4_boundary4_v8_sim`, draft_len=4 |",
        "| `int8_all` | always `int8`, draft_len=4 |",
        "| `fallback_int8_for_hard_categories` | int8 on long_context/retrieval_copy; "
        "else boundary4; draft_len=4 |",
        "| `category_adaptive_policy` | int8 on hard+structured suites; boundary4 on "
        "core_v2 and other suites; draft_len=4 |",
        "| `draft_len_adaptive_policy` | category-adaptive compressors; draft_len=4 on "
        "hard/tool_json; draft_len=8 on core_v2/code_structured |",
        "",
        f"- **max_new_tokens:** {manifest['max_new_tokens']}",
        "",
        "## 6. Exactness result",
        "",
        f"| ExactKV failures | **{agg['exactkv_failures']}** |",
        "",
        "## 7. Global acceptance leaderboard",
        "",
        "| Policy | Mean accept | Rejected | Corrections | Lossy div |",
        "|---|---:|---:|---:|---:|",
    ])
    for policy in EXP025_POLICIES:
        row = global_p.get(policy, {})
        lines.append(
            f"| `{policy}` | {_fmt_rate(row.get('mean_acceptance_rate'))} | "
            f"{row.get('total_rejected', 0)} | {row.get('total_corrections', 0)} | "
            f"{row.get('lossy_divergence_count', 0)} |"
        )

    lines.extend([
        "",
        "## 8. Per-suite acceptance",
        "",
        *_suite_summary_table(agg),
        "",
        "## 9. Per-category acceptance",
        "",
        *_primary_category_table(agg),
        "",
        "## 10. Rejection/correction summary",
        "",
        f"| Total cells | {agg['total_cells']} |",
    ])
    total_rej = sum(r.get("total_rejected", 0) for r in global_p.values())
    total_corr = sum(r.get("total_corrections", 0) for r in global_p.values())
    lines.append(f"| Total rejected (all policies) | {total_rej} |")
    lines.append(f"| Total corrections (all policies) | {total_corr} |")

    lines.extend([
        "",
        "## 11. Prompt-level win/loss analysis",
        "",
        "### vs `baseline_k8_v4`",
        "",
        *_win_loss_section(agg.get("comparisons_vs_baseline_k8_v4", [])),
        "",
        "### vs `baseline_boundary4`",
        "",
        *_win_loss_section(agg.get("comparisons_vs_baseline_boundary4", [])),
        "",
        "### vs `int8_all`",
        "",
        *_win_loss_section(agg.get("comparisons_vs_int8_all", [])),
        "",
        "## 12. Best policy vs baseline_k8_v4",
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
        "## 13. Best policy vs baseline_boundary4",
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
        "## 14. Best policy vs int8_all",
        "",
    ])
    if best_int8:
        lines.append(
            f"**`{best_int8.get('policy_name', '—')}`** — wins {best_int8.get('wins', 0)}, "
            f"losses {best_int8.get('losses', 0)}, ties {best_int8.get('ties', 0)}."
        )
    else:
        lines.append("_No comparison available._")

    lc_fb = next(
        (
            r
            for r in agg.get("per_category_by_policy", [])
            if r["policy_name"] == POLICY_FALLBACK_INT8_HARD
            and r["v10_suite"] == "long_context"
        ),
        None,
    )
    lc_b4 = next(
        (
            r
            for r in agg.get("per_category_by_policy", [])
            if r["policy_name"] == POLICY_BASELINE_BOUNDARY4
            and r["v10_suite"] == "long_context"
        ),
        None,
    )
    rc_fb = next(
        (
            r
            for r in agg.get("per_category_by_policy", [])
            if r["policy_name"] == POLICY_FALLBACK_INT8_HARD
            and r["v10_suite"] == "retrieval_copy"
        ),
        None,
    )
    rc_b4 = next(
        (
            r
            for r in agg.get("per_category_by_policy", [])
            if r["policy_name"] == POLICY_BASELINE_BOUNDARY4
            and r["v10_suite"] == "retrieval_copy"
        ),
        None,
    )

    lines.extend([
        "",
        "## 15. Whether Experiment 020 repair hypotheses generalize",
        "",
        _generalization_paragraph(agg),
        "",
        "**Hard-category fallback check:**",
        "",
    ])
    if lc_fb and lc_b4:
        lines.append(
            f"- `long_context`: fallback_int8 {_fmt_rate(lc_fb['mean_acceptance_rate'])} "
            f"vs boundary4 {_fmt_rate(lc_b4['mean_acceptance_rate'])}"
        )
    if rc_fb and rc_b4:
        lines.append(
            f"- `retrieval_copy`: fallback_int8 {_fmt_rate(rc_fb['mean_acceptance_rate'])} "
            f"vs boundary4 {_fmt_rate(rc_b4['mean_acceptance_rate'])}"
        )
    cat_g = global_p.get(POLICY_CATEGORY_ADAPTIVE, {}).get("mean_acceptance_rate")
    b4_g = global_p.get(POLICY_BASELINE_BOUNDARY4, {}).get("mean_acceptance_rate")
    if cat_g is not None and b4_g is not None:
        lines.append(
            f"- `category_adaptive_policy` global {_fmt_rate(cat_g)} vs boundary4 "
            f"{_fmt_rate(b4_g)}"
        )

    lines.extend([
        "",
        "## 16. What changed from the 25-prompt pilot",
        "",
        "- Panel expanded from 25 → **128** prompts across all V10 suites.",
        "- Added `int8_all` uniform baseline; removed `structured_safe_mode` from panel.",
        f"- `max_new_tokens` anchor **{MAX_NEW_TOKENS}** (pilot used 32).",
        "- Optional multi-model extension documented in manifest.",
        "",
        "## 17. What this proves",
        "",
        "- Repair policies can be evaluated at full V10-suite scale without touching "
        "verification.",
        "- Exactness gate can hold across policy × prompt grid.",
        "- Category-aware compressor selection effects are measurable on the full suite.",
        "",
        "## 18. What this does not prove",
        "",
        "- No production policy deployment; not enabled in core ExactKV.",
        "- No claim of speed, memory, or final accuracy improvement.",
        "- V10 suites are not universal public benchmarks.",
        "- Policies use fixed offline rules; no online margin gating.",
        "",
        "## 19. Limitations",
        "",
        "- Single or few models depending on RunPod budget.",
        "- `draft_len_adaptive_policy` uses mixed draft lengths by category.",
        "- Pilot used a subset overlapping hard suites; full suite adds multilingual, "
        "reasoning_math, core_v2 breadth.",
        "",
        "## 20. Recommendation for future policy integration",
        "",
    ])
    if agg["exactkv_failures"] == 0 and cat_g is not None and b4_g is not None:
        if cat_g > b4_g + 1e-9 or (
            lc_fb and lc_b4 and lc_fb["mean_acceptance_rate"] > lc_b4["mean_acceptance_rate"]
        ):
            lines.append(
                "**Deferred recommendation:** keep `category_adaptive_policy` and "
                "`fallback_int8_for_hard_categories` as optional experiment-layer hooks — "
                "**not** default core behavior without separate approval."
            )
        else:
            lines.append(
                "**No default core integration recommended** from this full-suite run; "
                "pilot gains did not fully survive at scale."
            )
    else:
        lines.append(
            "**No policy integration recommended** until exactness gate passes on "
            "published cells."
        )

    lines.extend([
        "",
        "## 21. VeriCache attribution",
        "",
        "ExactKV draft-verify-commit loop inspired by "
        "[VeriCache](https://arxiv.org/abs/2605.17613); policies are ExactKV "
        "experiment-layer selectors on built-in compressors only.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 025 full-suite repair policy validation"
    )
    default_device, default_dtype = _default_device_dtype()
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--dtype", default=default_dtype)
    parser.add_argument("--include-15b", action="store_true")
    parser.add_argument("--include-3b", action="store_true")
    parser.add_argument("--skip-15b", action="store_true")
    parser.add_argument("--skip-3b", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=_ROOT / "reports")
    parser.add_argument("--docs-dir", type=Path, default=_ROOT / "docs")
    args = parser.parse_args()

    prompts = load_full_v10_prompts()
    if len(prompts) != 128:
        print(f"Warning: expected 128 prompts, got {len(prompts)}", flush=True)

    try:
        runpod_meta = collect_runpod_meta()
    except Exception:
        runpod_meta = {"device_requested": args.device, "dtype_requested": args.dtype}

    model_runs = [
        run_full_suite_for_model(
            MODEL_PRIMARY,
            prompts,
            device=args.device,
            dtype=args.dtype,
        )
    ]
    skipped: list[str] = []

    if args.include_15b and not args.skip_15b:
        if model_runs[0]["aggregate"]["exactkv_failures"] == 0:
            try:
                model_runs.append(
                    run_full_suite_for_model(
                        MODEL_OPTIONAL_15B,
                        prompts,
                        device=args.device,
                        dtype=args.dtype,
                    )
                )
            except Exception as exc:
                skipped.append(f"{MODEL_OPTIONAL_15B}: {exc}")
                print(f"Optional 1.5B run skipped: {exc}", flush=True)
        else:
            skipped.append(f"{MODEL_OPTIONAL_15B}: primary had exactkv failures")

    if args.include_3b and not args.skip_3b:
        if all(r["aggregate"]["exactkv_failures"] == 0 for r in model_runs):
            try:
                model_runs.append(
                    run_full_suite_for_model(
                        MODEL_OPTIONAL_3B,
                        prompts,
                        device=args.device,
                        dtype=args.dtype,
                    )
                )
            except Exception as exc:
                skipped.append(f"{MODEL_OPTIONAL_3B}: {exc}")
                print(f"Optional 3B run skipped: {exc}", flush=True)
        else:
            skipped.append(f"{MODEL_OPTIONAL_3B}: prior runs had exactkv failures")

    artifact = build_artifact(
        model_runs,
        runpod_meta=runpod_meta,
        prompt_count=len(prompts),
        skipped_models=skipped,
    )
    assert_policy_artifact_safe(artifact)
    _assert_no_forbidden(artifact)

    json_path = args.reports_dir / "experiment_025_full_suite_repair_policy.json"
    csv_path = args.reports_dir / "experiment_025_full_suite_repair_policy.csv"
    md_path = args.docs_dir / "EXPERIMENT_025_FULL_SUITE_REPAIR_POLICY.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    write_csv(artifact, csv_path)
    md_path.write_text(generate_markdown_report(artifact), encoding="utf-8")

    agg = artifact["aggregate"]
    print(f"\nExperiment 025 complete: {agg['total_cells']} cells", flush=True)
    print(f"  exactkv_failures: {agg['exactkv_failures']}", flush=True)
    g = agg["global_by_policy"]
    for p in EXP025_POLICIES:
        print(f"  {p}: accept={g[p]['mean_acceptance_rate']:.3f}", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
