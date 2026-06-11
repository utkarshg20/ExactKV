#!/usr/bin/env python3
"""Experiment 016: Qwen2.5-3B validation on V10 suites (V11 Phase 2).

Loads all seven V10 prompt suites (128 prompts) × seven built-in compressors on
RunPod CUDA float16. No timing, throughput, latency, speedup, runtime_seconds,
or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.acceptance_tables import group_acceptance_by_compressor
from exactkv.benchmarks.reports import (
    build_run_manifest,
    load_json_report,
    validate_report,
    write_csv_report,
    write_json_report,
)
from exactkv.benchmarks.runner import RunConfig, run_one
from exactkv.benchmarks.sweeps import _compute_aggregate
from exactkv.benchmarks.v10_prompts import list_v10_suites, load_all_v10_prompts
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-3B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
EXPERIMENT_CLASS = "v11_qwen3b_v10_suites"

V10_SUITES = list_v10_suites()

COMPRESSORS = [
    "noop",
    "backend_passthrough",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "k_full_v8",
    "k8_v_full",
]

# Experiment 012 anchors (0.5B, V10 suites, draft_len=4, max_new_tokens=16).
_EXP012_ANCHORS = {
    "int8": 0.957,
    "k8_v4_sim": 0.914,
    "k8_v4_boundary4_v8_sim": 0.923,
    "k_full_v8": 0.996,
    "k8_v_full": 0.965,
    "noop": 1.000,
    "backend_passthrough": 1.000,
}

# Experiment 015 anchors (1.5B, V10 suites, draft_len=4, max_new_tokens=16).
_EXP015_ANCHORS = {
    "int8": 0.978,
    "k8_v4_sim": 0.942,
    "k8_v4_boundary4_v8_sim": 0.951,
    "k_full_v8": 0.996,
    "k8_v_full": 0.978,
    "noop": 1.000,
    "backend_passthrough": 1.000,
}

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

_LOW_N_THRESHOLD = 5


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


def _aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    acc_blocks = [r.get("exactkv", {}).get("acceptance", {}) for r in results]
    acceptance_rates = [a.get("acceptance_rate", 0.0) for a in acc_blocks]
    avg_lengths = [a.get("avg_accepted_per_round", 0.0) for a in acc_blocks]
    denom = max(n, 1)
    return {
        "num_runs": n,
        "mean_acceptance_rate": sum(acceptance_rates) / denom,
        "mean_average_accepted_length": sum(avg_lengths) / denom,
        "total_drafted": sum(a.get("total_drafted", 0) for a in acc_blocks),
        "total_accepted": sum(a.get("total_accepted", 0) for a in acc_blocks),
        "total_rejected": sum(a.get("total_rejected", 0) for a in acc_blocks),
        "total_corrections": sum(a.get("total_corrections", 0) for a in acc_blocks),
        "exactkv_failures": sum(1 for r in results if r.get("exactkv_failure", False)),
        "lossy_divergence_count": sum(
            1 for r in results
            if not r.get("lossy", {}).get("token_exact_match", True)
        ),
    }


def _group_acceptance(
    results: list[dict[str, Any]],
    key_fn,
    key_name: str,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    table = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {key_name: key}
        row.update(_aggregate_group(group))
        table.append(row)
    return table


def _group_compressor_category(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(r["compressor_name"], r.get("v10_primary_category", r.get("category", "")))].append(r)
    table = []
    for (comp, cat), group in sorted(groups.items()):
        row = {
            "compressor_name": comp,
            "primary_category": cat,
        }
        row.update(_aggregate_group(group))
        if row["num_runs"] < _LOW_N_THRESHOLD:
            row["low_n_warning"] = True
        table.append(row)
    return table


def _prompt_acceptance_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (prompt_id, compressor) acceptance for win/loss analysis."""
    rows = []
    for r in results:
        acc = r.get("exactkv", {}).get("acceptance", {})
        rows.append({
            "prompt_id": r["prompt_id"],
            "v10_suite": r.get("v10_suite", ""),
            "primary_category": r.get("v10_primary_category", r.get("category", "")),
            "compressor_name": r["compressor_name"],
            "acceptance_rate": acc.get("acceptance_rate", 0.0),
            "total_rejected": acc.get("total_rejected", 0),
            "total_corrections": acc.get("total_corrections", 0),
            "lossy_diverged": not r.get("lossy", {}).get("token_exact_match", True),
        })
    return rows


def _win_loss_pair(
    prompt_table: list[dict[str, Any]],
    comp_a: str,
    comp_b: str,
) -> dict[str, Any]:
    by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    for row in prompt_table:
        by_prompt[row["prompt_id"]][row["compressor_name"]] = row["acceptance_rate"]

    wins_a = wins_b = ties = 0
    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins_a": 0, "wins_b": 0, "ties": 0}
    )
    for pid, rates in sorted(by_prompt.items()):
        if comp_a not in rates or comp_b not in rates:
            continue
        ra, rb = rates[comp_a], rates[comp_b]
        cat = next(
            (r["primary_category"] for r in prompt_table if r["prompt_id"] == pid),
            "unknown",
        )
        if ra > rb + 1e-9:
            wins_a += 1
            per_category[cat]["wins_a"] += 1
        elif rb > ra + 1e-9:
            wins_b += 1
            per_category[cat]["wins_b"] += 1
        else:
            ties += 1
            per_category[cat]["ties"] += 1

    return {
        "compressor_a": comp_a,
        "compressor_b": comp_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "per_category": dict(per_category),
    }


def _lowest_acceptance_prompts(
    results: list[dict[str, Any]],
    compressor_name: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    subset = [r for r in results if r["compressor_name"] == compressor_name]
    ranked = sorted(
        subset,
        key=lambda r: r.get("exactkv", {}).get("acceptance", {}).get("acceptance_rate", 1.0),
    )
    out = []
    for r in ranked[:limit]:
        acc = r.get("exactkv", {}).get("acceptance", {})
        out.append({
            "prompt_id": r["prompt_id"],
            "v10_suite": r.get("v10_suite", ""),
            "primary_category": r.get("v10_primary_category", ""),
            "acceptance_rate": acc.get("acceptance_rate", 0.0),
        })
    return out


def _enrich_result(prompt_entry: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Attach V10 metadata to a standard runner result."""
    out = dict(cell)
    for key in (
        "v10_id",
        "v10_suite",
        "v10_suite_version",
        "v10_primary_category",
        "v10_secondary_tags",
    ):
        if key in prompt_entry:
            out[key] = prompt_entry[key]
    return out


def run_experiment_016(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
    dtype: str = DTYPE,
) -> dict[str, Any]:
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
            config = RunConfig(
                compressor_name=compressor_name,
                draft_len=DRAFT_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            cell = run_one(runtime, prompt_entry, config)
            results.append(_enrich_result(prompt_entry, cell))

    prompt_manifest = [
        {
            "id": p["v10_id"],
            "suite": p["v10_suite"],
            "suite_version": p["v10_suite_version"],
            "primary_category": p["v10_primary_category"],
            "secondary_tags": p.get("v10_secondary_tags"),
        }
        for p in prompts
    ]

    manifest = build_run_manifest(
        model_name=runtime.model_name,
        prompt_suite="v10_all",
        compressor_names=COMPRESSORS,
        draft_len=DRAFT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        device=str(runtime.device),
        dtype=dtype,
    )
    manifest["experiment"] = "016_qwen3b_v10_suites"
    manifest["experiment_class"] = EXPERIMENT_CLASS
    manifest["comparison_anchors"] = [
        "experiment_012_qwen05b_v10_suites",
        "experiment_015_qwen15b_v10_suites",
    ]
    manifest["v10_suites"] = V10_SUITES
    manifest["v10_prompt_count"] = len(prompts)
    manifest["v10_prompt_manifest"] = prompt_manifest

    aggregate = _compute_aggregate(
        results,
        num_prompts=len(prompts),
        compressor_names=COMPRESSORS,
        draft_lengths=[DRAFT_LEN],
    )
    aggregate["acceptance_by_compressor"] = group_acceptance_by_compressor(
        {"results": results}
    )
    aggregate["acceptance_by_suite"] = _group_acceptance(
        results, lambda r: r.get("v10_suite", "unknown"), "suite"
    )
    aggregate["acceptance_by_primary_category"] = _group_acceptance(
        results,
        lambda r: r.get("v10_primary_category", r.get("category", "unknown")),
        "primary_category",
    )
    aggregate["acceptance_by_compressor_and_category"] = _group_compressor_category(
        results
    )

    prompt_table = _prompt_acceptance_table(results)
    aggregate["win_loss_boundary4_vs_k8_v4_sim"] = _win_loss_pair(
        prompt_table, "k8_v4_boundary4_v8_sim", "k8_v4_sim"
    )
    aggregate["win_loss_int8_vs_boundary4"] = _win_loss_pair(
        prompt_table, "int8", "k8_v4_boundary4_v8_sim"
    )
    aggregate["lowest_acceptance_by_compressor"] = {
        comp: _lowest_acceptance_prompts(results, comp, limit=3)
        for comp in COMPRESSORS
        if comp not in ("noop", "backend_passthrough")
    }

    return {
        "manifest": manifest,
        "results": results,
        "aggregate": aggregate,
    }


def _fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def _suite_counts(prompts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for p in prompts:
        counts[p["v10_suite"]] += 1
    return dict(counts)


def _category_counts(prompts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for p in prompts:
        counts[p["v10_primary_category"]] += 1
    return dict(counts)


def generate_markdown_report(
    report: dict[str, Any],
    prompts: list[dict[str, Any]],
    runpod_meta: dict[str, Any] | None = None,
) -> str:
    agg = report["aggregate"]
    by_comp = agg["acceptance_by_compressor"]
    by_suite = agg["acceptance_by_suite"]
    by_cat = agg["acceptance_by_primary_category"]
    by_comp_cat = agg["acceptance_by_compressor_and_category"]
    wl_b4 = agg["win_loss_boundary4_vs_k8_v4_sim"]
    wl_i8 = agg["win_loss_int8_vs_boundary4"]
    suite_counts = _suite_counts(prompts)
    cat_counts = _category_counts(prompts)

    rp = runpod_meta or {}
    lines = [
        "# Experiment 016: Qwen2.5-3B V10 Suite Validation",
        "",
        "_Generated by `scripts/run_experiment_016_qwen3b_v10_suites.py`. "
        "V11 Phase 2 — 3B built-in scale validation on V10 suites._",
        "",
        "> This is **larger-model validation**, not a performance benchmark.",
        "> The V10 suites are **stronger and more diverse** than the old 34-prompt "
        "`core` suite, but are **still not a universal benchmark**.",
        "> ExactKV preserves the exactness gate: `exactkv_output_ids == full_output_ids`.",
        "> Simulated compressors (`_sim`) are **not** real packed-bit backends.",
        "> `total_kv_footprint_bytes` is a conservative accounting sum, "
        "**not** measured peak GPU memory.",
        "> **Active GPU memory is not reported.**",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> **External-paper results are not ExactKV results.**",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Validate whether V10/V11 findings from Experiments **012** (0.5B) and "
        "**015** (1.5B) transfer to **Qwen/Qwen2.5-3B** — same 128 prompts, "
        "built-in compressor panel, and anchor settings (`draft_len=4`, "
        "`max_new_tokens=16`).",
        "",
        "## 2. RunPod environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {rp.get('gpu', '—')} |",
        f"| Host | {rp.get('hostname', '—')} |",
        f"| torch | {rp.get('torch_version', '—')} |",
        f"| CUDA | {rp.get('cuda_version', '—')} |",
        f"| transformers | {rp.get('transformers_version', '—')} |",
        f"| dtype | {DTYPE} |",
        f"| device | cuda |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_016_qwen3b_v10_suites.py \\",
        "  --device cuda --dtype float16",
        "```",
        "",
        "Artifacts (gitignored): `reports/experiment_016_qwen3b_v10_suites.json`,",
        "`reports/experiment_016_qwen3b_v10_suites.csv`.",
        "",
        "## 3. Model and prompt suite summary",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model | `{MODEL_NAME}` |",
        f"| `draft_len` | {DRAFT_LEN} |",
        f"| `max_new_tokens` | {MAX_NEW_TOKENS} |",
        f"| Experiment class | `{EXPERIMENT_CLASS}` |",
        f"| Total prompts | **{len(prompts)}** |",
        f"| Total cells | **{agg['total_runs']}** ({len(prompts)} × {len(COMPRESSORS)}) |",
        "",
        "### Suite breakdown",
        "",
        "",
        "| Suite | Prompts |",
        "|---|---:|",
    ]
    for suite in V10_SUITES:
        lines.append(f"| `{suite}` | {suite_counts.get(suite, 0)} |")
    lines.append(f"| **Total** | **{len(prompts)}** |")
    lines.extend([
        "",
        "Primary-category distribution (prompts may appear once each):",
        "",
        "| `primary_category` | Count |",
        "|---|---:|",
    ])
    for cat, n in sorted(cat_counts.items()):
        flag = " ⚠️ low-n" if n < _LOW_N_THRESHOLD else ""
        lines.append(f"| `{cat}` | {n}{flag} |")

    lines.extend([
        "",
        "## 4. Compressor panel",
        "",
        "| Compressor | Role |",
        "|---|---|",
        "| `noop` | Lossless identity baseline |",
        "| `backend_passthrough` | V6 BackendAdapter PoC |",
        "| `int8` | Real symmetric INT8 |",
        "| `k8_v4_sim` | Simulated uniform K8/V4 |",
        "| `k8_v4_boundary4_v8_sim` | Layer-aware boundary V (N=4) |",
        "| `k_full_v8` | Real INT8 V, full K |",
        "| `k8_v_full` | Real INT8 K, full V |",
        "",
        "## 5. Exactness result",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total runs | {agg['total_runs']} |",
        f"| **ExactKV failures** | **{agg['exactkv_failures']}** |",
        "",
        "## 6. Global acceptance leaderboard",
        "",
        "| Compressor | Accept rate | Avg accept/round | Rejected | Corrections | Lossy div cells |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(by_comp, key=lambda r: -r["mean_acceptance_rate"]):
        div = sum(
            1 for r in report["results"]
            if r["compressor_name"] == row["compressor_name"]
            and not r.get("lossy", {}).get("token_exact_match", True)
        )
        lines.append(
            f"| `{row['compressor_name']}` | {_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{_fmt_rate(row['mean_average_accepted_length'])} | "
            f"{row['total_rejected']} | {row['total_corrections']} | {div} |"
        )

    lines.extend([
        "",
        "## 7. Per-suite acceptance leaderboard",
        "",
        "Mean acceptance across all compressors within each suite:",
        "",
        "| Suite | Runs | Mean accept | Lossy div |",
        "|---|---:|---:|---:|",
    ])
    for row in by_suite:
        lines.append(
            f"| `{row['suite']}` | {row['num_runs']} | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | "
            f"{row['lossy_divergence_count']} |"
        )

    lines.extend([
        "",
        "## 8. Per-category acceptance leaderboard",
        "",
        "Mean acceptance across all compressors within each `primary_category`:",
        "",
        "| Category | Runs | Mean accept | Low-n? |",
        "|---|---:|---:|---|",
    ])
    for row in by_cat:
        n_prompts = cat_counts.get(row["primary_category"], 0)
        low = "yes ⚠️" if n_prompts < _LOW_N_THRESHOLD else "no"
        lines.append(
            f"| `{row['primary_category']}` | {row['num_runs']} | "
            f"{_fmt_rate(row['mean_acceptance_rate'])} | {low} |"
        )

    lines.extend([
        "",
        "## 9. Divergence/rejection/correction summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Lossy divergence cells | {agg['lossy_divergence_count']} / {agg['total_runs']} |",
        f"| Total rejected (ExactKV) | {agg['total_rejected']} |",
        f"| Total corrections | {agg['total_corrections']} |",
        f"| Mean acceptance (all cells) | {_fmt_rate(agg['mean_acceptance_rate'])} |",
        "",
        "## 10. Prompt-level win/loss analysis",
        "",
        f"**`k8_v4_boundary4_v8_sim` vs `k8_v4_sim`:** "
        f"boundary4 wins **{wl_b4['wins_a']}**, k8_v4_sim wins **{wl_b4['wins_b']}**, "
        f"ties **{wl_b4['ties']}** (per prompt, n={len(prompts)}).",
        "",
        f"**`int8` vs `k8_v4_boundary4_v8_sim`:** "
        f"int8 wins **{wl_i8['wins_a']}**, boundary4 wins **{wl_i8['wins_b']}**, "
        f"ties **{wl_i8['ties']}**.",
        "",
        "## 11. Boundary4 vs k8_v4_sim by category",
        "",
        "| Category | boundary4 wins | k8_v4_sim wins | ties |",
        "|---|---:|---:|---:|",
    ])
    for cat in sorted(wl_b4["per_category"]):
        c = wl_b4["per_category"][cat]
        lines.append(
            f"| `{cat}` | {c['wins_a']} | {c['wins_b']} | {c['ties']} |"
        )

    lines.extend([
        "",
        "## 12. int8 vs boundary4 by category",
        "",
        "| Category | int8 wins | boundary4 wins | ties |",
        "|---|---:|---:|---:|",
    ])
    for cat in sorted(wl_i8["per_category"]):
        c = wl_i8["per_category"][cat]
        lines.append(
            f"| `{cat}` | {c['wins_a']} | {c['wins_b']} | {c['ties']} |"
        )

    lines.extend([
        "",
        "## 13. Comparison to Experiment 012 on Qwen2.5-0.5B",
        "",
        "Experiment **012** ran the same V10 suites at **0.5B** (float32, CPU-first).",
        "",
        "| Compressor | Exp 012 (0.5B) | Exp 016 (3B) | Δ (3B − 0.5B) |",
        "|---|---:|---:|---:|",
    ])
    exp_lookup = {r["compressor_name"]: r["mean_acceptance_rate"] for r in by_comp}
    for comp in ["int8", "k8_v4_sim", "k8_v4_boundary4_v8_sim", "k_full_v8", "k8_v_full"]:
        e12 = _EXP012_ANCHORS.get(comp)
        e16 = exp_lookup.get(comp)
        if e12 is not None and e16 is not None:
            delta = e16 - e12
            lines.append(
                f"| `{comp}` | {_fmt_rate(e12)} | {_fmt_rate(e16)} | {delta:+.3f} |"
            )

    b4 = exp_lookup.get("k8_v4_boundary4_v8_sim", 0)
    k84 = exp_lookup.get("k8_v4_sim", 0)
    margin = b4 - k84
    margin_012 = _EXP012_ANCHORS.get("k8_v4_boundary4_v8_sim", 0) - _EXP012_ANCHORS.get(
        "k8_v4_sim", 0
    )
    lines.extend([
        "",
        f"**boundary4 − k8_v4_sim margin:** Exp 016 **{margin:+.3f}** vs Exp 012 "
        f"**{margin_012:+.3f}**.",
        "",
        "## 14. Comparison to Experiment 015 on Qwen2.5-1.5B",
        "",
        "Experiment **015** ran the same V10 suites at **1.5B** (RunPod, float16 CUDA).",
        "",
        "| Compressor | Exp 015 (1.5B) | Exp 016 (3B) | Δ (3B − 1.5B) |",
        "|---|---:|---:|---:|",
    ])
    for comp in ["int8", "k8_v4_sim", "k8_v4_boundary4_v8_sim", "k_full_v8", "k8_v_full"]:
        e15 = _EXP015_ANCHORS.get(comp)
        e16 = exp_lookup.get(comp)
        if e15 is not None and e16 is not None:
            delta = e16 - e15
            lines.append(
                f"| `{comp}` | {_fmt_rate(e15)} | {_fmt_rate(e16)} | {delta:+.3f} |"
            )

    margin_015 = _EXP015_ANCHORS.get("k8_v4_boundary4_v8_sim", 0) - _EXP015_ANCHORS.get(
        "k8_v4_sim", 0
    )
    lines.extend([
        "",
        f"**boundary4 − k8_v4_sim margin:** Exp 016 **{margin:+.3f}** vs Exp 015 "
        f"**{margin_015:+.3f}**.",
        "",
        "## 15. Whether V10/V11 findings transfer to 3B",
        "",
    ])
    b4_wins = wl_b4["wins_a"] > wl_b4["wins_b"]
    int8_wins = wl_i8["wins_a"] > wl_i8["wins_b"]
    hardest_suite = min(by_suite, key=lambda r: r["mean_acceptance_rate"]) if by_suite else None
    lines.extend([
        f"- **Exactness gate:** `exactkv_failures == {agg['exactkv_failures']}`.",
        f"- **boundary4 vs k8_v4_sim:** boundary4 wins **{wl_b4['wins_a']}** / "
        f"**{len(prompts)}** prompts ({'ordering largely transfers' if b4_wins else 'weaker at 3B'}).",
        f"- **int8 vs boundary4:** int8 wins **{wl_i8['wins_a']}** prompts "
        f"({'int8 remains stronger' if int8_wins else 'mixed'}).",
    ])
    if hardest_suite:
        lines.append(
            f"- **Hardest suite (pooled):** `{hardest_suite['suite']}` "
            f"(mean accept {_fmt_rate(hardest_suite['mean_acceptance_rate'])})."
        )
    lines.extend([
        "",
        "## 16. What this proves",
        "",
        "- The exactness gate holds on **128 × 7 = 896** cells at **3B** on V10 suites.",
        "- Per-suite and per-category leaderboards are reproducible at 3B scale.",
        "- V10/V11 category and compressor ordering can be compared across 0.5B, 1.5B, and 3B.",
        "",
        "## 17. What this does not prove",
        "",
        "- That V10 suites are universal public benchmarks.",
        "- Production serving, throughput, or latency behaviour.",
        "- That `_sim` compressors are real packed-bit backends.",
        "- That restricted real backends (KVQuant/TurboQuant/KIVI) behave the same.",
        "- That external-paper compression results apply to ExactKV without separate runs.",
        "",
        "## 18. Limitations",
        "",
        "- Single 3B model; anchor `draft_len` / `max_new_tokens` only.",
        "- Some categories have low prompt counts (flagged above).",
        "- Built-in compressor panel only; no factory-only real backends in this phase.",
        "- RunPod single-GPU environment; OOM would block full-suite completion.",
        "",
        "## 19. Relation to V11 Phase 3",
        "",
        "Experiment 016 completes the **optional 3B built-in stretch** gate. "
        "Phase 3 (Experiment 017) is serving sidecar/probe feasibility refresh — "
        "do not start until this report is reviewed.",
        "",
        "## 20. VeriCache attribution",
        "",
        "The draft-then-verify algorithm is from **VeriCache** (Yao et al., "
        "arXiv:2605.17613, 2026). ExactKV implements a compressor-agnostic "
        "evaluation harness; Experiment 016 does not claim novel compression methods.",
        "",
    ])
    return "\n".join(lines)


def _collect_runpod_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        import socket
        import torch
        import transformers

        meta["hostname"] = socket.gethostname()
        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda
        meta["transformers_version"] = transformers.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 016 (3B V10 suites)")
    parser.add_argument(
        "--json-out",
        default="reports/experiment_016_qwen3b_v10_suites.json",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/experiment_016_qwen3b_v10_suites.csv",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/EXPERIMENT_016_QWEN3B_V10_SUITES.md",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
    )
    parser.add_argument(
        "--dtype",
        default=DTYPE,
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "Experiment 016 requires CUDA. Use RunPod GPU with --device cuda."
            )

    prompts = load_all_v10_prompts()
    expected = len(prompts) * len(COMPRESSORS)
    print(
        f"Experiment 016: {len(prompts)} prompts × {len(COMPRESSORS)} compressors "
        f"= {expected} cells"
    )
    if len(prompts) != 128:
        print(
            f"ERROR: expected 128 V10 prompts, got {len(prompts)}. "
            "Run validate_v10_prompt_suites.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading model {args.model} ({args.dtype}, {args.device}) ...")
    try:
        runtime = ModelRuntime(
            model_name=args.model, device=args.device, dtype=args.dtype
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                f"OOM loading {args.model} on {args.device}. "
                "Stop — do not reduce suite. Try a larger GPU."
            ) from exc
        raise

    try:
        report = run_experiment_016(runtime, prompts, dtype=args.dtype)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                f"OOM during Experiment 016 on {args.model}. "
                "Stop — do not reduce suite. Try a larger GPU."
            ) from exc
        raise
    _assert_no_forbidden_fields(report)

    runpod_meta = _collect_runpod_meta()
    report["manifest"]["runpod_meta"] = runpod_meta

    json_path = Path(args.json_out)
    csv_path = Path(args.csv_out)
    md_path = Path(args.markdown_out)

    write_json_report(report, json_path, manifest=report["manifest"])
    write_csv_report(report, csv_path)

    warnings = validate_report(load_json_report(json_path))
    if warnings:
        print("validate_report warnings:", warnings, file=sys.stderr)
        return 1

    md_path.write_text(
        generate_markdown_report(report, prompts, runpod_meta), encoding="utf-8"
    )

    agg = report["aggregate"]
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"exactkv_failures: {agg['exactkv_failures']}")

    if agg["exactkv_failures"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
