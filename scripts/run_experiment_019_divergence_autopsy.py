#!/usr/bin/env python3
"""Experiment 019: deep divergence autopsy and repair hypotheses (V11 Phase 5).

Mechanistic forensics at lossy-divergence and ExactKV rejection points.
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

from exactkv.analysis.divergence_autopsy import (
    AUTOPSY_SUITES,
    KVQUANT_NAME,
    REQUIRED_COMPRESSORS,
    aggregate_autopsy_results,
    assert_autopsy_artifact_safe,
    kvquant_available,
    load_autopsy_prompt_subset,
    resolve_compressor,
    run_autopsy_cell,
)
from exactkv.metrics.gpu_memory_pilot import collect_runpod_meta

MODEL_PRIMARY = "Qwen/Qwen2.5-0.5B"
MODEL_OPTIONAL = "Qwen/Qwen2.5-1.5B"
DTYPE = "float16"
DEVICE = "cuda"
DRAFT_LENGTHS = [4, 8]
MAX_NEW_TOKENS = 32
EXPERIMENT_CLASS = "v11_divergence_autopsy"
ATTENTION_PROMPT_LIMIT = 2

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


def run_autopsy_for_model(
    model_name: str,
    prompts: list[dict[str, Any]],
    compressors: list[str],
    *,
    device: str,
    dtype: str,
    attention_prompt_ids: set[str],
) -> dict[str, Any]:
    from exactkv.runtime.model_runtime import ModelRuntime

    print(f"Loading model {model_name} ({dtype}, {device}) ...")
    runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    compressor_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    total = len(prompts) * len(compressors) * len(DRAFT_LENGTHS)
    idx = 0
    for prompt_entry in prompts:
        for compressor_name in compressors:
            compressor = resolve_compressor(runtime, compressor_name, compressor_cache)
            for draft_len in DRAFT_LENGTHS:
                idx += 1
                collect_attn = prompt_entry["prompt_id"] in attention_prompt_ids
                print(
                    f"  [{idx}/{total}] {model_name} {prompt_entry['prompt_id']} × "
                    f"{compressor_name} draft={draft_len}",
                    flush=True,
                )
                cell = run_autopsy_cell(
                    runtime,
                    prompt_entry,
                    compressor,
                    draft_len=draft_len,
                    max_new_tokens=MAX_NEW_TOKENS,
                    collect_kv_errors=compressor_name != "noop",
                    collect_attention=collect_attn,
                )
                results.append(cell)

    aggregate = aggregate_autopsy_results(results)
    return {
        "model_name": model_name,
        "results": results,
        "aggregate": aggregate,
    }


def build_artifact(
    model_runs: list[dict[str, Any]],
    *,
    runpod_meta: dict[str, Any],
    compressors: list[str],
    prompt_count: int,
    kvquant_included: bool,
) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    for run in model_runs:
        all_results.extend(run["results"])

    agg = aggregate_autopsy_results(all_results)
    return {
        "manifest": {
            "experiment": "019_divergence_autopsy",
            "experiment_class": EXPERIMENT_CLASS,
            "artifact_type": "isolated_autopsy",
            "standard_report_schema_modified": False,
            "device": DEVICE,
            "dtype": DTYPE,
            "draft_lengths": DRAFT_LENGTHS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_suites": list(AUTOPSY_SUITES),
            "prompts_per_suite": 5,
            "prompt_count": prompt_count,
            "compressors": compressors,
            "kvquant_included": kvquant_included,
            "models_run": [r["model_name"] for r in model_runs],
            "runpod_meta": runpod_meta,
        },
        "model_runs": model_runs,
        "aggregate": agg,
        "note": (
            "Divergence forensics only. Repair policies are hypotheses unless "
            "separately implemented. Attention weights are not fabricated."
        ),
    }


def write_autopsy_csv(artifact: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in artifact["model_runs"]:
        for r in run["results"]:
            rows.append({
                "model_name": r["model_name"],
                "prompt_id": r["prompt_id"],
                "v10_suite": r.get("v10_suite", ""),
                "v10_primary_category": r.get("v10_primary_category", ""),
                "compressor_name": r["compressor_name"],
                "draft_len": r["draft_len"],
                "exactkv_failure": r["exactkv_failure"],
                "lossy_diverged": not r["lossy"]["token_exact_match"],
                "first_divergence_idx": r["lossy"]["first_divergence_idx"],
                "divergence_token_type": r["lossy"].get("divergence_token_type"),
                "acceptance_rate": r["exactkv"]["acceptance"]["acceptance_rate"],
                "rejection_events": len(r["autopsy"]["rejection_observations"]),
                "mean_k_cosine": (
                    r["autopsy"]["kv_layer_errors"]["mean_k_cosine"]
                    if r["autopsy"].get("kv_layer_errors")
                    else None
                ),
                "mean_v_cosine": (
                    r["autopsy"]["kv_layer_errors"]["mean_v_cosine"]
                    if r["autopsy"].get("kv_layer_errors")
                    else None
                ),
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
    hyps = agg.get("repair_hypotheses", [])

    lines = [
        "# Experiment 019: Deep Divergence Autopsy and Repair Hypotheses",
        "",
        "_Generated by `scripts/run_experiment_019_divergence_autopsy.py`. "
        "V11 Phase 5 — divergence forensics only._",
        "",
        "> This is **divergence forensics**, not a performance benchmark.",
        "> ExactKV preserves exact greedy output; this analysis studies **draft "
        "divergence before verification**.",
        "> **Attention weights are not fabricated**; if absent, deferral is stated.",
        "> **Repair policies are hypotheses only** unless separately implemented later.",
        "> ExactKV does **not** claim speedup, throughput, latency, runtime, "
        "tokens/sec, active GPU memory, or production readiness.",
        "> ExactKV does **not** claim compressed-KV improves final model accuracy.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Move beyond aggregate acceptance to identify where, why, and how compressed-KV "
        "drafts diverge. Produce mechanistic autopsy signals and concrete repair "
        "hypotheses for future compressor/policy work.",
        "",
        "## 2. Why this is needed after Experiments 012–018",
        "",
        "Experiments 012–016 established compressor rankings at scale; Experiment 013 "
        "added sensitivity forensics without logit margins or layer-wise KV error; "
        "Experiments 017–018 addressed serving and GPU memory — not mechanistic "
        "divergence repair paths.",
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
        f"| KVQuant included | {manifest['kvquant_included']} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_019_divergence_autopsy.py "
        "--device cuda --dtype float16",
        "```",
        "",
        "## 4. Prompt subset",
        "",
        f"- **Suites:** {', '.join(manifest['prompt_suites'])}",
        f"- **Per suite:** {manifest['prompts_per_suite']} (first ids by sorted order)",
        f"- **Total prompts:** {manifest['prompt_count']}",
        "",
        "## 5. Compressor panel",
        "",
        "| Compressor | Included |",
        "|---|---|",
    ]
    for c in manifest["compressors"]:
        lines.append(f"| `{c}` | yes |")

    lines.extend([
        "",
        f"- **draft_len:** {manifest['draft_lengths']}",
        f"- **max_new_tokens:** {manifest['max_new_tokens']}",
        "",
        "## 6. Exactness result",
        "",
        f"| ExactKV failures | **{agg['exactkv_failures']}** |",
        "",
        "## 7. Divergence summary",
        "",
        f"| Lossy divergence cells | {agg['lossy_divergence_cells']} / {agg['total_cells']} |",
        f"| Rejection events (ExactKV) | {agg['total_rejection_events']} |",
        f"| Lossy first-divergence events | {agg['total_lossy_divergence_events']} |",
        "",
        "## 8. First divergence by category",
        "",
        "| Suite | Divergence cells |",
        "|---|---:|",
    ])
    for suite, count in sorted(agg.get("first_divergence_by_suite", {}).items()):
        lines.append(f"| `{suite}` | {count} |")

    lines.extend([
        "",
        "## 9. First divergence by token type",
        "",
        "| Token type | Count |",
        "|---|---:|",
    ])
    for tt, count in sorted(agg.get("first_divergence_by_token_type", {}).items()):
        lines.append(f"| {tt} | {count} |")

    lines.extend([
        "",
        "## 10. Rejection/correction position summary",
        "",
        f"| Total rejection events | {agg['total_rejection_events']} |",
        "",
        "Rejection token types:",
        "",
        "| Token type | Count |",
        "|---|---:|",
    ])
    for tt, count in sorted(agg.get("rejection_token_types", {}).items()):
        lines.append(f"| {tt} | {count} |")

    lines.extend([
        "",
        "## 11. Drafter vs verifier top-k analysis",
        "",
        f"| Events with logit margin | {agg.get('logit_margin_count', 0)} |",
        f"| Mean logit margin (verifier top-1 − drafter) | "
        f"{_fmt_rate(agg.get('logit_margin_mean'))} |",
        f"| Low-margin events (<1.0) | {agg.get('logit_margin_low_count', 0)} |",
        f"| Drafter token in verifier top-5 | "
        f"{agg.get('drafter_in_verifier_top_k_count', 0)} |",
        "",
        "## 12. Logit-margin findings",
        "",
    ])
    if agg.get("logit_margin_low_count", 0) > 0:
        lines.append(
            "A measurable share of divergences occur when the verifier margin over the "
            "drafter token is small — corrections are often near-decision-boundary, not "
            "random token substitutions."
        )
    else:
        lines.append(
            "Few low-margin events in this panel; divergences may be driven more by "
            "KV drift than near-tie logits."
        )

    lines.extend([
        "",
        "## 13. Structured-output findings",
        "",
        "Structured suites (`tool_json`, `code_structured`) were analyzed with bracket "
        "depth, quote balance, and JSON-ish prefix heuristics on lossy output text.",
        "",
        "## 14. Layer/KV error findings",
        "",
    ])
    kv_rows = agg.get("kv_layer_error_summaries", [])
    if kv_rows:
        lines.extend([
            "| Compressor | Mean K cosine | Mean V cosine |",
            "|---|---:|---:|",
        ])
        by_comp: dict[str, list[dict[str, Any]]] = {}
        for row in kv_rows:
            by_comp.setdefault(row["compressor_name"], []).append(row)
        for comp, rows in sorted(by_comp.items()):
            mk = sum(r["mean_k_cosine"] for r in rows) / len(rows)
            mv = sum(r["mean_v_cosine"] for r in rows) / len(rows)
            lines.append(f"| `{comp}` | {mk:.4f} | {mv:.4f} |")
    else:
        lines.append("_No layer KV error metrics collected._")

    lines.extend([
        "",
        "## 15. Attention logging result",
        "",
    ])
    if agg.get("attention_weights_logged"):
        lines.append(
            "Prefill-only attention snapshots were obtained on a tiny subset "
            f"(≤{ATTENTION_PROMPT_LIMIT} prompts). Weights are observational only."
        )
    else:
        lines.append(
            "**Deferred:** true attention logging was not relied upon for this report. "
            "No attention weights were fabricated."
        )

    lines.extend([
        "",
        "## 16. Boundary4 vs k8_v4_sim autopsy",
        "",
        f"| Prompt×draft_len pairs compared | {len(agg.get('boundary4_vs_k8_v4_sim', []))} |",
        "",
        "## 17. Int8 autopsy",
        "",
    ])
    int8 = agg.get("by_compressor", {}).get("int8", {})
    if int8:
        lines.append(
            f"`int8`: {int8.get('lossy_div', 0)} lossy-divergence cells, "
            f"{int8.get('rejections', 0)} rejection events, "
            f"mean accept {_fmt_rate(int8.get('mean_acceptance_rate'))}."
        )
    else:
        lines.append("_int8 not in panel._")

    lines.extend([
        "",
        "## 18. KVQuant autopsy",
        "",
    ])
    if manifest["kvquant_included"]:
        kvq = agg.get("by_compressor", {}).get(KVQUANT_NAME, {})
        lines.append(
            f"`{KVQUANT_NAME}`: {kvq.get('lossy_div', 0)} lossy-divergence cells, "
            f"mean accept {_fmt_rate(kvq.get('mean_acceptance_rate'))}."
        )
    else:
        lines.append("KVQuant simquant **not included** (environment unavailable).")

    lines.extend([
        "",
        "## 19. Repair hypotheses",
        "",
        "| Policy | Rationale | Status |",
        "|---|---|---|",
    ])
    for h in hyps:
        lines.append(f"| {h['policy']} | {h['rationale']} | {h['status']} |")

    lines.extend([
        "",
        "## 20. What this proves",
        "",
        "- Divergence can be localized to token positions, categories, and token types.",
        "- Logit margins and top-k overlap quantify how close drafter proposals are to "
        "verifier consensus.",
        "- Layer-wise KV error summaries separate K vs V perturbation after prefill.",
        "",
        "## 21. What this does not prove",
        "",
        "- No claim that any repair policy improves speed, memory, or final accuracy.",
        "- No universal benchmark coverage; 25-prompt stratified subset only.",
        "- Attention maps are not used for acceptance decisions.",
        "",
        "## 22. Limitations",
        "",
        "- Auxiliary forward passes approximate drafter logits; sim compressors copy "
        "prefill next-token from full path.",
        "- Structured-output heuristics are not a JSON parser.",
        "- KV error metrics are post-prefill snapshots, not per-token drift.",
        "",
        "## 23. Relation to future repair-policy experiments",
        "",
        "Hypotheses in §19 should be validated in a separate experiment that "
        "**implements** a policy and measures acceptance/exactness — not in this "
        "forensics-only phase.",
        "",
        "## 24. Relation to v1.0.0 readiness",
        "",
        "Experiment 019 closes the V11 optional divergence deep-dive (D7/D8 partial). "
        "Launch package readiness remains Phase 6.",
        "",
        "## 25. VeriCache attribution",
        "",
        "ExactKV draft-verify-commit loop inspired by "
        "[VeriCache](https://arxiv.org/abs/2605.17613); this autopsy is ExactKV-specific "
        "forensics on Qwen2.5 built-in compressors.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 019 divergence autopsy")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--include-15b", action="store_true")
    parser.add_argument("--skip-15b", action="store_true")
    parser.add_argument("--include-kvquant", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=_ROOT / "reports")
    parser.add_argument("--docs-dir", type=Path, default=_ROOT / "docs")
    args = parser.parse_args()

    prompts = load_autopsy_prompt_subset(per_suite=5)
    compressors = list(REQUIRED_COMPRESSORS)
    kvquant_included = False
    if args.include_kvquant and kvquant_available():
        compressors.append(KVQUANT_NAME)
        kvquant_included = True

    attention_ids = {p["prompt_id"] for p in prompts[:ATTENTION_PROMPT_LIMIT]}

    try:
        runpod_meta = collect_runpod_meta()
    except Exception:
        runpod_meta = {}

    model_runs = [
        run_autopsy_for_model(
            MODEL_PRIMARY,
            prompts,
            compressors,
            device=args.device,
            dtype=args.dtype,
            attention_prompt_ids=attention_ids,
        )
    ]

    if args.include_15b and not args.skip_15b:
        if model_runs[0]["aggregate"]["exactkv_failures"] == 0:
            try:
                model_runs.append(
                    run_autopsy_for_model(
                        MODEL_OPTIONAL,
                        prompts,
                        [c for c in compressors if c != KVQUANT_NAME],
                        device=args.device,
                        dtype=args.dtype,
                        attention_prompt_ids=set(),
                    )
                )
            except Exception as exc:
                print(f"Optional 1.5B run skipped: {exc}", flush=True)

    artifact = build_artifact(
        model_runs,
        runpod_meta=runpod_meta,
        compressors=compressors,
        prompt_count=len(prompts),
        kvquant_included=kvquant_included,
    )
    assert_autopsy_artifact_safe(artifact)
    _assert_no_forbidden(artifact)

    json_path = args.reports_dir / "experiment_019_divergence_autopsy.json"
    csv_path = args.reports_dir / "experiment_019_divergence_autopsy.csv"
    md_path = args.docs_dir / "EXPERIMENT_019_DIVERGENCE_AUTOPSY.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    write_autopsy_csv(artifact, csv_path)
    md_path.write_text(generate_markdown_report(artifact), encoding="utf-8")

    agg = artifact["aggregate"]
    print(f"\nExperiment 019 complete: {agg['total_cells']} cells", flush=True)
    print(f"  exactkv_failures: {agg['exactkv_failures']}", flush=True)
    print(f"  lossy_divergence_cells: {agg['lossy_divergence_cells']}", flush=True)
    print(f"  rejection_events: {agg['total_rejection_events']}", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
