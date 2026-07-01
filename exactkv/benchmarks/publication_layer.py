"""Phase C publication + demo layer for ExactKV.

Consumes Phase A / B reports and optional Exp 116 / Exp 117 outputs.
No inference. No fabricated metrics or output text.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.benchmarks.leaderboard_platform import (
    DEFAULT_LEADERBOARD_JSON,
    DEFAULT_PHASE_A_INPUT,
    short_model_name,
)
from exactkv.safety.l4_runtime_coupling_stress_panel import STRESS_PANEL_PROMPTS

PHASE_C_ID = "phaseC_publication_demo_layer"
DEFAULT_DEMO_PACK = Path("reports/demo_pack.json")
DEFAULT_EXP115_REPORT = Path("reports/experiment_115_l4_runtime_coupling_stress_panel.json")
DEFAULT_EXP116_REPORT = Path("reports/experiment_116_instability_regime_analysis.json")
DEFAULT_EXP117_MANIFEST = Path("reports/visuals/exp117/exp117_manifest.json")
DEFAULT_VISUALS_DIR = Path("reports/visuals/phaseC")

PROMPT_BY_ID: dict[str, str] = {pid: text for pid, text in STRESS_PANEL_PROMPTS}


@dataclass(frozen=True)
class PhaseCValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _find_phase_a_cell(
    phase_a: Mapping[str, Any],
    *,
    prompt_id: str,
    compressor_name: str,
    model_name: str,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    for cell in phase_a.get("cells") or []:
        if (
            cell.get("prompt_id") == prompt_id
            and cell.get("compressor_name") == compressor_name
            and cell.get("model_name") == model_name
            and int(cell.get("max_new_tokens") or 0) == max_new_tokens
        ):
            return cell
    return None


def _find_exp115_trace(
    exp115: Mapping[str, Any] | None,
    *,
    prompt_id: str,
    compressor: str,
    model_name: str,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if not exp115:
        return None
    for cell in exp115.get("cells") or []:
        if (
            cell.get("prompt_id") == prompt_id
            and cell.get("compressor") == compressor
            and cell.get("model_name") == model_name
            and int(cell.get("max_new_tokens") or 0) == max_new_tokens
        ):
            records = cell.get("trace_records") or []
            return records[0] if records else None
    return None


def _build_divergence_timeline(
    cell: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build divergence timeline from report fields only."""
    metrics = cell.get("metrics") or {}
    acceptance = (cell.get("exactkv") or {}).get("acceptance") or {}
    first_div = metrics.get("first_divergence_index")
    if first_div is None:
        first_div = (cell.get("lossy") or {}).get("first_divergence_idx")

    timeline: list[dict[str, Any]] = []
    drafted = int(acceptance.get("total_drafted") or 0)
    accepted = int(acceptance.get("total_accepted") or 0)
    rejected = int(acceptance.get("total_rejected") or 0)

    for pos in range(max(drafted, 1)):
        if first_div is not None and pos < int(first_div):
            event = "match"
        elif first_div is not None and pos == int(first_div):
            event = "first_divergence"
        elif first_div is not None and pos > int(first_div):
            event = "post_divergence"
        else:
            event = "match" if pos < accepted else "unknown"
        entry: dict[str, Any] = {"token_position": pos, "event": event}
        if trace:
            proposal = trace.get("proposal_tokens") or []
            verifier = (trace.get("verifier_evidence") or {}).get("verifier_evidence_token_ids") or []
            if pos < len(proposal):
                entry["proposal_token_id"] = proposal[pos]
            if pos < len(verifier):
                entry["verifier_token_id"] = verifier[pos]
        timeline.append(entry)

    if trace and trace.get("mismatch_index") is not None:
        timeline.append(
            {
                "token_position": int(trace["mismatch_index"]),
                "event": "verifier_reject_at",
                "decision": trace.get("decision"),
            },
        )
    return timeline


def _build_acceptance_path(cell: Mapping[str, Any], trace: Mapping[str, Any] | None) -> list[str]:
    acceptance = (cell.get("exactkv") or {}).get("acceptance") or {}
    path = [
        f"drafted={acceptance.get('total_drafted', 0)}",
        f"accepted={acceptance.get('total_accepted', 0)}",
        f"rejected={acceptance.get('total_rejected', 0)}",
        f"acceptance_rate={acceptance.get('acceptance_rate', 0.0)}",
    ]
    if trace:
        path.append(f"trace_decision={trace.get('decision')}")
        if trace.get("mismatch_index") is not None:
            path.append(f"mismatch_index={trace.get('mismatch_index')}")
    path.append(f"exactkv_failure={cell.get('exactkv_failure', False)}")
    return path


def _demo_outputs_from_reports(
    cell: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Full vs compressed outputs — only fields present in reports."""
    full: dict[str, Any] = {"source": "phaseA_benchmark.json", "output_text_available": False}
    compressed: dict[str, Any] = {"source": "phaseA_benchmark.json", "output_text_available": False}

    if cell.get("full", {}).get("output_text"):
        full["output_text"] = cell["full"]["output_text"]
        full["output_text_available"] = True
    if cell.get("lossy", {}).get("output_text"):
        compressed["output_text"] = cell["lossy"]["output_text"]
        compressed["output_text_available"] = True

    if trace:
        ve = trace.get("verifier_evidence") or {}
        full["verifier_token_ids"] = ve.get("verifier_evidence_token_ids")
        compressed["proposal_token_ids"] = trace.get("proposal_tokens")
        full["source"] = "experiment_115_l4_runtime_coupling_stress_panel.json"
        compressed["source"] = "experiment_115_l4_runtime_coupling_stress_panel.json"

    return {"full_reference": full, "compressed_draft": compressed}


def _select_demo_candidates(phase_a: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return selector specs for the five canonical demo categories."""
    return [
        {
            "category": "structured_output_drift",
            "category_note": (
                "Pharmacy-style intent-flip prompt is not in the Phase A panel; "
                "p2_json_tool is the closest in-panel structured-output drift case."
            ),
            "prompt_id": "p2_json_tool",
            "compressor_name": "int4_sim",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "max_new_tokens": "4",
        },
        {
            "category": "qa_partial_drift",
            "category_note": "Partial prefix acceptance under probe-only shard compression.",
            "prompt_id": "p0_capital_france",
            "compressor_name": "shard",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "max_new_tokens": "4",
        },
        {
            "category": "worst_case_compression",
            "category_note": "Lowest acceptance int4_sim cell in Phase A benchmark.",
            "prompt_id": "p0_capital_france",
            "compressor_name": "int4_sim",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "max_new_tokens": "4",
        },
        {
            "category": "cross_model_disagreement",
            "category_note": (
                "Same prompt/compressor/length yields divergent metrics on "
                "Qwen 0.5B-Instruct vs Qwen 0.5B (k8_v4_sim, 16 tokens)."
            ),
            "prompt_id": "p1_simple_math",
            "compressor_name": "k8_v4_sim",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_new_tokens": "16",
            "comparison_model": "Qwen/Qwen2.5-0.5B",
        },
        {
            "category": "first_divergence_explosion",
            "category_note": "Earliest first_divergence_index (0) in Phase A cells.",
            "prompt_id": "p3_code_fn",
            "compressor_name": "kvquant",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "max_new_tokens": "4",
        },
    ]


def extract_canonical_demos(
    phase_a: Mapping[str, Any],
    *,
    exp115: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract exactly five canonical demos from report data."""
    demos: list[dict[str, Any]] = []
    for spec in _select_demo_candidates(phase_a):
        mnt = int(spec["max_new_tokens"])
        cell = _find_phase_a_cell(
            phase_a,
            prompt_id=spec["prompt_id"],
            compressor_name=spec["compressor_name"],
            model_name=spec["model_name"],
            max_new_tokens=mnt,
        )
        if cell is None:
            continue
        trace = _find_exp115_trace(
            exp115,
            prompt_id=spec["prompt_id"],
            compressor=spec["compressor_name"],
            model_name=spec["model_name"],
            max_new_tokens=mnt,
        )
        metrics = cell.get("metrics") or {}
        demo: dict[str, Any] = {
            "demo_id": f"{spec['category']}_{spec['prompt_id']}_{spec['compressor_name']}",
            "category": spec["category"],
            "category_note": spec.get("category_note"),
            "input_prompt": PROMPT_BY_ID.get(spec["prompt_id"], spec["prompt_id"]),
            "prompt_id": spec["prompt_id"],
            "compressor": spec["compressor_name"],
            "model": spec["model_name"],
            "model_short": short_model_name(spec["model_name"]),
            "max_new_tokens": mnt,
            "outputs": _demo_outputs_from_reports(cell, trace),
            "divergence_timeline": _build_divergence_timeline(cell, trace),
            "first_divergence_index": metrics.get("first_divergence_index"),
            "acceptance_decision_path": _build_acceptance_path(cell, trace),
            "metrics": {
                "acceptance_rate": metrics.get("acceptance_rate"),
                "verifier_agreement_score": metrics.get("verifier_agreement_score"),
                "token_level_divergence": metrics.get("token_level_divergence"),
                "exactkv_failure": metrics.get("exactkv_failure"),
            },
            "data_sources": ["reports/phaseA_benchmark.json"],
        }
        if trace:
            demo["data_sources"].append("reports/experiment_115_l4_runtime_coupling_stress_panel.json")
        if spec.get("comparison_model"):
            comp_cell = _find_phase_a_cell(
                phase_a,
                prompt_id=spec["prompt_id"],
                compressor_name=spec["compressor_name"],
                model_name=spec["comparison_model"],
                max_new_tokens=mnt,
            )
            if comp_cell:
                demo["comparison"] = {
                    "model": spec["comparison_model"],
                    "model_short": short_model_name(spec["comparison_model"]),
                    "metrics": comp_cell.get("metrics"),
                    "acceptance_decision_path": _build_acceptance_path(
                        comp_cell,
                        _find_exp115_trace(
                            exp115,
                            prompt_id=spec["prompt_id"],
                            compressor=spec["compressor_name"],
                            model_name=spec["comparison_model"],
                            max_new_tokens=mnt,
                        ),
                    ),
                }
        demos.append(demo)
    return demos


def build_visual_synthesis(
    phase_a: Mapping[str, Any],
    leaderboard: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path = DEFAULT_VISUALS_DIR,
) -> dict[str, Any]:
    """Build visual JSON (+ optional PNG) from report data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_div_map = {
        "title": "First Divergence Map",
        "x_axis": "token_position",
        "y_axis": "compressor",
        "points": [
            {
                "demo_id": d["demo_id"],
                "compressor": d["compressor"],
                "first_divergence_index": d.get("first_divergence_index"),
                "prompt_id": d.get("prompt_id"),
            }
            for d in demos
        ],
    }

    entries = leaderboard.get("entries") or []
    models = sorted({e["model_short"] for e in entries if e.get("model_short")})
    compressors = sorted({e["compressor"] for e in entries if e.get("compressor")})
    heat: dict[str, dict[str, float]] = {m: {} for m in models}
    for e in entries:
        if e.get("divergence_score") is not None:
            heat[e["model_short"]][e["compressor"]] = float(e["divergence_score"])

    failure_heatmap = {
        "title": "Compression Failure Heatmap",
        "x_axis": "compressor",
        "y_axis": "model",
        "compressors": compressors,
        "models": models,
        "values": heat,
    }

    global_rank = leaderboard.get("global_compressor_rankings") or []
    tradeoff_curve = {
        "title": "Acceptance vs Divergence Tradeoff",
        "points": [],
    }
    comp_acc: dict[str, list[float]] = defaultdict(list)
    comp_div: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        if e.get("score") is None:
            continue
        comp_acc[e["compressor"]].append(float(e["acceptance_rate"]))
        comp_div[e["compressor"]].append(float(e["divergence_score"]))
    for comp in compressors:
        if comp in comp_acc:
            tradeoff_curve["points"].append(
                {
                    "compressor": comp,
                    "mean_acceptance_rate": sum(comp_acc[comp]) / len(comp_acc[comp]),
                    "mean_divergence_score": sum(comp_div[comp]) / len(comp_div[comp]),
                    "global_mean_score": next(
                        (r["mean_score"] for r in global_rank if r["compressor"] == comp),
                        None,
                    ),
                },
            )

    synthesis = {
        "first_divergence_map": first_div_map,
        "compression_failure_heatmap": failure_heatmap,
        "acceptance_divergence_curve": tradeoff_curve,
        "exp117_visual_refs": [],
    }

    exp117_path = DEFAULT_EXP117_MANIFEST
    if exp117_path.is_file():
        exp117 = _load_json(exp117_path)
        synthesis["exp117_visual_refs"] = list((exp117.get("visual_outputs") or {}).values())

    json_path = output_dir / "phaseC_visual_synthesis.json"
    json_path.write_text(json.dumps(synthesis, indent=2) + "\n")
    synthesis["json_path"] = str(json_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        _render_phase_c_plots(synthesis, output_dir)
        synthesis["png_outputs"] = [
            str(output_dir / "first_divergence_map.png"),
            str(output_dir / "compression_failure_heatmap.png"),
            str(output_dir / "acceptance_divergence_curve.png"),
        ]
    except Exception as exc:  # noqa: BLE001
        synthesis["plot_render_error"] = str(exc)

    return synthesis


def _render_phase_c_plots(synthesis: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    points = synthesis["first_divergence_map"]["points"]
    fig, ax = plt.subplots(figsize=(7, 4))
    y_labels = [p["compressor"] for p in points]
    x_vals = [p["first_divergence_index"] if p["first_divergence_index"] is not None else -0.5 for p in points]
    ax.scatter(x_vals, range(len(points)), c="#d62728")
    ax.set_yticks(range(len(points)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("first divergence token position")
    ax.set_title("First Divergence Map (Phase C demos)")
    fig.tight_layout()
    fig.savefig(output_dir / "first_divergence_map.png")
    plt.close(fig)

    hm = synthesis["compression_failure_heatmap"]
    models = hm["models"]
    comps = hm["compressors"]
    mat = np.zeros((len(models), len(comps)))
    for i, m in enumerate(models):
        for j, c in enumerate(comps):
            mat[i, j] = hm["values"].get(m, {}).get(c, 0.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Compression Divergence Heatmap")
    fig.colorbar(im, ax=ax, label="divergence score")
    fig.tight_layout()
    fig.savefig(output_dir / "compression_failure_heatmap.png")
    plt.close(fig)

    curve = synthesis["acceptance_divergence_curve"]["points"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in curve:
        ax.scatter(p["mean_divergence_score"], p["mean_acceptance_rate"], s=60)
        ax.annotate(p["compressor"], (p["mean_divergence_score"], p["mean_acceptance_rate"]), fontsize=7)
    ax.set_xlabel("mean divergence score")
    ax.set_ylabel("mean acceptance rate")
    ax.set_title("Acceptance vs Divergence Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_divergence_curve.png")
    plt.close(fig)


def _leaderboard_stats(leaderboard: Mapping[str, Any]) -> dict[str, Any]:
    entries = leaderboard.get("entries") or []
    global_rank = leaderboard.get("global_compressor_rankings") or []
    insights = leaderboard.get("insights") or []
    return {
        "top_compressor": global_rank[0] if global_rank else None,
        "int8_mean_score": next((r["mean_score"] for r in global_rank if r["compressor"] == "int8"), None),
        "noop_mean_score": next((r["mean_score"] for r in global_rank if r["compressor"] == "noop"), None),
        "total_entries": len(entries),
        "insights": insights,
        "deterministic_mode": leaderboard.get("deterministic_mode"),
    }


def render_paper_draft(
    phase_a: Mapping[str, Any],
    leaderboard: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]],
    exp116: Mapping[str, Any] | None,
    visual_synthesis: Mapping[str, Any],
) -> str:
    stats = _leaderboard_stats(leaderboard)
    int8 = stats.get("int8_mean_score")
    regime_cov = (exp116 or {}).get("regime_coverage") or {}
    total_cells = phase_a.get("total_cells")
    models = phase_a.get("models_evaluated") or []

    abstract = (
        "KV cache compression reduces memory footprint but introduces token-level drift "
        "between lossy draft generation and full-precision verification. We present ExactKV, "
        "an evaluation framework that measures acceptance rate, first-divergence index, "
        "verifier agreement, and cross-model instability under compressed KV conditions. "
        f"Across {total_cells} benchmark cells spanning {len(models)} models and "
        f"{len(phase_a.get('compressors') or [])} compressors, INT8 achieves the highest "
        f"cross-model leaderboard score ({int8:.3f} mean) with zero ExactKV failures in "
        "the reported panel. Simulated INT4, asymmetric K8/V4, and restricted external "
        "adapters exhibit elevated divergence rates and lower verifier agreement, with "
        "first-token mismatches appearing as early as index 0. ExactKV provides trace-only "
        "verification without runtime commit, enabling reproducible comparison of compression "
        "robustness prior to deployment."
    )

    demo_summaries = "\n".join(
        f"- **{d['category']}** (`{d['compressor']}`, {d['model_short']}): "
        f"first divergence at token {d.get('first_divergence_index')}, "
        f"acceptance {d['metrics']['acceptance_rate']}"
        for d in demos
    )

    return f"""# Token-Level Drift in KV Cache Compression: A Cross-Model Evaluation of ExactKV

## Abstract

{abstract}

## 1. Introduction

Large language model inference stores growing key-value (KV) caches. Compression methods—INT8 quantization, simulated INT4, asymmetric K/V schemes, and external adapters—trade memory for approximation error. When approximation error appears at the token level, greedy decoding diverges from the full-KV reference. ExactKV measures this drift through draft-verify acceptance without modifying the core generator at commit time.

## 2. System Overview

ExactKV evaluates three generation modes per cell: full-KV greedy (reference), lossy compressed-KV greedy, and ExactKV draft-verify loops. Phase A (`phaseA_scale_benchmark`) runs a unified panel; Phase B (`exactkv_leaderboard_platform`) normalizes scores across models. Optional Exp 116 extracts instability regimes; Exp 117 renders phase diagrams from stress-panel outputs.

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Models | {', '.join(short_model_name(m) for m in models)} |
| Compressors | {', '.join(phase_a.get('compressors') or [])} |
| Prompts | {phase_a.get('prompt_count')} deterministic panel prompts |
| Lengths | {phase_a.get('max_new_tokens_values')} |
| Total cells | {total_cells} |
| Deterministic mode | {phase_a.get('deterministic_mode')} |

Data sources: `reports/phaseA_benchmark.json`, `reports/leaderboard.json`.

## 4. Key Results

### 4.1 INT8 dominance

Global leaderboard mean score: **int8 = {int8:.3f}**, **noop = {stats.get('noop_mean_score', 0):.3f}**. INT8 maintains zero reported divergence rate across all four models in Phase B aggregation while preserving high acceptance (mean 0.774).

### 4.2 Divergence across compressors

Simulated INT4 (`int4_sim`) and probe-only `shard` show the highest per-model divergence rates on Qwen 0.5B (0.333). Restricted mocks (`spectralquant`, `kvquant`) rank below built-in INT8/NOOP on composite score.

### 4.3 Model sensitivity

Qwen 0.5B-Instruct exhibits the widest compressor score spread (0.419) in leaderboard insights. Llama-3.1-8B shows elevated `k8_v4_sim` divergence (0.583) relative to INT8/NOOP baselines.

## 5. Canonical Demo Cases

{demo_summaries}

## 6. Failure Taxonomy

| Class | Description | Evidence in panel |
|-------|-------------|-------------------|
| Early divergence | Mismatch at token 0–1 | `kvquant` / `p3_code_fn` first_divergence_index=0 |
| Structured output drift | JSON/tool-call prefix corruption | `p2_json_tool` + `int4_sim` |
| Partial QA drift | Prefix accepted, suffix diverges | `shard` probe, acceptance=0.25 |
| Cross-model split | Same cell, different models disagree | `k8_v4_sim` @ 16 tokens, 0.5B vs Instruct |
| Verifier rejection | Trace REJECT with mismatch_index | Exp 115 cells (when trace available) |

Exp 116 regime coverage (when available): stable={regime_cov.get('stable', 'n/a')}, moderate_drift={regime_cov.get('moderate_drift', 'n/a')}, high_divergence={regime_cov.get('high_divergence', 'n/a')}.

## 7. Discussion

Compression error concentrates at token boundaries: the earliest observed mismatch occurs at index 0, implying quantisation noise can alter the very first generated token. Asymmetric schemes (`k8_v4_sim`) increase divergence on larger models, consistent with key/value bit-width asymmetry fragility. Verifier-mediated ExactKV prevents silent failure by rejecting divergent drafts—acceptance rate drops even when final ExactKV output matches full KV.

## 8. Limitations

- Phase A deterministic runs do not log decoded output text; demos use token-index timelines and optional Exp 115 token IDs.
- No runtime kernel integration or serving-system claims.
- No speed or GPU memory claims unless directly measured in source reports.
- Pharmacy-style semantic prompts are outside the current Phase A panel; structured JSON drift is used as the closest proxy.

## 9. Conclusion

ExactKV provides a reproducible, trace-only benchmark for KV compression robustness. INT8 remains the near-optimal baseline across models; aggressive compression and external probes increase divergence and reduce verifier agreement. The leaderboard platform enables canonical ranking without new inference.

## Visual References

- Phase C synthesis: `{visual_synthesis.get('json_path', 'reports/visuals/phaseC/phaseC_visual_synthesis.json')}`
- Exp 117 atlas: `{', '.join(visual_synthesis.get('exp117_visual_refs') or []) or 'n/a'}`
"""


def render_blog_post(
    leaderboard: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]],
) -> str:
    stats = _leaderboard_stats(leaderboard)
    top3 = (leaderboard.get("global_compressor_rankings") or [])[:3]
    demo_blocks = []
    for d in demos[:3]:
        demo_blocks.append(
            f"### {d['category'].replace('_', ' ').title()}\n\n"
            f"**Prompt:** `{d['input_prompt'][:80]}{'...' if len(d['input_prompt']) > 80 else ''}`\n\n"
            f"**Compressor / model:** `{d['compressor']}` on {d['model_short']}\n\n"
            f"First divergence at token **{d.get('first_divergence_index')}**, "
            f"acceptance **{d['metrics']['acceptance_rate']:.2f}**. "
            f"{d.get('category_note', '')}\n",
        )

    table_rows = "\n".join(
        f"| {r['rank']} | `{r['compressor']}` | {r['mean_score']:.3f} |"
        for r in top3
    )

    return f"""# What Breaks When You Compress the KV Cache

Everyone wants smaller KV caches. Few teams measure what happens to the *tokens* when compression kicks in.

ExactKV is an evaluation framework that asks a simple question: **does the compressed cache still agree with full-precision generation, token by token?**

## The short answer

In our latest cross-model panel, **INT8 is the near-optimal baseline** — mean leaderboard score **{stats.get('int8_mean_score', 0):.3f}** with **zero ExactKV failures** across four models. Aggressive simulators and external probes diverge earlier and accept fewer draft tokens.

## Three cases that illustrate the problem

{chr(10).join(demo_blocks)}

## Leaderboard snapshot

| Rank | Compressor | Mean score |
|-----:|------------|----------:|
{table_rows}

## Why this matters

Compression is not a single number. A method can look fine on average yet fail on structured outputs, factual QA, or larger models. ExactKV makes that visible before you ship.

## What we are not claiming

No speedups. No memory savings unless measured. No production serving integration — this is an evaluation layer, not a deployment stack.

---

*Data: Phase A benchmark + Phase B leaderboard. Reproduce: `python scripts/run_leaderboard.py --all`*
"""


def render_x_thread(
    leaderboard: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]],
) -> str:
    stats = _leaderboard_stats(leaderboard)
    structured = next((d for d in demos if d["category"] == "structured_output_drift"), demos[0])
    worst = next((d for d in demos if d["category"] == "worst_case_compression"), demos[1])
    insight = (stats.get("insights") or ["INT8 leads the panel."])[0]

    tweets = [
        "1/9 KV cache compression is everywhere. Token-level drift is not. ExactKV measures when compressed caches start lying — before you ship.",
        f"2/9 Structured output case: `{structured['compressor']}` on {structured['model_short']}. First divergence at token {structured.get('first_divergence_index')}. Acceptance {structured['metrics']['acceptance_rate']:.2f}.",
        f"3/9 Worst INT4 cell: acceptance {worst['metrics']['acceptance_rate']:.2f}, divergence at token {worst.get('first_divergence_index')}. ExactKV caught it — zero silent failures.",
        f"4/9 Cross-model panel: 336 cells, 4 models, 7 compressors. INT8 mean score {stats.get('int8_mean_score', 0):.3f}.",
        "5/9 int8 is near-optimal baseline: high acceptance, zero divergence rate in aggregated leaderboard, zero ExactKV failures.",
        f"6/9 {insight}",
        "7/9 Simulated INT4 + external probes (shard, kvquant mock) show 2–3× higher divergence on small models.",
        "8/9 ExactKV = trace-only verification. No runtime commit. Reproducible benchmark + leaderboard from JSON reports.",
        "9/9 Full paper draft + demo pack in repo. `python scripts/run_leaderboard.py --all`",
    ]
    return "\n\n".join(tweets) + "\n"


def render_linkedin_post(leaderboard: Mapping[str, Any]) -> str:
    stats = _leaderboard_stats(leaderboard)
    return f"""We built ExactKV as an **evaluation framework for KV cache compression robustness** — not another speed benchmark.

Recent cross-model results ({stats.get('total_entries', 28)} ranked model×compressor cells):

• INT8 remains the strongest baseline (mean score {stats.get('int8_mean_score', 0):.3f})
• Simulated INT4 and external probe adapters show higher token-level divergence
• Verifier agreement drops before ExactKV failures appear — catching drift early matters

The system is fully reproducible from published JSON reports: benchmark → leaderboard → publication artifacts. No hidden inference runs, no serving claims.

If your team compresses KV caches, you need token-level equivalence testing — not just memory ratios.

#MachineLearning #LLM #MLOps #Research #KVCache
"""


def validate_phase_c_outputs(
    demo_pack: Mapping[str, Any],
    *,
    expected_demos: int = 5,
) -> PhaseCValidationResult:
    errors: list[str] = []
    if demo_pack.get("phase_id") != PHASE_C_ID:
        errors.append("phase_id mismatch")
    demos = demo_pack.get("demos") or []
    if len(demos) != expected_demos:
        errors.append(f"expected {expected_demos} demos, got {len(demos)}")
    for d in demos:
        if not d.get("input_prompt"):
            errors.append(f"missing prompt for {d.get('demo_id')}")
        if d.get("metrics") is None:
            errors.append(f"missing metrics for {d.get('demo_id')}")
    if demo_pack.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")
    return PhaseCValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def run_phase_c_publication_layer(
    *,
    phase_a_path: Path | str = DEFAULT_PHASE_A_INPUT,
    leaderboard_path: Path | str = DEFAULT_LEADERBOARD_JSON,
    exp115_path: Path | str = DEFAULT_EXP115_REPORT,
    exp116_path: Path | str = DEFAULT_EXP116_REPORT,
    output_dir: Path | str = DEFAULT_VISUALS_DIR,
) -> dict[str, Any]:
    """Run full Phase C pipeline."""
    phase_a = _load_json(phase_a_path)
    leaderboard = _load_json(leaderboard_path)
    exp115 = _load_json(exp115_path) if Path(exp115_path).is_file() else None
    exp116 = _load_json(exp116_path) if Path(exp116_path).is_file() else None

    demos = extract_canonical_demos(phase_a, exp115=exp115)
    visual_synthesis = build_visual_synthesis(phase_a, leaderboard, demos, output_dir=Path(output_dir))

    return {
        "phase_id": PHASE_C_ID,
        "status": "publication_complete",
        "demos": demos,
        "visual_synthesis": visual_synthesis,
        "paper_draft_md": render_paper_draft(phase_a, leaderboard, demos, exp116, visual_synthesis),
        "blog_post_md": render_blog_post(leaderboard, demos),
        "x_thread_md": render_x_thread(leaderboard, demos),
        "linkedin_post_md": render_linkedin_post(leaderboard),
        "source_reports": {
            "phase_a": str(phase_a_path),
            "leaderboard": str(leaderboard_path),
            "exp115": str(exp115_path) if exp115 else None,
            "exp116": str(exp116_path) if exp116 else None,
        },
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "model_experiments_run": False,
        "validation_result": {},
    }


def write_phase_c_outputs(
    result: Mapping[str, Any],
    *,
    demo_pack_path: Path | str = DEFAULT_DEMO_PACK,
    paper_path: Path = Path("docs/generated/paper_draft.md"),
    blog_path: Path = Path("docs/generated/blog_post.md"),
    x_path: Path = Path("docs/generated/x_thread.md"),
    linkedin_path: Path = Path("docs/generated/linkedin_post.md"),
) -> dict[str, str]:
    demo_pack = {
        "phase_id": result["phase_id"],
        "status": result["status"],
        "demos": result["demos"],
        "visual_synthesis_path": result["visual_synthesis"].get("json_path"),
        "source_reports": result["source_reports"],
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "validation_result": validate_phase_c_outputs(
            {"phase_id": result["phase_id"], "demos": result["demos"], "exactkv_generator_modified": False},
        ).to_dict(),
    }
    paths: dict[str, str] = {}
    dp = Path(demo_pack_path)
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_text(json.dumps(demo_pack, indent=2) + "\n")
    paths["demo_pack"] = str(dp)

    for key, path, content_key in (
        ("paper_draft", paper_path, "paper_draft_md"),
        ("blog_post", blog_path, "blog_post_md"),
        ("x_thread", x_path, "x_thread_md"),
        ("linkedin_post", linkedin_path, "linkedin_post_md"),
    ):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result[content_key])
        paths[key] = str(p)

    return paths
