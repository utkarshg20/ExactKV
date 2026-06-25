"""ExactKV canonical leaderboard platform (Phase B).

Consumes Phase A benchmark JSON and optional Exp 116 instability analysis.
Pure aggregation — no inference, no ExactKVGenerator changes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LEADERBOARD_ID = "exactkv_leaderboard_platform"
DEFAULT_PHASE_A_INPUT = Path("reports/phaseA_benchmark.json")
DEFAULT_EXP116_INPUT = Path("reports/experiment_116_instability_regime_analysis.json")
DEFAULT_LEADERBOARD_JSON = Path("reports/leaderboard.json")
DEFAULT_LEADERBOARD_MD = Path("reports/leaderboard.md")

SCORE_WEIGHTS: dict[str, float] = {
    "acceptance_rate": 0.35,
    "verifier_agreement": 0.25,
    "first_divergence_normalized": 0.20,
    "exactkv_success": 0.10,
    "stability_score": 0.10,
}

CANONICAL_COMPRESSORS: tuple[str, ...] = (
    "noop",
    "int8",
    "int4_sim",
    "k8_v4_sim",
    "spectralquant",
    "kvquant",
    "shard",
)

CANONICAL_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

# Estimated compression ratio when Phase A memory fields absent.
_ESTIMATED_COMPRESSION_RATIO: dict[str, float] = {
    "noop": 1.0,
    "int8": 0.25,
    "int4_sim": 0.125,
    "k8_v4_sim": 0.1875,
    "spectralquant": 0.15,
    "kvquant": 0.125,
    "shard": 0.20,
}


@dataclass(frozen=True)
class LeaderboardValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def short_model_name(model: str) -> str:
    if "Instruct" in model and "0.5B" in model:
        return "Qwen 0.5B-Instruct"
    if "0.5B" in model:
        return "Qwen 0.5B"
    if "Llama" in model:
        return "Llama-3.1-8B"
    if "Mistral" in model:
        return "Mistral-7B"
    return model.split("/")[-1]


def load_phase_a_report(path: Path | str = DEFAULT_PHASE_A_INPUT) -> dict[str, Any]:
    report_path = Path(path)
    data = json.loads(report_path.read_text())
    if data.get("phase_id") != "phaseA_scale_benchmark":
        msg = f"expected phase_id phaseA_scale_benchmark, got {data.get('phase_id')}"
        raise ValueError(msg)
    return data


def load_exp116_instability_map(
    path: Path | str = DEFAULT_EXP116_INPUT,
) -> dict[tuple[str, str], float]:
    """Mean instability score per (model_name, compressor) from Exp 116."""
    report_path = Path(path)
    if not report_path.is_file():
        return {}
    data = json.loads(report_path.read_text())
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for d in data.get("cell_descriptors") or []:
        key = (str(d.get("model_name") or ""), str(d.get("compressor") or ""))
        buckets[key].append(float(d.get("instability_score") or 0.0))
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_first_divergence(
    mean_first_divergence_index: float | None,
    *,
    max_new_tokens: int = 16,
) -> float:
    """Later first divergence → higher score; no divergence → 1.0."""
    if mean_first_divergence_index is None:
        return 1.0
    denom = max(max_new_tokens, 1)
    return _clamp01(float(mean_first_divergence_index) / denom)


def compute_leaderboard_score(
    *,
    acceptance_rate: float,
    verifier_agreement: float,
    first_divergence_normalized: float,
    exactkv_failure_rate: float,
    stability_score: float,
) -> float:
    components = {
        "acceptance_rate": _clamp01(acceptance_rate),
        "verifier_agreement": _clamp01(verifier_agreement),
        "first_divergence_normalized": _clamp01(first_divergence_normalized),
        "exactkv_success": _clamp01(1.0 - exactkv_failure_rate),
        "stability_score": _clamp01(stability_score),
    }
    return sum(SCORE_WEIGHTS[k] * components[k] for k in SCORE_WEIGHTS)


def _resolve_compression_ratio(
    stats: Mapping[str, Any],
    compressor: str,
) -> float:
    ratio = stats.get("mean_compression_ratio")
    if ratio is not None:
        return float(ratio)
    return _ESTIMATED_COMPRESSION_RATIO.get(compressor, 0.5)


def _resolve_stability_score(
    model: str,
    compressor: str,
    stats: Mapping[str, Any],
    instability_map: Mapping[tuple[str, str], float],
    phase_a_instability: Mapping[str, float],
) -> float:
    if (model, compressor) in instability_map:
        return _clamp01(1.0 - instability_map[(model, compressor)])
    if compressor in phase_a_instability:
        return _clamp01(1.0 - phase_a_instability[compressor])
    return _clamp01(float(stats.get("divergence_stability_score") or 0.0))


def normalize_model_compressor_metrics(
    phase_a: Mapping[str, Any],
    *,
    exp116_path: Path | str = DEFAULT_EXP116_INPUT,
    models: Sequence[str] | None = None,
    compressors: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize metrics for each (model, compressor) pair from Phase A."""
    model_list = list(models or phase_a.get("models_evaluated") or CANONICAL_MODELS)
    compressor_list = list(compressors or phase_a.get("compressors") or CANONICAL_COMPRESSORS)
    per_model = phase_a.get("per_model_tables") or {}
    max_mnt = max(phase_a.get("max_new_tokens_values") or [16])
    resolutions = phase_a.get("compressor_resolutions") or {}
    phase_a_instability = phase_a.get("instability_scores_exp116") or {}
    instability_map = load_exp116_instability_map(exp116_path)

    rows: list[dict[str, Any]] = []
    for model in model_list:
        model_table = per_model.get(model) or {}
        comp_summary = model_table.get("compressor_summary") or {}
        for compressor in compressor_list:
            stats = comp_summary.get(compressor)
            resolution = resolutions.get(compressor) or {}
            if stats is None:
                rows.append(
                    {
                        "compressor": compressor,
                        "model": model,
                        "model_short": short_model_name(model),
                        "availability": "unavailable",
                        "score": None,
                        "acceptance_rate": None,
                        "divergence_score": None,
                        "verifier_agreement": None,
                        "failure_rate": None,
                        "stability_score": None,
                        "compression_ratio": None,
                        "mean_first_divergence_index": None,
                        "backend_tier": resolution.get("backend_tier"),
                        "probe_only": resolution.get("probe_only", False),
                    },
                )
                continue

            acceptance = float(stats.get("mean_acceptance_rate") or 0.0)
            verifier = float(stats.get("mean_verifier_agreement_score") or 0.0)
            failure = float(stats.get("exactkv_failure_rate") or 0.0)
            divergence_rate = float(stats.get("divergence_rate") or 0.0)
            first_div_norm = normalize_first_divergence(
                stats.get("mean_first_divergence_index"),
                max_new_tokens=max_mnt,
            )
            stability = _resolve_stability_score(
                model,
                compressor,
                stats,
                instability_map,
                phase_a_instability,
            )
            score = compute_leaderboard_score(
                acceptance_rate=acceptance,
                verifier_agreement=verifier,
                first_divergence_normalized=first_div_norm,
                exactkv_failure_rate=failure,
                stability_score=stability,
            )
            availability = "available"
            if resolution.get("probe_only"):
                availability = "probe_only"
            elif resolution.get("backend_tier") == "MOCK":
                availability = "mock_fallback"
            elif not resolution.get("adapter_available", True) and compressor not in (
                "noop",
                "int8",
                "int4_sim",
                "k8_v4_sim",
            ):
                availability = "restricted"

            rows.append(
                {
                    "compressor": compressor,
                    "model": model,
                    "model_short": short_model_name(model),
                    "availability": availability,
                    "score": round(score, 4),
                    "acceptance_rate": round(acceptance, 4),
                    "divergence_score": round(divergence_rate, 4),
                    "verifier_agreement": round(verifier, 4),
                    "failure_rate": round(failure, 4),
                    "stability_score": round(stability, 4),
                    "compression_ratio": round(_resolve_compression_ratio(stats, compressor), 4),
                    "mean_first_divergence_index": stats.get("mean_first_divergence_index"),
                    "first_divergence_normalized": round(first_div_norm, 4),
                    "backend_tier": resolution.get("backend_tier"),
                    "probe_only": resolution.get("probe_only", False),
                    "num_cells": int(stats.get("num_cells") or 0),
                },
            )
    return rows


def rank_leaderboard_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort available rows by score descending; unavailable rows last."""
    available = [r for r in rows if r.get("availability") != "unavailable" and r.get("score") is not None]
    unavailable = [r for r in rows if r.get("availability") == "unavailable" or r.get("score") is None]
    ranked = sorted(available, key=lambda r: float(r["score"]), reverse=True)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ranked, start=1):
        entry = dict(row)
        entry["rank"] = i
        out.append(entry)
    for row in unavailable:
        entry = dict(row)
        entry["rank"] = None
        out.append(entry)
    return out


def aggregate_global_compressor_rankings(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mean score per compressor across models."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("score") is None:
            continue
        buckets[str(row["compressor"])].append(float(row["score"]))
    summary = [
        {
            "compressor": comp,
            "mean_score": round(sum(scores) / len(scores), 4),
            "models_count": len(scores),
        }
        for comp, scores in sorted(buckets.items())
    ]
    summary.sort(key=lambda x: x["mean_score"], reverse=True)
    for i, row in enumerate(summary, start=1):
        row["rank"] = i
    return summary


def generate_insights(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_rankings: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Extract five insight bullets from computed leaderboard values only."""
    available = [r for r in rows if r.get("score") is not None]
    if not available:
        return ["No available leaderboard rows — Phase A benchmark may be empty."]

    insights: list[str] = []

    # 1. Top global compressor
    if global_rankings:
        top = global_rankings[0]
        insights.append(
            f"`{top['compressor']}` leads the cross-model mean score "
            f"({top['mean_score']:.3f} across {top['models_count']} models).",
        )

    # 2. int8 Pareto check vs noop on acceptance + failure
    int8_rows = [r for r in available if r["compressor"] == "int8"]
    noop_rows = [r for r in available if r["compressor"] == "noop"]
    if int8_rows and noop_rows:
        int8_mean_acc = sum(r["acceptance_rate"] for r in int8_rows) / len(int8_rows)
        noop_mean_acc = sum(r["acceptance_rate"] for r in noop_rows) / len(noop_rows)
        int8_fail = sum(r["failure_rate"] for r in int8_rows)
        if int8_fail == 0 and int8_mean_acc >= noop_mean_acc * 0.95:
            insights.append(
                f"`int8` is near Pareto-optimal: mean acceptance {int8_mean_acc:.3f} "
                f"vs noop {noop_mean_acc:.3f} with zero ExactKV failures across all models.",
            )
        else:
            insights.append(
                f"`int8` mean acceptance {int8_mean_acc:.3f} vs noop {noop_mean_acc:.3f}; "
                f"ExactKV failure sum {int8_fail:.0f}.",
            )

    # 3. spectralquant vs int4_sim
    sq = [r for r in available if r["compressor"] == "spectralquant"]
    i4 = [r for r in available if r["compressor"] == "int4_sim"]
    if sq and i4:
        sq_acc = sum(r["acceptance_rate"] for r in sq) / len(sq)
        i4_acc = sum(r["acceptance_rate"] for r in i4) / len(i4)
        sq_div = sum(r["divergence_score"] for r in sq) / len(sq)
        i4_div = sum(r["divergence_score"] for r in i4) / len(i4)
        if sq_div < i4_div:
            insights.append(
                f"`spectralquant` shows lower mean divergence ({sq_div:.3f}) than "
                f"`int4_sim` ({i4_div:.3f}) but acceptance "
                f"{'higher' if sq_acc > i4_acc else 'lower'} "
                f"({sq_acc:.3f} vs {i4_acc:.3f}).",
            )
        else:
            insights.append(
                f"`spectralquant` mean acceptance {sq_acc:.3f} vs `int4_sim` {i4_acc:.3f}; "
                f"divergence rates {sq_div:.3f} vs {i4_div:.3f}.",
            )

    # 4. shard probe instability
    shard = [r for r in available if r["compressor"] == "shard"]
    if shard:
        shard_div = sum(r["divergence_score"] for r in shard) / len(shard)
        shard_stab = sum(r["stability_score"] for r in shard) / len(shard)
        worst = min(shard, key=lambda r: r["score"])
        insights.append(
            f"`shard` probe-only rows show mean divergence {shard_div:.3f} and stability "
            f"{shard_stab:.3f}; weakest score on {worst['model_short']} "
            f"({worst['score']:.3f}).",
        )

    # 5. Model robustness spread
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in available:
        by_model[str(r["model_short"])].append(float(r["score"]))
    spreads = {
        m: (max(s) - min(s), sum(s) / len(s))
        for m, s in by_model.items()
        if s
    }
    if spreads:
        most_fragile = max(spreads.items(), key=lambda x: x[1][0])
        most_stable = max(spreads.items(), key=lambda x: x[1][1])
        insights.append(
            f"{most_stable[0]} has the highest mean score ({most_stable[1][1]:.3f}); "
            f"{most_fragile[0]} shows the widest compressor spread "
            f"({most_fragile[1][0]:.3f}).",
        )

    return insights[:5]


def _ascii_bar(value: float, width: int = 20) -> str:
    filled = int(round(_clamp01(value) * width))
    return "█" * filled + "░" * (width - filled)


def render_leaderboard_markdown(report: Mapping[str, Any]) -> str:
    """Render canonical leaderboard markdown."""
    entries = report.get("entries") or []
    global_rank = report.get("global_compressor_rankings") or []
    per_model = report.get("per_model_breakdown") or {}
    insights = report.get("insights") or []

    lines = [
        "# ExactKV Canonical Leaderboard",
        "",
        "KV compression ranking by token-level acceptance, divergence, verifier agreement, "
        "and stability. Derived from Phase A benchmark outputs only.",
        "",
        f"**Source:** `{report.get('source_phase_a', DEFAULT_PHASE_A_INPUT)}`",
        f"**Deterministic mode:** {report.get('deterministic_mode', 'unknown')}",
        f"**Generated:** {report.get('generated_at', 'n/a')}",
        "",
        "> No speedup, latency, throughput, or memory savings claims unless directly measured.",
        "",
        "## Global Ranked Table",
        "",
        "| Rank | Compressor | Model | Score | Acceptance | Divergence | Verifier | Failure | Stability | Availability |",
        "|-----:|------------|-------|------:|-----------:|-----------:|---------:|--------:|----------:|--------------|",
    ]

    for row in entries:
        if row.get("rank") is None:
            continue
        lines.append(
            "| {rank} | `{comp}` | {model} | {score:.3f} | {acc:.3f} | {div:.3f} | "
            "{ver:.3f} | {fail:.3f} | {stab:.3f} | {avail} |".format(
                rank=row["rank"],
                comp=row["compressor"],
                model=row["model_short"],
                score=row["score"],
                acc=row["acceptance_rate"],
                div=row["divergence_score"],
                ver=row["verifier_agreement"],
                fail=row["failure_rate"],
                stab=row["stability_score"],
                avail=row.get("availability", ""),
            ),
        )

    lines.extend(["", "## Global Compressor Mean Score", ""])
    for row in global_rank:
        bar = _ascii_bar(float(row["mean_score"]))
        lines.append(
            f"{row['rank']}. `{row['compressor']}` — {row['mean_score']:.3f}  `{bar}`",
        )

    lines.extend(["", "## Per-Model Breakdown", ""])
    for model_short, model_rows in sorted(per_model.items()):
        lines.extend([f"### {model_short}", ""])
        lines.append("| Rank | Compressor | Score | Acceptance | Divergence |")
        lines.append("|-----:|------------|------:|-----------:|-----------:|")
        sorted_rows = sorted(
            [r for r in model_rows if r.get("score") is not None],
            key=lambda r: float(r["score"]),
            reverse=True,
        )
        for i, row in enumerate(sorted_rows, start=1):
            lines.append(
                f"| {i} | `{row['compressor']}` | {row['score']:.3f} | "
                f"{row['acceptance_rate']:.3f} | {row['divergence_score']:.3f} |",
            )
        lines.append("")

    lines.extend(["## Insights", ""])
    for bullet in insights:
        lines.append(f"- {bullet}")
    lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            "```bash",
            report.get("reproducible_cli_command", "python scripts/run_leaderboard.py --all"),
            "```",
            "",
        ],
    )
    return "\n".join(lines)


def validate_leaderboard_report(report: Mapping[str, Any]) -> LeaderboardValidationResult:
    errors: list[str] = []
    for key in ("leaderboard_id", "entries", "global_compressor_rankings", "insights"):
        if key not in report:
            errors.append(f"missing key: {key}")
    if report.get("leaderboard_id") != LEADERBOARD_ID:
        errors.append("leaderboard_id mismatch")
    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")
    entries = report.get("entries") or []
    ranked = [e for e in entries if e.get("rank") is not None]
    if not ranked:
        errors.append("no ranked entries")
    if len(report.get("insights") or []) < 1:
        errors.append("insights required")
    return LeaderboardValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def run_leaderboard_platform(
    *,
    phase_a_path: Path | str = DEFAULT_PHASE_A_INPUT,
    exp116_path: Path | str = DEFAULT_EXP116_INPUT,
    filter_model: str | None = None,
    filter_compressor: str | None = None,
    deterministic_mode: bool | None = None,
) -> dict[str, Any]:
    """Build canonical leaderboard from Phase A (+ optional Exp 116)."""
    phase_a = load_phase_a_report(phase_a_path)
    if deterministic_mode is None:
        deterministic_mode = bool(phase_a.get("deterministic_mode"))

    models = list(phase_a.get("models_evaluated") or CANONICAL_MODELS)
    compressors = list(phase_a.get("compressors") or CANONICAL_COMPRESSORS)
    if filter_model:
        models = [m for m in models if filter_model in m or short_model_name(m) == filter_model]
    if filter_compressor:
        compressors = [c for c in compressors if c == filter_compressor]

    rows = normalize_model_compressor_metrics(
        phase_a,
        exp116_path=exp116_path,
        models=models,
        compressors=compressors,
    )
    ranked = rank_leaderboard_rows(rows)
    global_rankings = aggregate_global_compressor_rankings(ranked)

    per_model_breakdown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        per_model_breakdown[str(row["model_short"])].append(dict(row))

    insights = generate_insights(ranked, global_rankings=global_rankings)

    report = {
        "leaderboard_id": LEADERBOARD_ID,
        "status": "leaderboard_complete",
        "source_phase_a": str(phase_a_path),
        "source_exp116": str(exp116_path) if Path(exp116_path).is_file() else None,
        "deterministic_mode": deterministic_mode,
        "score_weights": dict(SCORE_WEIGHTS),
        "entries": ranked,
        "global_compressor_rankings": global_rankings,
        "per_model_breakdown": dict(per_model_breakdown),
        "insights": insights,
        "models_evaluated": models,
        "compressors_evaluated": compressors,
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "trace_only": True,
        "reproducible_cli_command": "python scripts/run_leaderboard.py --all",
        "validation_result": {},
    }
    report["validation_result"] = validate_leaderboard_report(report).to_dict()
    return report


def write_leaderboard_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path | str = DEFAULT_LEADERBOARD_JSON,
    markdown_path: Path | str = DEFAULT_LEADERBOARD_MD,
    write_json: bool = True,
    write_markdown: bool = True,
) -> tuple[Path | None, Path | None]:
    json_out: Path | None = None
    md_out: Path | None = None
    if write_json:
        json_out = Path(json_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n")
    if write_markdown:
        md_out = Path(markdown_path)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_leaderboard_markdown(report))
    return json_out, md_out
