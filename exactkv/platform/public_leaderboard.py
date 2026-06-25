"""Public leaderboard engine (Phase H).

Final standardized leaderboard over unified benchmark output. Uses the same
scoring function as Phase B (locked). No metric changes.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.benchmarks.leaderboard_platform import (
    DEFAULT_EXP116_INPUT,
    LEADERBOARD_ID,
    SCORE_WEIGHTS,
    aggregate_global_compressor_rankings,
    compute_leaderboard_score,
    generate_insights,
    load_phase_a_report,
    normalize_model_compressor_metrics,
    rank_leaderboard_rows,
    render_leaderboard_markdown,
    validate_leaderboard_report,
)

from exactkv.platform.leaderboard_aggregates import repair_phase_a_report_aggregates

PHASE_H_LEADERBOARD_ID = "phaseH_public_leaderboard"
DEFAULT_PUBLIC_LEADERBOARD_JSON = Path("reports/leaderboard.json")
DEFAULT_PUBLIC_LEADERBOARD_MD = Path("reports/leaderboard.md")
DEFAULT_PUBLIC_LEADERBOARD_CSV = Path("reports/leaderboard.csv")


@dataclass(frozen=True)
class PublicLeaderboardValidation:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_public_leaderboard(
    *,
    benchmark_path: Path | str | None = None,
    phase_a_path: Path | str | None = None,
    exp116_path: Path | str = DEFAULT_EXP116_INPUT,
    filter_model: str | None = None,
    filter_compressor: str | None = None,
) -> dict[str, Any]:
    """Build deterministic public leaderboard from benchmark / Phase A report."""
    if benchmark_path and Path(benchmark_path).is_file():
        data = json.loads(Path(benchmark_path).read_text())
        phase_a = data.get("phase_a_report") or {}
        if not phase_a and (run := data.get("benchmark_run")):
            phase_a = _phase_a_from_benchmark_run(data)
        if not phase_a:
            phase_a_path = phase_a_path or DEFAULT_LEADERBOARD_JSON.parent / "phaseA_benchmark.json"
    else:
        phase_a_path = phase_a_path or DEFAULT_LEADERBOARD_JSON.parent / "phaseA_benchmark.json"
        phase_a = load_phase_a_report(phase_a_path)

    if not phase_a:
        phase_a = load_phase_a_report(phase_a_path or "reports/phaseA_benchmark.json")

    phase_a = repair_phase_a_report_aggregates(phase_a)

    models = list(phase_a.get("models_evaluated") or [])
    compressors = list(phase_a.get("compressors") or [])
    if filter_model:
        models = [m for m in models if filter_model.lower() in m.lower()]
    if filter_compressor:
        compressors = [c for c in compressors if filter_compressor.lower() in c.lower()]

    rows = normalize_model_compressor_metrics(
        phase_a,
        exp116_path=exp116_path,
        models=models,
        compressors=compressors,
    )
    ranked = rank_leaderboard_rows(rows)
    global_rankings = aggregate_global_compressor_rankings(ranked)
    insights = generate_insights(ranked, global_rankings=global_rankings)

    from collections import defaultdict

    per_model_breakdown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        per_model_breakdown[str(row["model_short"])].append(dict(row))

    report = {
        "leaderboard_id": PHASE_H_LEADERBOARD_ID,
        "legacy_leaderboard_id": LEADERBOARD_ID,
        "status": "leaderboard_complete",
        "source_benchmark": str(benchmark_path) if benchmark_path else str(phase_a_path),
        "source_phase_a": phase_a.get("phase_id"),
        "deterministic_mode": bool(phase_a.get("deterministic_mode")),
        "score_weights": dict(SCORE_WEIGHTS),
        "score_formula": (
            "0.35*acceptance_rate + 0.25*verifier_agreement + "
            "0.20*(1-normalized_first_divergence) + 0.10*(1-failure_rate) + "
            "0.10*stability_score"
        ),
        "entries": ranked,
        "global_compressor_rankings": global_rankings,
        "per_model_breakdown": dict(per_model_breakdown),
        "insights": insights,
        "models_evaluated": models,
        "compressors_evaluated": compressors,
        "filters": {"model": filter_model, "compressor": filter_compressor},
        "exactkv_generator_modified": False,
        "reproducible_cli_command": "python scripts/exactkv.py run leaderboard",
    }
    report["validation_result"] = validate_public_leaderboard(report).to_dict()
    return report


def _phase_a_from_benchmark_run(benchmark_data: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct minimal Phase A shape from benchmark.json when embedded."""
    legacy_path = Path("reports/phaseA_benchmark.json")
    if legacy_path.is_file():
        return load_phase_a_report(legacy_path)
    return {}


def validate_public_leaderboard(report: Mapping[str, Any]) -> PublicLeaderboardValidation:
    legacy = dict(report)
    legacy["leaderboard_id"] = LEADERBOARD_ID
    result = validate_leaderboard_report(legacy)
    return PublicLeaderboardValidation(valid=result.valid, errors=result.errors)


def write_public_leaderboard_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path | str = DEFAULT_PUBLIC_LEADERBOARD_JSON,
    markdown_path: Path | str = DEFAULT_PUBLIC_LEADERBOARD_MD,
    csv_path: Path | str = DEFAULT_PUBLIC_LEADERBOARD_CSV,
) -> dict[str, Path]:
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    csv_out = Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(report, indent=2) + "\n")
    md_out.write_text(render_leaderboard_markdown(report))
    _write_leaderboard_csv(report, csv_out)
    return {
        "leaderboard_json": json_out,
        "leaderboard_md": md_out,
        "leaderboard_csv": csv_out,
    }


def _write_leaderboard_csv(report: Mapping[str, Any], path: Path) -> None:
    entries = report.get("entries") or []
    if not entries:
        path.write_text("")
        return
    fieldnames = [
        "rank",
        "compressor",
        "model",
        "model_short",
        "score",
        "acceptance_rate",
        "divergence_score",
        "verifier_agreement",
        "failure_rate",
        "stability_score",
        "compression_ratio",
        "first_divergence_normalized",
        "availability",
        "num_cells",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in entries:
            writer.writerow(row)


# Re-export locked scoring for tests and documentation
__all__ = [
    "PHASE_H_LEADERBOARD_ID",
    "SCORE_WEIGHTS",
    "compute_leaderboard_score",
    "run_public_leaderboard",
    "validate_public_leaderboard",
    "write_public_leaderboard_outputs",
]
