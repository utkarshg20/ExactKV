"""Leaderboard aggregate repair from Phase A / scale raw cells (Release Gate R1)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from exactkv.benchmarks.phase_a_scale_benchmark import build_per_model_tables


def repair_phase_a_report_aggregates(phase_a: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``per_model_tables`` from cells when incomplete or missing models.

    Sequential scale runs may append cells without updating derived tables.
    """
    repaired = dict(phase_a)
    cells = repaired.get("cells") or []
    if not cells:
        return repaired

    models_evaluated = list(repaired.get("models_evaluated") or [])
    if not models_evaluated:
        from collections import OrderedDict

        models_evaluated = list(
            OrderedDict.fromkeys(str(c.get("model_name") or "") for c in cells if c.get("model_name")),
        )
        repaired["models_evaluated"] = models_evaluated

    rebuilt = build_per_model_tables(cells)
    per_model = repaired.get("per_model_tables") or {}
    missing = [m for m in models_evaluated if m not in per_model]
    if missing or len(per_model) < len(rebuilt):
        repaired["per_model_tables"] = rebuilt
    return repaired


def count_cells_by_model(phase_a: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cell in phase_a.get("cells") or []:
        model = str(cell.get("model_name") or "")
        if model:
            counts[model] += 1
    return counts


def validate_leaderboard_against_raw(
    phase_a: dict[str, Any],
    leaderboard: dict[str, Any],
) -> list[str]:
    """Return error strings when public leaderboard disagrees with raw cell coverage."""
    errors: list[str] = []
    cell_counts = count_cells_by_model(phase_a)
    entries = leaderboard.get("entries") or []

    for model, count in sorted(cell_counts.items()):
        if count <= 0:
            continue
        model_rows = [e for e in entries if e.get("model") == model]
        scored = [r for r in model_rows if r.get("score") is not None]
        unavailable = [r for r in model_rows if r.get("availability") == "unavailable"]
        if not scored:
            errors.append(f"{model}: {count} raw cells but no scored leaderboard rows")
        if unavailable:
            errors.append(
                f"{model}: {len(unavailable)} unavailable row(s) despite {count} raw cells",
            )

    mistral_cells = sum(n for m, n in cell_counts.items() if "mistral" in m.lower())
    if mistral_cells > 0:
        mistral_scored = [
            e for e in entries
            if "mistral" in str(e.get("model", "")).lower()
            and e.get("score") is not None
            and e.get("availability") != "unavailable"
        ]
        if not mistral_scored:
            errors.append(f"Mistral: {mistral_cells} raw cells but no numeric public rows")

    return errors


def rebuild_scale_leaderboard_from_raw(
    raw_path: Path | str = Path("reports/scale_7b/raw.json"),
    *,
    write_raw_repairs: bool = True,
) -> dict[str, Any]:
    """Recompute scale_7b leaderboard from raw cells (no inference)."""
    from exactkv.platform.public_leaderboard import (  # noqa: PLC0415
        run_public_leaderboard,
        write_public_leaderboard_outputs,
    )

    raw_path = Path(raw_path)
    phase_a = json.loads(raw_path.read_text(encoding="utf-8"))
    repaired = repair_phase_a_report_aggregates(phase_a)

    if write_raw_repairs and repaired.get("per_model_tables") != phase_a.get("per_model_tables"):
        raw_path.write_text(json.dumps(repaired, indent=2) + "\n", encoding="utf-8")

    report = run_public_leaderboard(phase_a_path=raw_path)
    out_dir = raw_path.parent
    paths = write_public_leaderboard_outputs(
        report,
        json_path=out_dir / "leaderboard.json",
        markdown_path=out_dir / "leaderboard.md",
        csv_path=out_dir / "leaderboard.csv",
    )
    models_with_scores: set[str] = set()
    for row in report.get("entries") or []:
        if row.get("score") is not None and row.get("availability") != "unavailable":
            models_with_scores.add(str(row.get("model") or ""))
    return {
        "leaderboard_json": str(paths["leaderboard_json"]),
        "entries": len(report.get("entries") or []),
        "models_with_scores": sorted(m for m in models_with_scores if m),
    }
