#!/usr/bin/env python3
"""Validate external benchmark panel JSON artifacts under reports/external_panels."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REPORT_REQUIRED = (
    "phase_id",
    "dataset_family",
    "status",
    "deterministic_mode",
    "prompt_source",
    "generated_at",
    "manifest",
    "models_evaluated",
    "context_buckets",
    "max_new_tokens_values",
    "compressors",
    "total_cells",
    "cells_run",
    "cells_skipped",
    "exactkv_failures",
    "compressor_summary",
    "bucket_summary",
    "category_summary",
    "cells",
)

MERGED_REPORT_REQUIRED = (
    "phase_id",
    "dataset_family",
    "merge_group",
    "status",
    "deterministic_mode",
    "prompt_source",
    "generated_at",
    "source_files",
    "models_evaluated",
    "total_cells",
    "cells_run",
    "cells_skipped",
    "exactkv_failures",
    "divergence_rate",
    "bucket_summary",
    "category_summary",
    "cells",
)

CELL_REQUIRED_OK = (
    "model_name",
    "compressor_name",
    "context_bucket",
    "max_new_tokens",
    "prompt_id",
    "status",
    "metrics",
    "exactkv_failure",
    "dataset_family",
)

METRICS_REQUIRED = (
    "token_level_divergence",
    "acceptance_rate",
    "exactkv_failure",
)

METRICS_NULLABLE = frozenset({"first_divergence_index"})

SUMMARY_NULLABLE = frozenset({"mean_first_divergence_index"})

PANEL_FILE_GLOB = "*_raw.json"


def _is_bad_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isnan(float(value)) or math.isinf(float(value))
    return False


def _discover_panel_files(input_dir: Path) -> list[Path]:
    skip = {
        "analysis_pack.json",
        "case_studies_extracted.json",
        "summary_all.json",
        "validation_report.json",
    }
    return [p for p in sorted(input_dir.glob(PANEL_FILE_GLOB)) if p.name not in skip]


def _is_merged_report(report: dict[str, Any]) -> bool:
    return bool(report.get("merge_group")) or "_merged_raw" in str(report.get("_source_name", ""))


def _derive_compressors(cells: list[dict[str, Any]]) -> list[str]:
    return sorted({str(c.get("compressor_name")) for c in cells if c.get("compressor_name")})


def _compute_divergence_rate(cells: list[dict[str, Any]]) -> float:
    ok = [c for c in cells if c.get("status") == "ok"]
    if not ok:
        return 0.0
    div = sum(1 for c in ok if (c.get("metrics") or {}).get("token_level_divergence"))
    return div / len(ok)


def _compressor_divergence_rates(cells: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("status") != "ok":
            continue
        name = str(cell.get("compressor_name") or "unknown")
        buckets.setdefault(name, []).append(cell)
    out: dict[str, float] = {}
    for name, group in sorted(buckets.items()):
        div = sum(1 for c in group if (c.get("metrics") or {}).get("token_level_divergence"))
        out[name] = div / max(len(group), 1)
    return out


def _bucket_divergence_rates(cells: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("status") != "ok":
            continue
        bucket = cell.get("context_bucket")
        if bucket is None:
            continue
        buckets.setdefault(str(int(bucket)), []).append(cell)
    out: dict[str, float] = {}
    for key, group in sorted(buckets.items(), key=lambda x: int(x[0])):
        div = sum(1 for c in group if (c.get("metrics") or {}).get("token_level_divergence"))
        out[key] = div / max(len(group), 1)
    return out


def validate_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    merged = _is_merged_report(report) or path.name.endswith("_merged_raw.json")
    required = MERGED_REPORT_REQUIRED if merged else REPORT_REQUIRED

    for field in required:
        if field not in report:
            issues.append(f"missing report field: {field}")

    if report.get("phase_id") and report["phase_id"] != "external_benchmark_panel":
        issues.append(f"unexpected phase_id: {report['phase_id']!r}")

    prompt_source = report.get("prompt_source")
    if prompt_source not in ("pilot", "hf", "export"):
        issues.append(f"invalid prompt_source: {prompt_source!r}")

    deterministic = bool(report.get("deterministic_mode"))
    cells = report.get("cells") or []
    ok_cells = [c for c in cells if c.get("status") == "ok"]

    for cell in ok_cells:
        for field in CELL_REQUIRED_OK:
            if cell.get(field) is None:
                issues.append(
                    f"cell {cell.get('prompt_id', '?')} missing {field}",
                )

        metrics = cell.get("metrics") or {}
        for field in METRICS_REQUIRED:
            if field not in metrics:
                issues.append(
                    f"cell {cell.get('prompt_id', '?')} metrics missing {field}",
                )

        for key, value in metrics.items():
            if key in METRICS_NULLABLE:
                continue
            if value is None:
                issues.append(
                    f"cell {cell.get('prompt_id', '?')} metrics.{key} is None",
                )
            elif _is_bad_number(value):
                issues.append(
                    f"cell {cell.get('prompt_id', '?')} metrics.{key} is NaN/Inf",
                )

        top_fail = bool(cell.get("exactkv_failure"))
        metric_fail = bool(metrics.get("exactkv_failure"))
        if top_fail != metric_fail:
            issues.append(
                f"cell {cell.get('prompt_id', '?')} exactkv_failure mismatch "
                f"(top={top_fail}, metrics={metric_fail})",
            )

        timing = cell.get("timing_ms")
        if not isinstance(timing, dict) or "total_cell" not in timing:
            issues.append(
                f"cell {cell.get('prompt_id', '?')} missing timing_ms.total_cell",
            )
        elif not deterministic:
            total = timing.get("total_cell")
            if not isinstance(total, (int, float)) or total <= 0:
                issues.append(
                    f"cell {cell.get('prompt_id', '?')} invalid GPU timing_ms.total_cell",
                )

    reported_failures = int(report.get("exactkv_failures") or 0)
    computed_failures = sum(1 for c in ok_cells if c.get("exactkv_failure"))
    if reported_failures != computed_failures:
        issues.append(
            f"exactkv_failures mismatch (report={reported_failures}, cells={computed_failures})",
        )

    computed_div = _compute_divergence_rate(cells)
    if merged:
        reported_div = float(report.get("divergence_rate", -1))
        if abs(reported_div - computed_div) > 1e-9:
            issues.append(
                f"divergence_rate mismatch (report={reported_div}, cells={computed_div})",
            )
        timing = report.get("timing") or {}
        if not timing.get("count"):
            warnings.append("merged report missing timing.count")
    else:
        for comp_name, stats in (report.get("compressor_summary") or {}).items():
            expected = _compressor_divergence_rates(cells).get(comp_name)
            if expected is None:
                warnings.append(f"compressor_summary has unknown compressor {comp_name!r}")
                continue
            actual = float(stats.get("divergence_rate", -1))
            if abs(actual - expected) > 1e-9:
                issues.append(
                    f"compressor_summary[{comp_name}] divergence_rate "
                    f"{actual} != computed {expected}",
                )
            for key, value in stats.items():
                if key in SUMMARY_NULLABLE:
                    continue
                if value is None:
                    issues.append(f"compressor_summary[{comp_name}].{key} is None")
                elif _is_bad_number(value):
                    issues.append(f"compressor_summary[{comp_name}].{key} is NaN/Inf")

    for bucket, stats in (report.get("bucket_summary") or {}).items():
        expected = _bucket_divergence_rates(cells).get(str(bucket))
        if expected is None:
            warnings.append(f"bucket_summary has unknown bucket {bucket!r}")
            continue
        actual = float(stats.get("divergence_rate", -1))
        if abs(actual - expected) > 1e-9:
            issues.append(
                f"bucket_summary[{bucket}] divergence_rate {actual} != computed {expected}",
            )

    models = report.get("models_evaluated") or []
    compressors = report.get("compressors") or []
    if ok_cells:
        model_names = {c.get("model_name") for c in ok_cells}
        compressor_names = {c.get("compressor_name") for c in ok_cells}
        if set(models) != model_names:
            warnings.append(
                f"models_evaluated {sorted(models)} != cell models {sorted(model_names)}",
            )
        if not merged:
            if set(compressors) != compressor_names:
                warnings.append(
                    f"compressors {sorted(compressors)} != cell compressors {sorted(compressor_names)}",
                )
        elif not compressors:
            derived = _derive_compressors(ok_cells)
            if derived:
                warnings.append(
                    f"merged report has no compressors list; derived {derived} from cells",
                )

    manifest = report.get("manifest") or {}
    if not merged and prompt_source == "pilot" and report.get("dataset_family"):
        family = str(report["dataset_family"])
        suite = str(manifest.get("prompt_suite") or "")
        if family not in suite:
            warnings.append(
                f"manifest.prompt_suite {suite!r} does not mention family {family!r}",
            )

    gpu_like = any(
        isinstance((c.get("timing_ms") or {}).get("total_cell"), (int, float))
        and (c.get("timing_ms") or {}).get("total_cell", 0) > 100
        for c in ok_cells
    )
    if deterministic and gpu_like:
        warnings.append("deterministic_mode report contains GPU-like timing (>100ms)")
    if not deterministic and not gpu_like and ok_cells:
        warnings.append("GPU report has only stub timing (<=100ms) on all ok cells")

    return {
        "path": rel,
        "dataset_family": report.get("dataset_family"),
        "merged": merged,
        "deterministic_mode": deterministic,
        "prompt_source": prompt_source,
        "cells_run": report.get("cells_run"),
        "exactkv_failures": reported_failures,
        "computed_divergence_rate": computed_div,
        "issues": issues,
        "warnings": warnings,
        "valid": not issues,
    }


def build_validation_report(input_dir: Path) -> dict[str, Any]:
    files = _discover_panel_files(input_dir)
    file_results = []
    for path in files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            file_results.append(
                {
                    "path": str(path),
                    "valid": False,
                    "issues": [f"invalid JSON: {exc}"],
                    "warnings": [],
                },
            )
            continue
        if not isinstance(report, dict) or "cells" not in report:
            file_results.append(
                {
                    "path": str(path),
                    "valid": False,
                    "issues": ["not an external panel report (missing cells)"],
                    "warnings": [],
                },
            )
            continue
        file_results.append(validate_report(path, report))

    valid_count = sum(1 for r in file_results if r.get("valid"))
    issue_count = sum(len(r.get("issues") or []) for r in file_results)
    warning_count = sum(len(r.get("warnings") or []) for r in file_results)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "files_scanned": len(file_results),
        "files_valid": valid_count,
        "files_invalid": len(file_results) - valid_count,
        "total_issues": issue_count,
        "total_warnings": warning_count,
        "claim_boundary": (
            "Validator checks ExactKV drift panel schema and internal consistency. "
            "It does not certify official LongBench/RULER/BFCL/HumanEval/MBPP scores."
        ),
        "files": file_results,
        "overall_valid": valid_count == len(file_results) and issue_count == 0,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External panel artifact validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"**Input:** `{report['input_dir']}`",
        "",
        f"**Overall:** {'PASS' if report['overall_valid'] else 'FAIL'} "
        f"({report['files_valid']}/{report['files_scanned']} files valid, "
        f"{report['total_issues']} issues, {report['total_warnings']} warnings)",
        "",
        report["claim_boundary"],
        "",
        "## Per-file results",
        "",
        "| File | Family | Mode | Source | Cells | Valid | Issues | Warnings |",
        "|------|--------|------|--------|------:|-------|-------:|---------:|",
    ]
    for entry in report["files"]:
        lines.append(
            "| {path} | {family} | {mode} | {source} | {cells} | {valid} | {issues} | {warnings} |".format(
                path=entry.get("path", "?"),
                family=entry.get("dataset_family") or "-",
                mode="deterministic" if entry.get("deterministic_mode") else "gpu",
                source=entry.get("prompt_source") or "-",
                cells=entry.get("cells_run") or "-",
                valid="yes" if entry.get("valid") else "no",
                issues=len(entry.get("issues") or []),
                warnings=len(entry.get("warnings") or []),
            ),
        )

    failed = [e for e in report["files"] if not e.get("valid")]
    if failed:
        lines.extend(["", "## Issues", ""])
        for entry in failed:
            lines.append(f"### `{entry.get('path')}`")
            for issue in entry.get("issues") or []:
                lines.append(f"- {issue}")
            for warning in entry.get("warnings") or []:
                lines.append(f"- (warning) {warning}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/external_panels",
        help="Directory containing external panel JSON artifacts",
    )
    args = parser.parse_args()
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        print(f"input directory not found: {input_dir}", file=sys.stderr)
        return 1

    report = build_validation_report(input_dir)
    out_json = input_dir / "validation_report.json"
    out_md = input_dir / "validation_report.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out_md)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(
        f"Validation: {'PASS' if report['overall_valid'] else 'FAIL'} "
        f"({report['files_valid']}/{report['files_scanned']} files)",
    )
    return 0 if report["overall_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
