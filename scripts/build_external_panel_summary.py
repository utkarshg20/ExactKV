#!/usr/bin/env python3
"""Merge external panel GPU artifacts and build summary_all.json + README.md."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_DIR = Path("reports/external_panels")
LOG_DIR = PANEL_DIR / "logs"

# Canonical merged outputs per family/run-type
MERGE_GROUPS: dict[str, list[str]] = {
    "longbench_pilot": ["longbench_pilot_*_raw.json"],
    "longbench_hf": ["longbench_hf_*_raw.json"],
    "ruler_2048_4096": ["ruler_2048_4096_*_raw.json"],
    "ruler_8192": ["ruler_8192_*_raw.json"],
    "bfcl": ["bfcl_*_raw.json"],
    "humaneval": ["humaneval_*_raw.json"],
    # Legacy deterministic smoke (keep if no GPU files)
    "longbench_smoke_det": ["longbench_raw.json"],
    "ruler_smoke_det": ["ruler_raw.json"],
    "bfcl_smoke_det": ["bfcl_raw.json"],
    "humaneval_smoke_det": ["humaneval_raw.json"],
}


def _glob_reports(pattern: str) -> list[Path]:
    return [p for p in sorted(PANEL_DIR.glob(pattern)) if "_merged_raw" not in p.name]


def _is_gpu_report(report: dict[str, Any]) -> bool:
    return not report.get("deterministic_mode", True)


def _timing_stats(cells: list[dict[str, Any]]) -> dict[str, float | None]:
    times = [
        float((c.get("timing_ms") or {}).get("total_cell", 0))
        for c in cells
        if c.get("status") == "ok" and (c.get("timing_ms") or {}).get("total_cell")
    ]
    times = [t for t in times if t > 0]
    if not times:
        return {"mean_ms": None, "p90_ms": None, "count": 0}
    times_sorted = sorted(times)
    p90_idx = max(0, int(len(times_sorted) * 0.9) - 1)
    return {
        "mean_ms": round(statistics.mean(times_sorted), 3),
        "p90_ms": round(times_sorted[p90_idx], 3),
        "count": len(times_sorted),
    }


def _summarize_report(report: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    ok_cells = [c for c in report.get("cells", []) if c.get("status") == "ok"]
    metrics = [c.get("metrics") or {} for c in ok_cells]
    n = max(len(metrics), 1)
    div = sum(1 for m in metrics if m.get("token_level_divergence"))
    return {
        "source_path": str(source_path),
        "phase_id": report.get("phase_id"),
        "dataset_family": report.get("dataset_family"),
        "deterministic_mode": report.get("deterministic_mode"),
        "smoke": report.get("smoke"),
        "prompt_source": report.get("prompt_source"),
        "models_evaluated": report.get("models_evaluated"),
        "context_buckets": report.get("context_buckets"),
        "max_new_tokens_values": report.get("max_new_tokens_values"),
        "total_cells": report.get("total_cells"),
        "cells_run": report.get("cells_run"),
        "cells_skipped": report.get("cells_skipped"),
        "exactkv_failures": report.get("exactkv_failures"),
        "divergence_rate": div / n if metrics else None,
        "mean_acceptance_rate": (
            sum(m.get("acceptance_rate", 0.0) for m in metrics) / n if metrics else None
        ),
        "timing": _timing_stats(ok_cells),
        "bucket_summary": report.get("bucket_summary"),
        "category_summary": report.get("category_summary"),
        "compressor_summary": report.get("compressor_summary"),
        "generated_at": report.get("generated_at"),
        "limitations_note": report.get("limitations_note"),
    }


def merge_reports(paths: list[Path], *, group_name: str) -> dict[str, Any] | None:
    gpu_paths = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _is_gpu_report(r):
            gpu_paths.append((p, r))

    if not gpu_paths:
        return None

    merged_cells: list[dict[str, Any]] = []
    models: list[str] = []
    skipped = 0
    for p, r in gpu_paths:
        merged_cells.extend(r.get("cells") or [])
        models.extend(r.get("models_evaluated") or [])
        skipped += int(r.get("cells_skipped") or 0)

    ok_cells = [c for c in merged_cells if c.get("status") == "ok"]
    metrics = [c.get("metrics") or {} for c in ok_cells]
    n = max(len(metrics), 1)
    div = sum(1 for m in metrics if m.get("token_level_divergence"))

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in ok_cells:
        b = c.get("context_bucket")
        if b is not None:
            by_bucket[str(int(b))].append(c.get("metrics") or {})
        cat = str(c.get("task_category") or c.get("category") or "unknown")
        by_cat[cat].append(c.get("metrics") or {})

    bucket_summary = {}
    for b, ms in sorted(by_bucket.items(), key=lambda x: int(x[0])):
        nn = max(len(ms), 1)
        bucket_summary[b] = {
            "num_cells": len(ms),
            "divergence_rate": sum(1 for m in ms if m.get("token_level_divergence")) / nn,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in ms) / nn,
        }

    category_summary = {}
    for cat, ms in sorted(by_cat.items()):
        nn = max(len(ms), 1)
        category_summary[cat] = {
            "num_cells": len(ms),
            "divergence_rate": sum(1 for m in ms if m.get("token_level_divergence")) / nn,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in ms) / nn,
        }

    family = gpu_paths[0][1].get("dataset_family", group_name.split("_")[0])
    merged = {
        "phase_id": "external_benchmark_panel",
        "dataset_family": family,
        "merge_group": group_name,
        "status": "benchmark_complete",
        "deterministic_mode": False,
        "prompt_source": gpu_paths[0][1].get("prompt_source"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p, _ in gpu_paths],
        "models_evaluated": sorted(set(models)),
        "total_cells": len(merged_cells),
        "cells_run": len(ok_cells),
        "cells_skipped": skipped,
        "exactkv_failures": sum(1 for c in ok_cells if c.get("exactkv_failure")),
        "divergence_rate": div / n,
        "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics) / n,
        "timing": _timing_stats(ok_cells),
        "bucket_summary": bucket_summary,
        "category_summary": category_summary,
        "cells": merged_cells,
        "limitations_note": (
            "Merged ExactKV drift panel on external benchmark prompts. "
            "Not official LongBench/RULER/BFCL/HumanEval scores."
        ),
    }
    return merged


def collect_workflow_logs() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not LOG_DIR.is_dir():
        return entries
    for mf in sorted(LOG_DIR.glob("workflow_manifest_*.jsonl")):
        for line in mf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append({"raw": line, "source": str(mf)})
    return entries


def build_summary(*, write_readme: bool = False) -> dict[str, Any]:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    per_file: list[dict[str, Any]] = []
    merged_groups: dict[str, Any] = {}
    skipped_groups: list[str] = []

    for group, patterns in MERGE_GROUPS.items():
        paths: list[Path] = []
        for pat in patterns:
            paths.extend(_glob_reports(pat))
        paths = sorted(set(paths))

        file_summaries = []
        for p in paths:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            file_summaries.append(_summarize_report(r, source_path=p))

        per_file.extend(file_summaries)

        merged = merge_reports(paths, group_name=group)
        if merged:
            out_path = PANEL_DIR / f"{group}_merged_raw.json"
            out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            merged_groups[group] = {
                "merged_path": str(out_path),
                "cells_run": merged["cells_run"],
                "total_cells": merged["total_cells"],
                "exactkv_failures": merged["exactkv_failures"],
                "divergence_rate": merged["divergence_rate"],
                "mean_acceptance_rate": merged["mean_acceptance_rate"],
                "timing": merged["timing"],
                "prompt_source": merged.get("prompt_source"),
                "models_evaluated": merged.get("models_evaluated"),
            }
        elif patterns[0].endswith("_raw.json") and not group.endswith("_det"):
            skipped_groups.append(group)

    workflow_steps = collect_workflow_logs()
    failed_steps = [s for s in workflow_steps if s.get("status") == "failed"]
    skipped_steps = [s for s in workflow_steps if s.get("status") == "skipped"]

    summary_all = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "ExactKV drift panels on external benchmark prompts. "
            "Not official LongBench/RULER/BFCL/HumanEval leaderboard scores."
        ),
        "merged_groups": merged_groups,
        "per_file_summaries": per_file,
        "workflow_steps": workflow_steps,
        "failed_steps": failed_steps,
        "skipped_steps": skipped_steps,
        "skipped_merge_groups": skipped_groups,
    }

    out = PANEL_DIR / "summary_all.json"
    out.write_text(json.dumps(summary_all, indent=2) + "\n", encoding="utf-8")

    if write_readme:
        _write_readme(summary_all)

    return summary_all


def _write_readme(summary: dict[str, Any]) -> None:
    lines = [
        "# External benchmark GPU panels",
        "",
        "**Claim boundary:** These are ExactKV drift panels on external benchmark prompts. "
        "They are **not** official LongBench, RULER, BFCL, or HumanEval leaderboard scores.",
        "",
        f"Generated: {summary.get('generated_at')}",
        "",
        "## Commands (conservative RunPod A5000 workflow)",
        "",
        "```bash",
        "bash scripts/run_external_gpu_workflow.sh",
        "```",
        "",
        "Per-step commands:",
        "",
        "```bash",
        "python3 scripts/run_external_panel.py --family longbench --device cuda --dtype float16 \\",
        "  --max-prompts 6 --context-buckets 2048,4096 --max-new-tokens 16,32",
        "",
        "python3 scripts/export_longbench_subset.py --max-per-subset 2",
        "python3 scripts/run_external_panel.py --family longbench --prompt-source hf --device cuda --dtype float16 \\",
        "  --max-prompts 12 --context-buckets 2048,4096 --max-new-tokens 16,32",
        "",
        "python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 \\",
        "  --context-buckets 2048,4096 --max-new-tokens 16,32",
        "",
        "python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 \\",
        "  --context-buckets 8192 --max-new-tokens 16,32",
        "",
        "python3 scripts/run_external_panel.py --family bfcl --device cuda --dtype float16 \\",
        "  --max-prompts 25 --context-buckets 1024,2048 --max-new-tokens 16,32",
        "",
        "python3 scripts/run_external_panel.py --family humaneval --device cuda --dtype float16 \\",
        "  --max-prompts 20 --context-buckets 1024,2048 --max-new-tokens 32",
        "```",
        "",
        "Models were run **one at a time** (Llama-3.1-8B, then Mistral-7B) to reduce VRAM/disk pressure.",
        "",
        "**GPU scope (this run):** Successful panels used **Llama-3.1-8B only** on RunPod A5000. "
        "Mistral-7B failed with disk quota exceeded after Llama weights filled the 50 GB volume. "
        "LongBench HF export skipped (`datasets` not installed). "
        "BFCL/HumanEval used bundled pilot JSONL (4 prompts each).",
        "",
        "## Merged GPU results",
        "",
        "| Group | Prompt source | Cells (ok/total) | Divergence rate | Mean acceptance | ExactKV failures | Mean ms | P90 ms |",
        "|-------|---------------|------------------:|----------------:|----------------:|-----------------:|--------:|-------:|",
    ]

    for group, info in sorted((summary.get("merged_groups") or {}).items()):
        if not info:
            continue
        t = info.get("timing") or {}
        lines.append(
            f"| {group} | {info.get('prompt_source', 'pilot')} | "
            f"{info.get('cells_run')}/{info.get('total_cells')} | "
            f"{(info.get('divergence_rate') or 0):.3f} | "
            f"{(info.get('mean_acceptance_rate') or 0):.3f} | "
            f"{info.get('exactkv_failures', 0)} | "
            f"{t.get('mean_ms', 'n/a')} | {t.get('p90_ms', 'n/a')} |",
        )

    lines.extend([
        "",
        "## Per-file artifacts",
        "",
    ])
    for fs in summary.get("per_file_summaries") or []:
        if fs.get("deterministic_mode"):
            continue
        t = fs.get("timing") or {}
        lines.append(
            f"- `{fs.get('source_path')}`: "
            f"family={fs.get('dataset_family')} source={fs.get('prompt_source')} "
            f"cells={fs.get('cells_run')}/{fs.get('total_cells')} "
            f"div={(fs.get('divergence_rate') or 0):.3f} "
            f"accept={(fs.get('mean_acceptance_rate') or 0):.3f} "
            f"failures={fs.get('exactkv_failures')} "
            f"mean_ms={t.get('mean_ms')} p90_ms={t.get('p90_ms')}",
        )

    failed = summary.get("failed_steps") or []
    skipped = summary.get("skipped_steps") or []
    lines.extend(["", "## Failed workflow steps", ""])
    if failed:
        for s in failed:
            detail = str(s.get("detail") or "")
            if "Mistral" in str(s.get("step", "")):
                detail = "disk quota exceeded loading Mistral weights; " + detail
            lines.append(f"- **{s.get('step')}**: {detail}")
    else:
        lines.append("- None recorded")

    lines.extend(["", "## Skipped workflow steps", ""])
    if skipped:
        for s in skipped:
            lines.append(f"- **{s.get('step')}**: {s.get('detail')}")
    else:
        lines.append("- None recorded")

    skipped_merge = summary.get("skipped_merge_groups") or []
    if skipped_merge:
        lines.extend(["", "## Merge groups without GPU data", ""])
        for g in skipped_merge:
            lines.append(f"- {g}")

    lines.extend([
        "",
        "## Logs",
        "",
        "Workflow logs: `reports/external_panels/logs/`",
        "",
        "Summary JSON: `reports/external_panels/summary_all.json`",
        "",
    ])

    (PANEL_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    summary = build_summary(write_readme=args.write_readme)
    print(json.dumps({k: summary[k] for k in ("merged_groups", "failed_steps", "skipped_steps")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
