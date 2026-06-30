#!/usr/bin/env python3
"""Recommend next external-panel / MBPP commands from existing artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RULER_16K_MAX_MEAN_MS = 18000.0
RULER_16K_MAX_P90_MS = 22000.0


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _family_present(input_dir: Path, family: str, *, gpu_only: bool = False) -> bool:
    for path in sorted(input_dir.glob(f"{family}*.json")):
        if path.name in {"analysis_pack.json", "summary_all.json", "validation_report.json"}:
            continue
        report = _load_json(path)
        if not report or "cells" not in report:
            continue
        if gpu_only and report.get("deterministic_mode"):
            continue
        if report.get("cells_run", 0) > 0:
            return True
    return False


def _summary_group(input_dir: Path, key: str) -> dict[str, Any] | None:
    summary_path = input_dir / "summary_all.json"
    summary = _load_json(summary_path)
    if not summary:
        return None
    return (summary.get("merged_groups") or {}).get(key)


def _mistral_failed(input_dir: Path) -> bool:
    readme = input_dir / "README.md"
    if readme.is_file() and "Mistral" in readme.read_text(encoding="utf-8"):
        if "disk quota exceeded" in readme.read_text(encoding="utf-8").lower():
            return True
    summary = _load_json(input_dir / "summary_all.json") or {}
    for entry in (summary.get("failed_steps") or []):
        if "mistral" in str(entry).lower():
            return True
    for path in input_dir.glob("*Mistral*.json"):
        report = _load_json(path)
        if report and report.get("cells_run", 0) == 0:
            return True
    return False


def _ruler_8192_acceptable(input_dir: Path) -> bool:
    group = _summary_group(input_dir, "ruler_8192")
    if not group:
        return False
    timing = group.get("timing") or {}
    mean_ms = float(timing.get("mean_ms") or 0)
    p90_ms = float(timing.get("p90_ms") or 0)
    failures = int(group.get("exactkv_failures") or 0)
    return failures == 0 and mean_ms <= RULER_16K_MAX_MEAN_MS and p90_ms <= RULER_16K_MAX_P90_MS


def build_next_run_plan(input_dir: Path) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []
    blockers: list[str] = []

    has_hf_longbench = _family_present(input_dir, "longbench", gpu_only=True) and any(
        (_load_json(p) or {}).get("prompt_source") == "hf"
        for p in input_dir.glob("longbench*.json")
    )
    if not has_hf_longbench:
        recommendations.append(
            {
                "priority": 1,
                "title": "Real Hugging Face LongBench (after datasets install)",
                "reason": "Pilot LongBench GPU panels exist; HF export not yet run.",
                "commands": [
                    "pip install datasets",
                    "python3 scripts/export_longbench_subset.py --max-per-subset 2 "
                    "--output benchmarks/prompts/longbench_export.jsonl",
                    "python3 scripts/run_external_panel.py --family longbench --prompt-source hf "
                    "--device cuda --dtype float16 --max-prompts 12 "
                    "--context-buckets 2048,4096 --max-new-tokens 16,32",
                ],
                "blockers": ["Requires `pip install datasets` on GPU host"],
            },
        )

    if _mistral_failed(input_dir):
        recommendations.append(
            {
                "priority": 2,
                "title": "Mistral-7B external panel rerun (after disk fix)",
                "reason": "Prior RunPod workflow failed with disk quota exceeded after Llama cache.",
                "commands": [
                    "# Clear HF cache or expand volume before rerun",
                    "rm -rf ~/.cache/huggingface/hub/models--*",
                    "bash scripts/run_external_gpu_workflow.sh  # or rerun families one model at a time",
                ],
                "blockers": ["RunPod 50 GB volume; run Llama and Mistral sequentially with cache cleanup"],
            },
        )

    bfcl_group = _summary_group(input_dir, "bfcl")
    bfcl_cells = int((bfcl_group or {}).get("cells_run") or 0)
    has_bfcl_50 = (input_dir / "bfcl_export_50_raw.json").is_file()
    if not has_bfcl_50 and bfcl_cells < 600:
        recommendations.append(
            {
                "priority": 3,
                "title": "Larger BFCL panel on GPU (50+ real BFCL v3 prompts)",
                "reason": f"Current BFCL merged panel has {bfcl_cells} ok cells; export panel not found.",
                "commands": [
                    "python3 scripts/export_bfcl_subset.py --max-per-category 13 --max-total 50",
                    "python3 scripts/run_external_panel.py --family bfcl --prompt-source export "
                    "--device cuda --dtype float16 --max-prompts 50 "
                    "--context-buckets 1024,2048 --max-new-tokens 16,32 "
                    "--output-json reports/external_panels/bfcl_export_50_raw.json",
                ],
                "blockers": [],
            },
        )

    has_mbpp_gpu = _family_present(input_dir, "mbpp", gpu_only=True) or (
        input_dir / "mbpp_gpu_raw.json"
    ).is_file()
    if not has_mbpp_gpu:
        has_mbpp_offline = _family_present(input_dir, "mbpp")
        reason = (
            "MBPP pilot loader added; offline smoke exists."
            if has_mbpp_offline
            else "MBPP pilot loader added; no artifacts yet."
        )
        recommendations.append(
            {
                "priority": 4,
                "title": "MBPP smoke (bundled pilot, token drift only)",
                "reason": reason,
                "commands": (
                    ["python3 scripts/run_external_panel.py --family mbpp --deterministic-mode --smoke"]
                    if not has_mbpp_offline
                    else []
                )
                + [
                    "python3 scripts/run_external_panel.py --family mbpp --device cuda --dtype float16 "
                    "--max-prompts 6 --context-buckets 512,1024 --max-new-tokens 16,32",
                ],
                "blockers": [],
            },
        )

    has_ruler_8192 = _summary_group(input_dir, "ruler_8192") is not None
    has_ruler_16k = any(
        16384 in ((_load_json(p) or {}).get("context_buckets") or [])
        for p in input_dir.glob("ruler*.json")
    )
    if has_ruler_8192 and not has_ruler_16k:
        if _ruler_8192_acceptable(input_dir):
            recommendations.append(
                {
                    "priority": 5,
                    "title": "RULER 16K bucket (conditional on 8K timing)",
                    "reason": "8192 bucket completed with acceptable timing and zero ExactKV failures.",
                    "commands": [
                        "python3 scripts/run_external_panel.py --family ruler --device cuda --dtype float16 "
                        "--context-buckets 16384 --max-new-tokens 16,32",
                    ],
                    "blockers": [],
                },
            )
        else:
            blockers.append(
                "Skip RULER 16K until 8192 mean/p90 timing and failure checks pass "
                f"(threshold mean<={RULER_16K_MAX_MEAN_MS}ms p90<={RULER_16K_MAX_P90_MS}ms).",
            )
    elif not has_ruler_8192:
        blockers.append("RULER 16K deferred until an 8192 GPU panel exists.")

    recommendations.sort(key=lambda r: r["priority"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "claim_boundary": (
            "Planner recommends ExactKV drift panel commands only. "
            "It does not schedule official benchmark score reproduction or code execution for MBPP."
        ),
        "recommendations": recommendations,
        "blockers": blockers,
        "maintenance_commands": [
            "python3 scripts/validate_external_panel_artifacts.py --input reports/external_panels",
            "python3 scripts/plan_next_external_runs.py --input reports/external_panels",
        ],
    }


def write_markdown(plan: dict[str, Any], path: Path) -> None:
    lines = [
        "# External panel next-run plan",
        "",
        f"Generated: {plan['generated_at']}",
        "",
        plan["claim_boundary"],
        "",
        "## Recommended commands (priority order)",
        "",
    ]
    if not plan["recommendations"]:
        lines.append("_No pending recommendations; existing artifacts cover the current roadmap._")
        lines.append("")
    for rec in plan["recommendations"]:
        lines.extend(
            [
                f"### {rec['priority']}. {rec['title']}",
                "",
                rec["reason"],
                "",
                "```bash",
                *rec["commands"],
                "```",
                "",
            ],
        )
        if rec.get("blockers"):
            lines.append("Blockers:")
            for blocker in rec["blockers"]:
                lines.append(f"- {blocker}")
            lines.append("")

    if plan.get("blockers"):
        lines.extend(["## Global blockers", ""])
        for blocker in plan["blockers"]:
            lines.append(f"- {blocker}")
        lines.append("")

    lines.extend(["## Maintenance", ""])
    lines.append("```bash")
    lines.extend(plan.get("maintenance_commands") or [])
    lines.append("```")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/external_panels",
        help="Directory containing external panel artifacts",
    )
    args = parser.parse_args()
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        print(f"input directory not found: {input_dir}", file=sys.stderr)
        return 1

    plan = build_next_run_plan(input_dir)
    out_md = input_dir / "next_run_plan.md"
    write_markdown(plan, out_md)
    print(f"Wrote {out_md}")
    print(f"Recommendations: {len(plan['recommendations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
