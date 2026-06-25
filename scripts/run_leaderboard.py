#!/usr/bin/env python3
"""ExactKV canonical leaderboard runner (Phase B).

Consumes Phase A benchmark JSON and emits ranked leaderboard artifacts.

Usage:
    python scripts/run_leaderboard.py --all
    python scripts/run_leaderboard.py --json --markdown
    python scripts/run_leaderboard.py --filter-model "Qwen 0.5B"
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.leaderboard_platform import (  # noqa: E402
    DEFAULT_EXP116_INPUT,
    DEFAULT_LEADERBOARD_JSON,
    DEFAULT_LEADERBOARD_MD,
    DEFAULT_PHASE_A_INPUT,
    LEADERBOARD_ID,
    run_leaderboard_platform,
    validate_leaderboard_report,
    write_leaderboard_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="ExactKV canonical leaderboard (Phase B)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write JSON + markdown (default when no export flags given)",
    )
    parser.add_argument("--json", action="store_true", help="Write reports/leaderboard.json")
    parser.add_argument("--markdown", action="store_true", help="Write reports/leaderboard.md")
    parser.add_argument(
        "--phase-a-input",
        type=Path,
        default=DEFAULT_PHASE_A_INPUT,
    )
    parser.add_argument(
        "--exp116-input",
        type=Path,
        default=DEFAULT_EXP116_INPUT,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_LEADERBOARD_JSON,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_LEADERBOARD_MD,
    )
    parser.add_argument(
        "--deterministic-mode",
        action="store_true",
        help="Tag output as deterministic (does not re-run Phase A)",
    )
    parser.add_argument("--filter-model", default=None, help="Filter by model name substring")
    parser.add_argument("--filter-compressor", default=None, help="Filter to one compressor")
    args = parser.parse_args()

    export_json = args.json or args.all or not (args.json or args.markdown)
    export_md = args.markdown or args.all or not (args.json or args.markdown)

    report = run_leaderboard_platform(
        phase_a_path=args.phase_a_input,
        exp116_path=args.exp116_input,
        filter_model=args.filter_model,
        filter_compressor=args.filter_compressor,
        deterministic_mode=args.deterministic_mode or None,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    json_path, md_path = write_leaderboard_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
        write_json=export_json,
        write_markdown=export_md,
    )

    validation = validate_leaderboard_report(report)
    print(f"leaderboard_id={LEADERBOARD_ID}")
    print(f"status={report['status']}")
    print(f"ranked_entries={len([e for e in report['entries'] if e.get('rank')])}")
    print(f"insights={len(report.get('insights') or [])}")
    if json_path:
        print(f"wrote_json={json_path}")
    if md_path:
        print(f"wrote_markdown={md_path}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
