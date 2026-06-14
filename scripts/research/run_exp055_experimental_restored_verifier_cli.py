#!/usr/bin/env python3
"""Experiment 055: Explicit CLI flag for experimental restored-verifier runtime (Phase 13B).

Requires ``--experimental-restored-verifier`` to enable. **Not** default CLI behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import transformers

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS  # noqa: E402
from exactkv.cache.offline_verifier import VERIFIER_SOURCE  # noqa: E402
from exactkv.runtime.experimental_cli import (  # noqa: E402
    EXPERIMENT_055_ID,
    EXP055_CLAIM_NOTE,
    add_experimental_restored_verifier_cli_args,
    format_cli_summary,
    report_to_exp055_json,
    run_experimental_restored_verifier_from_cli,
    validate_exp055_report,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_055_experimental_restored_verifier_cli.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp 055 explicit CLI for experimental restored-verifier runtime"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="JSON report path (default: reports/experiment_055_... or --output)",
    )
    add_experimental_restored_verifier_cli_args(parser)
    return parser


def run_exp055_cli(argv: Sequence[str] | None = None) -> tuple[int, dict[str, Any]]:
    """Execute Exp 055 CLI flow; returns exit code and JSON report dict."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    json_out = args.json_out
    if args.output:
        json_out = Path(args.output)
    if json_out is None:
        json_out = DEFAULT_JSON

    resolution, result = run_experimental_restored_verifier_from_cli(args)
    json_report = report_to_exp055_json(resolution, result)
    json_report["transformers_version"] = transformers.__version__
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp055_report(json_report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(format_cli_summary(resolution, result))
    print(f"Wrote {json_out}")
    return (0 if result.status in ("pass", "disabled") else 1), json_report


def main() -> int:
    code, _report = run_exp055_cli()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
