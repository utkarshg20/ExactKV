#!/usr/bin/env python3
"""Validate project lineage artifacts (Release Gate R2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.lineage_validator import validate_project_lineage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check project lineage completeness")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--json-out", type=Path, default=_ROOT / "reports" / "project_lineage_validation.json")
    args = parser.parse_args()

    report = validate_project_lineage(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print("ExactKV project lineage validator")
    for issue in report.issues:
        print(f"  [{issue.severity.upper()}] {issue.check}: {issue.detail}")
    if report.valid:
        print("PASSED: project lineage complete")
        return 0
    print("FAILED: lineage validation errors")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
