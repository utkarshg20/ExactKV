#!/usr/bin/env python3
"""Validate public release artifact consistency (Phase J)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.public_release_validator import validate_public_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public release consistency")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--json-out", type=Path, default=_ROOT / "reports" / "public_release_validation.json")
    args = parser.parse_args()

    report = validate_public_release(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print("ExactKV public release validator")
    for issue in report.issues:
        print(f"  [{issue.severity.upper()}] {issue.check}: {issue.detail}")
    if report.valid:
        print("PASSED: public release artifacts consistent")
        return 0
    errors = sum(1 for i in report.issues if i.severity == "error")
    print(f"FAILED: {errors} error(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
