#!/usr/bin/env python3
"""Validate Phase K launch pack artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.launch_pack_validator import validate_launch_pack  # noqa: E402


def main() -> int:
    report = validate_launch_pack(_ROOT)
    print("ExactKV launch pack validator")
    if report.valid:
        print("PASSED: launch pack complete")
        return 0
    print("FAILED:")
    for issue in report.issues:
        if issue.severity == "error":
            print(f"  [{issue.check}] {issue.detail}")
    out = _ROOT / "reports" / "launch_pack_validation.json"
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
