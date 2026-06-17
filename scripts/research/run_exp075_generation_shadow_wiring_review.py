#!/usr/bin/env python3
"""Experiment 075: generation-shadow wiring review (Phase 16J)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.generation_shadow_review import (  # noqa: E402
    DEFAULT_EXP075_REPORT,
    run_exp075_generation_shadow_review,
    validate_exp075_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 075 generation-shadow wiring review")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP075_REPORT)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    report = run_exp075_generation_shadow_review(root=args.repo_root)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp075_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 075: {report['status']} inspected={len(report['files_inspected'])} "
        f"recommended={report['recommended_next_level']}"
    )
    print(f"  missing={report.get('files_missing', [])}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
