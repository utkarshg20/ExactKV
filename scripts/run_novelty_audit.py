#!/usr/bin/env python3
"""Phase I — generate novelty audit reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.platform.novelty_audit import (  # noqa: E402
    build_novelty_audit,
    render_novelty_audit_markdown,
    write_novelty_audit_matrix_csv,
)

DEFAULT_JSON = _ROOT / "reports" / "novelty_audit.json"
DEFAULT_MD = _ROOT / "docs" / "NOVELTY_AUDIT.md"
DEFAULT_CSV = _ROOT / "reports" / "novelty_audit_matrix.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase I novelty audit")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    report = build_novelty_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_novelty_audit_markdown(report), encoding="utf-8")
    write_novelty_audit_matrix_csv(report.get("prior_art_systems") or [], args.csv_out)

    summary = report.get("summary") or {}
    print(f"phase_id={report.get('phase_id')}")
    print(f"status={report.get('status')}")
    print(f"prior_art_count={summary.get('prior_art_count')}")
    print(f"verified_sources={summary.get('verified_sources')}")
    print(f"source_pending={summary.get('source_pending')}")
    print(f"closest_prior_art={report.get('closest_prior_art')}")
    print(f"wrote_json={args.json_out}")
    print(f"wrote_md={args.md_out}")
    print(f"wrote_csv={args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
