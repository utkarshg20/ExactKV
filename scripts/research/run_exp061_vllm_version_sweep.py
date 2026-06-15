#!/usr/bin/env python3
"""Experiment 061: isolated vLLM version compatibility sweep (Phase 15B-unblock).

Reads sweep manifest/logs and writes a normalized JSON report.
Does **not** install vLLM into system Python.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.integrations.vllm_probe import (  # noqa: E402
    DEFAULT_EXP061_REPORT,
    DEFAULT_SWEEP_LOG_DIR,
    DEFAULT_SWEEP_MANIFEST,
    SYSTEM_PYTHON,
    build_exp061_report,
    load_candidate_results,
    load_sweep_manifest,
    validate_exp061_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 061 vLLM version sweep report")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP061_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SWEEP_MANIFEST)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_SWEEP_LOG_DIR)
    parser.add_argument("--system-python", type=Path, default=SYSTEM_PYTHON)
    args = parser.parse_args()

    manifest = load_sweep_manifest(args.manifest)
    candidate_results = load_candidate_results(manifest=manifest, log_root=args.log_root)
    if manifest.get("candidate_results") and not candidate_results:
        candidate_results = list(manifest["candidate_results"])

    report = build_exp061_report(
        candidate_results=candidate_results,
        candidates=list(manifest.get("candidates") or []),
        excluded_versions=list(manifest.get("excluded_versions") or []),
        system_python=args.system_python,
        manifest=manifest,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["sweep_manifest"] = str(args.manifest)

    schema_errors = validate_exp061_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 061: any_candidate_passed={report['any_candidate_passed']} "
        f"winning={report['winning_candidate']}"
    )
    print(f"  candidates_tested={report['candidates']}")
    print(f"  generation_smoke_passed={report['generation_smoke_passed']}")
    if report["blockers"]:
        print(f"  blockers={report['blockers'][:3]}")
    print(f"  next={report['recommended_next_step']}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
