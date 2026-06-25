#!/usr/bin/env python3
"""Experiment 117: instability visualization engine (Phase 21P).

Generates phase diagrams and heatmaps from Exp 115 + Exp 116. No inference.
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

from exactkv.analysis.l4_instability_visualization_engine import (  # noqa: E402
    DEFAULT_EXP115_REPORT,
    DEFAULT_EXP116_REPORT,
    DEFAULT_EXP117_OUTPUT_DIR,
    EXPERIMENT_117_ID,
    run_exp117_instability_visualization_engine,
    validate_exp117_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 117: instability visualization engine (Phase 21P)",
    )
    parser.add_argument("--exp115-input", type=Path, default=DEFAULT_EXP115_REPORT)
    parser.add_argument("--exp116-input", type=Path, default=DEFAULT_EXP116_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP117_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = run_exp117_instability_visualization_engine(
        exp115_path=args.exp115_input,
        exp116_path=args.exp116_input,
        output_dir=args.output_dir,
    )
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = args.output_dir / "exp117_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    validation = validate_exp117_manifest(manifest)
    print(f"Wrote visuals to {args.output_dir}")
    print(f"experiment_id={EXPERIMENT_117_ID}")
    print(f"status={manifest['status']}")
    print(f"visual_count={len(manifest.get('visual_outputs') or {})}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
