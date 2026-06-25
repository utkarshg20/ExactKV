#!/usr/bin/env python3
"""Phase C publication + demo layer runner.

Generates demo pack, paper draft, blog, social posts, and visual synthesis
from Phase A + B reports. No inference.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.publication_layer import (  # noqa: E402
    DEFAULT_DEMO_PACK,
    DEFAULT_EXP115_REPORT,
    DEFAULT_EXP116_REPORT,
    DEFAULT_LEADERBOARD_JSON,
    DEFAULT_PHASE_A_INPUT,
    DEFAULT_VISUALS_DIR,
    PHASE_C_ID,
    run_phase_c_publication_layer,
    validate_phase_c_outputs,
    write_phase_c_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase C publication + demo layer")
    parser.add_argument("--phase-a-input", type=Path, default=DEFAULT_PHASE_A_INPUT)
    parser.add_argument("--leaderboard-input", type=Path, default=DEFAULT_LEADERBOARD_JSON)
    parser.add_argument("--exp115-input", type=Path, default=DEFAULT_EXP115_REPORT)
    parser.add_argument("--exp116-input", type=Path, default=DEFAULT_EXP116_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VISUALS_DIR)
    parser.add_argument("--demo-pack", type=Path, default=DEFAULT_DEMO_PACK)
    args = parser.parse_args()

    result = run_phase_c_publication_layer(
        phase_a_path=args.phase_a_input,
        leaderboard_path=args.leaderboard_input,
        exp115_path=args.exp115_input,
        exp116_path=args.exp116_input,
        output_dir=args.output_dir,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    paths = write_phase_c_outputs(result, demo_pack_path=args.demo_pack)
    validation = validate_phase_c_outputs(
        {
            "phase_id": PHASE_C_ID,
            "demos": result["demos"],
            "exactkv_generator_modified": False,
        },
    )

    print(f"phase_id={PHASE_C_ID}")
    print(f"status={result['status']}")
    print(f"demos={len(result['demos'])}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    if result["visual_synthesis"].get("png_outputs"):
        print(f"visual_pngs={len(result['visual_synthesis']['png_outputs'])}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
