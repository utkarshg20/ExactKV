#!/usr/bin/env python3
"""Phase D runtime KV probe runner — systems credibility instrumentation."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.runtime.kv_probe_layer import (  # noqa: E402
    DEFAULT_LAYER_DRIFT_REPORT,
    DEFAULT_MEMORY_PROFILE_REPORT,
    DEFAULT_RUNTIME_PROBE_REPORT,
    DEFAULT_VISUALS_DIR,
    PHASE_D_ID,
    run_phase_d_runtime_probe,
    validate_phase_d_report,
    write_phase_d_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase D runtime KV probe")
    parser.add_argument(
        "--deterministic-mode",
        action="store_true",
        help="Hash-seeded synthetic probes (no GPU)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-runtime", type=Path, default=DEFAULT_RUNTIME_PROBE_REPORT)
    parser.add_argument("--output-memory", type=Path, default=DEFAULT_MEMORY_PROFILE_REPORT)
    parser.add_argument("--output-layer", type=Path, default=DEFAULT_LAYER_DRIFT_REPORT)
    parser.add_argument("--visuals-dir", type=Path, default=DEFAULT_VISUALS_DIR)
    args = parser.parse_args()

    report = run_phase_d_runtime_probe(
        deterministic_mode=args.deterministic_mode,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    paths = write_phase_d_reports(
        report,
        runtime_path=args.output_runtime,
        memory_path=args.output_memory,
        layer_path=args.output_layer,
        visuals_dir=args.visuals_dir,
    )

    validation = validate_phase_d_report(report)
    print(f"phase_id={PHASE_D_ID}")
    print(f"status={report['status']}")
    print(f"total_cells={report['total_cells']}")
    print(f"deterministic_mode={report['deterministic_mode']}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
