#!/usr/bin/env python3
"""7B/8B scale benchmark runner CLI (Phase H+)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmark.scale_7b_benchmark import run_scale_7b_benchmark  # noqa: E402
from exactkv.configs.load_scale_config import DEFAULT_SCALE_CONFIG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 7B/8B scale benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_SCALE_CONFIG)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default=None, help="Override config device")
    args = parser.parse_args()

    summary = run_scale_7b_benchmark(
        config_path=args.config,
        force_deterministic=args.deterministic if args.deterministic else None,
    )
    print(f"phase_id={summary['phase_id']}")
    print(f"status={summary['status']}")
    print(f"device={summary['device']}")
    print(f"total_cells={summary['total_cells']}")
    for name, path in summary.get("outputs", {}).items():
        print(f"wrote_{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
