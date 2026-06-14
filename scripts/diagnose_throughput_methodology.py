#!/usr/bin/env python3
"""Print throughput methodology checklist and optional diagnostic JSON stub.

Does **not** run model inference by default. Phase 11I methodology helper only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.throughput_contract import (
    ThroughputClaimStatus,
    build_default_diagnostic_plan,
    build_diagnostic_result_stub,
    validate_throughput_benchmark_plan,
    validate_throughput_diagnostic_result,
)

_CHECKLIST = (
    "Exactness gate (exactkv_failures == 0) before timing interpretation",
    "Warmup runs before measured trials",
    "CUDA/device synchronization when timing GPU paths",
    "Explicit baseline arm (e.g. full_greedy)",
    "Named metrics: TOKENS_PER_SECOND, TOTAL_SECONDS, VERIFY_SECONDS, DECODE_SECONDS",
    "Sample count >= 3 for any claim-ready panel",
    "Hardware metadata recorded (GPU, dtype, torch, transformers)",
    "Negative or neutral results reported — hide_negative_results must stay False",
    "No speedup claim unless claim_status == CLAIM_ALLOWED",
    "Placeholder modes (batched/serving/remote) are not runtime support",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ExactKV throughput methodology checklist (Phase 11I). No inference."
    )
    parser.add_argument(
        "--stub",
        metavar="PATH",
        help="Write blocked diagnostic JSON stub to PATH (no model inference)",
    )
    args = parser.parse_args(argv)

    plan = build_default_diagnostic_plan()
    plan_errors = validate_throughput_benchmark_plan(plan)
    if plan_errors:
        print("Plan validation errors:", file=sys.stderr)
        for err in plan_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("ExactKV throughput benchmark methodology (Phase 11I)")
    print(f"Default claim_status: {ThroughputClaimStatus.DIAGNOSTIC_ONLY.value}")
    print(f"Default mode: {plan.mode.value}")
    print()
    print("Required checks before any throughput/latency/speedup claim:")
    for item in _CHECKLIST:
        print(f"  [ ] {item}")
    print()
    print(
        "Current ExactKV timing diagnostics do not support a speedup claim "
        "(see Exp 030 diagnostic panel)."
    )

    if args.stub:
        stub = build_diagnostic_result_stub(negative_vs_baseline=True)
        stub_errors = validate_throughput_diagnostic_result(stub)
        if stub_errors:
            print("Stub validation errors:", file=sys.stderr)
            for err in stub_errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        path = Path(args.stub)
        path.write_text(json.dumps(stub.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote diagnostic stub to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
