#!/usr/bin/env python3
"""ExactKV live terminal demo — single-act streaming crash test.

Default: one continuous in-place replay (drift → reject/commit → match).
Optional `--mode cases` for site case-study carousel.
Launch cut: `--speed hero` (pharmacy semantic crash + 6%→90% scale punch).

Usage::

    python3 scripts/exactkv_live_demo.py --speed hero
    python3 scripts/exactkv_live_demo.py --speed launch
    python3 scripts/exactkv_live_demo.py --mode cases --speed cinematic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.demo.case_study_loader import (  # noqa: E402
    DEFAULT_CASE_STUDIES_JSON,
    load_case_studies,
    select_cases,
)
from exactkv.demo.live_terminal import run_live_demo  # noqa: E402
from exactkv.demo.streaming_demo import run_streaming_demo  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ExactKV live terminal demo")
    p.add_argument(
        "--mode",
        choices=("stream", "cases"),
        default="stream",
        help="stream=single in-place crash test (default); cases=site carousel",
    )
    p.add_argument(
        "--scenario",
        choices=("weather", "hero"),
        default=None,
        help="stream scenario (default: weather; auto-hero when --speed hero)",
    )
    p.add_argument("--json", type=Path, default=DEFAULT_CASE_STUDIES_JSON)
    p.add_argument("--case", metavar="PROMPT_ID")
    p.add_argument("--compressor", help="with --case (default int4_sim)")
    p.add_argument("--index", type=int)
    p.add_argument("--list-cases", action="store_true")
    p.add_argument(
        "--speed",
        choices=("instant", "fast", "cinematic", "launch", "social", "hero", "default"),
        default="launch",
    )
    p.add_argument("--no-delay", action="store_true")
    p.add_argument("--plain", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Preserve hero scenario even when --no-delay forces instant pacing.
    scenario = args.scenario
    if scenario is None and args.speed == "hero":
        scenario = "hero"
    speed = "instant" if args.no_delay else args.speed

    if args.case is not None or args.index is not None:
        args.mode = "cases"

    if args.mode == "stream":
        run_streaming_demo(
            no_delay=args.no_delay,
            plain=args.plain,
            speed=speed,
            scenario=scenario,
        )
        return 0

    all_cases = load_case_studies(args.json)
    if args.list_cases:
        for i, c in enumerate(all_cases):
            print(f"{i:2d}  {c.prompt_id:<40}  {c.title}  [{c.compressor_name}]")
        return 0

    if args.case:
        try:
            cases = select_cases(all_cases, prompt_id=args.case, compressor=args.compressor)
        except KeyError:
            print(f"unknown prompt_id: {args.case}", file=sys.stderr)
            return 1
    elif args.index is not None:
        cases = select_cases(all_cases, index=args.index)
    else:
        cases = select_cases(all_cases)

    run_live_demo(
        cases,
        no_delay=args.no_delay,
        plain=args.plain,
        speed=speed if speed not in {"launch", "hero"} else "cinematic",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
