"""ExactKV command-line interface.

Subcommands
-----------
list-compressors  Print all registered compressors with their capabilities.
bench             Run a single-compressor benchmark over a prompt suite.
sweep             Run a multi-compressor × multi-draft-length sweep.
analyze           Analyse an existing JSON report (no model re-run).
report            Render an existing JSON report to a Markdown document.

Usage::

    python -m exactkv list-compressors
    python -m exactkv bench   --model Qwen/Qwen2.5-0.5B --suite smoke ...
    python -m exactkv sweep   --model Qwen/Qwen2.5-0.5B --suite smoke ...
    python -m exactkv analyze --report reports/sweep.json ...
    python -m exactkv report  --report reports/sweep.json --markdown-out docs/sweep.md

Design constraints
------------------
* No timing, latency, throughput, or speedup output.
* Compressor name is validated BEFORE the model is loaded so unknown-compressor
  errors are fast (no wasted GPU/CPU time).
* int4_sim is flagged as simulated in all outputs; no real memory-savings claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Prompt loader helper
# ---------------------------------------------------------------------------

def _load_prompts(suite: str | None, suite_file: str | None) -> list[dict]:
    """Load prompts from --suite-file (overrides --suite) or the named suite."""
    if suite_file:
        from exactkv.benchmarks.prompts import load_prompts
        return load_prompts(suite_file)
    from exactkv.benchmarks.prompts import list_suites, load_suite
    name = suite or "smoke"
    # Delegate to the registry; raises ValueError with helpful message if unknown.
    return load_suite(name)


# ---------------------------------------------------------------------------
# Compressor validation (fast — before model load)
# ---------------------------------------------------------------------------

def _validate_compressors(names: list[str]) -> bool:
    """Return True if all names are registered; print error and return False otherwise."""
    import exactkv.compressors  # registers built-ins
    from exactkv.compressors import list_compressors
    available = list_compressors()
    ok = True
    for name in names:
        if name not in available:
            print(
                f"Error: Unknown compressor {name!r}. "
                f"Available: {available}",
                file=sys.stderr,
            )
            ok = False
    return ok


# ---------------------------------------------------------------------------
# list-compressors
# ---------------------------------------------------------------------------

def _cmd_list_compressors(args: argparse.Namespace) -> int:
    import exactkv.compressors  # registers built-ins
    from exactkv.compressors import get_compressor, list_compressors

    names = list_compressors()
    print(f"Registered compressors ({len(names)} total):\n")

    def _fmt_bits(b: int | None) -> str:
        return "full" if b is None else str(b)

    for name in names:
        comp = get_compressor(name)
        caps = getattr(comp, "capabilities", None)
        print(f"  {name}")
        if caps is not None:
            print(f"    compressor_type           : {caps.compressor_type}")
            print(f"    is_simulated              : {caps.is_simulated}")
            print(f"    supports_real_bytes_claim : {caps.supports_real_bytes_claim}")
            print(f"    key_bit_width             : {_fmt_bits(caps.key_bit_width)}")
            print(f"    value_bit_width           : {_fmt_bits(caps.value_bit_width)}")
            print(f"    asymmetric                : {caps.asymmetric}")
            print(f"    supports_quantization     : {caps.supports_quantization}")
            print(f"    supports_token_dropping   : {caps.supports_token_dropping}")
            if caps.notes:
                note = caps.notes if len(caps.notes) <= 90 else caps.notes[:87] + "..."
                print(f"    notes                     : {note}")
        print()

    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def _cmd_bench(args: argparse.Namespace) -> int:
    # 1. Validate compressor before model load
    if not _validate_compressors([args.compressor]):
        return 1

    # 2. Load prompts
    try:
        prompts = _load_prompts(args.suite, args.suite_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading prompts: {exc}", file=sys.stderr)
        return 1

    # 3. Load model
    from exactkv.runtime.model_runtime import ModelRuntime
    print(f"Loading {args.model} …")
    rt = ModelRuntime(model_name=args.model, device=args.device, dtype=args.dtype)

    # 4. Run
    from exactkv.benchmarks.runner import RunConfig, run_suite
    config = RunConfig(
        compressor_name=args.compressor,
        draft_len=args.draft_len,
        max_new_tokens=args.max_new_tokens,
    )
    suite_label = args.suite or "custom"
    print(
        f"Running bench: {len(prompts)} prompt(s), "
        f"compressor={args.compressor}, draft_len={args.draft_len}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    report = run_suite(rt, prompts, config)

    # 5. Build manifest and write reports
    from exactkv.benchmarks.reports import (
        build_run_manifest,
        write_csv_report,
        write_json_report,
    )
    manifest = build_run_manifest(
        model_name=args.model,
        prompt_suite=suite_label,
        compressor_names=[args.compressor],
        draft_len=args.draft_len,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.json_out:
        write_json_report(report, args.json_out, manifest=manifest)
        print(f"JSON: {args.json_out}")
    if args.csv_out:
        write_csv_report(report, args.csv_out)
        print(f"CSV:  {args.csv_out}")

    # 6. Summary (no timing fields)
    agg = report["aggregate"]
    results = report.get("results", [])
    acc_rates = [r["exactkv"]["acceptance"]["acceptance_rate"] for r in results]
    mean_ar = sum(acc_rates) / max(len(acc_rates), 1) if acc_rates else 0.0
    lossy_div = sum(1 for r in results if not r["lossy"]["token_exact_match"])

    print("\n── Bench summary " + "─" * 40)
    print(f"  Total prompts     : {agg['total_prompts']}")
    print(f"  ExactKV failures  : {agg['exactkv_failures']}")
    print(f"  Mean accept rate  : {mean_ar:.3f}")
    print(f"  Lossy divergences : {lossy_div}")

    # V5 workspace memory note (only when fields are populated from Phase A)
    if results:
        total_fp = results[0].get("memory", {}).get("total_kv_footprint_bytes", 0)
        if total_fp > 0:
            print(
                "  Workspace memory  : stored/materialized/metadata/total "
                "included in report (accounting totals, not measured GPU memory)"
            )

    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _cmd_sweep(args: argparse.Namespace) -> int:
    # 1. Parse and validate compressors
    compressor_names = [c.strip() for c in args.compressors.split(",") if c.strip()]
    if not compressor_names:
        print("Error: --compressors must not be empty.", file=sys.stderr)
        return 1
    if not _validate_compressors(compressor_names):
        return 1

    # 2. Parse draft lengths
    try:
        draft_lengths = [int(d.strip()) for d in args.draft_lengths.split(",") if d.strip()]
    except ValueError as exc:
        print(f"Error: --draft-lengths must be comma-separated integers: {exc}", file=sys.stderr)
        return 1
    if any(d < 1 for d in draft_lengths):
        print("Error: all draft lengths must be >= 1.", file=sys.stderr)
        return 1

    # 3. Load prompts
    try:
        prompts = _load_prompts(args.suite, args.suite_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading prompts: {exc}", file=sys.stderr)
        return 1

    # 4. Load model
    from exactkv.runtime.model_runtime import ModelRuntime
    print(f"Loading {args.model} …")
    rt = ModelRuntime(model_name=args.model, device=args.device, dtype=args.dtype)

    # 5. Run sweep
    from exactkv.benchmarks.sweeps import run_sweep
    total_cells = len(prompts) * len(compressor_names) * len(draft_lengths)
    print(
        f"Running sweep: {total_cells} cell(s) "
        f"({len(prompts)} prompt(s) × {len(compressor_names)} compressor(s) "
        f"× {len(draft_lengths)} draft_len(s))"
    )
    sweep = run_sweep(
        runtime=rt,
        prompts=prompts,
        compressor_names=compressor_names,
        draft_lengths=draft_lengths,
        max_new_tokens=args.max_new_tokens,
        prompt_suite=args.suite or "custom",
    )

    # 6. Write reports
    from exactkv.benchmarks.reports import write_csv_report, write_json_report
    if args.json_out:
        write_json_report(sweep, args.json_out)
        print(f"JSON: {args.json_out}")
    if args.csv_out:
        write_csv_report(sweep, args.csv_out)
        print(f"CSV:  {args.csv_out}")

    # 7. Summary (no timing fields)
    agg = sweep["aggregate"]
    print("\n── Sweep summary " + "─" * 40)
    print(f"  Total runs        : {agg['total_runs']}")
    print(f"  ExactKV failures  : {agg['exactkv_failures']}")
    print(f"  Lossy divergences : {agg['lossy_divergence_count']}")
    print(f"  Mean accept rate  : {agg['mean_acceptance_rate']:.3f}")
    print(f"  Total drafted     : {agg['total_drafted']}")
    print(f"  Total accepted    : {agg['total_accepted']}")
    print(f"  Total rejected    : {agg['total_rejected']}")
    print(f"  Total corrections : {agg['total_corrections']}")

    # V5 workspace memory note (only when fields are populated from Phase A)
    sweep_results = sweep.get("results", [])
    if sweep_results:
        total_fp = sweep_results[0].get("memory", {}).get("total_kv_footprint_bytes", 0)
        if total_fp > 0:
            print(
                "  Workspace memory  : stored/materialized/metadata/total "
                "included in report (accounting totals, not measured GPU memory)"
            )

    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> int:
    from exactkv.analysis.acceptance_tables import (
        build_acceptance_table,
        write_acceptance_table_csv,
    )
    from exactkv.analysis.failure_report import build_failure_report, write_failure_report_json
    from exactkv.benchmarks.reports import load_json_report

    # 1. Load report
    try:
        report = load_json_report(args.report)
    except FileNotFoundError:
        print(f"Error: Report file not found: {args.report}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in {args.report}: {exc}", file=sys.stderr)
        return 1

    # 2. Analyse
    fr = build_failure_report(report)
    table = build_acceptance_table(report)

    # 3. Write outputs
    if args.acceptance_csv:
        write_acceptance_table_csv(table, args.acceptance_csv)
        print(f"Acceptance CSV: {args.acceptance_csv}")
    if args.failure_json:
        write_failure_report_json(fr, args.failure_json)
        print(f"Failure JSON:   {args.failure_json}")

    # 4. Summary
    print("\n── Analysis summary " + "─" * 37)
    print(f"  Status            : {fr['status']}")
    print(f"  ExactKV failures  : {fr['exactkv_failure_count']}")
    print(f"  Lossy divergences : {fr['lossy_divergence_count']}")

    return 0 if fr["status"] == "pass" else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _cmd_report(args: argparse.Namespace) -> int:
    """Render an existing JSON report to a Markdown document.

    Does not re-run the model.  No timing, throughput, latency, or speedup
    output is produced.
    """
    from exactkv.benchmarks.reports import load_json_report
    from exactkv.analysis.failure_report import build_failure_report
    from exactkv.reporting.markdown import write_markdown_report

    # 1. Load report
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report file not found: {report_path}", file=sys.stderr)
        return 1
    try:
        report = load_json_report(report_path)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in {report_path}: {exc}", file=sys.stderr)
        return 1

    # 2. Render and write Markdown
    out_path = Path(args.markdown_out)
    include_examples = not getattr(args, "no_examples", False)
    write_markdown_report(
        report,
        path=out_path,
        title=args.title or None,
        include_examples=include_examples,
        max_examples=args.max_examples,
    )

    # 3. Short summary (no timing fields)
    fr = build_failure_report(report)
    print(f"Input report  : {report_path}")
    print(f"Markdown out  : {out_path}")
    print(f"ExactKV failures  : {fr['exactkv_failure_count']}")
    print(f"Lossy divergences : {fr['lossy_divergence_count']}")
    print(f"Workspace memory  : included in Markdown report")

    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exactkv",
        description=(
            "ExactKV CLI — correctness-first KV-cache compression benchmark.\n"
            "Reports acceptance and exactness. Does NOT report throughput or speedup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── list-compressors ──────────────────────────────────────────────────
    sub.add_parser(
        "list-compressors",
        help="List registered compressors with their capabilities.",
    )

    # ── Common model / prompt arguments ──────────────────────────────────
    def _add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                       help="HuggingFace model name or path (default: Qwen/Qwen2.5-0.5B)")
        p.add_argument("--device", default="auto",
                       help="Device: cpu, cuda, auto (default: auto)")
        p.add_argument("--dtype", default="float32",
                       choices=["float32", "float16", "bfloat16", "auto"],
                       help="Model dtype (default: float32)")
        p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    def _add_prompt_args(p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group()
        g.add_argument(
            "--suite", default="smoke",
            help=(
                "Named prompt suite: smoke (fast CI, default), core, "
                "structured, code, stress. "
                "Use --suite-file for a custom JSONL file."
            ),
        )
        g.add_argument("--suite-file", dest="suite_file", metavar="PATH",
                       help="Path to a custom JSONL prompt file (overrides --suite)")

    def _add_output_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json-out", dest="json_out", metavar="PATH",
                       help="Write JSON report to this path (parent dirs created automatically)")
        p.add_argument("--csv-out", dest="csv_out", metavar="PATH",
                       help="Write CSV report to this path (parent dirs created automatically)")

    # ── bench ─────────────────────────────────────────────────────────────
    p_bench = sub.add_parser(
        "bench",
        help="Run a single-compressor benchmark over a prompt suite.",
    )
    _add_model_args(p_bench)
    _add_prompt_args(p_bench)
    _add_output_args(p_bench)
    p_bench.add_argument("--compressor", default="int8",
                         help="Compressor name (default: int8)")
    p_bench.add_argument("--draft-len", dest="draft_len", type=int, default=8,
                         help="Draft length (default: 8)")
    p_bench.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16,
                         help="Max new tokens per prompt (default: 16)")

    # ── sweep ─────────────────────────────────────────────────────────────
    p_sweep = sub.add_parser(
        "sweep",
        help="Run a multi-compressor × multi-draft-length sweep.",
    )
    _add_model_args(p_sweep)
    _add_prompt_args(p_sweep)
    _add_output_args(p_sweep)
    p_sweep.add_argument("--compressors", default="noop,int8",
                         help="Comma-separated compressor names (default: noop,int8)")
    p_sweep.add_argument("--draft-lengths", dest="draft_lengths", default="4,8",
                         help="Comma-separated draft lengths (default: 4,8)")
    p_sweep.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16,
                         help="Max new tokens per prompt (default: 16)")

    # ── analyze ───────────────────────────────────────────────────────────
    p_analyze = sub.add_parser(
        "analyze",
        help="Analyse an existing JSON report without re-running the model.",
    )
    p_analyze.add_argument("--report", required=True, metavar="PATH",
                           help="Path to a JSON report written by bench or sweep.")
    p_analyze.add_argument("--acceptance-csv", dest="acceptance_csv", metavar="PATH",
                           help="Write acceptance table to this CSV path.")
    p_analyze.add_argument("--failure-json", dest="failure_json", metavar="PATH",
                           help="Write failure report to this JSON path.")

    # ── report ────────────────────────────────────────────────────────────
    p_report = sub.add_parser(
        "report",
        help=(
            "Render an existing JSON report to a Markdown document. "
            "Does not re-run the model. "
            "Reports exactness and acceptance — no speedup or throughput output."
        ),
    )
    p_report.add_argument("--report", required=True, metavar="PATH",
                          help="Path to a JSON report written by bench or sweep.")
    p_report.add_argument("--markdown-out", dest="markdown_out", required=True,
                          metavar="PATH",
                          help="Output Markdown file path (parent dirs created automatically).")
    p_report.add_argument("--title", default=None, metavar="TEXT",
                          help="Report title (default: 'ExactKV Benchmark Report').")
    p_report.add_argument("--max-examples", dest="max_examples", type=int, default=3,
                          metavar="INT",
                          help="Maximum examples per example section (default: 3).")
    p_report.add_argument("--no-examples", dest="no_examples", action="store_true",
                          help="Disable lossy-divergence and rejection example blocks.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Integer exit code (0 = success, non-zero = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "list-compressors":
            return _cmd_list_compressors(args)
        elif args.command == "bench":
            return _cmd_bench(args)
        elif args.command == "sweep":
            return _cmd_sweep(args)
        elif args.command == "analyze":
            return _cmd_analyze(args)
        elif args.command == "report":
            return _cmd_report(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
