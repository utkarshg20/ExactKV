#!/usr/bin/env python3
"""ExactKV unified CLI (Phase H).

One-command reproducibility for benchmark, leaderboard, publication, and plots.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_PLOTS_DIR = Path("reports/plots")
DEFAULT_BENCHMARK_JSON = Path("reports/benchmark.json")
DEFAULT_LEADERBOARD_JSON = Path("reports/leaderboard.json")


def _cmd_run_benchmark(args: argparse.Namespace) -> int:
    from exactkv.benchmark.unified_benchmark_runner import (  # noqa: PLC0415
        run_unified_benchmark,
        write_benchmark_outputs,
    )

    result = run_unified_benchmark(
        compressors=args.compressors.split(",") if args.compressors else None,
        models=args.models.split(",") if args.models else None,
        device=args.device,
        dtype=args.dtype,
        deterministic_mode=args.deterministic,
        local_files_only=args.local_files_only,
    )
    paths = write_benchmark_outputs(result)
    print(f"status={result.benchmark_run.status}")
    print(f"total_cells={result.benchmark_run.total_cells}")
    print(f"config_hash={result.benchmark_run.config_hash}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    if not result.validation.get("valid", True):
        print("validation_errors:", result.validation.get("errors"))
        return 1
    return 0


def _cmd_run_leaderboard(args: argparse.Namespace) -> int:
    from exactkv.platform.public_leaderboard import (  # noqa: PLC0415
        run_public_leaderboard,
        validate_public_leaderboard,
        write_public_leaderboard_outputs,
    )

    benchmark_path = args.benchmark_input or (
        DEFAULT_BENCHMARK_JSON if DEFAULT_BENCHMARK_JSON.is_file() else None
    )
    report = run_public_leaderboard(
        benchmark_path=benchmark_path,
        phase_a_path=args.phase_a_input,
        filter_model=args.filter_model,
        filter_compressor=args.filter_compressor,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    paths = write_public_leaderboard_outputs(report)
    validation = validate_public_leaderboard(report)
    print(f"status={report['status']}")
    print(f"ranked_entries={len(report.get('entries') or [])}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


def _cmd_run_full(args: argparse.Namespace) -> int:
    from exactkv.benchmark.unified_benchmark_runner import (  # noqa: PLC0415
        run_unified_benchmark,
        write_benchmark_outputs,
    )
    from exactkv.engine.unified_truth_engine import (  # noqa: PLC0415
        run_phase_g_unified_truth_engine,
        write_phase_g_outputs,
    )
    from exactkv.platform.public_leaderboard import (  # noqa: PLC0415
        run_public_leaderboard,
        write_public_leaderboard_outputs,
    )

    bench = run_unified_benchmark(
        device=args.device,
        deterministic_mode=args.deterministic,
        local_files_only=args.local_files_only,
    )
    write_benchmark_outputs(bench)

    lb = run_public_leaderboard(benchmark_path=DEFAULT_BENCHMARK_JSON)
    lb["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_public_leaderboard_outputs(lb)

    truth = run_phase_g_unified_truth_engine(
        phase_a_path=Path("reports/phaseA_benchmark.json")
        if not args.deterministic
        else DEFAULT_BENCHMARK_JSON,
    )
    truth["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_phase_g_outputs(truth)

    _cmd_plot_all(args)
    print("full_pipeline=complete")
    return 0


def _cmd_run_full_scale_7b(args: argparse.Namespace) -> int:
    from exactkv.benchmark.scale_7b_benchmark import run_scale_7b_benchmark  # noqa: PLC0415
    from exactkv.configs.load_scale_config import DEFAULT_SCALE_CONFIG  # noqa: PLC0415

    summary = run_scale_7b_benchmark(
        config_path=args.config or DEFAULT_SCALE_CONFIG,
        force_deterministic=True if args.deterministic else None,
    )
    print(f"phase_id={summary['phase_id']}")
    print(f"status={summary['status']}")
    print(f"total_cells={summary['total_cells']}")
    for name, path in summary.get("outputs", {}).items():
        print(f"wrote_{name}={path}")
    return 0


def _cmd_run_publish(args: argparse.Namespace) -> int:
    from exactkv.platform.release_packager import build_release_package  # noqa: PLC0415

    manifest = build_release_package(release_dir=args.release_dir)
    print(f"release_dir={manifest['release_dir']}")
    for f in manifest.get("files", []):
        print(f"wrote={args.release_dir / f}")
    return 0


def _cmd_run_spectralquant_check(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    from exactkv.adapters.spectralquant_real_adapter import (  # noqa: PLC0415
        SpectralQuantRealKVCompressor,
        spectralquant_available,
    )

    available = spectralquant_available()
    comp = SpectralQuantRealKVCompressor()
    k = torch.randn(1, 4, 16, 32)
    v = torch.randn(1, 4, 16, 32)
    out = comp.compress(k, v, seed=0)
    print(f"spectralquant_available={available}")
    print(f"approximation_mode={out.metadata.get('approximation_mode')}")
    print(f"compression_ratio={out.metadata.get('compression_ratio')}")
    print(f"supports_gpu={comp.supports_gpu()}")
    return 0


def _cmd_run_shard_analysis(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    from exactkv.adapters.shard_real_adapter import ShardRealKVCompressor  # noqa: PLC0415

    comp = ShardRealKVCompressor()
    k = torch.randn(1, 4, 16, 32)
    v = torch.randn(1, 4, 16, 32)
    out = comp.compress(k, v, seed=0)
    print(f"stability_score_estimate={out.metadata.get('stability_score_estimate')}")
    print(f"divergence_risk_estimate={out.metadata.get('divergence_risk_estimate')}")
    print(f"acceptance_proxy_score={out.metadata.get('acceptance_proxy_score')}")
    print(f"probe_only={out.metadata.get('probe_only')}")
    return 0


def _cmd_export_paper(args: argparse.Namespace) -> int:
    from exactkv.benchmarks.publication_layer import (  # noqa: PLC0415
        run_phase_c_publication_layer,
        write_phase_c_outputs,
    )

    result = run_phase_c_publication_layer(
        phase_a_path=args.phase_a_input or Path("reports/phaseA_benchmark.json"),
        leaderboard_path=args.leaderboard_input or DEFAULT_LEADERBOARD_JSON,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    paths = write_phase_c_outputs(result)
    print(f"status={result['status']}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    return 0


def _cmd_plot_all(args: argparse.Namespace) -> int:
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    sources = [
        Path("reports/phaseG_divergence_map.png"),
        Path("reports/visuals/phaseD/layer_drift_heatmap.png"),
        Path("reports/visuals/phaseD/divergence_compression_curve.png"),
        Path("reports/visuals/phaseD/memory_proxy_bars.png"),
        Path("reports/visuals/phaseC/phaseC_first_divergence_map.png"),
        Path("reports/visuals/exp117/exp117_phase_diagram.png"),
    ]
    for src in sources:
        if src.is_file():
            dst = plots_dir / src.name
            shutil.copy2(src, dst)
            copied += 1

    manifest = {
        "plots_dir": str(plots_dir),
        "copied": copied,
        "sources": [str(s) for s in sources if s.is_file()],
    }
    (plots_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"plots_copied={copied}")
    print(f"wrote_manifest={plots_dir / 'plots_manifest.json'}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exactkv", description="ExactKV Phase H unified CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run benchmark workflows")
    run_sub = run.add_subparsers(dest="run_cmd", required=True)

    bench = run_sub.add_parser("benchmark", help="Run unified benchmark matrix")
    bench.add_argument("--deterministic", action="store_true", help="Hash-seeded synthetic cells")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--dtype", default="float32")
    bench.add_argument("--compressors", default=None, help="Comma-separated compressor names")
    bench.add_argument("--models", default=None, help="Comma-separated model names")
    bench.add_argument("--local-files-only", action="store_true")
    bench.set_defaults(func=_cmd_run_benchmark)

    lb = run_sub.add_parser("leaderboard", help="Build public leaderboard")
    lb.add_argument("--benchmark-input", type=Path, default=None)
    lb.add_argument("--phase-a-input", type=Path, default=None)
    lb.add_argument("--filter-model", default=None)
    lb.add_argument("--filter-compressor", default=None)
    lb.set_defaults(func=_cmd_run_leaderboard)

    full = run_sub.add_parser("full", help="Benchmark + leaderboard + Phase G + plots")
    full.add_argument("--deterministic", action="store_true")
    full.add_argument("--device", default="cpu")
    full.add_argument("--local-files-only", action="store_true")
    full.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    full.set_defaults(func=_cmd_run_full)

    scale = run_sub.add_parser("full-scale-7b", help="7B/8B scale benchmark from YAML config")
    scale.add_argument("--config", type=Path, default=None)
    scale.add_argument("--deterministic", action="store_true")
    scale.set_defaults(func=_cmd_run_full_scale_7b)

    publish = run_sub.add_parser("publish", help="Build public release package")
    publish.add_argument("--release-dir", type=Path, default=Path("reports/public_release"))
    publish.set_defaults(func=_cmd_run_publish)

    sq = run_sub.add_parser("spectralquant-check", help="Check SpectralQuant real adapter")
    sq.set_defaults(func=_cmd_run_spectralquant_check)

    shard = run_sub.add_parser("shard-analysis", help="Run Shard probe analysis")
    shard.set_defaults(func=_cmd_run_shard_analysis)

    export = sub.add_parser("export", help="Export publication artifacts")
    export_sub = export.add_subparsers(dest="export_cmd", required=True)
    paper = export_sub.add_parser("paper", help="Export paper/blog/social from reports")
    paper.add_argument("--phase-a-input", type=Path, default=None)
    paper.add_argument("--leaderboard-input", type=Path, default=None)
    paper.set_defaults(func=_cmd_export_paper)

    plot = sub.add_parser("plot", help="Plot utilities")
    plot_sub = plot.add_subparsers(dest="plot_cmd", required=True)
    all_plots = plot_sub.add_parser("all", help="Collect all standard plots")
    all_plots.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    all_plots.set_defaults(func=_cmd_plot_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
