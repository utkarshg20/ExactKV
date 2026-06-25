"""Unified benchmark runner (Phase H).

Single entry point for ExactKV compression benchmarks. Delegates inference to
the existing Phase A engine without modifying it. Uses Phase G divergence
authority definitions when enriching cells from disk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.benchmarks.phase_a_scale_benchmark import (
    PHASE_A_ALL_COMPRESSORS,
    PHASE_A_MAX_NEW_TOKENS,
    PHASE_A_MODELS,
    default_phase_a_prompts,
    run_phase_a_scale_benchmark,
    validate_phase_a_report,
)
from exactkv.engine.unified_truth_engine import FirstDivergenceAuthority
from exactkv.schema.benchmark_schema import (
    BenchmarkCell,
    BenchmarkConfig,
    BenchmarkRun,
    aggregate_failure_rate,
    cells_from_phase_a_report,
    resolve_git_commit,
)

PHASE_H_BENCHMARK_ID = "phaseH_unified_benchmark_runner"
DEFAULT_BENCHMARK_JSON = Path("reports/benchmark.json")
DEFAULT_BENCHMARK_MD = Path("reports/benchmark.md")


@dataclass
class UnifiedBenchmarkResult:
    """Phase H benchmark output contract."""

    benchmark_run: BenchmarkRun
    phase_a_report: dict[str, Any]
    compressor_registry: list[str]
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": PHASE_H_BENCHMARK_ID,
            "status": self.benchmark_run.status,
            "benchmark_run": self.benchmark_run.to_dict(),
            "compressor_registry": self.compressor_registry,
            "source_phase_a_id": self.phase_a_report.get("phase_id"),
            "validation": self.validation,
            "exactkv_generator_modified": False,
            "runtime_commit_authorized": False,
            "reproducible_cli_command": "python scripts/exactkv.py run benchmark",
        }


def _authority_enrich_phase_a_report(report: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical divergence authority fields without changing Phase A logic."""
    authority = FirstDivergenceAuthority()
    for cell in report.get("cells") or []:
        full = cell.get("full") or {}
        exactkv = cell.get("exactkv") or {}
        lossy = cell.get("lossy") or {}
        baseline = {"token_ids": list(full.get("output_ids") or [])}
        compressed = {"token_ids": list(exactkv.get("output_ids") or lossy.get("output_ids") or [])}
        div = authority.compute(baseline, compressed)
        metrics = dict(cell.get("metrics") or {})
        metrics["first_divergence_index"] = div.canonical_first_divergence_index
        metrics["divergence_type"] = div.divergence_type
        cell["metrics"] = metrics
    return report


def run_unified_benchmark(
    *,
    compressors: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    prompts: Sequence[Mapping[str, str]] | None = None,
    max_new_tokens_values: Sequence[int] | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    deterministic_mode: bool = False,
    draft_len: int = 4,
    local_files_only: bool = False,
) -> UnifiedBenchmarkResult:
    """Execute unified benchmark matrix via Phase A backend."""
    from exactkv.registry.compressor_registry import list_compressors  # noqa: PLC0415

    prompt_list = list(prompts or default_phase_a_prompts())
    config = BenchmarkConfig(
        models=tuple(models or PHASE_A_MODELS),
        compressors=tuple(compressors or PHASE_A_ALL_COMPRESSORS),
        prompt_ids=tuple(p["prompt_id"] for p in prompt_list),
        max_new_tokens_values=tuple(max_new_tokens_values or PHASE_A_MAX_NEW_TOKENS),
        device=device,
        dtype=dtype,
        deterministic_mode=deterministic_mode,
        draft_len=draft_len,
    )

    phase_a_report = run_phase_a_scale_benchmark(
        models=config.models,
        compressors=config.compressors,
        prompts=prompt_list,
        max_new_tokens_values=config.max_new_tokens_values,
        device=device,
        dtype=dtype,
        deterministic_mode=deterministic_mode,
        draft_len=draft_len,
        local_files_only=local_files_only,
    )
    phase_a_report = _authority_enrich_phase_a_report(phase_a_report)
    validation = validate_phase_a_report(phase_a_report)

    cells = cells_from_phase_a_report(phase_a_report)
    run = BenchmarkRun(
        run_id=f"benchmark_{config.config_hash()}",
        config=config,
        config_hash=config.config_hash(),
        git_commit=resolve_git_commit(),
        cells=cells,
        status=str(phase_a_report.get("status") or "benchmark_complete"),
        exactkv_failure_rate=aggregate_failure_rate(cells),
        total_cells=len(cells),
    )

    return UnifiedBenchmarkResult(
        benchmark_run=run,
        phase_a_report=phase_a_report,
        compressor_registry=list_compressors(),
        validation=validation.to_dict() if hasattr(validation, "to_dict") else dict(validation),
    )


def load_unified_benchmark_from_disk(path: Path | str = DEFAULT_BENCHMARK_JSON) -> UnifiedBenchmarkResult:
    """Load prior benchmark run from ``reports/benchmark.json``."""
    from exactkv.registry.compressor_registry import list_compressors  # noqa: PLC0415

    data = json.loads(Path(path).read_text())
    run_data = data.get("benchmark_run") or {}
    cfg = run_data.get("config") or {}
    config = BenchmarkConfig(
        models=tuple(cfg.get("models") or ()),
        compressors=tuple(cfg.get("compressors") or ()),
        prompt_ids=tuple(cfg.get("prompt_ids") or ()),
        max_new_tokens_values=tuple(cfg.get("max_new_tokens_values") or ()),
        device=str(cfg.get("device") or "cpu"),
        dtype=str(cfg.get("dtype") or "float32"),
        deterministic_mode=bool(cfg.get("deterministic_mode")),
        draft_len=int(cfg.get("draft_len") or 4),
    )
    cells = [BenchmarkCell(**c) for c in run_data.get("cells") or []]
    run = BenchmarkRun(
        run_id=str(run_data.get("run_id") or ""),
        config=config,
        config_hash=str(run_data.get("config_hash") or config.config_hash()),
        git_commit=run_data.get("git_commit"),
        cells=cells,
        status=str(run_data.get("status") or data.get("status") or ""),
        exactkv_failure_rate=float(run_data.get("exactkv_failure_rate") or 0.0),
        total_cells=int(run_data.get("total_cells") or len(cells)),
    )
    return UnifiedBenchmarkResult(
        benchmark_run=run,
        phase_a_report={},
        compressor_registry=list_compressors(),
        validation=data.get("validation") or {},
    )


def render_benchmark_markdown(result: UnifiedBenchmarkResult) -> str:
    run = result.benchmark_run
    lines = [
        "# ExactKV Unified Benchmark",
        "",
        f"**Status:** {run.status}",
        f"**Run ID:** `{run.run_id}`",
        f"**Config hash:** `{run.config_hash}`",
        f"**Git commit:** `{run.git_commit or 'unknown'}`",
        f"**Total cells:** {run.total_cells}",
        f"**ExactKV failure rate:** {run.exactkv_failure_rate:.4f}",
        "",
        "## Compressors",
        "",
        ", ".join(f"`{c}`" for c in run.config.compressors),
        "",
        "## Models",
        "",
        ", ".join(f"`{m}`" for m in run.config.models),
        "",
        "## Aggregate acceptance (by compressor)",
        "",
        "| Compressor | Mean acceptance | Mean first divergence | Failures |",
        "|------------|----------------:|----------------------:|---------:|",
    ]
    from collections import defaultdict

    by_comp: dict[str, list[BenchmarkCell]] = defaultdict(list)
    for cell in run.cells:
        by_comp[cell.compressor].append(cell)

    for comp in sorted(by_comp):
        rows = by_comp[comp]
        acc = sum(c.acceptance_rate for c in rows) / len(rows)
        divs = [c.first_divergence for c in rows if c.first_divergence is not None]
        mean_div = sum(divs) / len(divs) if divs else float("nan")
        fails = sum(1 for c in rows if c.exactkv_failure)
        div_s = f"{mean_div:.2f}" if divs else "n/a"
        lines.append(f"| `{comp}` | {acc:.3f} | {div_s} | {fails} |")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python scripts/exactkv.py run benchmark --deterministic",
            "```",
            "",
        ],
    )
    return "\n".join(lines)


def write_benchmark_outputs(
    result: UnifiedBenchmarkResult,
    *,
    json_path: Path | str = DEFAULT_BENCHMARK_JSON,
    markdown_path: Path | str = DEFAULT_BENCHMARK_MD,
) -> dict[str, Path]:
    json_out = Path(json_path)
    md_out = Path(markdown_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    md_out.write_text(render_benchmark_markdown(result))
    return {"benchmark_json": json_out, "benchmark_md": md_out}
