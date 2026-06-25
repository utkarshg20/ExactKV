"""7B/8B scale benchmark runner (Phase H+).

Delegates matrix execution to Phase A via unified benchmark runner.
Streams incremental cell results to disk. No Phase A–G modifications.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from exactkv.benchmarks.phase_a_scale_benchmark import default_phase_a_prompts
from exactkv.benchmark.unified_benchmark_runner import (
    run_unified_benchmark,
)
from exactkv.configs.load_scale_config import (
    DEFAULT_SCALE_CONFIG,
    load_scale_config,
    map_compressors,
    resolve_device,
)
from exactkv.engine.unified_truth_engine import FirstDivergenceAuthority
from exactkv.platform.public_leaderboard import (
    run_public_leaderboard,
    write_public_leaderboard_outputs,
)
from exactkv.schema.benchmark_schema import BenchmarkCell, cells_from_phase_a_report

PHASE_H_PLUS_SCALE_ID = "phaseH_plus_scale_7b_8b_benchmark"


def expand_prompt_suite(count: int) -> list[dict[str, str]]:
    """Expand base panel to ``count`` deterministic prompt entries."""
    base = default_phase_a_prompts()
    if count <= len(base):
        return base[:count]
    out: list[dict[str, str]] = []
    for i in range(count):
        src = base[i % len(base)]
        out.append(
            {
                "prompt_id": f"p{i:02d}_{src['prompt_id']}",
                "category": src.get("category", "scale_expanded"),
                "prompt": src["prompt"],
            },
        )
    return out


def _append_incremental(path: Path, cell: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(cell) + "\n")


def _render_scale_plots(
    cells: Sequence[BenchmarkCell],
    output_dir: Path,
) -> dict[str, str]:
    """Write divergence heatmap and model comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from collections import defaultdict

        by_model: dict[str, list[float]] = defaultdict(list)
        for c in cells:
            if c.first_divergence is not None:
                short = c.model.split("/")[-1][:20]
                by_model[short].append(float(c.first_divergence))

        if by_model:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = sorted(by_model)
            means = [float(np.mean(by_model[m])) for m in labels]
            ax.bar(range(len(labels)), means, color="#2ca02c")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("mean first_divergence_index")
            ax.set_title("Scale 7B/8B — Model Comparison")
            fig.tight_layout()
            cmp_path = output_dir / "model_comparison.png"
            fig.savefig(cmp_path)
            plt.close(fig)
            result["model_comparison"] = str(cmp_path)

        matrix: dict[tuple[str, str], list[float]] = defaultdict(list)
        for c in cells:
            if c.first_divergence is not None:
                matrix[(c.model, c.compressor)].append(float(c.first_divergence))
        models = sorted({m for m, _ in matrix})
        comps = sorted({c for _, c in matrix})
        if models and comps:
            mat = np.full((len(models), len(comps)), np.nan)
            for i, m in enumerate(models):
                for j, comp in enumerate(comps):
                    vals = matrix.get((m, comp), [])
                    if vals:
                        mat[i, j] = float(np.mean(vals))
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.split("/")[-1][:18] for m in models])
            ax.set_xticks(range(len(comps)))
            ax.set_xticklabels(comps, rotation=45, ha="right")
            ax.set_title("Scale 7B/8B — Divergence Heatmap")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            heat_path = output_dir / "divergence_heatmap.png"
            fig.savefig(heat_path)
            plt.close(fig)
            result["divergence_heatmap"] = str(heat_path)
    except ImportError as exc:
        result["plot_error"] = str(exc)

    return result


def run_scale_7b_benchmark(
    *,
    config_path: Path | str | None = None,
    force_deterministic: bool | None = None,
) -> dict[str, Any]:
    """Execute scale benchmark from YAML/JSON config."""
    cfg = load_scale_config(config_path or DEFAULT_SCALE_CONFIG)
    device = resolve_device(str(cfg.get("device") or "auto"))
    deterministic = bool(force_deterministic if force_deterministic is not None else cfg.get("deterministic_fallback"))
    if device == "cpu" and cfg.get("deterministic_fallback", True):
        deterministic = True

    prompt_count = int(cfg.get("prompts") or 50)
    prompts = expand_prompt_suite(prompt_count)
    compressors = map_compressors(
        list(cfg.get("compressors") or []),
        cfg.get("compressor_map") or {},
    )
    models = list(cfg.get("models") or [])
    mnt_values = [int(x) for x in (cfg.get("max_new_tokens") or [4, 8, 16])]

    output_dir = Path(cfg.get("output_dir") or "reports/scale_7b")
    output_dir.mkdir(parents=True, exist_ok=True)
    incremental_path = output_dir / "raw_incremental.jsonl"

    if incremental_path.exists():
        incremental_path.unlink()

    result = run_unified_benchmark(
        compressors=compressors,
        models=models,
        prompts=prompts,
        max_new_tokens_values=mnt_values,
        device=device,
        dtype=str(cfg.get("dtype") or "float32"),
        deterministic_mode=deterministic,
        local_files_only=bool(cfg.get("local_files_only")),
        draft_len=int(cfg.get("draft_len") or 4),
    )

    authority = FirstDivergenceAuthority()
    enriched_cells: list[dict[str, Any]] = []
    for cell_dict in result.phase_a_report.get("cells") or []:
        full = cell_dict.get("full") or {}
        exactkv = cell_dict.get("exactkv") or {}
        lossy = cell_dict.get("lossy") or {}
        div = authority.compute(
            {"token_ids": list(full.get("output_ids") or [])},
            {"token_ids": list(exactkv.get("output_ids") or lossy.get("output_ids") or [])},
        )
        metrics = dict(cell_dict.get("metrics") or {})
        metrics["first_divergence_index"] = div.canonical_first_divergence_index
        metrics["divergence_type"] = div.divergence_type
        cell_dict["metrics"] = metrics
        enriched_cells.append(cell_dict)
        _append_incremental(incremental_path, cell_dict)

    result.phase_a_report["cells"] = enriched_cells
    schema_cells = cells_from_phase_a_report(result.phase_a_report)

    raw_path = output_dir / "raw.json"
    raw_path.write_text(json.dumps(result.phase_a_report, indent=2) + "\n")

    lb = run_public_leaderboard(phase_a_path=raw_path)
    lb_paths = write_public_leaderboard_outputs(
        lb,
        json_path=output_dir / "leaderboard.json",
        markdown_path=output_dir / "leaderboard.md",
        csv_path=output_dir / "leaderboard.csv",
    )

    plots = _render_scale_plots(schema_cells, output_dir)

    summary = {
        "phase_id": PHASE_H_PLUS_SCALE_ID,
        "status": "scale_benchmark_complete",
        "device": device,
        "deterministic_mode": deterministic,
        "models": models,
        "compressors_public": list(cfg.get("compressors") or []),
        "compressors_phase_a": compressors,
        "prompt_count": prompt_count,
        "total_cells": len(schema_cells),
        "config_path": str(config_path or DEFAULT_SCALE_CONFIG),
        "outputs": {
            "raw": str(raw_path),
            "raw_incremental": str(incremental_path),
            "leaderboard_json": str(lb_paths["leaderboard_json"]),
            **plots,
        },
        "divergence_authority": "FirstDivergenceAuthority",
        "exactkv_generator_modified": False,
        "reproducible_cli_command": "python scripts/exactkv.py run full-scale-7b",
    }
    (output_dir / "scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
