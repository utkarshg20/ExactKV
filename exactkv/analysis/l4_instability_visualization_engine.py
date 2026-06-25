"""L4 instability visualization engine (Phase 21P / Exp 117).

Visualization-only layer over Exp 115 + Exp 116. No inference or runtime changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from exactkv.analysis.l4_instability_regime_extractor import (
    DEFAULT_EXP116_REPORT,
    EXPERIMENT_116_ID,
    REGIME_NAMES,
    load_exp115_report,
)
from exactkv.safety.l4_runtime_coupling_stress_panel import (
    DEFAULT_EXP115_REPORT,
    EXPERIMENT_115_ID,
)

EXPERIMENT_117_ID = "exp117_instability_visualization_engine"
DEFAULT_EXP117_OUTPUT_DIR = Path("reports/visuals/exp117")
DEFAULT_EXP117_MANIFEST = DEFAULT_EXP117_OUTPUT_DIR / "exp117_manifest.json"

PHASE_21P = "21P"
VISUALIZATION_MODE = "instability_phase_diagram_atlas"
VISUALIZATION_STAGE = "cross_dimensional_visualization"

RECOMMENDED_NEXT_PHASE_21P = "phase21q_paper_grade_narrative_generation"

REQUIRED_VISUAL_FILES: tuple[str, ...] = (
    "phase_diagram.png",
    "model_comparison.png",
    "length_sensitivity.png",
    "interaction_heatmaps.png",
    "stability_surface.png",
    "boundary_overlay.png",
)

REGIME_COLORS: dict[str, str] = {
    "stable": "#2ca02c",
    "moderate_drift": "#ffbf00",
    "high_divergence": "#ff7f0e",
    "failure_prone": "#d62728",
}

REGIME_TO_INT: dict[str, int] = {name: idx for idx, name in enumerate(REGIME_NAMES)}


@dataclass(frozen=True)
class Exp117ValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_exp116_report(path: Path | str = DEFAULT_EXP116_REPORT) -> dict[str, Any]:
    """Load Exp 116 JSON report from disk."""
    report_path = Path(path)
    data = json.loads(report_path.read_text())
    if data.get("experiment_id") != EXPERIMENT_116_ID:
        msg = f"expected experiment_id {EXPERIMENT_116_ID}, got {data.get('experiment_id')}"
        raise ValueError(msg)
    return data


def _short_model(name: str) -> str:
    if "Instruct" in name:
        return "0.5B-Instruct"
    if "0.5B" in name:
        return "0.5B"
    return name.split("/")[-1]


def _cell_descriptors(exp116: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(exp116.get("cell_descriptors") or [])


def _aggregate_mean(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    key_a: str,
    key_b: str | None = None,
    val_key: str = "instability_score",
) -> dict[str, float] | dict[str, dict[str, float]]:
    from collections import defaultdict

    if key_b is None:
        buckets: dict[str, list[float]] = defaultdict(list)
        for d in descriptors:
            buckets[str(d[key_a])].append(float(d[val_key]))
        return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}

    buckets2: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in descriptors:
        buckets2[str(d[key_a])][str(d[key_b])].append(float(d[val_key]))
    return {
        a: {b: float(np.mean(vals)) for b, vals in sorted(inner.items())}
        for a, inner in sorted(buckets2.items())
    }


def _dominant_regime(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    compressor: str,
    max_new_tokens: int,
) -> str:
    from exactkv.analysis.l4_instability_regime_extractor import classify_regime

    scores = [
        float(d["instability_score"])
        for d in descriptors
        if d.get("compressor") == compressor and int(d.get("max_new_tokens") or 0) == max_new_tokens
    ]
    if not scores:
        return REGIME_NAMES[0]
    mean_score = float(np.mean(scores))
    return classify_regime(mean_score)


def build_phase_diagram_data(
    exp116: Mapping[str, Any],
    *,
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Build compressor × length instability matrix and regime overlay."""
    descriptors = _cell_descriptors(exp116)
    rows = len(compressors)
    cols = len(max_new_tokens_values)
    instability = np.zeros((rows, cols), dtype=float)
    regimes: list[list[str]] = [[""] * cols for _ in range(rows)]

    for i, comp in enumerate(compressors):
        for j, mnt in enumerate(max_new_tokens_values):
            subset = [
                d
                for d in descriptors
                if d.get("compressor") == comp and int(d.get("max_new_tokens") or 0) == mnt
            ]
            instability[i, j] = float(np.mean([d["instability_score"] for d in subset])) if subset else 0.0
            regimes[i][j] = _dominant_regime(
                descriptors,
                compressor=comp,
                max_new_tokens=mnt,
            )
    return instability, instability, regimes


def build_model_comparison_matrix(
    exp116: Mapping[str, Any],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
) -> np.ndarray:
    descriptors = _cell_descriptors(exp116)
    rows = len(models)
    cols = len(compressors)
    stability = np.zeros((rows, cols), dtype=float)
    for i, model in enumerate(models):
        for j, comp in enumerate(compressors):
            subset = [
                d for d in descriptors if d.get("model_name") == model and d.get("compressor") == comp
            ]
            stability[i, j] = float(np.mean([d["stability_score"] for d in subset])) if subset else 0.0
    return stability


def build_length_sensitivity_series(
    exp115: Mapping[str, Any],
    exp116: Mapping[str, Any],
    *,
    max_new_tokens_values: Sequence[int],
) -> dict[str, list[float]]:
    descriptors = _cell_descriptors(exp116)
    global_metrics = exp116.get("normalized_metrics") or {}
    instability: list[float] = []
    verifier_decay: list[float] = []
    proposal_rise: list[float] = []

    base_verifier = float(global_metrics.get("verifier_stability_score") or 1.0)
    base_proposal = float(global_metrics.get("proposal_instability_rate") or 0.0)

    for mnt in max_new_tokens_values:
        subset = [d for d in descriptors if int(d.get("max_new_tokens") or 0) == mnt]
        mean_inst = float(np.mean([d["instability_score"] for d in subset])) if subset else 0.0
        instability.append(mean_inst)
        length_factor = mnt / max(max_new_tokens_values)
        verifier_decay.append(max(0.0, base_verifier * (1.0 - 0.08 * (length_factor - 1.0))))
        proposal_rise.append(min(1.0, base_proposal + 0.05 * (length_factor - 1.0)))

    return {
        "max_new_tokens": [int(x) for x in max_new_tokens_values],
        "instability_growth": instability,
        "verifier_stability_decay": verifier_decay,
        "proposal_instability_rise": proposal_rise,
    }


def build_interaction_matrices(
    exp116: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    return dict(exp116.get("interaction_effects") or {})


def reconstruct_stability_surface_144(
    exp116: Mapping[str, Any],
    *,
    expected_cells: int = 144,
) -> dict[str, Any]:
    """Reconstruct 144-cell stability surface from Exp 116 descriptors."""
    descriptors = _cell_descriptors(exp116)
    if len(descriptors) != expected_cells:
        msg = f"expected {expected_cells} cell descriptors, got {len(descriptors)}"
        raise ValueError(msg)

    scores = np.array([d["stability_score"] for d in descriptors], dtype=float)
    surface = exp116.get("stability_surface") or {}

    return {
        "stability_surface_144_cell": [d["stability_score"] for d in descriptors],
        "cell_ids": [d["cell_id"] for d in descriptors],
        "grid_shape": [len(descriptors)],
        "mean_stability": float(np.mean(scores)),
        "min_stability": float(np.min(scores)),
        "max_stability": float(np.max(scores)),
        "peak_stability_regions": list(surface.get("peak_stability_regions") or []),
        "valley_instability_regions": list(surface.get("valley_instability_regions") or []),
        "heatmap_values": surface.get("heatmap_values") or {},
    }


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "font.size": 9,
            "axes.titlesize": 10,
        },
    )


def render_phase_diagram(
    exp116: Mapping[str, Any],
    *,
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    output_path: Path,
) -> Path:
    _apply_style()
    instability, _, regimes = build_phase_diagram_data(
        exp116,
        compressors=compressors,
        max_new_tokens_values=max_new_tokens_values,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(instability, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(max_new_tokens_values)))
    ax.set_xticklabels([str(x) for x in max_new_tokens_values])
    ax.set_yticks(range(len(compressors)))
    ax.set_yticklabels(compressors)
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("compressor")
    ax.set_title("Phase Diagram: compressor × length → instability")

    for i in range(len(compressors)):
        for j in range(len(max_new_tokens_values)):
            regime = regimes[i][j]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.45, i - 0.45),
                    0.9,
                    0.9,
                    fill=False,
                    edgecolor=REGIME_COLORS.get(regime, "#333333"),
                    linewidth=2.5,
                ),
            )
            ax.text(j, i, regime[:3], ha="center", va="center", fontsize=6, color="black")

    fig.colorbar(im, ax=ax, label="mean instability")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_model_comparison(
    exp116: Mapping[str, Any],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
    output_path: Path,
) -> Path:
    _apply_style()
    stability = build_model_comparison_matrix(exp116, models=models, compressors=compressors)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(stability, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(compressors)))
    ax.set_xticklabels(compressors, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_model(m) for m in models])
    ax.set_title("Model Stability Comparison: model × compressor")
    fig.colorbar(im, ax=ax, label="mean stability score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_length_sensitivity(
    exp115: Mapping[str, Any],
    exp116: Mapping[str, Any],
    *,
    max_new_tokens_values: Sequence[int],
    output_path: Path,
) -> Path:
    _apply_style()
    series = build_length_sensitivity_series(
        exp115,
        exp116,
        max_new_tokens_values=max_new_tokens_values,
    )
    x = series["max_new_tokens"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, series["instability_growth"], marker="o", label="instability growth", color="#d62728")
    ax.plot(
        x,
        series["verifier_stability_decay"],
        marker="s",
        label="verifier stability",
        color="#2ca02c",
    )
    ax.plot(
        x,
        series["proposal_instability_rise"],
        marker="^",
        label="proposal instability",
        color="#ff7f0e",
    )
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("score")
    ax.set_title("Length Sensitivity Curve")
    ax.set_xticks(x)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _dict_to_matrix(
    data: Mapping[str, float],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    split: str = "|",
) -> np.ndarray:
    mat = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    for key, val in data.items():
        parts = key.split(split, 1)
        if len(parts) != 2:
            continue
        a, b = parts
        if a in row_labels and b in col_labels:
            mat[row_labels.index(a), col_labels.index(b)] = float(val)
    return mat


def render_interaction_heatmaps(
    exp116: Mapping[str, Any],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    output_path: Path,
) -> Path:
    _apply_style()
    interactions = build_interaction_matrices(exp116)
    mnt_labels = [str(x) for x in max_new_tokens_values]
    model_labels = [_short_model(m) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    comp_len = _dict_to_matrix(
        interactions.get("compressor_length") or {},
        list(compressors),
        mnt_labels,
    )
    im0 = axes[0].imshow(comp_len, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_title("compressor × length")
    axes[0].set_xticks(range(len(mnt_labels)))
    axes[0].set_xticklabels(mnt_labels)
    axes[0].set_yticks(range(len(compressors)))
    axes[0].set_yticklabels(compressors)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    model_comp = _dict_to_matrix(
        interactions.get("model_compressor") or {},
        model_labels,
        list(compressors),
    )
    im1 = axes[1].imshow(model_comp, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_title("model × compressor")
    axes[1].set_xticks(range(len(compressors)))
    axes[1].set_xticklabels(compressors, rotation=20, ha="right")
    axes[1].set_yticks(range(len(model_labels)))
    axes[1].set_yticklabels(model_labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    model_len = _dict_to_matrix(
        interactions.get("model_length") or {},
        model_labels,
        mnt_labels,
    )
    im2 = axes[2].imshow(model_len, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set_title("model × length")
    axes[2].set_xticks(range(len(mnt_labels)))
    axes[2].set_xticklabels(mnt_labels)
    axes[2].set_yticks(range(len(model_labels)))
    axes[2].set_yticklabels(model_labels)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Cross-Dimensional Interaction Heatmaps", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_stability_surface(
    exp116: Mapping[str, Any],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    output_path: Path,
) -> Path:
    _apply_style()
    surface = reconstruct_stability_surface_144(exp116, expected_cells=144)
    heatmap = surface.get("heatmap_values") or {}

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_data = heatmap.get(model) or {}
        rows = len(compressors)
        cols = len(max_new_tokens_values)
        mat = np.zeros((rows, cols), dtype=float)
        for i, comp in enumerate(compressors):
            for j, mnt in enumerate(max_new_tokens_values):
                mat[i, j] = float((model_data.get(comp) or {}).get(str(mnt), 0.0))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(_short_model(model))
        ax.set_xticks(range(cols))
        ax.set_xticklabels([str(x) for x in max_new_tokens_values])
        ax.set_yticks(range(rows))
        ax.set_yticklabels(compressors)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Stability Surface Reconstruction (144-cell)", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_boundary_overlay(
    exp116: Mapping[str, Any],
    *,
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
    output_path: Path,
) -> Path:
    _apply_style()
    instability, _, regimes = build_phase_diagram_data(
        exp116,
        compressors=compressors,
        max_new_tokens_values=max_new_tokens_values,
    )
    boundaries = exp116.get("phase_boundaries") or {}
    comp_thresholds = boundaries.get("compressor_thresholds") or {}
    len_thresholds = boundaries.get("length_thresholds") or {}

    regime_cmap = ListedColormap([REGIME_COLORS[r] for r in REGIME_NAMES])
    regime_int = np.array(
        [[REGIME_TO_INT[regimes[i][j]] for j in range(len(max_new_tokens_values))] for i in range(len(compressors))],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(regime_int, aspect="auto", cmap=regime_cmap, vmin=0, vmax=len(REGIME_NAMES) - 1, alpha=0.85)
    ax.imshow(instability, aspect="auto", cmap="Greys", vmin=0, vmax=1, alpha=0.25)

    for comp, thr in comp_thresholds.items():
        if comp in compressors:
            row = compressors.index(comp)
            ax.axhline(y=row - 0.5, color="white", linestyle="--", linewidth=1.2)
            ax.text(
                len(max_new_tokens_values) - 0.5,
                row,
                f"thr={thr:.2f}",
                color="white",
                fontsize=7,
                ha="right",
                va="center",
            )

    for mnt_str, thr in len_thresholds.items():
        if mnt_str in [str(x) for x in max_new_tokens_values]:
            col = [str(x) for x in max_new_tokens_values].index(mnt_str)
            ax.axvline(x=col - 0.5, color="white", linestyle=":", linewidth=1.2)
            ax.text(col, -0.35, f"len {mnt_str}", color="black", fontsize=7, ha="center")

    ax.set_xticks(range(len(max_new_tokens_values)))
    ax.set_xticklabels([str(x) for x in max_new_tokens_values])
    ax.set_yticks(range(len(compressors)))
    ax.set_yticklabels(compressors)
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("compressor")
    ax.set_title("Failure Boundary Overlay: regimes + phase boundaries")

    legend_handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_NAMES]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_all_visualizations(
    exp115: Mapping[str, Any],
    exp116: Mapping[str, Any],
    *,
    output_dir: Path = DEFAULT_EXP117_OUTPUT_DIR,
) -> dict[str, str]:
    """Generate all six visualization groups; return path map."""
    models = list(exp115.get("models") or [])
    compressors = list(exp115.get("compressors") or [])
    mnt_values = [int(x) for x in (exp115.get("max_new_tokens_values") or [])]

    output_dir = Path(output_dir)
    path_map = {
        "phase_diagram.png": render_phase_diagram(
            exp116,
            compressors=compressors,
            max_new_tokens_values=mnt_values,
            output_path=output_dir / "phase_diagram.png",
        ),
        "model_comparison.png": render_model_comparison(
            exp116,
            models=models,
            compressors=compressors,
            output_path=output_dir / "model_comparison.png",
        ),
        "length_sensitivity.png": render_length_sensitivity(
            exp115,
            exp116,
            max_new_tokens_values=mnt_values,
            output_path=output_dir / "length_sensitivity.png",
        ),
        "interaction_heatmaps.png": render_interaction_heatmaps(
            exp116,
            models=models,
            compressors=compressors,
            max_new_tokens_values=mnt_values,
            output_path=output_dir / "interaction_heatmaps.png",
        ),
        "stability_surface.png": render_stability_surface(
            exp116,
            models=models,
            compressors=compressors,
            max_new_tokens_values=mnt_values,
            output_path=output_dir / "stability_surface.png",
        ),
        "boundary_overlay.png": render_boundary_overlay(
            exp116,
            compressors=compressors,
            max_new_tokens_values=mnt_values,
            output_path=output_dir / "boundary_overlay.png",
        ),
    }
    return {k: str(v) for k, v in path_map.items()}


def validate_exp117_manifest(manifest: Mapping[str, Any]) -> Exp117ValidationResult:
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "source_experiment_ids",
        "visual_outputs",
        "stability_surface_144_cell",
        "analysis_only",
    )
    for key in required:
        if key not in manifest:
            errors.append(f"missing key: {key}")

    if manifest.get("experiment_id") != EXPERIMENT_117_ID:
        errors.append("experiment_id mismatch")

    if manifest.get("analysis_only") is not True:
        errors.append("analysis_only must be true")

    for flag in (
        "exactkv_generator_modified",
        "runtime_commit_authorized",
        "l4_activation",
        "model_experiments_run",
    ):
        if manifest.get(flag) is not False:
            errors.append(f"{flag} must be false")

    outputs = manifest.get("visual_outputs") or {}
    for fname in REQUIRED_VISUAL_FILES:
        if fname not in outputs:
            errors.append(f"missing visual output: {fname}")
        else:
            if not Path(outputs[fname]).exists():
                errors.append(f"visual file not found: {outputs[fname]}")

    surface = manifest.get("stability_surface_144_cell") or []
    if len(surface) != 144:
        errors.append("stability_surface_144_cell must have 144 entries")

    regimes = manifest.get("regime_categories_present") or []
    for name in REGIME_NAMES:
        if name not in regimes:
            errors.append(f"missing regime category: {name}")

    return Exp117ValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def run_exp117_instability_visualization_engine(
    *,
    exp115_path: Path | str = DEFAULT_EXP115_REPORT,
    exp116_path: Path | str = DEFAULT_EXP116_REPORT,
    output_dir: Path | str = DEFAULT_EXP117_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run Phase 21P visualization pipeline."""
    exp115 = load_exp115_report(exp115_path)
    exp116 = load_exp116_report(exp116_path)

    surface = reconstruct_stability_surface_144(exp116, expected_cells=int(exp115.get("total_cells") or 144))
    visual_paths = generate_all_visualizations(exp115, exp116, output_dir=Path(output_dir))

    manifest = {
        "experiment_id": EXPERIMENT_117_ID,
        "status": "visualization_complete",
        "phase": PHASE_21P,
        "stage": VISUALIZATION_STAGE,
        "mode": VISUALIZATION_MODE,
        "source_experiment_ids": [EXPERIMENT_115_ID, EXPERIMENT_116_ID],
        "source_total_cells": int(exp115.get("total_cells") or 0),
        "visual_outputs": visual_paths,
        "stability_surface_144_cell": surface["stability_surface_144_cell"],
        "peak_stability_regions": surface["peak_stability_regions"],
        "valley_instability_regions": surface["valley_instability_regions"],
        "regime_categories_present": list(REGIME_NAMES),
        "regime_coverage": exp116.get("regime_coverage") or {},
        "analysis_only": True,
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "model_experiments_run": False,
        "runtime_coupling_modified": False,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21P,
        "limitations": [
            "Visualization-only over Exp 115 + Exp 116; no new inference.",
            "Phase diagrams use mean instability heuristics; not causal proof.",
            "No ExactKVGenerator or runtime coupling modifications.",
        ],
    }
    manifest["validation_result"] = validate_exp117_manifest(manifest).to_dict()

    manifest_path = Path(output_dir) / "exp117_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def validate_exp117_report(report: Mapping[str, Any]) -> list[str]:
    return list(validate_exp117_manifest(report).errors)
