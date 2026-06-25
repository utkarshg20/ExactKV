"""L4 instability regime extraction (Phase 21O / Exp 116).

Post-hoc analysis over Exp 115 stress panel outputs only. No inference, no
runtime coupling changes, no generator modifications.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exactkv.safety.l4_runtime_coupling_stress_panel import (
    DEFAULT_EXP115_REPORT,
    DIVERGENCE_DECISIONS,
    EXPERIMENT_115_ID,
)

EXPERIMENT_116_ID = "exp116_instability_regime_analysis"
DEFAULT_EXP116_REPORT = Path(
    "reports/experiment_116_instability_regime_analysis.json",
)
PHASE_21O = "21O"
ANALYSIS_MODE = "instability_regime_extraction"
ANALYSIS_STAGE = "failure_topology_mapping"

RECOMMENDED_NEXT_PHASE_21O = "phase21p_instability_visualization_engine"

REGIME_STABLE = "stable"
REGIME_MODERATE_DRIFT = "moderate_drift"
REGIME_HIGH_DIVERGENCE = "high_divergence"
REGIME_FAILURE_PRONE = "failure_prone"

REGIME_NAMES: tuple[str, ...] = (
    REGIME_STABLE,
    REGIME_MODERATE_DRIFT,
    REGIME_HIGH_DIVERGENCE,
    REGIME_FAILURE_PRONE,
)

DIVERGENCE_DECISION_SET = DIVERGENCE_DECISIONS

ANALYSIS_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "exactkv_generator_modified",
    "runtime_commit_authorized",
    "l4_activation",
    "model_experiments_run",
)


@dataclass(frozen=True)
class CellDescriptor:
    cell_id: str
    model_name: str
    prompt_id: str
    compressor: str
    max_new_tokens: int
    instability_score: float
    stability_score: float
    decision: str
    prefix_length: int
    proposal_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp116ValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_exp115_report(path: Path | str = DEFAULT_EXP115_REPORT) -> dict[str, Any]:
    """Load Exp 115 JSON report from disk."""
    report_path = Path(path)
    data = json.loads(report_path.read_text())
    if data.get("experiment_id") != EXPERIMENT_115_ID:
        msg = f"expected experiment_id {EXPERIMENT_115_ID}, got {data.get('experiment_id')}"
        raise ValueError(msg)
    return data


def _cell_id(cell: Mapping[str, Any]) -> str:
    return (
        f"{cell.get('model_name')}|{cell.get('prompt_id')}|"
        f"{cell.get('compressor')}|{cell.get('max_new_tokens')}"
    )


def _group_key(cell: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(cell.get("prompt_id") or ""),
        str(cell.get("compressor") or ""),
        int(cell.get("max_new_tokens") or 0),
    )


def _primary_trace(cell: Mapping[str, Any]) -> dict[str, Any] | None:
    records = cell.get("trace_records") or []
    if not records:
        return None
    rec = records[0]
    proposal = rec.get("proposal_tokens") or []
    return {
        "decision": str(rec.get("decision") or ""),
        "prefix_length": int(rec.get("prefix_length") or 0),
        "proposal_token_count": len(proposal),
        "mismatch_index": rec.get("mismatch_index"),
    }


def normalize_exp115_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Extract normalized global metrics from Exp 115 report."""
    cross = report.get("cross_model_metrics") or {}
    return {
        "cross_model_agreement_rate": float(cross.get("cross_model_agreement_rate") or 0.0),
        "cross_model_prefix_stability": float(cross.get("cross_model_prefix_stability") or 0.0),
        "cross_model_failure_delta": float(cross.get("cross_model_failure_delta") or 0.0),
        "verifier_stability_score": float(report.get("verifier_stability_score") or 0.0),
        "proposal_instability_rate": float(report.get("proposal_instability_rate") or 0.0),
    }


def build_cross_model_group_context(
    cells: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Per (prompt, compressor, length) cross-model agreement context."""
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        trace = _primary_trace(cell)
        if trace is None or not cell.get("generation_completed"):
            continue
        buckets.setdefault(_group_key(cell), []).append(
            {
                "model_name": cell.get("model_name"),
                "decision": trace["decision"],
                "prefix_length": trace["prefix_length"],
                "proposal_tokens": tuple(
                    (cell.get("trace_records") or [{}])[0].get("proposal_tokens") or [],
                ),
            },
        )

    context: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, entries in buckets.items():
        if len(entries) < 2:
            context[key] = {
                "cross_model_agreement_rate": 1.0,
                "cross_model_prefix_stability": 1.0,
                "proposal_instability_rate": 0.0,
            }
            continue
        decisions = {e["decision"] for e in entries}
        prefixes = {e["prefix_length"] for e in entries}
        proposals = {e["proposal_tokens"] for e in entries}
        context[key] = {
            "cross_model_agreement_rate": 1.0 if len(decisions) == 1 else 0.0,
            "cross_model_prefix_stability": 1.0 if len(prefixes) == 1 else 0.0,
            "proposal_instability_rate": 0.0 if len(proposals) == 1 else 1.0,
        }
    return context


def compute_cell_instability_score(
    cell: Mapping[str, Any],
    *,
    group_context: Mapping[tuple[str, str, int], Mapping[str, Any]],
    global_metrics: Mapping[str, float],
) -> float:
    """Deterministic per-cell instability score in [0, 1] (higher = less stable)."""
    score = 0.0
    trace = _primary_trace(cell)

    if not cell.get("generation_completed"):
        return 1.0

    if (cell.get("exactkv_failures") or 0) > 0:
        score += 0.35

    if trace is None:
        return min(1.0, score + 0.5)

    decision = trace["decision"]
    if decision == "REJECT":
        score += 0.30
    elif decision == "INVALID_TRACE":
        score += 0.40
    elif decision == "BLOCK_MISSING_EVIDENCE":
        score += 0.25
    elif decision != "ACCEPT_PREFIX":
        score += 0.20

    prop_len = max(trace["proposal_token_count"], 1)
    prefix_ratio = trace["prefix_length"] / prop_len
    score += (1.0 - prefix_ratio) * 0.20

    ctx = group_context.get(_group_key(cell), {})
    if ctx.get("cross_model_agreement_rate", 1.0) < 1.0:
        score += 0.15
    if ctx.get("proposal_instability_rate", 0.0) > 0.0:
        score += 0.10

    score += (1.0 - global_metrics.get("verifier_stability_score", 1.0)) * 0.05
    return min(1.0, max(0.0, score))


def classify_regime(instability_score: float) -> str:
    if instability_score < 0.25:
        return REGIME_STABLE
    if instability_score < 0.50:
        return REGIME_MODERATE_DRIFT
    if instability_score < 0.75:
        return REGIME_HIGH_DIVERGENCE
    return REGIME_FAILURE_PRONE


def build_cell_descriptors(
    cells: Sequence[Mapping[str, Any]],
    *,
    group_context: Mapping[tuple[str, str, int], Mapping[str, Any]],
    global_metrics: Mapping[str, float],
) -> tuple[CellDescriptor, ...]:
    descriptors: list[CellDescriptor] = []
    for cell in cells:
        instability = compute_cell_instability_score(
            cell,
            group_context=group_context,
            global_metrics=global_metrics,
        )
        trace = _primary_trace(cell) or {
            "decision": "",
            "prefix_length": 0,
            "proposal_token_count": 0,
        }
        descriptors.append(
            CellDescriptor(
                cell_id=_cell_id(cell),
                model_name=str(cell.get("model_name") or ""),
                prompt_id=str(cell.get("prompt_id") or ""),
                compressor=str(cell.get("compressor") or ""),
                max_new_tokens=int(cell.get("max_new_tokens") or 0),
                instability_score=round(instability, 6),
                stability_score=round(1.0 - instability, 6),
                decision=str(trace["decision"]),
                prefix_length=int(trace["prefix_length"]),
                proposal_token_count=int(trace["proposal_token_count"]),
            ),
        )
    return tuple(descriptors)


def extract_instability_regimes(
    descriptors: Sequence[CellDescriptor],
) -> dict[str, list[str]]:
    regimes: dict[str, list[str]] = {name: [] for name in REGIME_NAMES}
    for desc in descriptors:
        regime = classify_regime(desc.instability_score)
        regimes[regime].append(desc.cell_id)
    for name in REGIME_NAMES:
        regimes[name].sort()
    return regimes


def _mean_instability(
    descriptors: Sequence[CellDescriptor],
    *,
    key_fn: Callable[[CellDescriptor], str],
) -> dict[str, float]:
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for desc in descriptors:
        buckets[key_fn(desc)].append(desc.instability_score)
    return {
        k: round(sum(v) / len(v), 6) if v else 0.0
        for k, v in sorted(buckets.items())
    }


def compute_phase_boundaries(
    descriptors: Sequence[CellDescriptor],
) -> dict[str, Any]:
    """Detect stable→unstable transition thresholds across dimensions."""
    compressor_scores = _mean_instability(descriptors, key_fn=lambda d: d.compressor)
    length_scores = _mean_instability(
        descriptors,
        key_fn=lambda d: str(d.max_new_tokens),
    )
    model_scores = _mean_instability(descriptors, key_fn=lambda d: d.model_name)

    compressor_thresholds: dict[str, float] = {}
    prev_score: float | None = None
    for comp, score in compressor_scores.items():
        if prev_score is not None and score - prev_score >= 0.15:
            compressor_thresholds[comp] = round(score, 6)
        prev_score = score

    length_thresholds: dict[str, float] = {}
    prev_len_score: float | None = None
    for length, score in length_scores.items():
        if prev_len_score is not None and score - prev_len_score >= 0.10:
            length_thresholds[length] = round(score, 6)
        prev_len_score = score

    model_sensitivity_order = [
        m for m, _ in sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "compressor_thresholds": compressor_thresholds,
        "length_thresholds": length_thresholds,
        "model_sensitivity_order": model_sensitivity_order,
        "compressor_mean_instability": compressor_scores,
        "length_mean_instability": length_scores,
        "model_mean_instability": model_scores,
    }


def compute_failure_taxonomy(
    cells: Sequence[Mapping[str, Any]],
    descriptors: Sequence[CellDescriptor],
    *,
    group_context: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, int]:
    taxonomy = {
        "verifier_mismatch": 0,
        "proposal_instability": 0,
        "compressor_drift": 0,
        "length_collapse": 0,
        "cross_model_divergence": 0,
    }
    compressor_means = _mean_instability(descriptors, key_fn=lambda d: d.compressor)
    length_means = _mean_instability(
        descriptors,
        key_fn=lambda d: str(d.max_new_tokens),
    )
    high_compressors = {
        c for c, s in compressor_means.items() if s >= 0.40
    }
    high_lengths = {
        length for length, s in length_means.items() if s >= 0.35
    }

    for cell, desc in zip(cells, descriptors, strict=False):
        trace = _primary_trace(cell)
        if trace is None:
            continue
        decision = trace["decision"]
        if decision == "REJECT":
            taxonomy["verifier_mismatch"] += 1
        if decision == "BLOCK_MISSING_EVIDENCE":
            taxonomy["verifier_mismatch"] += 1
        ctx = group_context.get(_group_key(cell), {})
        if ctx.get("proposal_instability_rate", 0.0) > 0.0:
            taxonomy["proposal_instability"] += 1
        if ctx.get("cross_model_agreement_rate", 1.0) < 1.0:
            taxonomy["cross_model_divergence"] += 1
        if desc.compressor in high_compressors and decision in DIVERGENCE_DECISION_SET:
            taxonomy["compressor_drift"] += 1
        if str(desc.max_new_tokens) in high_lengths and desc.instability_score >= 0.50:
            taxonomy["length_collapse"] += 1
    return taxonomy


def compute_interaction_effects(
    descriptors: Sequence[CellDescriptor],
) -> dict[str, dict[str, float]]:
    def _pair_mean(
        key_a: Callable[[CellDescriptor], str],
        key_b: Callable[[CellDescriptor], str],
    ) -> dict[str, float]:
        from collections import defaultdict

        buckets: dict[str, list[float]] = defaultdict(list)
        for desc in descriptors:
            pair_key = f"{key_a(desc)}|{key_b(desc)}"
            buckets[pair_key].append(desc.instability_score)
        return {
            k: round(sum(v) / len(v), 6) if v else 0.0
            for k, v in sorted(buckets.items())
        }

    return {
        "compressor_length": _pair_mean(
            lambda d: d.compressor,
            lambda d: str(d.max_new_tokens),
        ),
        "model_compressor": _pair_mean(
            lambda d: d.model_name,
            lambda d: d.compressor,
        ),
        "model_length": _pair_mean(
            lambda d: d.model_name,
            lambda d: str(d.max_new_tokens),
        ),
    }


def compute_stability_surface(
    descriptors: Sequence[CellDescriptor],
    *,
    models: Sequence[str],
    compressors: Sequence[str],
    max_new_tokens_values: Sequence[int],
) -> dict[str, Any]:
    """144-cell stability surface with peak/valley regions."""
    grid: list[dict[str, Any]] = []
    heatmap_values: dict[str, dict[str, dict[str, float]]] = {
        m: {c: {} for c in compressors} for m in models
    }

    for model in models:
        for comp in compressors:
            for mnt in max_new_tokens_values:
                matching = [
                    d
                    for d in descriptors
                    if d.model_name == model
                    and d.compressor == comp
                    and d.max_new_tokens == mnt
                ]
                if not matching:
                    continue
                mean_stability = round(
                    sum(d.stability_score for d in matching) / len(matching),
                    6,
                )
                heatmap_values[model][comp][str(mnt)] = mean_stability
                for d in matching:
                    grid.append(
                        {
                            "cell_id": d.cell_id,
                            "model_name": d.model_name,
                            "compressor": d.compressor,
                            "max_new_tokens": d.max_new_tokens,
                            "stability_score": d.stability_score,
                        },
                    )

    sorted_by_stability = sorted(descriptors, key=lambda d: d.stability_score, reverse=True)
    peak_count = max(1, len(sorted_by_stability) // 10)
    valley_count = max(1, len(sorted_by_stability) // 10)

    return {
        "grid": grid,
        "heatmap_values": heatmap_values,
        "peak_stability_regions": [
            d.cell_id for d in sorted_by_stability[:peak_count]
        ],
        "valley_instability_regions": [
            d.cell_id for d in sorted_by_stability[-valley_count:]
        ],
    }


def run_exp116_instability_regime_extraction(
    exp115_report: Mapping[str, Any] | None = None,
    *,
    exp115_path: Path | str = DEFAULT_EXP115_REPORT,
) -> dict[str, Any]:
    """Run full Phase 21O analysis pipeline over Exp 115 output."""
    source = (
        dict(exp115_report)
        if exp115_report is not None
        else load_exp115_report(exp115_path)
    )
    cells = list(source.get("cells") or [])
    models = list(source.get("models") or [])
    compressors = list(source.get("compressors") or [])
    mnt_values = [int(x) for x in (source.get("max_new_tokens_values") or [])]

    global_metrics = normalize_exp115_metrics(source)
    group_context = build_cross_model_group_context(cells)
    descriptors = build_cell_descriptors(
        cells,
        group_context=group_context,
        global_metrics=global_metrics,
    )

    instability_regimes = extract_instability_regimes(descriptors)
    phase_boundaries = compute_phase_boundaries(descriptors)
    failure_taxonomy = compute_failure_taxonomy(
        cells,
        descriptors,
        group_context=group_context,
    )
    interaction_effects = compute_interaction_effects(descriptors)
    stability_surface = compute_stability_surface(
        descriptors,
        models=models,
        compressors=compressors,
        max_new_tokens_values=mnt_values,
    )

    regime_coverage = {name: len(ids) for name, ids in instability_regimes.items()}

    report = {
        "experiment_id": EXPERIMENT_116_ID,
        "status": "analysis_complete",
        "phase": PHASE_21O,
        "stage": ANALYSIS_STAGE,
        "mode": ANALYSIS_MODE,
        "source_experiment_id": EXPERIMENT_115_ID,
        "source_total_cells": len(cells),
        "normalized_metrics": global_metrics,
        "instability_regimes": instability_regimes,
        "regime_coverage": regime_coverage,
        "phase_boundaries": phase_boundaries,
        "failure_taxonomy": failure_taxonomy,
        "interaction_effects": interaction_effects,
        "stability_surface": stability_surface,
        "cell_descriptors": [d.to_dict() for d in descriptors],
        "analysis_only": True,
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "model_experiments_run": False,
        "runtime_coupling_modified": False,
        "new_inference_paths": False,
        "allowed_next_phase": RECOMMENDED_NEXT_PHASE_21O,
        "limitations": [
            "Post-hoc analysis over Exp 115 only; no new inference.",
            "Regime classification is deterministic heuristic; not causal proof.",
            "Failure taxonomy counts may overlap across modes.",
            "No ExactKVGenerator or runtime coupling modifications.",
        ],
    }
    report["validation_result"] = validate_exp116_report(report).to_dict()
    return report


def validate_exp116_report(report: Mapping[str, Any]) -> Exp116ValidationResult:
    errors: list[str] = []

    required = (
        "experiment_id",
        "status",
        "source_experiment_id",
        "source_total_cells",
        "normalized_metrics",
        "instability_regimes",
        "phase_boundaries",
        "failure_taxonomy",
        "interaction_effects",
        "stability_surface",
        "analysis_only",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_116_ID:
        errors.append("experiment_id mismatch")

    if report.get("source_experiment_id") != EXPERIMENT_115_ID:
        errors.append("source_experiment_id must be exp115")

    for flag in ANALYSIS_FORBIDDEN_FLAGS:
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")

    if report.get("analysis_only") is not True:
        errors.append("analysis_only must be true")

    regimes = report.get("instability_regimes") or {}
    for name in REGIME_NAMES:
        if name not in regimes:
            errors.append(f"missing regime: {name}")

    total_classified = sum(len(v) for v in regimes.values())
    if total_classified != int(report.get("source_total_cells") or 0):
        errors.append("regime classification does not cover all source cells")

    boundaries = report.get("phase_boundaries") or {}
    for key in (
        "compressor_thresholds",
        "length_thresholds",
        "model_sensitivity_order",
    ):
        if key not in boundaries:
            errors.append(f"missing phase_boundaries.{key}")

    taxonomy = report.get("failure_taxonomy") or {}
    for key in (
        "verifier_mismatch",
        "proposal_instability",
        "compressor_drift",
        "length_collapse",
        "cross_model_divergence",
    ):
        if key not in taxonomy:
            errors.append(f"missing failure_taxonomy.{key}")

    interactions = report.get("interaction_effects") or {}
    for key in ("compressor_length", "model_compressor", "model_length"):
        if key not in interactions:
            errors.append(f"missing interaction_effects.{key}")

    surface = report.get("stability_surface") or {}
    for key in ("grid", "heatmap_values", "peak_stability_regions", "valley_instability_regions"):
        if key not in surface:
            errors.append(f"missing stability_surface.{key}")

    return Exp116ValidationResult(valid=len(errors) == 0, errors=tuple(errors))
