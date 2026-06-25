"""Phase G — Unified Truth + Divergence Authority Engine.

Consolidates Phase A–F and leaderboard reports into one authoritative correctness
and divergence system. Disk-only inputs; no inference; no Phase A–F modifications.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

PHASE_G_ID = "phaseG_unified_truth_authority_engine"

DEFAULT_PHASE_A_INPUT = Path("reports/phaseA_benchmark.json")
DEFAULT_PHASE_D_INPUT = Path("reports/phaseD_runtime_probe.json")
DEFAULT_PHASE_F_INPUT = Path("reports/phaseF_kernel_benchmark.json")
DEFAULT_LEADERBOARD_INPUT = Path("reports/leaderboard.json")
DEFAULT_PHASE_G_TRUTH_REPORT = Path("reports/phaseG_unified_truth.json")
DEFAULT_KERNEL_CONSISTENCY_REPORT = Path("reports/phaseG_kernel_consistency.json")
DEFAULT_UNIFIED_DIVERGENCE_MAP = Path("reports/phaseG_divergence_map.png")

FAILURE_REGIME_STABLE = "stable"
FAILURE_REGIME_DRIFT_PRONE = "drift_prone"
FAILURE_REGIME_COMPRESSION_BREAK = "compression_break"
FAILURE_REGIME_KERNEL_DIVERGENT = "kernel_divergent"

FAILURE_REGIMES: tuple[str, ...] = (
    FAILURE_REGIME_STABLE,
    FAILURE_REGIME_DRIFT_PRONE,
    FAILURE_REGIME_COMPRESSION_BREAK,
    FAILURE_REGIME_KERNEL_DIVERGENT,
)

DIVERGENCE_TYPE_NONE = "none"
DIVERGENCE_TYPE_TOKEN_MISMATCH = "token_mismatch"
DIVERGENCE_TYPE_LENGTH_DRIFT = "length_drift"
DIVERGENCE_TYPE_KERNEL_INCONSISTENCY = "kernel_inconsistency"
DIVERGENCE_TYPE_VERIFIER_DISAGREEMENT = "verifier_disagreement"

# Phase A compressor → Phase D probe mode (when linkable)
PHASE_A_TO_PROBE_MODE: dict[str, str] = {
    "noop": "noop",
    "int8": "int8_sim",
    "int4_sim": "int4_sim",
    "k8_v4_sim": "int4_sim",
    "shard": "kv_dropout_sim",
}

# Phase A compressor → Phase E/F kernel mode (when linkable)
PHASE_A_TO_KERNEL_MODE: dict[str, str] = {
    "noop": "noop",
    "int8": "int8",
    "int4_sim": "int4",
    "k8_v4_sim": "int4",
    "spectralquant": "int4",
    "kvquant": "int4",
    "shard": "block_sparse",
}

HIDDEN_DRIFT_THRESHOLD = 0.05
ACCEPTANCE_DRIFT_THRESHOLD = 0.99


@dataclass(frozen=True)
class PhaseGValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FirstDivergenceResult:
    """Canonical first-divergence authority output — single system definition."""

    canonical_first_divergence_index: int | None
    divergence_type: str
    token_exact_match: bool
    baseline_length: int
    compressed_length: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FirstDivergenceAuthority:
    """Official divergence definition for ExactKV.

    ``baseline`` and ``compressed`` are dicts with optional keys:
    - token_ids: list[int]
    - verifier_token_ids: list[int] (compressed path only)
    """

    def compute(
        self,
        baseline: Mapping[str, Any],
        compressed: Mapping[str, Any],
    ) -> FirstDivergenceResult:
        base_ids = list(baseline.get("token_ids") or [])
        comp_ids = list(compressed.get("token_ids") or [])
        verifier_ids = list(compressed.get("verifier_token_ids") or [])

        base_len = len(base_ids)
        comp_len = len(comp_ids)

        if base_len != comp_len:
            idx = min(base_len, comp_len)
            return FirstDivergenceResult(
                canonical_first_divergence_index=idx,
                divergence_type=DIVERGENCE_TYPE_LENGTH_DRIFT,
                token_exact_match=False,
                baseline_length=base_len,
                compressed_length=comp_len,
                details={"reason": "sequence_length_mismatch"},
            )

        for i, (a, b) in enumerate(zip(base_ids, comp_ids)):
            if a != b:
                div_type = DIVERGENCE_TYPE_TOKEN_MISMATCH
                if verifier_ids and i < len(verifier_ids) and verifier_ids[i] != a:
                    div_type = DIVERGENCE_TYPE_VERIFIER_DISAGREEMENT
                return FirstDivergenceResult(
                    canonical_first_divergence_index=i,
                    divergence_type=div_type,
                    token_exact_match=False,
                    baseline_length=base_len,
                    compressed_length=comp_len,
                    details={
                        "baseline_token_id": a,
                        "compressed_token_id": b,
                        "verifier_token_id": (
                            verifier_ids[i] if i < len(verifier_ids) else None
                        ),
                    },
                )

        if verifier_ids and verifier_ids != base_ids:
            for i, (a, v) in enumerate(zip(base_ids, verifier_ids)):
                if a != v:
                    return FirstDivergenceResult(
                        canonical_first_divergence_index=i,
                        divergence_type=DIVERGENCE_TYPE_VERIFIER_DISAGREEMENT,
                        token_exact_match=True,
                        baseline_length=base_len,
                        compressed_length=comp_len,
                        details={
                            "baseline_token_id": a,
                            "verifier_token_id": v,
                        },
                    )

        return FirstDivergenceResult(
            canonical_first_divergence_index=None,
            divergence_type=DIVERGENCE_TYPE_NONE,
            token_exact_match=True,
            baseline_length=base_len,
            compressed_length=comp_len,
            details={},
        )


@dataclass
class UnifiedTruthRecord:
    """Single merged truth record across Phase A–F sources."""

    record_id: str
    model_name: str
    compressor_name: str
    prompt_id: str
    max_new_tokens: int
    source_phases: list[str]
    phase_a: dict[str, Any]
    phase_d: dict[str, Any] | None
    phase_e_kernel: dict[str, Any] | None
    phase_f_kernel: dict[str, Any] | None
    leaderboard: dict[str, Any] | None
    divergence_authority: dict[str, Any]
    failure_regime: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KernelConsistencyReport:
    """Cross-kernel (Phase E torch vs Phase F triton) consistency."""

    phase_e_id: str
    phase_f_id: str
    kv_shape: list[int]
    modes: list[dict[str, Any]]
    overall_consistent: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRegimeMap:
    stable: list[str]
    drift_prone: list[str]
    compression_break: list[str]
    kernel_divergent: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _record_id(
    model_name: str,
    compressor_name: str,
    prompt_id: str,
    max_new_tokens: int,
) -> str:
    return f"{model_name}|{compressor_name}|{prompt_id}|mnt{max_new_tokens}"


def _index_phase_d_cells(phase_d: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in phase_d.get("cells") or []:
        key = (
            str(cell.get("model_name") or ""),
            str(cell.get("prompt_id") or ""),
            str(cell.get("compression_mode") or ""),
        )
        out[key] = cell
    return out


def _index_leaderboard(leaderboard: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in leaderboard.get("entries") or []:
        key = (str(entry.get("model") or ""), str(entry.get("compressor") or ""))
        out[key] = entry
    return out


def _extract_phase_f_kernel_rows(phase_f: Mapping[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Split Phase F benchmark into Phase E (torch) and Phase F (triton) rows by mode."""
    torch_rows: dict[str, dict[str, Any]] = {}
    triton_rows: dict[str, dict[str, Any]] = {}
    for row in phase_f.get("benchmarks") or []:
        mode = str(row.get("mode") or "")
        backend = str(row.get("backend") or "")
        if backend == "torch":
            torch_rows[mode] = row
        elif backend == "triton":
            triton_rows[mode] = row
    return torch_rows, triton_rows


def _phase_a_baseline_compressed(cell: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    full = cell.get("full") or {}
    exactkv = cell.get("exactkv") or {}
    lossy = cell.get("lossy") or {}
    baseline = {"token_ids": list(full.get("output_ids") or [])}
    compressed: dict[str, Any] = {
        "token_ids": list(exactkv.get("output_ids") or lossy.get("output_ids") or []),
    }
    return baseline, compressed


def _authority_from_phase_a_cell(
    authority: FirstDivergenceAuthority,
    cell: Mapping[str, Any],
) -> FirstDivergenceResult:
    baseline, compressed = _phase_a_baseline_compressed(cell)
    result = authority.compute(baseline, compressed)
    metrics = cell.get("metrics") or {}
    reported_idx = metrics.get("first_divergence_index")
    if reported_idx is None:
        reported_idx = lossy_idx if (lossy_idx := (cell.get("lossy") or {}).get("first_divergence_idx")) is not None else None
    if reported_idx is not None and result.canonical_first_divergence_index is None:
        result = FirstDivergenceResult(
            canonical_first_divergence_index=int(reported_idx),
            divergence_type=DIVERGENCE_TYPE_TOKEN_MISMATCH,
            token_exact_match=bool(metrics.get("token_level_divergence") is False),
            baseline_length=result.baseline_length,
            compressed_length=result.compressed_length,
            details={**result.details, "reconciled_from_phase_a_metrics": True},
        )
    elif reported_idx is not None and result.canonical_first_divergence_index is not None:
        if int(reported_idx) != int(result.canonical_first_divergence_index):
            result.details["phase_a_reported_index"] = reported_idx
    return result


def validate_kernel_consistency(
    e_output: Mapping[str, Any],
    f_output: Mapping[str, Any],
) -> KernelConsistencyReport:
    """Validate Phase E (torch) vs Phase F (triton) kernel benchmark rows."""
    phase_e_id = str(e_output.get("phase_e_id") or "phaseE_kv_compression_kernel")
    phase_f_id = str(f_output.get("phase_f_id") or "phaseF_triton_kv_compression_kernel")
    kv_shape = list(e_output.get("kv_shape") or f_output.get("kv_shape") or [])

    torch_rows, triton_rows = _extract_phase_f_kernel_rows(e_output)
    if not torch_rows and e_output is f_output:
        torch_rows, triton_rows = _extract_phase_f_kernel_rows(f_output)

    mode_reports: list[dict[str, Any]] = []
    all_consistent = True

    for mode in sorted(set(torch_rows) | set(triton_rows)):
        e_row = torch_rows.get(mode)
        f_row = triton_rows.get(mode)
        if e_row is None or f_row is None:
            mode_reports.append(
                {
                    "mode": mode,
                    "status": "incomplete",
                    "consistent": False,
                    "reason": "missing_torch_or_triton_row",
                },
            )
            all_consistent = False
            continue

        e_ratio = float(e_row.get("compression_ratio") or 0.0)
        f_ratio = float(f_row.get("compression_ratio") or 0.0)
        e_mem_before = int(e_row.get("memory_before") or 0)
        f_mem_before = int(f_row.get("memory_before") or 0)
        e_mem_after = int(e_row.get("memory_after") or 0)
        f_mem_after = int(f_row.get("memory_after") or 0)

        ratio_match = abs(e_ratio - f_ratio) < 1e-9
        memory_match = e_mem_before == f_mem_before and e_mem_after == f_mem_after
        token_equivalence_ratio = 1.0 if ratio_match and memory_match else 0.0

        e_lat = float(e_row.get("latency_ms") or 0.0)
        f_lat = float(f_row.get("latency_ms") or 0.0)
        divergence_delta = abs(e_lat - f_lat)
        speedup_x = round(e_lat / f_lat, 4) if f_lat > 0 else None

        exec_f = str(f_row.get("execution_backend") or f_row.get("backend") or "")
        kernel_fallback = mode == "block_sparse" and exec_f == "torch"

        consistent = ratio_match and memory_match and e_row.get("status") == "ok" and f_row.get("status") == "ok"
        if not consistent:
            all_consistent = False

        mode_reports.append(
            {
                "mode": mode,
                "status": "ok",
                "consistent": consistent,
                "token_equivalence_ratio": token_equivalence_ratio,
                "divergence_delta_latency_ms": round(divergence_delta, 6),
                "speedup_triton_vs_torch": speedup_x,
                "memory_consistent": memory_match,
                "compression_ratio_torch": e_ratio,
                "compression_ratio_triton": f_ratio,
                "shape_parity": {"kv_shape": kv_shape},
                "execution_backend_triton_path": exec_f,
                "kernel_fallback_to_torch": kernel_fallback,
                "divergence_type_if_inconsistent": (
                    None if consistent else DIVERGENCE_TYPE_KERNEL_INCONSISTENCY
                ),
            },
        )

    return KernelConsistencyReport(
        phase_e_id=phase_e_id,
        phase_f_id=phase_f_id,
        kv_shape=kv_shape,
        modes=mode_reports,
        overall_consistent=all_consistent,
        note="Phase E derived from Phase F torch backend rows; parity on compression_ratio and memory bytes.",
    )


def _classify_failure_regime(
    *,
    record_id: str,
    divergence: FirstDivergenceResult,
    phase_a: Mapping[str, Any],
    phase_d: Mapping[str, Any] | None,
    kernel_mode: str | None,
    kernel_consistency: KernelConsistencyReport,
) -> str:
    if phase_a.get("exactkv_failure"):
        return FAILURE_REGIME_COMPRESSION_BREAK

    kernel_mode_row = next(
        (m for m in kernel_consistency.modes if m.get("mode") == kernel_mode),
        None,
    )
    if kernel_mode_row and not kernel_mode_row.get("consistent", True):
        return FAILURE_REGIME_KERNEL_DIVERGENT

    if divergence.divergence_type in (
        DIVERGENCE_TYPE_TOKEN_MISMATCH,
        DIVERGENCE_TYPE_LENGTH_DRIFT,
        DIVERGENCE_TYPE_VERIFIER_DISAGREEMENT,
    ):
        if str(phase_a.get("compressor_name") or "") != "noop":
            return FAILURE_REGIME_COMPRESSION_BREAK
        return FAILURE_REGIME_DRIFT_PRONE

    metrics = phase_a.get("metrics") or {}
    acceptance = float(metrics.get("acceptance_rate") or 1.0)
    hidden_drift = 0.0
    if phase_d:
        dm = phase_d.get("divergence_metrics") or {}
        hidden_drift = float(dm.get("cosine_hidden_drift_proxy") or 0.0)
        stab = phase_d.get("stability_metrics") or {}
        mean_inst = float(stab.get("mean_layer_instability") or 0.0)
        if mean_inst > HIDDEN_DRIFT_THRESHOLD or hidden_drift > HIDDEN_DRIFT_THRESHOLD:
            return FAILURE_REGIME_DRIFT_PRONE

    if acceptance < ACCEPTANCE_DRIFT_THRESHOLD:
        return FAILURE_REGIME_DRIFT_PRONE

    return FAILURE_REGIME_STABLE


def build_failure_regime_map(records: Sequence[UnifiedTruthRecord]) -> FailureRegimeMap:
    buckets: dict[str, list[str]] = {r: [] for r in FAILURE_REGIMES}
    for rec in records:
        buckets.setdefault(rec.failure_regime, []).append(rec.record_id)
    return FailureRegimeMap(
        stable=sorted(buckets[FAILURE_REGIME_STABLE]),
        drift_prone=sorted(buckets[FAILURE_REGIME_DRIFT_PRONE]),
        compression_break=sorted(buckets[FAILURE_REGIME_COMPRESSION_BREAK]),
        kernel_divergent=sorted(buckets[FAILURE_REGIME_KERNEL_DIVERGENT]),
    )


def build_unified_truth_records(
    phase_a: Mapping[str, Any],
    phase_d: Mapping[str, Any],
    phase_f: Mapping[str, Any],
    leaderboard: Mapping[str, Any],
    *,
    authority: FirstDivergenceAuthority | None = None,
    kernel_consistency: KernelConsistencyReport | None = None,
) -> list[UnifiedTruthRecord]:
    auth = authority or FirstDivergenceAuthority()
    k_report = kernel_consistency or validate_kernel_consistency(phase_f, phase_f)

    d_index = _index_phase_d_cells(phase_d)
    lb_index = _index_leaderboard(leaderboard)
    torch_rows, triton_rows = _extract_phase_f_kernel_rows(phase_f)

    records: list[UnifiedTruthRecord] = []
    cells = sorted(
        phase_a.get("cells") or [],
        key=lambda c: (
            str(c.get("model_name") or ""),
            str(c.get("compressor_name") or ""),
            str(c.get("prompt_id") or ""),
            int(c.get("max_new_tokens") or 0),
        ),
    )

    for cell in cells:
        model = str(cell.get("model_name") or "")
        compressor = str(cell.get("compressor_name") or "")
        prompt_id = str(cell.get("prompt_id") or "")
        mnt = int(cell.get("max_new_tokens") or 0)
        rid = _record_id(model, compressor, prompt_id, mnt)

        probe_mode = PHASE_A_TO_PROBE_MODE.get(compressor)
        phase_d_cell = (
            d_index.get((model, prompt_id, probe_mode)) if probe_mode else None
        )

        kernel_mode = PHASE_A_TO_KERNEL_MODE.get(compressor)
        phase_e_row = torch_rows.get(kernel_mode) if kernel_mode else None
        phase_f_row = triton_rows.get(kernel_mode) if kernel_mode else None

        div_result = _authority_from_phase_a_cell(auth, cell)
        regime = _classify_failure_regime(
            record_id=rid,
            divergence=div_result,
            phase_a=cell,
            phase_d=phase_d_cell,
            kernel_mode=kernel_mode,
            kernel_consistency=k_report,
        )

        sources = ["phaseA"]
        if phase_d_cell:
            sources.append("phaseD")
        if phase_e_row:
            sources.append("phaseE")
        if phase_f_row:
            sources.append("phaseF")
        lb_entry = lb_index.get((model, compressor))
        if lb_entry:
            sources.append("phaseB")

        phase_a_slice = {
            "exactkv_failure": cell.get("exactkv_failure"),
            "metrics": cell.get("metrics"),
            "acceptance": (cell.get("exactkv") or {}).get("acceptance"),
            "token_exact_match": (cell.get("exactkv") or {}).get("token_exact_match"),
        }
        phase_d_slice = None
        if phase_d_cell:
            phase_d_slice = {
                "compression_mode": phase_d_cell.get("compression_mode"),
                "divergence_metrics": phase_d_cell.get("divergence_metrics"),
                "stability_metrics": phase_d_cell.get("stability_metrics"),
                "first_divergence_index": phase_d_cell.get("first_divergence_index"),
                "memory_proxy": phase_d_cell.get("memory_proxy"),
            }

        records.append(
            UnifiedTruthRecord(
                record_id=rid,
                model_name=model,
                compressor_name=compressor,
                prompt_id=prompt_id,
                max_new_tokens=mnt,
                source_phases=sources,
                phase_a=phase_a_slice,
                phase_d=phase_d_slice,
                phase_e_kernel=phase_e_row,
                phase_f_kernel=phase_f_row,
                leaderboard=lb_entry,
                divergence_authority=div_result.to_dict(),
                failure_regime=regime,
            ),
        )

    return records


def render_phase_g_divergence_map(
    records: Sequence[UnifiedTruthRecord],
    *,
    output_path: Path | str = DEFAULT_UNIFIED_DIVERGENCE_MAP,
) -> dict[str, Any]:
    """Render divergence heatmap from unified records (matplotlib optional)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"output_path": str(output_path)}

    matrix_data: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for rec in records:
        div = rec.divergence_authority
        idx = div.get("canonical_first_divergence_index")
        matrix_data[(rec.model_name, rec.compressor_name)].append(
            float(idx) if idx is not None else None,
        )

    models = sorted({m for m, _ in matrix_data})
    compressors = sorted({c for _, c in matrix_data})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        mat = np.full((len(models), len(compressors)), np.nan)
        for i, model in enumerate(models):
            for j, comp in enumerate(compressors):
                vals = [v for v in matrix_data.get((model, comp), []) if v is not None]
                if vals:
                    mat[i, j] = float(sum(vals) / len(vals))

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split("/")[-1][:24] for m in models])
        ax.set_xticks(range(len(compressors)))
        ax.set_xticklabels(compressors, rotation=45, ha="right")
        ax.set_title("Phase G — Canonical First Divergence Index (mean per model×compressor)")
        fig.colorbar(im, ax=ax, label="first_divergence_index")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        result["status"] = "ok"
    except ImportError as exc:
        result["status"] = "skipped"
        result["plot_error"] = str(exc)

    return result


def run_phase_g_unified_truth_engine(
    *,
    phase_a_path: Path | str = DEFAULT_PHASE_A_INPUT,
    phase_d_path: Path | str = DEFAULT_PHASE_D_INPUT,
    phase_f_path: Path | str = DEFAULT_PHASE_F_INPUT,
    leaderboard_path: Path | str = DEFAULT_LEADERBOARD_INPUT,
    divergence_map_path: Path | str = DEFAULT_UNIFIED_DIVERGENCE_MAP,
) -> dict[str, Any]:
    """Build unified truth report from Phase A–F disk artifacts only."""
    phase_a = _load_json(phase_a_path)
    phase_d = _load_json(phase_d_path)
    phase_f = _load_json(phase_f_path)
    leaderboard = _load_json(leaderboard_path)

    kernel_report = validate_kernel_consistency(phase_f, phase_f)
    records = build_unified_truth_records(
        phase_a,
        phase_d,
        phase_f,
        leaderboard,
        kernel_consistency=kernel_report,
    )
    failure_map = build_failure_regime_map(records)
    visual = render_phase_g_divergence_map(records, output_path=divergence_map_path)

    regime_counts = {r: 0 for r in FAILURE_REGIMES}
    for rec in records:
        regime_counts[rec.failure_regime] = regime_counts.get(rec.failure_regime, 0) + 1

    report = {
        "phase_id": PHASE_G_ID,
        "status": "unified_truth_complete",
        "authority_engine": "FirstDivergenceAuthority",
        "divergence_types": [
            DIVERGENCE_TYPE_NONE,
            DIVERGENCE_TYPE_TOKEN_MISMATCH,
            DIVERGENCE_TYPE_LENGTH_DRIFT,
            DIVERGENCE_TYPE_KERNEL_INCONSISTENCY,
            DIVERGENCE_TYPE_VERIFIER_DISAGREEMENT,
        ],
        "inputs": {
            "phase_a": str(phase_a_path),
            "phase_d": str(phase_d_path),
            "phase_f": str(phase_f_path),
            "leaderboard": str(leaderboard_path),
            "phase_e_note": "Derived from phase_f torch backend rows (no separate phaseE JSON)",
        },
        "source_totals": {
            "phase_a_cells": len(phase_a.get("cells") or []),
            "phase_d_cells": len(phase_d.get("cells") or []),
            "unified_records": len(records),
            "leaderboard_entries": len(leaderboard.get("entries") or []),
        },
        "failure_regime_counts": regime_counts,
        "failure_regime_map": failure_map.to_dict(),
        "kernel_consistency": kernel_report.to_dict(),
        "records": [r.to_dict() for r in records],
        "divergence_map": visual,
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "inference_runs_required": False,
        "reproducible_cli_command": "python scripts/run_phase_g_unified_truth_engine.py",
        "limitations_note": (
            "Unified truth is synthesized from existing Phase A–F reports only. "
            "Phase D links apply where compressor→probe mode mapping exists. "
            "Kernel rows are mode-level benchmarks, not per-cell inference."
        ),
    }
    report["validation_result"] = validate_phase_g_report(report).to_dict()
    return report


def validate_phase_g_report(report: Mapping[str, Any]) -> PhaseGValidationResult:
    errors: list[str] = []
    if report.get("phase_id") != PHASE_G_ID:
        errors.append(f"unexpected phase_id: {report.get('phase_id')}")
    records = report.get("records") or []
    if not records:
        errors.append("no unified records")
    for rec in records[:5]:
        if "divergence_authority" not in rec:
            errors.append("record missing divergence_authority")
            break
        da = rec["divergence_authority"]
        if "canonical_first_divergence_index" not in da:
            errors.append("divergence_authority missing canonical_first_divergence_index")
            break
        if "divergence_type" not in da:
            errors.append("divergence_authority missing divergence_type")
            break
    kc = report.get("kernel_consistency") or {}
    if "modes" not in kc:
        errors.append("kernel_consistency missing modes")
    frm = report.get("failure_regime_map") or {}
    for key in FAILURE_REGIMES:
        if key not in frm:
            errors.append(f"failure_regime_map missing {key}")
    return PhaseGValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def write_phase_g_outputs(
    report: Mapping[str, Any],
    *,
    truth_path: Path | str = DEFAULT_PHASE_G_TRUTH_REPORT,
    kernel_path: Path | str = DEFAULT_KERNEL_CONSISTENCY_REPORT,
) -> dict[str, Path]:
    truth_out = Path(truth_path)
    kernel_out = Path(kernel_path)
    truth_out.parent.mkdir(parents=True, exist_ok=True)
    kernel_out.parent.mkdir(parents=True, exist_ok=True)

    kernel_payload = dict(report.get("kernel_consistency") or {})
    kernel_payload["phase_g_id"] = PHASE_G_ID
    kernel_out.write_text(json.dumps(kernel_payload, indent=2) + "\n")

    truth_out.write_text(json.dumps(report, indent=2) + "\n")
    return {
        "phaseG_unified_truth": truth_out,
        "phaseG_kernel_consistency": kernel_out,
    }
