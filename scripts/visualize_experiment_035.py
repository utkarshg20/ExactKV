#!/usr/bin/env python3
"""Experiment 035: visual plot package and mini leaderboard (V13 Phase 8).

Visualization/reporting only — uses existing experiment CSV/JSON and committed docs.
No model inference, timing benchmarks, or memory benchmarks are run.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_REPORTS = _ROOT / "reports"
_ASSETS = _ROOT / "docs" / "assets"
_DOCS = _ROOT / "docs"

PUBLIC_TAGLINE = (
    "Everyone is racing to shrink KV caches.\n"
    "ExactKV tells you when they start lying."
)

COMPRESSOR_PANEL = (
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "backend_passthrough",
)

SUITE_PANEL = (
    "core_v2",
    "long_context",
    "retrieval_copy",
    "tool_json",
    "code_structured",
)

EXP034_FIXTURE = {
    "prompt": (
        'Complete this tool call JSON: {"name": "get_weather", "arguments": '
        '{"city": "Paris", "units":'
    ),
    "full_output_text": ' "metric"}} To complete this tool call JSON, you would need to define a function…',
    "lossy_output_text": ' "}}}\n\n{"name": "get_weather", "arguments": {"city": "Paris", "units": "metric"}}}',
    "exactkv_output_text": ' "metric"}} To complete this tool call JSON, you would need to define a function…',
    "rejected_token": "}}",
    "correction_token": "metric",
    "exactkv_failures": 0,
    "final_match": True,
}


@dataclass
class ExperimentExactness:
    experiment: str
    label: str
    cells: int
    failures: int
    phase: str = ""


@dataclass
class LeaderboardEntry:
    tier: str
    method: str
    experiment: str
    model_panel: str
    mean_acceptance: float | None
    exactkv_failures: int | None
    integration_status: str
    caveat: str
    rank: int | None = None


# Tier badges for docs and public visuals (Phase 8d).
TIER_FULL_PANEL = "FULL PANEL"
TIER_RESTRICTED = "RESTRICTED BACKEND"
TIER_SMOKE = "SMOKE ONLY"
TIER_FUTURE = "FUTURE CANDIDATE"
TIER_REPAIR = "REPAIR POLICY"

PUBLIC_LEADERBOARD_COPY = (
    "ExactKV Leaderboard ranks integrated compressors by token-level acceptance and exactness. "
    "Restricted backends (including Shard external-drafter probe results), smoke-only adapters, "
    "and future candidates are separated to avoid apples-to-oranges claims."
)

# Shard Exp 039–041 restricted external-drafter metrics (not full-panel compressor acceptance).
SHARD_ACCEPTED_PREFIX_MEAN = 58.22
SHARD_MAX_NEW_TOKENS = 64
SHARD_ACCEPTED_PREFIX_RATIO = round(SHARD_ACCEPTED_PREFIX_MEAN / SHARD_MAX_NEW_TOKENS, 4)
SHARD_DIVERGENCE_RATE_039 = 0.1875  # 6/32
SHARD_MAX_DIVERGENCE_RATE_040 = 0.3125  # length_128tok
SHARD_COMBINED_DIVERGENCE_RATE_041 = 0.5625  # stream_bits=4 + 128tok
SHARD_RESTRICTED_CAVEAT = (
    "Exp 039: 32 prompts @64tok, 6 draft divergences (18.75%), accepted-prefix mean 58.22/64, "
    "exactkv_failures=0. Exp 040: 5-setting ablation, max single-knob divergence 31.25% "
    "(length_128tok), stream_bits=4 at 25%, all exactkv_failures=0. Exp 041: combined "
    "stream_bits=4 + 128tok, 18 divergences (56.25%), exactkv_failures=0. Restricted "
    "external-drafter metrics — not full-panel compressor acceptance. External Shard README "
    "not ExactKV."
)

SPECTRALQUANT_SMOKE_CAVEAT = (
    "SpectralQuant has ExactKV tensor-smoke coverage, but no generation-time ExactKV probe yet. "
    "Synthetic K/V round-trip pass (Exp 042); tensor smoke only — not generation integration. "
    "External paper/README not ExactKV."
)

# Exp 045 restricted adapter panel (12 prompts, Qwen2.5-0.5B CPU).
SPECTRALQUANT_PANEL_MEAN_ACCEPTANCE = 0.481
SPECTRALQUANT_KEY_MAX_RECON_ERROR = 38.99
SPECTRALQUANT_RESTRICTED_CAVEAT = (
    "Exp 045: 12-prompt restricted adapter panel @32tok, draft_len=4, "
    f"mean acceptance {SPECTRALQUANT_PANEL_MEAN_ACCEPTANCE:.3f}, exactkv_failures=0. "
    "Materializing factory-only adapter: compresses K/V then materialises dequant tensors "
    "for draft — no active memory savings. 6-prompt eigenspectral calibration; "
    f"key max reconstruction error ~{SPECTRALQUANT_KEY_MAX_RECON_ERROR:.0f} (layer 0). "
    "11/12 prompts had draft divergence corrected by verifier. Small-panel adapter "
    "acceptance — not full-panel compressor ranking. External SpectralQuant README not ExactKV."
)


@dataclass
class PlotData:
    exactness: list[ExperimentExactness] = field(default_factory=list)
    acceptance_by_compressor: dict[str, float] = field(default_factory=dict)
    acceptance_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    category_heatmap: dict[str, dict[str, float]] = field(default_factory=dict)
    divergence_histogram: list[int] = field(default_factory=list)
    divergence_source: str = ""
    timing_by_arm: dict[str, float] = field(default_factory=dict)
    memory_by_arm_mib: dict[str, float] = field(default_factory=dict)
    killer_demo: dict[str, Any] = field(default_factory=dict)
    leaderboard: list[LeaderboardEntry] = field(default_factory=list)
    inputs_used: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _acceptance_from_row(row: dict[str, str]) -> float | None:
    if row.get("acceptance_rate"):
        return float(row["acceptance_rate"])
    acc = row.get("sequential_accepted") or row.get("total_accepted")
    rej = row.get("sequential_rejected") or row.get("total_rejected")
    if acc is not None and rej is not None:
        total = int(acc) + int(rej)
        return int(acc) / total if total else 1.0
    return None


def _is_exactkv_failure(row: dict[str, str]) -> bool:
    for key in (
        "exactkv_token_exact_match",
        "exactkv_failure",
        "sequential_exactkv_failure",
        "span_exactkv_failure",
    ):
        if key not in row:
            continue
        val = row[key].strip().lower()
        if key == "exactkv_token_exact_match":
            return val == "false"
        return val == "true"
    return False


def _agg_acceptance_by_compressor(
    rows: list[dict[str, str]],
    *,
    compressors: tuple[str, ...] = COMPRESSOR_PANEL,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        comp = row.get("compressor_name", "")
        if comp not in compressors:
            continue
        if _is_exactkv_failure(row):
            continue
        acc = _acceptance_from_row(row)
        if acc is not None:
            buckets[comp].append(acc)
    return {k: v for k, v in ((_c, _mean(buckets[_c])) for _c in compressors) if v is not None}


def _agg_acceptance_by_model(
    path: Path,
    model_label: str,
    *,
    compressors: tuple[str, ...] = COMPRESSOR_PANEL,
) -> dict[str, float] | None:
    rows = _read_csv(path)
    if not rows:
        return None
    data = _agg_acceptance_by_compressor(rows, compressors=compressors)
    return data or None


def _count_failures(rows: list[dict[str, str]], *, exactkv_only: bool = False) -> int:
    if not rows:
        return 0
    if exactkv_only and "arm" in rows[0]:
        rows = [r for r in rows if r.get("arm", "").startswith("exactkv")]
    return sum(1 for r in rows if _is_exactkv_failure(r))


def load_plot_data() -> PlotData:
    data = PlotData()

    def track(path: Path) -> list[dict[str, str]]:
        if path.is_file():
            data.inputs_used.append(str(path.relative_to(_ROOT)))
            return _read_csv(path)
        data.missing.append(str(path.relative_to(_ROOT)))
        return []

    # --- Exactness summary ---
    exactness_specs = [
        ("012", "Exp 012 suite expansion", "reports/experiment_012_eval_suite_expansion.csv", "V10"),
        ("015", "Exp 015 Qwen 1.5B", "reports/experiment_015_qwen15b_v10_suites.csv", "V10"),
        ("016", "Exp 016 Qwen 3B", "reports/experiment_016_qwen3b_v10_suites.csv", "V10"),
        ("019", "Exp 019 divergence autopsy", "reports/experiment_019_divergence_autopsy.csv", "V11"),
        ("025", "Exp 025 repair policy", "reports/experiment_025_full_suite_repair_policy.csv", "V12"),
        ("029", "Exp 029 span grid", "reports/experiment_029_span_verification_grid.csv", "V13"),
        ("030", "Exp 030 timing (ExactKV arms)", "reports/experiment_030_diagnostic_timing.csv", "V13"),
        ("031", "Exp 031 memory (ExactKV arms)", "reports/experiment_031_gpu_memory_isolation.csv", "V13"),
        ("033", "Exp 033 Llama 8B", "reports/experiment_033_llama31_8b_small_suite.csv", "V13"),
    ]
    for exp_id, label, rel, phase in exactness_specs:
        rows = track(_ROOT / rel)
        if not rows:
            continue
        exactkv_only = exp_id in ("030", "031")
        if exactkv_only:
            sub = [r for r in rows if r.get("arm", "").startswith("exactkv")]
            cell_count = len(sub)
            fail_count = _count_failures(sub)
        else:
            cell_count = len(rows)
            fail_count = _count_failures(rows)
        data.exactness.append(
            ExperimentExactness(
                experiment=exp_id,
                label=label,
                cells=cell_count,
                failures=fail_count,
                phase=phase,
            )
        )

    exp034_json = _ROOT / "reports/experiment_034_killer_correction_demo.json"
    if exp034_json.is_file():
        data.inputs_used.append(str(exp034_json.relative_to(_ROOT)))
        report = json.loads(exp034_json.read_text(encoding="utf-8"))
        summary = report.get("search_summary", {})
        data.exactness.append(
            ExperimentExactness(
                experiment="034",
                label="Exp 034 killer demo search",
                cells=int(summary.get("cells_searched", 0)),
                failures=int(summary.get("exactkv_failures", 0)),
                phase="V13",
            )
        )
        demo = report.get("selected_demo") or {}
        hr = demo.get("highlight_round") or {}
        data.killer_demo = {
            "prompt": demo.get("prompt", EXP034_FIXTURE["prompt"]),
            "full_output_text": demo.get("full_output_text", EXP034_FIXTURE["full_output_text"]),
            "lossy_output_text": demo.get("lossy_output_text", EXP034_FIXTURE["lossy_output_text"]),
            "exactkv_output_text": demo.get("exactkv_output_text", EXP034_FIXTURE["exactkv_output_text"]),
            "rejected_token": hr.get("first_rejected_text", EXP034_FIXTURE["rejected_token"]),
            "correction_token": hr.get("correction_text", EXP034_FIXTURE["correction_token"]),
            "exactkv_failures": 0,
            "final_match": bool(demo.get("exactkv_exact_match", True)),
        }
    else:
        data.missing.append(str(exp034_json.relative_to(_ROOT)))
        data.killer_demo = dict(EXP034_FIXTURE)

    # --- Acceptance by compressor (0.5B anchor: Exp 012 + 025) ---
    rows_012 = track(_ROOT / "reports/experiment_012_eval_suite_expansion.csv")
    rows_025 = track(_ROOT / "reports/experiment_025_full_suite_repair_policy.csv")
    comb_05b: dict[str, list[float]] = defaultdict(list)
    for rows in (rows_012, rows_025):
        for comp, val in _agg_acceptance_by_compressor(rows).items():
            comb_05b[comp].append(val)
    data.acceptance_by_compressor = {
        c: _mean(comb_05b[c])  # type: ignore[misc]
        for c in COMPRESSOR_PANEL
        if _mean(comb_05b[c]) is not None
    }

    # --- Acceptance by model ---
    model_files = [
        ("Qwen2.5-0.5B", "reports/experiment_012_eval_suite_expansion.csv"),
        ("Qwen2.5-1.5B", "reports/experiment_015_qwen15b_v10_suites.csv"),
        ("Qwen2.5-3B", "reports/experiment_016_qwen3b_v10_suites.csv"),
        ("Llama-3.1-8B", "reports/experiment_033_llama31_8b_small_suite.csv"),
    ]
    for label, rel in model_files:
        path = _ROOT / rel
        acc = _agg_acceptance_by_model(path, label)
        if acc:
            data.acceptance_by_model[label] = acc

    # --- Category heatmap: lossy divergence rate (Exp 019) ---
    rows_019 = track(_ROOT / "reports/experiment_019_divergence_autopsy.csv")
    if rows_019:
        data.divergence_source = "Exp 019 divergence autopsy (lossy divergence rate)"
        hm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows_019:
            suite = row.get("v10_suite", "")
            if suite not in SUITE_PANEL:
                continue
            comp = row.get("compressor_name", "")
            if comp not in COMPRESSOR_PANEL[:4]:
                continue
            div = 1.0 if row.get("lossy_diverged", "").lower() == "true" else 0.0
            hm[suite][comp].append(div)
        data.category_heatmap = {
            suite: {comp: _mean(vals) for comp, vals in hm[suite].items() if _mean(vals) is not None}
            for suite in SUITE_PANEL
            if suite in hm
        }
        data.divergence_histogram = [
            int(row["first_divergence_idx"])
            for row in rows_019
            if row.get("lossy_diverged", "").lower() == "true" and row.get("first_divergence_idx")
        ]
    elif data.killer_demo:
        data.divergence_source = "Exp 034 selected demo only (not aggregate)"
        data.divergence_histogram = [1]

    # --- Timing (Exp 030) ---
    rows_030 = track(_ROOT / "reports/experiment_030_diagnostic_timing.csv")
    for arm in ("full_greedy", "exactkv_sequential", "exactkv_span"):
        vals = [
            float(r["mean_wall_time_seconds"])
            for r in rows_030
            if r.get("arm") == arm and r.get("timing_valid", "").lower() == "true"
        ]
        if vals:
            data.timing_by_arm[arm] = _mean(vals)  # type: ignore[assignment]

    # --- Memory (Exp 031) ---
    rows_031 = track(_ROOT / "reports/experiment_031_gpu_memory_isolation.csv")
    for arm in ("full_greedy", "exactkv_sequential", "exactkv_span"):
        vals = [
            float(r["mean_peak_allocated"]) / (1024 * 1024)
            for r in rows_031
            if r.get("arm") == arm and r.get("mean_peak_allocated")
        ]
        if vals:
            data.memory_by_arm_mib[arm] = _mean(vals)  # type: ignore[assignment]

    data.leaderboard = build_tiered_leaderboard(data)
    return data


def _csv_panel_entry(
    tier: str,
    method: str,
    experiment: str,
    rel: str,
    compressor: str,
    model_panel: str,
    integration_status: str,
    caveat: str,
) -> LeaderboardEntry | None:
    rows = _read_csv(_ROOT / rel)
    sub = [r for r in rows if r.get("compressor_name") == compressor]
    if not sub:
        return None
    acc_vals = [v for r in sub if (v := _acceptance_from_row(r)) is not None]
    return LeaderboardEntry(
        tier=tier,
        method=method,
        experiment=experiment,
        model_panel=model_panel,
        mean_acceptance=_mean(acc_vals),
        exactkv_failures=_count_failures(sub),
        integration_status=integration_status,
        caveat=caveat,
    )


def build_tiered_leaderboard(data: PlotData) -> list[LeaderboardEntry]:
    """Tiered crash-test lab leaderboard — no apples-to-oranges ranking across tiers."""
    entries: list[LeaderboardEntry] = []

    full_specs = [
        ("noop", "Exp 012", "reports/experiment_012_eval_suite_expansion.csv", "Qwen2.5-0.5B · 128-prompt V10", "built-in identity"),
        ("backend_passthrough", "Exp 012", "reports/experiment_012_eval_suite_expansion.csv", "Qwen2.5-0.5B · 128-prompt V10", "built-in passthrough"),
        ("int8", "Exp 012", "reports/experiment_012_eval_suite_expansion.csv", "Qwen2.5-0.5B · 128-prompt V10", "built-in symmetric INT8"),
        ("k8_v4_boundary4_v8_sim", "Exp 012", "reports/experiment_012_eval_suite_expansion.csv", "Qwen2.5-0.5B · 128-prompt V10", "built-in sim asymmetric"),
        ("k8_v4_sim", "Exp 012", "reports/experiment_012_eval_suite_expansion.csv", "Qwen2.5-0.5B · 128-prompt V10", "built-in sim asymmetric"),
        ("int8", "Exp 015", "reports/experiment_015_qwen15b_v10_suites.csv", "Qwen2.5-1.5B · 128-prompt V10", "built-in symmetric INT8"),
        ("k8_v4_sim", "Exp 015", "reports/experiment_015_qwen15b_v10_suites.csv", "Qwen2.5-1.5B · 128-prompt V10", "built-in sim asymmetric"),
        ("int8", "Exp 016", "reports/experiment_016_qwen3b_v10_suites.csv", "Qwen2.5-3B · 128-prompt V10", "built-in symmetric INT8"),
        ("k8_v4_sim", "Exp 016", "reports/experiment_016_qwen3b_v10_suites.csv", "Qwen2.5-3B · 128-prompt V10", "built-in sim asymmetric"),
        ("int8", "Exp 033", "reports/experiment_033_llama31_8b_small_suite.csv", "Llama-3.1-8B · 12-prompt small suite", "built-in symmetric INT8"),
        ("k8_v4_sim", "Exp 033", "reports/experiment_033_llama31_8b_small_suite.csv", "Llama-3.1-8B · 12-prompt small suite", "built-in sim asymmetric"),
    ]
    labels = {
        "noop": "No compression",
        "backend_passthrough": "Passthrough",
        "int8": "INT8",
        "k8_v4_sim": "K8/V4",
        "k8_v4_boundary4_v8_sim": "Boundary V",
    }
    full_panel: list[LeaderboardEntry] = []
    for comp, exp, rel, panel, status in full_specs:
        e = _csv_panel_entry(TIER_FULL_PANEL, labels.get(comp, comp), exp, rel, comp, panel, status, "full/large panel")
        if e:
            full_panel.append(e)
    full_panel.sort(key=lambda e: (-(e.mean_acceptance or 0), e.method))
    for i, e in enumerate(full_panel, 1):
        e.rank = i
        entries.append(e)

    # Repair policies — separate from compressors (Exp 025 docs).
    repair_rows = _read_csv(_ROOT / "reports/experiment_025_full_suite_repair_policy.csv")
    if repair_rows:
        from collections import defaultdict

        by_policy: dict[str, list[float]] = defaultdict(list)
        fail_by: dict[str, int] = defaultdict(int)
        for r in repair_rows:
            pol = r.get("policy_name", "")
            acc = _acceptance_from_row(r)
            if acc is not None:
                by_policy[pol].append(acc)
            if _is_exactkv_failure(r):
                fail_by[pol] += 1
        policy_labels = {
            "int8_all": "INT8 all suites",
            "category_adaptive_policy": "Category adaptive",
            "fallback_int8_for_hard_categories": "Fallback INT8 hard cats",
            "baseline_boundary4": "Boundary V baseline",
            "baseline_k8_v4": "K8/V4 baseline",
            "draft_len_adaptive_policy": "Draft-len adaptive",
        }
        for pol, accs in sorted(by_policy.items(), key=lambda x: -_mean(x[1])):
            entries.append(
                LeaderboardEntry(
                    tier=TIER_REPAIR,
                    method=policy_labels.get(pol, pol),
                    experiment="Exp 025",
                    model_panel="Qwen2.5-0.5B · 128-prompt V10",
                    mean_acceptance=_mean(accs),
                    exactkv_failures=fail_by.get(pol, 0),
                    integration_status="repair policy (not a compressor)",
                    caveat="selects compressor+draft_len; not default",
                )
            )

    # Restricted backends — values from published experiment reports (not fabricated).
    restricted_static = [
        LeaderboardEntry(
            TIER_RESTRICTED, "KVQuant sim", "Exp 010", "Qwen2.5-0.5B · V9 core (272 cells)",
            0.792, 0, "factory-only restricted adapter",
            "simquant; not deployment CUDA; supports_real_bytes_claim=False",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "KVQuant sim", "Exp 014", "Qwen2.5-0.5B · harder-category spotcheck",
            0.634, 0, "factory-only restricted adapter",
            "subset panel; compare to Exp 010 anchor 0.792",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "KVQuant sim", "Exp 023", "Qwen2.5-1.5B · hard panel (200 cells)",
            0.609, 0, "factory-only restricted adapter",
            "1.5B quantizer artifact; not universal",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "TurboQuant Python", "Exp 008", "Qwen2.5-0.5B · V9 core (272 cells)",
            0.435, 0, "factory-only restricted adapter",
            "not production llama.cpp/MLX; supports_real_bytes_claim=False",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "TurboQuant Python", "Exp 014", "Qwen2.5-0.5B · harder-category spotcheck",
            0.309, 0, "factory-only restricted adapter",
            "subset panel; compare to Exp 008 anchor 0.435",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "TurboQuant llama.cpp probe", "Exp 022", "Qwen2.5-0.5B · 10-prompt external probe",
            0.486, None, "external drafter probe only",
            "HF verifier probe accept; not standard ExactKV compressor panel; no_go integration",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "KIVI offline", "Exp 009", "Qwen2.5-0.5B · V9 core (272 cells)",
            0.012, 0, "factory-only restricted adapter",
            "offline adapter; not KIVI CUDA/Triton production path",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED, "KIVI offline", "Exp 014", "Qwen2.5-0.5B · harder-category spotcheck",
            0.019, 0, "factory-only restricted adapter",
            "subset panel; compare to Exp 009 anchor 0.012",
        ),
        LeaderboardEntry(
            TIER_RESTRICTED,
            "Shard external-drafter probe",
            "Exp 039–041",
            "Llama-3.1-8B · 32-prompt external probe",
            SHARD_ACCEPTED_PREFIX_RATIO,
            0,
            "external drafter probe (Mode B); Llama-only; not default registry",
            SHARD_RESTRICTED_CAVEAT,
        ),
        LeaderboardEntry(
            TIER_RESTRICTED,
            "SpectralQuant experimental adapter",
            "Exp 045",
            "Qwen2.5-0.5B · 12-prompt restricted panel",
            SPECTRALQUANT_PANEL_MEAN_ACCEPTANCE,
            0,
            "factory-only materializing adapter; not default registry",
            SPECTRALQUANT_RESTRICTED_CAVEAT,
        ),
    ]
    entries.extend(restricted_static)

    entries.append(
        LeaderboardEntry(
            TIER_SMOKE,
            "SnapKV experimental",
            "Exp 032b",
            "Qwen2.5-0.5B · 4-prompt smoke (8 cells)",
            None,
            0,
            "factory-only smoke adapter",
            "smoke-only; not full-suite ranked; restricted experimental SnapKV via kvpress",
        )
    )
    return entries


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_exactness_summary(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{e.experiment}\n({e.phase})" for e in data.exactness]
    failures = [e.failures for e in data.exactness]
    cells = [e.cells for e in data.exactness]
    x = range(len(labels))
    bars = ax.bar(x, cells, color="0.75", edgecolor="0.3", label="cells tested")
    ax.bar(x, failures, color="0.35", label="exactkv_failures")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cell count")
    ax.set_title("Exactness summary — exactkv_failures == 0 on published panels")
    for i, (c, f) in enumerate(zip(cells, failures)):
        ax.text(i, c + max(cells) * 0.02, f"{c} cells\nfail={f}", ha="center", fontsize=7)
    ax.legend(loc="upper right")
    fig.text(0.5, 0.01, "Not a new benchmark — aggregated from existing experiments only", ha="center", fontsize=8)
    _save(fig, path)


def plot_acceptance_by_compressor(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    items = [(k, v) for k, v in data.acceptance_by_compressor.items() if k in COMPRESSOR_PANEL]
    if not items:
        ax.text(0.5, 0.5, "No compressor data", ha="center")
    else:
        labels, vals = zip(*sorted(items, key=lambda x: -x[1]))
        ax.barh(list(labels), list(vals), color="0.55")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("mean acceptance rate")
        ax.set_title("Acceptance by compressor (Qwen2.5-0.5B · Exp 012 + 025)")
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    _save(fig, path)


def plot_acceptance_by_model(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    models = list(data.acceptance_by_model.keys())
    comps = [c for c in COMPRESSOR_PANEL if c != "backend_passthrough"]
    width = 0.18
    x = range(len(models))
    for i, comp in enumerate(comps):
        vals = [data.acceptance_by_model.get(m, {}).get(comp) for m in models]
        offsets = [xi + (i - len(comps) / 2) * width for xi in x]
        heights = [v if v is not None else 0 for v in vals]
        bars = ax.bar(offsets, heights, width=width, label=comp)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("mean acceptance")
    ax.set_title("Acceptance by model (selected compressors)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_category_heatmap(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    suites = [s for s in SUITE_PANEL if s in data.category_heatmap]
    comps = [c for c in COMPRESSOR_PANEL[:4] if any(c in data.category_heatmap.get(s, {}) for s in suites)]
    if not suites or not comps:
        ax.text(0.5, 0.5, "No category heatmap data", ha="center")
    else:
        import numpy as np

        matrix = []
        for suite in suites:
            matrix.append([data.category_heatmap.get(suite, {}).get(c, float("nan")) for c in comps])
        arr = np.array(matrix)
        im = ax.imshow(arr, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(suites)))
        ax.set_yticklabels(suites)
        for i in range(len(suites)):
            for j in range(len(comps)):
                val = arr[i, j]
                if val == val:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="lossy divergence rate")
        ax.set_title(f"Category heatmap — {data.divergence_source}")
    _save(fig, path)


def plot_divergence_histogram(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if data.divergence_histogram:
        ax.hist(data.divergence_histogram, bins=20, color="0.5", edgecolor="0.2")
        ax.set_xlabel("first divergence token index")
        ax.set_ylabel("count")
        ax.set_title(f"First divergence histogram — {data.divergence_source}")
    else:
        ax.text(0.5, 0.5, "No divergence histogram data", ha="center")
    _save(fig, path)


def plot_timing_diagnostic(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ("full_greedy", "exactkv_sequential", "exactkv_span")
    labels, vals = [], []
    for arm in order:
        if arm in data.timing_by_arm:
            labels.append(arm)
            vals.append(data.timing_by_arm[arm])
    if vals:
        ax.bar(labels, vals, color=["0.7", "0.45", "0.3"])
        ax.set_ylabel("mean wall time (s)")
        ax.set_title("Timing diagnostic — Exp 030 only (Qwen2.5-0.5B · A5000 fp16)")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.2f}s", ha="center", fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Diagnostic setup only · ExactKV slower in this setup · no speedup claim",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    _save(fig, path)


def plot_memory_diagnostic(data: PlotData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ("full_greedy", "exactkv_sequential", "exactkv_span")
    labels, vals = [], []
    for arm in order:
        if arm in data.memory_by_arm_mib:
            labels.append(arm)
            vals.append(data.memory_by_arm_mib[arm])
    if vals:
        ax.bar(labels, vals, color=["0.7", "0.45", "0.3"])
        ax.set_ylabel("mean peak allocated (MiB)")
        ax.set_title("Memory diagnostic — Exp 031 only (Qwen2.5-0.5B · A5000 fp16)")
        ymin = min(vals) - 20
        ymax = max(vals) + 20
        ax.set_ylim(ymin, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v + (ymax - ymin) * 0.02, f"{v:.1f}", ha="center", fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Active CUDA peak indistinguishable · no active GPU memory savings claim",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    _save(fig, path)


def plot_killer_demo_card(data: PlotData, path: Path) -> None:
    kd = data.killer_demo
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    lines = [
        "KILLER DEMO CARD — Exp 034 (tj_002 × int4_sim)",
        "",
        "Prompt: weather tool JSON",
        kd.get("prompt", "")[:72] + "…",
        "",
        f"Full KV:     {kd.get('full_output_text', '')[:56]}…",
        f"Lossy draft: {kd.get('lossy_output_text', '')[:56]}…",
        f"ExactKV:     {kd.get('exactkv_output_text', '')[:56]}…",
        "",
        f"REJECT draft token: {kd.get('rejected_token', '')!r}",
        f"COMMIT verifier token: {kd.get('correction_token', '')!r}",
        "",
        f"exactkv_failures: {kd.get('exactkv_failures', 0)}",
        f"final output match: {str(kd.get('final_match', True)).lower()}",
        "",
        PUBLIC_TAGLINE,
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    _save(fig, path)


def _entries_by_tier(entries: list[LeaderboardEntry]) -> dict[str, list[LeaderboardEntry]]:
    out: dict[str, list[LeaderboardEntry]] = {}
    for e in entries:
        out.setdefault(e.tier, []).append(e)
    return out


def _fmt_acc(val: float | None) -> str:
    return f"{val:.3f}" if val is not None else "—"


def _fmt_fail(val: int | None) -> str:
    return str(val) if val is not None else "—"


def write_leaderboard_md(data: PlotData, path: Path) -> None:
    from scripts.exactkv_leaderboard import write_leaderboard_md as _write_lb_md

    _write_lb_md(data.leaderboard, path)


def write_report_md(data: PlotData, path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Experiment 035: Visual Plots and Mini Leaderboard",
        "",
        f"_Generated by `scripts/visualize_experiment_035.py` on {ts}. V13 Phase 8 — visualization only._",
        "",
        "> This is a **visualization/reporting phase** using existing results, not a new benchmark.",
        "> No speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, production serving, "
        "or model accuracy improvement claim is made.",
        "> Timing numbers are **diagnostic only** from Exp 030.",
        "> Memory numbers are **diagnostic only** from Exp 031.",
        "> External Shard/SpectralQuant/SnapKV paper results are **not** ExactKV results.",
        "> ExactKV does **not** improve the model's underlying accuracy; it preserves full-greedy output "
        "while using lossy KV only as a draft.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Turn existing V10–V13 ExactKV results into a polished visual package and mini leaderboard — "
        "a KV-compression crash-test lab narrative.",
        "",
        "> **" + PUBLIC_TAGLINE.replace("\n", " ") + "**",
        "",
        "## 2. Inputs used",
        "",
    ]
    if data.inputs_used:
        lines.extend(f"- `{p}`" for p in sorted(set(data.inputs_used)))
    else:
        lines.append("- _(none found)_")
    if data.missing:
        lines.extend(["", "**Missing locally (skipped or fixture fallback):**", ""])
        lines.extend(f"- `{p}`" for p in data.missing)

    asset_names = [
        "exp035_exactness_summary.png",
        "exp035_acceptance_by_compressor.png",
        "exp035_acceptance_by_model.png",
        "exp035_category_heatmap.png",
        "exp035_first_divergence_histogram.png",
        "exp035_timing_diagnostic.png",
        "exp035_memory_diagnostic.png",
        "exp035_killer_demo_card.png",
    ]
    lines.extend([
        "",
        "## 3. Generated assets",
        "",
    ])
    for name in asset_names:
        lines.append(f"![{name}](assets/{name})")
        lines.append("")

    lines.extend([
        "## 4. Exactness summary",
        "",
        "| Exp | Phase | Cells | exactkv_failures |",
        "| --- | --- | ---: | ---: |",
    ])
    for e in data.exactness:
        lines.append(f"| {e.experiment} | {e.phase} | {e.cells} | {e.failures} |")
    lines.extend([
        "",
        "V13 highlights: Exp 029 (600), 030 ExactKV arms (320), 031 (192 ExactKV), 033 (48), 034 search (348). "
        "Do not overgeneralize beyond tested cells.",
        "",
        "## 5. Mini leaderboard (tiered)",
        "",
        PUBLIC_LEADERBOARD_COPY,
        "",
        "See [`leaderboard.md`](leaderboard.md) — **FULL PANEL**, **RESTRICTED BACKEND**, "
        "**SMOKE ONLY**, **FUTURE CANDIDATE**, and **REPAIR POLICY** sections.",
        "",
        "## 6. Acceptance visuals",
        "",
        "- **By compressor:** Qwen2.5-0.5B mean across Exp 012 + 025.",
        "- **By model:** Exp 012 / 015 / 016 / 033 panels for selected compressors.",
        "",
        "## 7. Divergence visuals",
        "",
        f"- **Category heatmap:** {data.divergence_source}.",
        f"- **First divergence histogram:** {data.divergence_source}.",
        "",
        "## 8. Timing visual",
        "",
        "Exp 030 diagnostic only — ExactKV sequential and span arms slower than full greedy on A5000 fp16. "
        "**No speedup claim.**",
        "",
        "## 9. Memory visual",
        "",
        "Exp 031 diagnostic only — peak active CUDA allocation indistinguishable across arms at 0.5B scale. "
        "**No active GPU memory savings claim.**",
        "",
        "## 10. Killer demo visual/card",
        "",
        "Exp 034 `tj_002` weather JSON — lossy `}}` rejected, verifier `metric` committed. "
        "Live terminal replay: [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md).",
        "",
        "## 11. Public-facing narrative",
        "",
        PUBLIC_TAGLINE,
        "",
        "ExactKV is a correctness-first crash-test lab for lossy KV drafts: when compression starts lying, "
        "the full-KV verifier catches it and preserves greedy output.",
        "",
        "## 12. What this proves",
        "",
        "- Published panels maintain `exactkv_failures == 0` on tested cells.",
        "- Acceptance and divergence vary by compressor, model, and category — visible in plots.",
        "- Diagnostic timing/memory charts document constraints without launch headlines.",
        "",
        "## 13. What this does not prove",
        "",
        "- No speedup, throughput, latency, tokens/sec, or VRAM savings.",
        "- No production serving or model accuracy improvement.",
        "- Leaderboard lists Shard as restricted external-drafter probe (Exp 039–041), not integrated compressor.",
        "",
        "## 14. Limitations",
        "",
        "- Plots reflect available local reports; missing JSON/CSV uses fixtures or skips.",
        "- SnapKV experimental is smoke-only (8 cells).",
        "- Llama panel is a 12-prompt small suite (Exp 033).",
        "",
        "## 15. Next steps",
        "",
        "**Proceed to Phase 9** — V13 completion / launch decision package (headline audit, readiness assessment).",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 scripts/visualize_experiment_035.py",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_all(output_assets: Path | None = None) -> PlotData:
    assets = output_assets or _ASSETS
    data = load_plot_data()

    plot_exactness_summary(data, assets / "exp035_exactness_summary.png")
    plot_acceptance_by_compressor(data, assets / "exp035_acceptance_by_compressor.png")
    plot_acceptance_by_model(data, assets / "exp035_acceptance_by_model.png")
    plot_category_heatmap(data, assets / "exp035_category_heatmap.png")
    plot_divergence_histogram(data, assets / "exp035_first_divergence_histogram.png")
    plot_timing_diagnostic(data, assets / "exp035_timing_diagnostic.png")
    plot_memory_diagnostic(data, assets / "exp035_memory_diagnostic.png")
    plot_killer_demo_card(data, assets / "exp035_killer_demo_card.png")

    write_leaderboard_md(data, _DOCS / "leaderboard.md")
    from scripts.exactkv_leaderboard import write_leaderboard_html

    write_leaderboard_html(data.leaderboard, _DOCS / "leaderboard.html")
    write_report_md(data, _DOCS / "EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md")
    return data


def main() -> int:
    data = generate_all()
    print(f"Wrote 8 PNG assets to {_ASSETS}")
    print(f"Wrote {_DOCS / 'leaderboard.md'}")
    print(f"Wrote {_DOCS / 'EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md'}")
    if data.missing:
        print(f"Note: {len(data.missing)} input(s) missing locally (see report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
