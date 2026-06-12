#!/usr/bin/env python3
"""Experiment 036 / V13 Phase 8b: public visual polish package.

Renders launch-quality cards from existing experiment data only.
No model inference, timing benchmarks, or memory benchmarks are run.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.figure import Figure

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.visualize_experiment_035 import (  # noqa: E402
    EXP034_FIXTURE,
    PUBLIC_LEADERBOARD_COPY,
    PUBLIC_TAGLINE,
    TIER_FULL_PANEL,
    TIER_FUTURE,
    TIER_RESTRICTED,
    TIER_SMOKE,
    LeaderboardEntry,
    PlotData,
    _entries_by_tier,
    load_plot_data,
)

_ASSETS = _ROOT / "docs" / "assets"
_DOCS = _ROOT / "docs"
_THREAD_DIR = _ASSETS / "public_exactkv_launch_thread_cards"

# --- Design system (dark launch cards) ---
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BORDER = "#30363d"

COMPRESSOR_LABELS = {
    "noop": "No compression",
    "backend_passthrough": "Passthrough",
    "int8": "INT8",
    "k8_v4_sim": "K8/V4",
    "k8_v4_boundary4_v8_sim": "Boundary V",
    "snapkv_experimental": "SnapKV (smoke)",
}

PUBLIC_EXACTNESS_EXPS = (
    ("012", "Exp 012"),
    ("015", "Exp 015"),
    ("016", "Exp 016"),
    ("025", "Exp 025"),
    ("029", "Exp 029"),
    ("030", "Exp 030"),
    ("031", "Exp 031"),
    ("033", "Exp 033"),
    ("034", "Exp 034"),
)


def _new_fig(w: float, h: float) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def _card_bg(ax: Any, x: float, y: float, w: float, h: float) -> None:
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=BORDER,
        facecolor=CARD,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(rect)


def _footer(ax: Any, text: str, y: float = 0.04) -> None:
    ax.text(0.5, y, text, ha="center", va="bottom", fontsize=9, color=MUTED, transform=ax.transAxes)


def load_public_data() -> PlotData:
    return load_plot_data()


def _exactness_for_public(data: PlotData) -> list[tuple[str, str, int, int]]:
    by_id = {e.experiment: e for e in data.exactness}
    out: list[tuple[str, str, int, int]] = []
    for exp_id, label in PUBLIC_EXACTNESS_EXPS:
        e = by_id.get(exp_id)
        if e:
            out.append((exp_id, label, e.cells, e.failures))
    return out


def _tier_badge_color(tier: str) -> str:
    return {
        TIER_FULL_PANEL: ACCENT,
        TIER_RESTRICTED: AMBER,
        TIER_SMOKE: AMBER,
        TIER_FUTURE: MUTED,
    }.get(tier, MUTED)


def render_hero_card(path: Path) -> None:
    fig, ax = _new_fig(12, 7)
    _card_bg(ax, 0.05, 0.08, 0.9, 0.84)
    ax.text(0.5, 0.82, "ExactKV", ha="center", fontsize=42, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(
        0.5,
        0.70,
        "A crash-test lab for KV-cache compression",
        ha="center",
        fontsize=20,
        color=TEXT,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.60,
        "Compressed KV drafts.  Full KV verifies.",
        ha="center",
        fontsize=16,
        color=MUTED,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.46,
        PUBLIC_TAGLINE.replace("\n", "\n"),
        ha="center",
        fontsize=15,
        fontstyle="italic",
        color=TEXT,
        transform=ax.transAxes,
        linespacing=1.5,
    )
    ax.text(
        0.5,
        0.28,
        "exactkv_failures == 0  on tested panels",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color=GREEN,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.18,
        "No speed / memory claims without measurement",
        ha="center",
        fontsize=12,
        color=AMBER,
        transform=ax.transAxes,
    )
    _footer(ax, "Visualization only · not a new benchmark")
    _save(fig, path)


def render_killer_correction_card(data: PlotData, path: Path) -> None:
    kd = data.killer_demo or EXP034_FIXTURE
    fig, ax = _new_fig(12, 8)
    _card_bg(ax, 0.04, 0.06, 0.92, 0.88)

    ax.text(0.5, 0.90, "When compressed KV starts lying", ha="center", fontsize=22, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.84, "Exp 034 · weather tool JSON · int4_sim", ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    prompt = 'Complete tool JSON: {"name": "get_weather", … "units":'
    ax.text(0.08, 0.76, "Prompt", fontsize=11, color=MUTED, transform=ax.transAxes)
    ax.text(0.08, 0.71, prompt, fontsize=12, color=TEXT, family="monospace", transform=ax.transAxes)

    # Lossy draft box
    lossy_rect = mpatches.FancyBboxPatch(
        (0.08, 0.48),
        0.35,
        0.18,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor=RED,
        facecolor="#2d1b1b",
        transform=ax.transAxes,
    )
    ax.add_patch(lossy_rect)
    ax.text(0.245, 0.62, "Lossy compressed KV draft", ha="center", fontsize=10, color=RED, transform=ax.transAxes)
    rej = kd.get("rejected_token", "}}")
    ax.text(0.245, 0.54, f'… "{rej}"', ha="center", fontsize=28, fontweight="bold", color=RED, transform=ax.transAxes)

    # Arrow reject
    ax.annotate(
        "",
        xy=(0.52, 0.57),
        xytext=(0.45, 0.57),
        arrowprops=dict(arrowstyle="->", color=RED, lw=2.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
    )
    ax.text(0.50, 0.62, "REJECT", ha="center", fontsize=11, fontweight="bold", color=RED, transform=ax.transAxes)

    # Verifier correction box
    corr_rect = mpatches.FancyBboxPatch(
        (0.55, 0.48),
        0.35,
        0.18,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor=GREEN,
        facecolor="#1b2d1f",
        transform=ax.transAxes,
    )
    ax.add_patch(corr_rect)
    ax.text(0.725, 0.62, "Full-KV verifier commits", ha="center", fontsize=10, color=GREEN, transform=ax.transAxes)
    corr = kd.get("correction_token", "metric")
    ax.text(0.725, 0.54, f'"{corr}"', ha="center", fontsize=28, fontweight="bold", color=GREEN, transform=ax.transAxes)

    ax.text(0.5, 0.38, "ExactKV failures: 0", ha="center", fontsize=16, color=GREEN, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.32, "Final output match: true", ha="center", fontsize=14, color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.24, "Rejected draft token never committed to authoritative KV", ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    ax.text(
        0.5,
        0.12,
        PUBLIC_TAGLINE.replace("\n", "  ·  "),
        ha="center",
        fontsize=11,
        fontstyle="italic",
        color=ACCENT,
        transform=ax.transAxes,
    )
    _footer(ax, "Correctness demo · not timing or memory evidence")
    _save(fig, path)


def render_exactness_wall(data: PlotData, path: Path) -> None:
    rows = _exactness_for_public(data)
    fig, ax = _new_fig(12, 7)
    _card_bg(ax, 0.05, 0.10, 0.9, 0.82)
    ax.text(0.5, 0.86, "Exactness wall", ha="center", fontsize=24, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.80, "tested panels only", ha="center", fontsize=12, color=AMBER, transform=ax.transAxes)

    n = len(rows)
    cols = 3
    for i, (exp_id, label, cells, failures) in enumerate(rows):
        col = i % cols
        row = i // cols
        x = 0.12 + col * 0.30
        y = 0.58 - row * 0.22
        tile = mpatches.FancyBboxPatch(
            (x, y),
            0.26,
            0.16,
            boxstyle="round,pad=0.008",
            linewidth=1,
            edgecolor=GREEN if failures == 0 else RED,
            facecolor=CARD,
            transform=ax.transAxes,
        )
        ax.add_patch(tile)
        ax.text(x + 0.13, y + 0.12, label, ha="center", fontsize=13, fontweight="bold", color=TEXT, transform=ax.transAxes)
        ax.text(x + 0.13, y + 0.07, f"{cells:,} cells", ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)
        ax.text(x + 0.13, y + 0.02, f"failures = {failures}", ha="center", fontsize=11, fontweight="bold", color=GREEN, transform=ax.transAxes)

    ax.text(0.5, 0.14, "exactkv_failures == 0 across published panels shown", ha="center", fontsize=13, color=GREEN, transform=ax.transAxes)
    _footer(ax, "Do not overgeneralize beyond tested cells")
    _save(fig, path)


def render_leaderboard_card(data: PlotData, path: Path) -> None:
    by_tier = _entries_by_tier(data.leaderboard)
    fig, ax = _new_fig(12, 10)
    _card_bg(ax, 0.04, 0.05, 0.92, 0.90)
    ax.text(0.5, 0.96, "ExactKV crash-test leaderboard", ha="center", fontsize=20, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.92, PUBLIC_LEADERBOARD_COPY[:90] + "…", ha="center", fontsize=8, color=MUTED, transform=ax.transAxes)

    y = 0.86

    def draw_tier_header(tier: str) -> None:
        nonlocal y
        ax.text(0.08, y, tier, fontsize=11, fontweight="bold", color=_tier_badge_color(tier), transform=ax.transAxes)
        y -= 0.035

    def draw_row(entry: LeaderboardEntry, *, show_bar: bool) -> None:
        nonlocal y
        label = entry.method
        if entry.rank is not None:
            label = f"{entry.rank}. {label}"
        ax.text(0.10, y, label, fontsize=10, fontweight="bold", color=TEXT, transform=ax.transAxes)
        if show_bar and entry.mean_acceptance is not None:
            bar_w = min(entry.mean_acceptance, 1.0) * 0.28
            ax.add_patch(
                mpatches.Rectangle((0.52, y - 0.012), bar_w, 0.022, facecolor=ACCENT, transform=ax.transAxes)
            )
            ax.text(0.52 + bar_w + 0.01, y, f"{entry.mean_acceptance:.0%}", fontsize=9, color=TEXT, transform=ax.transAxes)
        elif entry.mean_acceptance is not None:
            ax.text(0.52, y, f"{entry.mean_acceptance:.0%}", fontsize=9, color=TEXT, transform=ax.transAxes)
        else:
            ax.text(0.52, y, "—", fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(0.10, y - 0.022, f"{entry.experiment} · {entry.model_panel[:42]}", fontsize=7, color=MUTED, transform=ax.transAxes)
        y -= 0.055

    # FULL PANEL — top 5 ranked only on card
    full = by_tier.get(TIER_FULL_PANEL, [])[:5]
    if full:
        draw_tier_header(TIER_FULL_PANEL)
        for e in full:
            draw_row(e, show_bar=True)
        y -= 0.01

    # RESTRICTED — top 4 by acceptance
    restricted = sorted(
        [e for e in by_tier.get(TIER_RESTRICTED, []) if e.mean_acceptance is not None],
        key=lambda e: -(e.mean_acceptance or 0),
    )[:4]
    if restricted:
        draw_tier_header(TIER_RESTRICTED)
        for e in restricted:
            draw_row(e, show_bar=False)
        y -= 0.01

    # SMOKE + FUTURE — text only (not ranked)
    for tier in (TIER_SMOKE, TIER_FUTURE):
        tier_rows = by_tier.get(tier, [])
        if not tier_rows:
            continue
        draw_tier_header(tier)
        for e in tier_rows:
            ax.text(0.10, y, f"{e.method} — {e.integration_status}", fontsize=9, color=TEXT, transform=ax.transAxes)
            ax.text(0.10, y - 0.022, e.caveat[:70], fontsize=7, color=MUTED, transform=ax.transAxes)
            y -= 0.05
        y -= 0.01

    _footer(ax, "Tiered sections · SnapKV smoke not ranked vs full panel")
    _save(fig, path)


def render_timing_truth_card(data: PlotData, path: Path) -> None:
    timing = data.timing_by_arm
    full = timing.get("full_greedy", 0.565)
    seq = timing.get("exactkv_sequential", 1.496)
    span = timing.get("exactkv_span", 1.646)

    fig, ax = _new_fig(11, 6.5)
    _card_bg(ax, 0.05, 0.08, 0.9, 0.84)
    ax.text(0.5, 0.86, "ExactKV tells the truth about speed too", ha="center", fontsize=20, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.79, "Exp 030 diagnostic · Qwen2.5-0.5B · A5000 fp16", ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    labels = ["Full KV\ngreedy", "ExactKV\nsequential", "ExactKV\nspan"]
    vals = [full, seq, span]
    colors = [ACCENT, AMBER, AMBER]
    xpos = [0.25, 0.50, 0.75]
    max_v = max(vals) * 1.15
    for x, lab, val, col in zip(xpos, labels, vals, colors):
        h = 0.45 * (val / max_v)
        ax.add_patch(mpatches.Rectangle((x - 0.08, 0.22), 0.16, h, facecolor=col, transform=ax.transAxes))
        ax.text(x, 0.22 + h + 0.03, f"{val:.2f}s", ha="center", fontsize=14, fontweight="bold", color=TEXT, transform=ax.transAxes)
        ax.text(x, 0.14, lab, ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    ax.text(0.5, 0.38, "Full greedy faster than ExactKV in this setup", ha="center", fontsize=13, color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.32, "No speedup claim · diagnostic only", ha="center", fontsize=12, fontweight="bold", color=AMBER, transform=ax.transAxes)
    _footer(ax, "Hardware/model/prompt specific · not production benchmark")
    _save(fig, path)


def render_memory_truth_card(data: PlotData, path: Path) -> None:
    mem = data.memory_by_arm_mib
    full = mem.get("full_greedy", 1195.3)
    seq = mem.get("exactkv_sequential", 1194.9)
    span = mem.get("exactkv_span", 1195.1)

    fig, ax = _new_fig(11, 6.5)
    _card_bg(ax, 0.05, 0.08, 0.9, 0.84)
    ax.text(0.5, 0.86, "Active CUDA peak: measured honestly", ha="center", fontsize=20, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.5, 0.79, "Exp 031 diagnostic · Qwen2.5-0.5B · A5000 fp16", ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    labels = ["Full KV", "ExactKV\nsequential", "ExactKV\nspan"]
    vals = [full, seq, span]
    xpos = [0.25, 0.50, 0.75]
    base = min(vals) - 30
    scale = 0.45 / (max(vals) - base)
    for x, lab, val in zip(xpos, labels, vals):
        h = (val - base) * scale
        ax.add_patch(mpatches.Rectangle((x - 0.08, 0.22), 0.16, h, facecolor=ACCENT, transform=ax.transAxes))
        ax.text(x, 0.22 + h + 0.03, f"{val:.0f} MiB", ha="center", fontsize=13, fontweight="bold", color=TEXT, transform=ax.transAxes)
        ax.text(x, 0.14, lab, ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    ax.text(
        0.5,
        0.38,
        "V5 accounting improves; active CUDA peak did not move at 0.5B scale",
        ha="center",
        fontsize=12,
        color=TEXT,
        transform=ax.transAxes,
    )
    ax.text(0.5, 0.32, "No active GPU memory savings claim", ha="center", fontsize=12, fontweight="bold", color=AMBER, transform=ax.transAxes)
    _footer(ax, "Model weights dominate peak · diagnostic only")
    _save(fig, path)


def render_one_page_summary(data: PlotData, path: Path) -> None:
    fig, ax = _new_fig(14, 10)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.5, 0.96, "ExactKV — one-page summary", ha="center", fontsize=24, fontweight="bold", color=ACCENT, transform=ax.transAxes)
    ax.text(0.5, 0.91, PUBLIC_TAGLINE.replace("\n", "  ·  "), ha="center", fontsize=11, color=MUTED, transform=ax.transAxes)

    tiles = [
        (0.03, 0.52, 0.46, 0.36, "Hero", "Crash-test lab · exactkv_failures == 0"),
        (0.51, 0.52, 0.46, 0.36, "Killer demo", f"Reject '{data.killer_demo.get('rejected_token', '}}')}' → commit '{data.killer_demo.get('correction_token', 'metric')}'"),
        (0.03, 0.08, 0.30, 0.40, "Exactness", "9 experiments · 0 failures"),
        (0.35, 0.08, 0.30, 0.40, "Speed truth", "ExactKV slower in Exp 030 setup"),
        (0.67, 0.08, 0.30, 0.40, "Memory truth", "Peak indistinguishable · Exp 031"),
    ]
    for x, y, w, h, title, body in tiles:
        _card_bg(ax, x, y, w, h)
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", fontsize=14, fontweight="bold", color=ACCENT, transform=ax.transAxes)
        ax.text(x + 0.02, y + h / 2 - 0.02, body, ha="left", va="center", fontsize=11, color=TEXT, transform=ax.transAxes, wrap=True)
    _footer(ax, "Public visual package · Exp 036 · tested panels only")
    _save(fig, path)


def render_thread_cards(data: PlotData, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        ("thread_01_hook.png", render_hero_card),
        ("thread_02_killer.png", lambda p: render_killer_correction_card(data, p)),
        ("thread_03_exactness.png", lambda p: render_exactness_wall(data, p)),
        ("thread_04_leaderboard.png", lambda p: render_leaderboard_card(data, p)),
        ("thread_05_timing.png", lambda p: render_timing_truth_card(data, p)),
        ("thread_06_memory.png", lambda p: render_memory_truth_card(data, p)),
    ]
    paths: list[Path] = []
    for name, fn in cards:
        p = out_dir / name
        fn(p)
        paths.append(p)
    return paths


def write_public_visual_package_md(data: PlotData, thread_paths: list[Path]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Public Visual Package (V13 Phase 8b / Exp 036)",
        "",
        f"_Generated by `scripts/render_public_visuals_036.py` on {ts}._",
        "",
        "> This is a **visualization/reporting phase** using existing results, not a new benchmark.",
        "> No speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, production serving, "
        "or model accuracy improvement claim is made.",
        "> Timing numbers are **diagnostic only** from Exp 030.",
        "> Memory numbers are **diagnostic only** from Exp 031.",
        "> External Shard/SpectralQuant/SnapKV paper results are **not** ExactKV results.",
        "> ExactKV preserves full-greedy output while using lossy KV only as a draft.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Launch-quality visual assets for README hero sections, launch posts, and social media — "
        "polished cards built from existing V10–V13 experiment data.",
        "",
        f"> **{PUBLIC_TAGLINE.replace(chr(10), ' ')}**",
        "",
        "## 2. Assets generated",
        "",
        "### Public-ready (Exp 036)",
        "",
        "| Asset | Use |",
        "| --- | --- |",
        "| [`public_exactkv_hero_card.png`](assets/public_exactkv_hero_card.png) | README hero, launch header |",
        "| [`public_killer_correction_card.png`](assets/public_killer_correction_card.png) | Centerpiece · social/thread |",
        "| [`public_exactness_wall.png`](assets/public_exactness_wall.png) | Exactness gate summary |",
        "| [`public_leaderboard.png`](assets/public_leaderboard.png) | Tiered crash-test leaderboard (FULL / RESTRICTED / SMOKE / FUTURE) |",
        "| [`public_timing_truth_card.png`](assets/public_timing_truth_card.png) | Honest speed diagnostic |",
        "| [`public_memory_truth_card.png`](assets/public_memory_truth_card.png) | Honest memory diagnostic |",
        "| [`public_exactkv_one_page_summary.png`](assets/public_exactkv_one_page_summary.png) | One-pager overview |",
        "",
        "### Launch thread cards",
        "",
    ]
    for p in thread_paths:
        rel = p.relative_to(_DOCS)
        lines.append(f"- [`{p.name}`]({rel})")
    lines.extend([
        "",
        "## 3. Data sources",
        "",
        "Same inputs as Exp 035: reports CSV/JSON from Exp 012, 015, 016, 019, 025, 029, 030, 031, 033, 034.",
        "",
        "## 4. How to regenerate",
        "",
        "```bash",
        "python3 scripts/render_public_visuals_036.py",
        "```",
        "",
        "## 5. Public-ready assets",
        "",
        "All `public_*.png` files in `docs/assets/` listed above — designed for external audiences.",
        "",
        "## 6. Internal-only assets",
        "",
        "Exp 035 research figures remain internal documentation style:",
        "",
        "- `exp035_exactness_summary.png`",
        "- `exp035_acceptance_by_compressor.png`",
        "- `exp035_acceptance_by_model.png`",
        "- `exp035_category_heatmap.png`",
        "- `exp035_first_divergence_histogram.png`",
        "- `exp035_timing_diagnostic.png`",
        "- `exp035_memory_diagnostic.png`",
        "- `exp035_killer_demo_card.png`",
        "",
        "See [`EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md`](EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md).",
        "",
        "## 7. Allowed claims",
        "",
        "- `exactkv_failures == 0` on tested panels shown.",
        "- Lossy KV can draft wrong tokens; ExactKV rejects and corrects (Exp 034).",
        "- Mean acceptance varies by compressor on tested panels.",
        "- Exp 030: ExactKV slower than full greedy in diagnostic setup (honesty framing).",
        "- Exp 031: active CUDA peak indistinguishable at 0.5B scale (no savings claim).",
        "",
        "## 8. Forbidden claims",
        "",
        "- Speedup, throughput, latency, tokens/sec, or runtime improvement.",
        "- Active GPU memory savings or production serving readiness.",
        "- Model accuracy improvement.",
        "- Shard/SpectralQuant as integrated ExactKV results.",
        "",
        "## 9. Suggested launch use",
        "",
        "1. **README hero:** `public_exactkv_hero_card.png`",
        "2. **Twitter/thread:** `public_exactkv_launch_thread_cards/thread_*.png` in order",
        "3. **Blog centerpiece:** `public_killer_correction_card.png`",
        "4. **Honesty section:** timing + memory truth cards",
        "5. **Leaderboard:** [`leaderboard.md`](leaderboard.md) + `public_leaderboard.png` (tiered)",
        "6. **Live demo / video:** [`EXACTKV_CRASH_TEST_VIDEO.md`](EXACTKV_CRASH_TEST_VIDEO.md)",
        "",
        PUBLIC_LEADERBOARD_COPY,
        "",
    ])
    (_DOCS / "PUBLIC_VISUAL_PACKAGE.md").write_text("\n".join(lines), encoding="utf-8")


def generate_all() -> PlotData:
    data = load_public_data()
    render_hero_card(_ASSETS / "public_exactkv_hero_card.png")
    render_killer_correction_card(data, _ASSETS / "public_killer_correction_card.png")
    render_exactness_wall(data, _ASSETS / "public_exactness_wall.png")
    render_leaderboard_card(data, _ASSETS / "public_leaderboard.png")
    render_timing_truth_card(data, _ASSETS / "public_timing_truth_card.png")
    render_memory_truth_card(data, _ASSETS / "public_memory_truth_card.png")
    render_one_page_summary(data, _ASSETS / "public_exactkv_one_page_summary.png")
    thread_paths = render_thread_cards(data, _THREAD_DIR)
    write_public_visual_package_md(data, thread_paths)
    return data


def main() -> int:
    data = generate_all()
    print(f"Wrote public visuals to {_ASSETS}")
    print(f"Wrote thread cards to {_THREAD_DIR}")
    print(f"Wrote {_DOCS / 'PUBLIC_VISUAL_PACKAGE.md'}")
    if data.missing:
        print(f"Note: {len(data.missing)} input(s) missing (fixtures used where applicable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
