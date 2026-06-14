#!/usr/bin/env python3
"""ExactKV crash-test leaderboard — terminal + static HTML (V13 Phase 8f).

Reads local experiment CSV reports and renders a tiered leaderboard.
No hosted backend, no model inference, no timing/memory benchmarks.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.visualize_experiment_035 import (  # noqa: E402
    PUBLIC_LEADERBOARD_COPY,
    PUBLIC_TAGLINE,
    TIER_FULL_PANEL,
    TIER_FUTURE,
    TIER_REPAIR,
    TIER_RESTRICTED,
    TIER_SMOKE,
    LeaderboardEntry,
    PlotData,
    _entries_by_tier,
    _fmt_acc,
    _fmt_fail,
    build_tiered_leaderboard,
    load_plot_data,
)

_TITLE = "KV Compression Crash-Test Leaderboard"
_SUBTITLE = (
    "Compressors ranked by acceptance and exactness under full-KV verification"
)
_FOOTER = (
    "No speedup, memory savings, or serving claims. "
    "ExactKV measures behavioral equivalence to full KV."
)
_DISCLAIMER = (
    "Not a speed leaderboard. ExactKV measures when lossy KV drafts start lying — "
    "not throughput, latency, or production readiness."
)
_NOT_APPLES = (
    "Tiers are not apples-to-apples: full-panel compressors are ranked; "
    "restricted backends, smoke-only adapters, and future candidates are separate."
)

_TIER_SECTIONS = [
    (TIER_FULL_PANEL, "FULL PANEL", "Ranked compressors on integrated full/large panels."),
    (TIER_REPAIR, "REPAIR POLICY", "Adaptive repair policies — not default compressors."),
    (TIER_RESTRICTED, "RESTRICTED BACKEND", "Factory-only or probe adapters — separate tier."),
    (TIER_SMOKE, "SMOKE ONLY", "Smoke adapters — not ranked against full-panel compressors."),
    (TIER_FUTURE, "FUTURE CANDIDATE", "Candidates without ExactKV panel metrics yet."),
]

_WATCH_INTERVAL_SEC = 4

_TIER_TAB_ORDER = [
    (TIER_FULL_PANEL, "full-suite", "Full-suite integrated"),
    (TIER_REPAIR, "repair", "Repair policies"),
    (TIER_RESTRICTED, "restricted", "Restricted backends"),
    (TIER_SMOKE, "smoke", "Smoke-only adapters"),
    (TIER_FUTURE, "future", "Future candidates"),
]


def load_leaderboard_entries() -> tuple[list[LeaderboardEntry], PlotData]:
    data = load_plot_data()
    if not data.leaderboard:
        data.leaderboard = build_tiered_leaderboard(data)
    return data.leaderboard, data


def short_model(model_panel: str) -> str:
    if "Llama" in model_panel:
        return "Llama 8B"
    if "1.5B" in model_panel:
        return "Qwen 1.5B"
    if "3B" in model_panel:
        return "Qwen 3B"
    if "0.5B" in model_panel:
        return "Qwen 0.5B"
    return model_panel.split("·")[0].strip()


def panel_scope(entry: LeaderboardEntry) -> str:
    panel = entry.model_panel.lower()
    if entry.tier == TIER_SMOKE or "smoke" in panel:
        return "smoke"
    if "small suite" in panel or "12-prompt" in panel:
        return "small"
    if "spotcheck" in panel or "probe" in panel or "hard panel" in panel:
        return "subset"
    if "128-prompt" in panel or "272 cells" in panel:
        return "full"
    if entry.tier == TIER_FUTURE:
        return "candidate"
    return "panel"


def entry_badges(entry: LeaderboardEntry) -> list[str]:
    badges: list[str] = []
    tier_badge = {
        TIER_FULL_PANEL: "FULL",
        TIER_REPAIR: "REPAIR",
        TIER_RESTRICTED: "RESTRICTED",
        TIER_SMOKE: "SMOKE",
        TIER_FUTURE: "FUTURE",
    }.get(entry.tier, entry.tier)
    badges.append(tier_badge)

    method_l = entry.method.lower()
    if "shard external-drafter" in method_l:
        badges.extend([
            "EXTERNAL DRAFTER",
            "LLAMA ONLY",
            "NOT DEFAULT",
            "NO SPEED CLAIM",
            "NO MEMORY CLAIM",
        ])
        return badges

    if "spectralquant" in method_l:
        badges.extend([
            "TENSOR PROBE",
            "NOT GENERATION",
            "NOT DEFAULT",
            "NO SPEED CLAIM",
            "NO MEMORY CLAIM",
        ])
        return badges

    status_l = entry.integration_status.lower()
    caveat_l = entry.caveat.lower()
    method_l = entry.method.lower()

    simulated = (
        "sim" in method_l
        or "sim " in status_l
        or "simquant" in caveat_l
        or "sim asymmetric" in status_l
    )
    if simulated:
        badges.append("SIMULATED")

    no_real_byte = (
        "supports_real_bytes_claim=false" in caveat_l.replace(" ", "")
        or (entry.tier == TIER_RESTRICTED and "factory-only" in status_l)
    )
    if no_real_byte:
        badges.append("NO REAL-BYTE CLAIM")
    elif entry.tier in (TIER_FULL_PANEL, TIER_REPAIR) and not simulated:
        badges.append("REAL-BYTE")

    if entry.tier == TIER_FUTURE:
        badges.append("NOT INTEGRATED")

    return badges


def _fmt_badges(badges: list[str], *, plain: bool = False) -> str:
    if plain:
        return "[" + " · ".join(badges) + "]"
    return " ".join(f"[{b}]" for b in badges)


def _emit_title_block(emit: Any, *, plain: bool) -> None:
    if plain:
        emit("EXACTKV CRASH-TEST LEADERBOARD")
        emit(_SUBTITLE)
        emit("")
        return
    width = 72
    bar = "═" * (width - 2)
    emit(f"╔{bar}╗")
    title = "EXACTKV CRASH-TEST LEADERBOARD"
    emit(f"║ {title.center(width - 4)} ║")
    sub = _SUBTITLE
    if len(sub) > width - 6:
        words = sub.split()
        line = ""
        for word in words:
            chunk = (line + " " + word).strip()
            if len(chunk) > width - 6:
                emit(f"║ {line.ljust(width - 4)} ║")
                line = word
            else:
                line = chunk
        if line:
            emit(f"║ {line.ljust(width - 4)} ║")
    else:
        emit(f"║ {sub.center(width - 4)} ║")
    emit(f"╚{bar}╝")
    emit("")


def _emit_section_header(emit: Any, title: str, hint: str, *, plain: bool) -> None:
    if plain:
        emit(f"── {title} ──")
        emit(f"   {hint}")
    else:
        emit(f"┌─ {title} " + "─" * max(0, 68 - len(title)))
        emit(f"│  {hint}")
        emit("└" + "─" * 70)
    emit("")


def _terminal_table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
) -> list[str]:
    if not rows:
        return ["  (no rows — local CSV reports may be missing)"]

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, w in enumerate(widths):
            cell = cells[i] if i < len(cells) else ""
            if len(cell) > w:
                cell = cell[: w - 1] + "…"
            parts.append(cell.ljust(w))
        return "│ " + " │ ".join(parts) + " │"

    out = [hline("┌", "┬", "┐"), fmt_row(headers), hline("├", "┼", "┤")]
    out.extend(fmt_row(r) for r in rows)
    out.append(hline("└", "┴", "┘"))
    return out


def render_terminal(
    entries: list[LeaderboardEntry],
    *,
    out: TextIO | None = None,
    plain: bool = False,
    watch: bool = False,
) -> str:
    if out is None:
        out = sys.stdout
    buf = io.StringIO()
    lines: list[str] = []
    by_tier = _entries_by_tier(entries)

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text, file=out)
        print(text, file=buf)

    if watch and not plain:
        print("\033[2J\033[H", end="", file=out)

    _emit_title_block(emit, plain=plain)
    emit("> " + PUBLIC_TAGLINE.replace("\n", "\n> "))
    emit("")
    emit(_NOT_APPLES)
    emit("")

    for tier, section_title, hint in _TIER_SECTIONS:
        rows = by_tier.get(tier, [])
        _emit_section_header(emit, section_title, hint, plain=plain)

        if tier == TIER_FULL_PANEL:
            table_rows = [
                [
                    str(e.rank or "—"),
                    e.method,
                    short_model(e.model_panel),
                    _fmt_acc(e.mean_acceptance),
                    _fmt_fail(e.exactkv_failures),
                    _fmt_badges(entry_badges(e), plain=plain),
                ]
                for e in rows
            ]
            for row in _terminal_table(
                ["#", "Compressor", "Model", "Accept", "Fail", "Badges"],
                table_rows,
                [3, 14, 10, 8, 6, 28],
            ):
                emit(row)
        elif tier == TIER_REPAIR:
            if rows:
                for e in rows:
                    emit(
                        f"  • {e.method:<26} accept={_fmt_acc(e.mean_acceptance)}  "
                        f"failures={_fmt_fail(e.exactkv_failures)}  "
                        f"{_fmt_badges(entry_badges(e), plain=plain)}"
                    )
            else:
                emit("  (no repair policy rows)")
        elif tier == TIER_RESTRICTED:
            if rows:
                for e in rows:
                    emit(
                        f"  • {e.method:<22} {short_model(e.model_panel):<10} "
                        f"accept={_fmt_acc(e.mean_acceptance)}  "
                        f"failures={_fmt_fail(e.exactkv_failures)}  "
                        f"{_fmt_badges(entry_badges(e), plain=plain)}"
                    )
            else:
                emit("  (no restricted backend rows)")
        elif tier == TIER_SMOKE:
            if rows:
                for e in rows:
                    if e.method.lower() == "spectralquant":
                        emit(
                            f"  • {e.method}: {e.model_panel} — "
                            f"tensor smoke passed  accept=N/A  failures=N/A  "
                            f"{_fmt_badges(entry_badges(e), plain=plain)}"
                        )
                    else:
                        emit(
                            f"  • {e.method}: {e.model_panel} — "
                            f"{_fmt_fail(e.exactkv_failures)} failures  "
                            f"{_fmt_badges(entry_badges(e), plain=plain)}"
                        )
            else:
                emit("  (no smoke rows)")
        elif tier == TIER_FUTURE:
            if rows:
                for e in rows:
                    emit(
                        f"  • {e.method}: {e.caveat}  "
                        f"{_fmt_badges(entry_badges(e), plain=plain)}"
                    )
            else:
                emit("  (no future candidates)")
        emit("")

    if watch:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        emit(f"Last refresh: {ts}  (Ctrl+C to exit)")
        emit("")

    emit(_FOOTER)
    emit(_DISCLAIMER)
    emit("Canonical: docs/leaderboard.md · Public: docs/leaderboard.html")
    return buf.getvalue()


def render_summary(entries: list[LeaderboardEntry]) -> str:
    by_tier = _entries_by_tier(entries)
    full = by_tier.get(TIER_FULL_PANEL, [])
    repair = by_tier.get(TIER_REPAIR, [])
    restricted = by_tier.get(TIER_RESTRICTED, [])
    smoke = by_tier.get(TIER_SMOKE, [])
    future = by_tier.get(TIER_FUTURE, [])

    tiers_with_rows = sum(1 for t, _, _ in _TIER_SECTIONS if by_tier.get(t))

    best_line = "(no full-panel rows)"
    if full:
        top = full[0]
        best_line = (
            f"{top.method} on {short_model(top.model_panel)} "
            f"(accept={_fmt_acc(top.mean_acceptance)}, "
            f"failures={_fmt_fail(top.exactkv_failures)})"
        )

    all_failures = [
        e.exactkv_failures for e in entries if e.exactkv_failures is not None
    ]
    nonzero = [f for f in all_failures if f > 0]
    if not all_failures:
        fail_line = "no failure counts in local reports"
    elif not nonzero:
        fail_line = f"0 ExactKV failures across {len(all_failures)} measured rows"
    else:
        fail_line = f"{len(nonzero)} row(s) with nonzero failures (max {max(nonzero)})"

    top_caveat = (
        "Restricted backends are factory-only — not ranked against full-panel compressors."
        if restricted
        else "Smoke-only and future tiers are separated from full-panel ranking."
    )

    lines = [
        "EXACTKV LEADERBOARD — LAUNCH SUMMARY",
        f"Best full-panel: {best_line}",
        f"Tiers with data: {tiers_with_rows} of {len(_TIER_SECTIONS)}",
        f"Full-panel ranked rows: {len(full)}",
        f"Repair policies: {len(repair)} (separate tier)",
        f"ExactKV failures: {fail_line}",
        f"Restricted backends: {len(restricted)}",
        f"Smoke-only adapters: {len(smoke)}",
        f"Future candidates: {len(future)}",
        f"Top caveat: {top_caveat}",
        _FOOTER,
    ]
    return "\n".join(lines) + "\n"


def run_watch(
    *,
    plain: bool,
    once: bool,
    interval_sec: int = _WATCH_INTERVAL_SEC,
) -> int:
    try:
        while True:
            entries, _data = load_leaderboard_entries()
            render_terminal(entries, plain=plain, watch=True)
            if once:
                break
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nWatch stopped.", file=sys.stderr)
    return 0


def write_leaderboard_md(entries: list[LeaderboardEntry], path: Path) -> None:
    lines = [
        f"# {_TITLE}",
        "",
        "_Generated by `scripts/exactkv_leaderboard.py` from local experiment reports._",
        "",
        f"> {_SUBTITLE}.",
        "",
        "> " + PUBLIC_TAGLINE.replace("\n", "\n> "),
        "",
        PUBLIC_LEADERBOARD_COPY,
        "",
        "## Ranking policy",
        "",
        "- **Full-panel results** are ranked by token-level acceptance and ExactKV failures.",
        "- **Repair policies** are a separate tier — adaptive selectors, not default compressors.",
        "- **Restricted backends** are listed separately; not ranked against full-panel compressors. Shard shows restricted external-drafter metrics (accepted-prefix mean, divergence rate) — not standard compressor acceptance.",
        "- **Smoke-only adapters** are diagnostic probes — not ranked against full-panel compressors.",
        "- **Future candidates** have no ExactKV panel or tensor-smoke metrics yet.",
        "",
        "> " + _NOT_APPLES,
        "",
        "> " + _FOOTER,
        "",
    ]
    by_tier = _entries_by_tier(entries)
    for tier, _slug, label in _TIER_TAB_ORDER:
        rows = by_tier.get(tier, [])
        if not rows:
            continue
        lines.extend([f"## {label}", ""])
        if tier == TIER_FULL_PANEL:
            lines.append(
                "| Rank | Compressor | Model | Acceptance | Failures | Scope | Experiment | Badges |"
            )
            lines.append("| ---: | --- | --- | ---: | ---: | --- | --- | --- |")
            for r in rows:
                badges = ", ".join(entry_badges(r))
                lines.append(
                    f"| {r.rank or '—'} | {r.method} | {short_model(r.model_panel)} | "
                    f"{_fmt_acc(r.mean_acceptance)} | {_fmt_fail(r.exactkv_failures)} | "
                    f"{panel_scope(r)} | {r.experiment} | {badges} |"
                )
        else:
            lines.append(
                "| Method | Model / panel | Acceptance | Failures | Experiment | Badges | Caveat |"
            )
            lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
            for r in rows:
                badges = ", ".join(entry_badges(r))
                lines.append(
                    f"| {r.method} | {r.model_panel} | {_fmt_acc(r.mean_acceptance)} | "
                    f"{_fmt_fail(r.exactkv_failures)} | {r.experiment} | {badges} | {r.caveat} |"
                )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- Tiers prevent apples-to-oranges ranking (full panel vs smoke vs restricted).",
        "- **TurboQuant / KIVI / KVQuant** are factory-only restricted adapters.",
        "- **SnapKV experimental** is smoke-only (8 cells).",
        "- **SpectralQuant** has ExactKV tensor-smoke coverage, but no generation-time ExactKV probe yet (Exp 042).",
        "- **Shard** has restricted external-drafter probe results under ExactKV verification (Exp 039–041) — not a full-panel integrated compressor.",
        "- External Shard, SpectralQuant, SnapKV paper, or kvpress results are **not** ExactKV results.",
        "- Regenerate: `python3 scripts/exactkv_leaderboard.py --md --html`",
        "- Live terminal: `python3 scripts/exactkv_leaderboard.py --watch`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _badge_class(badge: str) -> str:
    mapping = {
        "FULL": "tier-full",
        "REPAIR": "tier-repair",
        "RESTRICTED": "tier-restricted",
        "SMOKE": "tier-smoke",
        "FUTURE": "tier-future",
        "SIMULATED": "sim",
        "REAL-BYTE": "realbyte",
        "NO REAL-BYTE CLAIM": "nobytes",
        "EXTERNAL DRAFTER": "tier-restricted",
        "LLAMA ONLY": "tier-restricted",
        "NOT DEFAULT": "nobytes",
        "NO SPEED CLAIM": "nobytes",
        "NO MEMORY CLAIM": "nobytes",
        "TENSOR PROBE": "tier-smoke",
        "NOT GENERATION": "nobytes",
    }
    return mapping.get(badge, "default")


def write_leaderboard_html(entries: list[LeaderboardEntry], path: Path) -> None:
    by_tier = _entries_by_tier(entries)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tab_buttons = []
    tab_panels = []
    for i, (tier, slug, label) in enumerate(_TIER_TAB_ORDER):
        rows = by_tier.get(tier, [])
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab{active}" data-tab="{slug}" type="button">{html.escape(label)}</button>'
        )

        if tier == TIER_FULL_PANEL:
            head = (
                "<tr><th>Rank</th><th>Compressor</th><th>Model</th>"
                "<th>Acceptance</th><th>Failures</th><th>Scope</th><th>Badges</th></tr>"
            )
            body_rows = []
            for r in rows:
                badges_html = "".join(
                    f'<span class="badge {_badge_class(b)}">{html.escape(b)}</span>'
                    for b in entry_badges(r)
                )
                body_rows.append(
                    f"<tr><td>{r.rank or '—'}</td><td>{html.escape(r.method)}</td>"
                    f"<td>{html.escape(short_model(r.model_panel))}</td>"
                    f"<td class='num'>{_fmt_acc(r.mean_acceptance)}</td>"
                    f"<td class='num'>{_fmt_fail(r.exactkv_failures)}</td>"
                    f"<td>{html.escape(panel_scope(r))}</td>"
                    f"<td class='badges'>{badges_html}</td></tr>"
                )
        else:
            head = (
                "<tr><th>Method</th><th>Panel</th><th>Acceptance</th>"
                "<th>Failures</th><th>Experiment</th><th>Badges</th><th>Caveat</th></tr>"
            )
            body_rows = []
            for r in rows:
                badges_html = "".join(
                    f'<span class="badge {_badge_class(b)}">{html.escape(b)}</span>'
                    for b in entry_badges(r)
                )
                body_rows.append(
                    f"<tr><td>{html.escape(r.method)}</td>"
                    f"<td>{html.escape(r.model_panel)}</td>"
                    f"<td class='num'>{_fmt_acc(r.mean_acceptance)}</td>"
                    f"<td class='num'>{_fmt_fail(r.exactkv_failures)}</td>"
                    f"<td>{html.escape(r.experiment)}</td>"
                    f"<td class='badges'>{badges_html}</td>"
                    f"<td class='caveat'>{html.escape(r.caveat)}</td></tr>"
                )

        table = (
            f'<table><thead>{head}</thead><tbody>{"".join(body_rows)}</tbody></table>'
            if body_rows
            else '<p class="empty">No rows for this tier in local reports.</p>'
        )
        tab_panels.append(
            f'<section class="panel{active}" id="tab-{slug}" role="tabpanel">{table}</section>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(_TITLE)}</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --border: #30363d;
      --amber: #d29922;
      --red: #f85149;
      --sticky-bg: rgba(13, 17, 23, 0.92);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .sticky-header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--sticky-bg);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 1rem 1.25rem 0.75rem;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 1.25rem 3rem; }}
    h1 {{ font-size: 1.65rem; margin: 0 0 0.25rem; letter-spacing: -0.02em; }}
    .subtitle {{ color: var(--muted); font-size: 1rem; max-width: 52rem; margin: 0; }}
    .tagline {{
      margin: 1rem 0 0.75rem;
      padding: 0.75rem 1rem;
      border-left: 3px solid var(--accent);
      background: var(--card);
      border-radius: 0 8px 8px 0;
      font-size: 0.92rem;
    }}
    .not-apples {{
      margin: 0.75rem 0 0;
      padding: 0.65rem 0.85rem;
      background: rgba(210, 153, 34, 0.08);
      border: 1px solid rgba(210, 153, 34, 0.25);
      border-radius: 6px;
      color: #e3b341;
      font-size: 0.85rem;
    }}
    .disclaimer {{ color: var(--muted); font-size: 0.82rem; margin: 0.5rem 0 0; }}
    .content {{ padding-top: 1.25rem; }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin: 0;
      padding: 0.75rem 0 0;
    }}
    .tab {{
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 0.5rem 0.9rem;
      border-radius: 999px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 500;
      transition: border-color 0.15s, color 0.15s;
    }}
    .tab:hover {{ border-color: var(--accent); color: var(--text); }}
    .tab.active {{
      background: rgba(88, 166, 255, 0.12);
      color: var(--text);
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.2);
    }}
    .panel {{ display: none; margin-top: 1.25rem; }}
    .panel.active {{ display: block; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      background: var(--card);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    th, td {{ padding: 0.65rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: #1c2128; color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(88, 166, 255, 0.06); }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .caveat {{ color: var(--muted); font-size: 0.82rem; max-width: 22rem; }}
    .badges {{ white-space: normal; }}
    .badge {{
      display: inline-block;
      font-size: 0.66rem;
      font-weight: 700;
      padding: 0.18rem 0.45rem;
      border-radius: 999px;
      margin: 0.12rem 0.2rem 0.12rem 0;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .tier-full {{ background: rgba(63, 185, 80, 0.18); color: var(--green); border: 1px solid rgba(63,185,80,0.4); }}
    .tier-repair {{ background: rgba(88, 166, 255, 0.14); color: var(--accent); border: 1px solid rgba(88,166,255,0.35); }}
    .tier-restricted {{ background: rgba(210, 153, 34, 0.14); color: var(--amber); border: 1px solid rgba(210,153,34,0.4); }}
    .tier-smoke {{ background: rgba(139, 148, 158, 0.18); color: #c9d1d9; border: 1px solid var(--border); }}
    .tier-future {{ background: rgba(248, 81, 73, 0.12); color: #ff7b72; border: 1px solid rgba(248,81,73,0.35); }}
    .sim {{ background: rgba(139, 148, 158, 0.14); color: #b1bac4; border: 1px solid var(--border); }}
    .realbyte {{ background: rgba(63, 185, 80, 0.1); color: #56d364; border: 1px solid rgba(63,185,80,0.3); }}
    .nobytes {{ background: rgba(210, 153, 34, 0.12); color: var(--amber); border: 1px solid rgba(210,153,34,0.3); }}
    .empty {{ color: var(--muted); padding: 1rem; }}
    footer {{
      margin-top: 2.5rem;
      padding: 1rem 1.1rem;
      font-size: 0.8rem;
      color: var(--muted);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      line-height: 1.6;
    }}
    footer strong {{ color: var(--text); font-weight: 600; }}
  </style>
</head>
<body>
  <header class="sticky-header">
    <div class="wrap" style="padding:0;">
      <h1>{html.escape(_TITLE)}</h1>
      <p class="subtitle">{html.escape(_SUBTITLE)}.</p>
      <div class="tagline">{html.escape(PUBLIC_TAGLINE)}</div>
      <p class="not-apples">{html.escape(_NOT_APPLES)}</p>
      <p class="disclaimer">{html.escape(_DISCLAIMER)}</p>
      <nav class="tabs" role="tablist">
        {"".join(tab_buttons)}
      </nav>
    </div>
  </header>
  <div class="wrap content">
    {"".join(tab_panels)}

    <footer>
      <strong>Caveat</strong> — {html.escape(_FOOTER)}<br />
      Generated {generated} by <code>scripts/exactkv_leaderboard.py</code> from local reports.
      Not a hosted live backend. Shard has restricted external-drafter probe results; SpectralQuant has tensor-smoke coverage only (Exp 042); external Shard/SpectralQuant README results are not ExactKV results.
    </footer>
  </div>
  <script>
    document.querySelectorAll('.tab').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def write_manifest(entries: list[LeaderboardEntry], path: Path) -> None:
    """Optional local JSON for debugging (gitignored reports/)."""
    payload = {
        "title": _TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "tier": e.tier,
                "method": e.method,
                "model": short_model(e.model_panel),
                "model_panel": e.model_panel,
                "mean_acceptance": e.mean_acceptance,
                "exactkv_failures": e.exactkv_failures,
                "experiment": e.experiment,
                "scope": panel_scope(e),
                "badges": entry_badges(e),
                "rank": e.rank,
            }
            for e in entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ExactKV crash-test leaderboard renderer")
    parser.add_argument("--terminal", action="store_true", help="Print terminal table (default)")
    parser.add_argument("--md", action="store_true", help="Write docs/leaderboard.md")
    parser.add_argument("--html", action="store_true", help="Write docs/leaderboard.html")
    parser.add_argument("--json", action="store_true", help="Write reports/leaderboard_manifest.json")
    parser.add_argument("--all", action="store_true", help="Terminal + md + html + json")
    parser.add_argument("--plain", action="store_true", help="Plain terminal output (no ANSI/box drawing)")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh terminal dashboard every few seconds (Ctrl+C to exit)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="With --watch: render one frame and exit (for tests)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact launch/demo summary (~10 lines)",
    )
    parser.add_argument(
        "--md-path",
        type=Path,
        default=_ROOT / "docs" / "leaderboard.md",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=_ROOT / "docs" / "leaderboard.html",
    )
    args = parser.parse_args()

    if args.watch:
        if args.md or args.html or args.json or args.all:
            entries, data = load_leaderboard_entries()
            if args.md:
                write_leaderboard_md(entries, args.md_path)
                print(f"Wrote {args.md_path}")
            if args.html:
                write_leaderboard_html(entries, args.html_path)
                print(f"Wrote {args.html_path}")
            if args.json or args.all:
                manifest = _ROOT / "reports" / "leaderboard_manifest.json"
                write_manifest(entries, manifest)
                print(f"Wrote {manifest}")
            if data.missing:
                print(
                    f"Note: {len(data.missing)} expected CSV(s) missing — some rows omitted.",
                    file=sys.stderr,
                )
        return run_watch(plain=args.plain, once=args.once)

    entries, data = load_leaderboard_entries()

    if args.summary:
        print(render_summary(entries), end="")
    else:
        write_outputs = args.md or args.html or args.json or args.all
        do_terminal = args.terminal or args.all or not write_outputs
        if do_terminal:
            render_terminal(entries, plain=args.plain)

    if args.md or args.all:
        write_leaderboard_md(entries, args.md_path)
        print(f"Wrote {args.md_path}")
    if args.html or args.all:
        write_leaderboard_html(entries, args.html_path)
        print(f"Wrote {args.html_path}")
    if args.json or args.all:
        manifest = _ROOT / "reports" / "leaderboard_manifest.json"
        write_manifest(entries, manifest)
        print(f"Wrote {manifest}")

    if data.missing:
        print(
            f"Note: {len(data.missing)} expected CSV(s) missing — some rows omitted.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
