#!/usr/bin/env python3
"""ExactKV crash-test leaderboard — terminal + static HTML (V13 Phase 8f).

Reads local experiment CSV reports and renders a tiered leaderboard.
No hosted backend, no model inference, no timing/memory benchmarks.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
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
    "Compressors ranked by token-level acceptance, first divergence, and exactness "
    "under full-KV verification."
)
_DISCLAIMER = (
    "Not a speed leaderboard. ExactKV measures when lossy KV drafts start lying — "
    "not throughput, latency, or production readiness."
)

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
        TIER_FULL_PANEL: "FULL PANEL",
        TIER_REPAIR: "REPAIR POLICY",
        TIER_RESTRICTED: "RESTRICTED",
        TIER_SMOKE: "SMOKE ONLY",
        TIER_FUTURE: "FUTURE",
    }.get(entry.tier, entry.tier)
    badges.append(tier_badge)

    status_l = entry.integration_status.lower()
    caveat_l = entry.caveat.lower()
    method_l = entry.method.lower()

    if (
        "sim" in method_l
        or "sim " in status_l
        or "simquant" in caveat_l
        or "sim asymmetric" in status_l
    ):
        badges.append("SIMULATED")

    if "supports_real_bytes_claim=false" in caveat_l.replace(" ", ""):
        badges.append("NO REAL-BYTE CLAIM")
    elif entry.tier == TIER_RESTRICTED and "factory-only" in status_l:
        badges.append("NO REAL-BYTE CLAIM")

    if entry.tier == TIER_FUTURE:
        badges.append("NOT INTEGRATED")

    return badges


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
) -> str:
    if out is None:
        out = sys.stdout
    lines: list[str] = []
    by_tier = _entries_by_tier(entries)

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text, file=out)

    emit("EXACTKV CRASH-TEST LEADERBOARD")
    emit(_SUBTITLE)
    emit("")
    emit("> " + PUBLIC_TAGLINE.replace("\n", "\n> "))
    emit("")

    # --- FULL PANEL ---
    emit("FULL PANEL RESULTS")
    full_rows = by_tier.get(TIER_FULL_PANEL, [])
    table_rows = [
        [
            e.method,
            short_model(e.model_panel),
            _fmt_acc(e.mean_acceptance),
            _fmt_fail(e.exactkv_failures),
            panel_scope(e),
        ]
        for e in full_rows
    ]
    for row in _terminal_table(
        ["Compressor", "Model", "Acceptance", "Failures", "Status"],
        table_rows,
        [14, 12, 10, 10, 8],
    ):
        emit(row)
    emit("")

    # --- RESTRICTED ---
    emit("RESTRICTED BACKENDS")
    restricted = by_tier.get(TIER_RESTRICTED, [])
    if restricted:
        for e in restricted:
            acc = _fmt_acc(e.mean_acceptance)
            fail = _fmt_fail(e.exactkv_failures)
            emit(
                f"  • {e.method:<22} {short_model(e.model_panel):<10} "
                f"accept={acc}  failures={fail}  [{e.experiment}]"
            )
        emit("  Factory-only adapters — not production runtimes. Tier: RESTRICTED · SIMULATED")
    else:
        emit("  (no restricted backend rows)")
    emit("")

    # --- SMOKE ---
    emit("SMOKE ONLY")
    smoke = by_tier.get(TIER_SMOKE, [])
    for e in smoke:
        fail = _fmt_fail(e.exactkv_failures)
        emit(f"  • {e.method}: {e.model_panel} — {fail} failures · not full-suite ranked")
    if not smoke:
        emit("  (no smoke rows)")
    emit("")

    # --- FUTURE ---
    emit("FUTURE CANDIDATES")
    future = by_tier.get(TIER_FUTURE, [])
    for e in future:
        emit(f"  • {e.method}: {e.caveat}")
    if not future:
        emit("  (no future candidates)")
    emit("")

    # --- REPAIR (compact) ---
    repair = by_tier.get(TIER_REPAIR, [])
    if repair:
        emit("REPAIR POLICIES (not compressors — separate tier)")
        for e in repair[:4]:
            emit(
                f"  • {e.method:<24} accept={_fmt_acc(e.mean_acceptance)}  "
                f"failures={_fmt_fail(e.exactkv_failures)}"
            )
        if len(repair) > 4:
            emit(f"  … +{len(repair) - 4} more in docs/leaderboard.html")
        emit("")

    emit(_DISCLAIMER)
    emit("Canonical: docs/leaderboard.md · Public: docs/leaderboard.html")
    return "\n".join(lines) + "\n"


def write_leaderboard_md(entries: list[LeaderboardEntry], path: Path) -> None:
    lines = [
        f"# {_TITLE}",
        "",
        "_Generated by `scripts/exactkv_leaderboard.py` from local experiment reports._",
        "",
        f"> {_SUBTITLE}",
        "",
        "> " + PUBLIC_TAGLINE.replace("\n", "\n> "),
        "",
        PUBLIC_LEADERBOARD_COPY,
        "",
        "> No speedup, throughput, latency, tokens/sec, active GPU memory savings, production serving, "
        "or model accuracy improvement claim is made.",
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
        "- **Shard / SpectralQuant** are future candidates without ExactKV panel numbers.",
        "- External Shard, SpectralQuant, SnapKV paper, or kvpress results are **not** ExactKV results.",
        "- Regenerate: `python3 scripts/exactkv_leaderboard.py`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _badge_class(badge: str) -> str:
    mapping = {
        "FULL PANEL": "tier-full",
        "REPAIR POLICY": "tier-repair",
        "RESTRICTED": "tier-restricted",
        "SMOKE ONLY": "tier-smoke",
        "FUTURE": "tier-future",
        "SIMULATED": "sim",
        "NO REAL-BYTE CLAIM": "nobytes",
        "NOT INTEGRATED": "future",
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.35rem; letter-spacing: -0.02em; }}
    .subtitle {{ color: var(--muted); font-size: 1.05rem; max-width: 52rem; }}
    .tagline {{
      margin: 1.25rem 0;
      padding: 0.85rem 1rem;
      border-left: 3px solid var(--accent);
      background: var(--card);
      border-radius: 0 8px 8px 0;
      font-size: 0.95rem;
    }}
    .disclaimer {{ color: var(--muted); font-size: 0.85rem; margin-top: 1.5rem; }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin: 1.5rem 0 0;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
    }}
    .tab {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
    }}
    .tab:hover {{ border-color: var(--accent); color: var(--text); }}
    .tab.active {{
      background: var(--card);
      color: var(--text);
      border-color: var(--accent);
    }}
    .panel {{ display: none; margin-top: 1rem; }}
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
      font-size: 0.68rem;
      font-weight: 600;
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      margin: 0.1rem 0.15rem 0.1rem 0;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .tier-full {{ background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.35); }}
    .tier-repair {{ background: rgba(88, 166, 255, 0.12); color: var(--accent); border: 1px solid rgba(88,166,255,0.3); }}
    .tier-restricted {{ background: rgba(210, 153, 34, 0.12); color: var(--amber); border: 1px solid rgba(210,153,34,0.35); }}
    .tier-smoke {{ background: rgba(139, 148, 158, 0.15); color: var(--muted); border: 1px solid var(--border); }}
    .tier-future {{ background: rgba(248, 81, 73, 0.1); color: #ff7b72; border: 1px solid rgba(248,81,73,0.3); }}
    .sim {{ background: rgba(139, 148, 158, 0.12); color: var(--muted); border: 1px solid var(--border); }}
    .nobytes {{ background: rgba(210, 153, 34, 0.1); color: var(--amber); border: 1px solid rgba(210,153,34,0.25); }}
    .empty {{ color: var(--muted); padding: 1rem; }}
    footer {{ margin-top: 2rem; font-size: 0.8rem; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(_TITLE)}</h1>
    <p class="subtitle">{html.escape(_SUBTITLE)}</p>
    <div class="tagline">{html.escape(PUBLIC_TAGLINE)}</div>
    <p class="disclaimer">{html.escape(_DISCLAIMER)}</p>

    <nav class="tabs" role="tablist">
      {"".join(tab_buttons)}
    </nav>
    {"".join(tab_panels)}

    <footer>
      Generated {generated} by <code>scripts/exactkv_leaderboard.py</code> from local reports.
      Not a hosted live backend. No speedup, throughput, latency, tokens/sec, VRAM, or serving claims.
      External Shard/SpectralQuant results are not ExactKV results.
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
    parser.add_argument("--plain", action="store_true", help="Plain terminal output")
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

    do_all = args.all or not (args.terminal or args.md or args.html or args.json)
    entries, data = load_leaderboard_entries()

    if do_all or args.terminal:
        render_terminal(entries, plain=args.plain)
    if do_all or args.md:
        write_leaderboard_md(entries, args.md_path)
        print(f"Wrote {args.md_path}")
    if do_all or args.html:
        write_leaderboard_html(entries, args.html_path)
        print(f"Wrote {args.html_path}")
    if args.json or args.all:
        manifest = _ROOT / "reports" / "leaderboard_manifest.json"
        write_manifest(entries, manifest)
        print(f"Wrote {manifest}")

    if data.missing:
        print(f"Note: {len(data.missing)} expected CSV(s) missing — some rows omitted.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
