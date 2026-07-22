#!/usr/bin/env python3
"""Embed static leaderboard + case-study HTML into site/index.html (no-JS fallback)."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "site" / "index.html"
LEADERBOARD = ROOT / "site" / "data" / "leaderboard.json"
CASES = ROOT / "site" / "data" / "case_studies.json"

LEADERBOARD_MARK = ("<!-- SYNC:LEADERBOARD_TBODY -->", "<!-- /SYNC:LEADERBOARD_TBODY -->")
CASES_MARK = ("<!-- SYNC:CASE_STUDIES -->", "<!-- /SYNC:CASE_STUDIES -->")


def _fmt_num(v: object, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{float(v):.{digits}f}"
    return html.escape(str(v))


def render_leaderboard_rows(data: dict) -> str:
    # Public site shows headline compressors only (noop / int8 / int4_sim).
    # Diagnostic fallback/proxy and probe-first slots stay in JSON, not the table.
    rows = list(data.get("entries") or [])
    lines: list[str] = []
    for e in rows:
        lines.append(
            f'          <tr>'
            f"<td>{html.escape(str(e.get('rank', '-')))}</td>"
            f"<td><code>{html.escape(str(e.get('compressor', '-')))}</code></td>"
            f"<td>{html.escape(str(e.get('model_short') or e.get('model') or '-'))}</td>"
            f'<td class="num">{_fmt_num(e.get("score"))}</td>'
            f'<td class="num">{_fmt_num(e.get("acceptance_rate"))}</td>'
            f'<td class="num">{_fmt_num(e.get("divergence_score"))}</td>'
            f"</tr>"
        )
    return "\n".join(lines)


def render_case_studies(data: dict) -> str:
    cases = (data.get("case_studies") or [])[:8]
    lines: list[str] = []
    if data.get("note"):
        lines.append(f'      <p class="viz-caption">{html.escape(str(data["note"]))}</p>')
    for c in cases:
        title = f"{c.get('dataset_family') or 'panel'} · {c.get('task_category') or c.get('prompt_id') or ''}"
        meta_parts = [
            c.get("panel"),
            c.get("compressor_name"),
            (c.get("model_name") or "").split("/")[-1] or None,
            f"{c.get('context_bucket')} ctx" if c.get("context_bucket") else None,
            f"fdi={c.get('first_divergence_index')}" if c.get("first_divergence_index") is not None else None,
        ]
        meta = " · ".join(str(p) for p in meta_parts if p)
        lines.extend(
            [
                '      <article class="case-card">',
                f"        <h3>{html.escape(title)}</h3>",
                f'        <p class="case-meta"><code>{html.escape(meta)}</code></p>',
                '        <div class="case-cols">',
                '          <div class="case-col">',
                '            <span class="case-label">Full KV</span>',
                f"            <pre>{html.escape(str(c.get('full_snippet') or '-'))}</pre>",
                "          </div>",
                '          <div class="case-col case-col-lossy">',
                '            <span class="case-label">Lossy draft</span>',
                f"            <pre>{html.escape(str(c.get('lossy_snippet') or '-'))}</pre>",
                "          </div>",
                '          <div class="case-col">',
                '            <span class="case-label">ExactKV out</span>',
                f"            <pre>{html.escape(str(c.get('exactkv_snippet') or '-'))}</pre>",
                "          </div>",
                "        </div>",
                "      </article>",
            ]
        )
    return "\n".join(lines)


def _patch(text: str, start: str, end: str, body: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{start}\n{body}\n      {end}"
    if not pattern.search(text):
        raise ValueError(f"missing sync markers: {start}")
    return pattern.sub(replacement, text, count=1)


def main() -> int:
    lb = json.loads(LEADERBOARD.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    page = INDEX.read_text(encoding="utf-8")
    page = _patch(page, *LEADERBOARD_MARK, render_leaderboard_rows(lb))
    page = _patch(page, *CASES_MARK, render_case_studies(cases))
    INDEX.write_text(page, encoding="utf-8")
    n_lb = len(lb.get("entries") or [])
    n_cs = min(8, len(cases.get("case_studies") or []))
    print(f"embedded static HTML: {n_lb} leaderboard rows, {n_cs} case studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
