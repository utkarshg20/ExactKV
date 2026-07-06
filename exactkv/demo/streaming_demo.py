"""Single-act streaming terminal demo — cinematic crash-test replay."""
from __future__ import annotations

import re
import shutil
import sys
import time
from typing import TextIO

from exactkv.demo.case_study_loader import CLOSING_LINES, PUBLIC_TAGLINE
from exactkv.demo.live_terminal import SPEED_PROFILES, LiveFrame, TerminalStyle, _emit, progress_bar

SNIPPET_LINES = 5

META = {
    "panel": "BFCL tool-call panel · ctx 2048 · 64 max new tokens",
    "prompt_label": "int4_sim 4× quant · greedy decode · compressed KV draft",
    "prompt": (
        "User: Return a complete weather JSON tool result for Paris, France. "
        "Include units, conditions, humidity, wind, and a 2-day forecast."
    ),
    "compressor": "int4_sim · 4× quant",
    "model": "Mistral-7B-Instruct",
}

# Longer realistic tool JSON; four drifts on high-impact tool-call fields.
TIMELINE: tuple[tuple[str, ...], ...] = (
    ("sync", '{"tool":"get_weather","city":"Paris","country":"France","units":"'),
    ("drift", "imperial", "metric", "4× quant flipped units argmax"),
    ("sync", '","temp_c":'),
    ("drift", "22", "18", "wrong temperature from compressed logits"),
    ("sync", ',"feels_like_c":16,"conditions":"'),
    ("drift", "overcast", "clear skies", "open-text field drift under int4_sim"),
    ("sync", '","humidity_pct":'),
    ("drift", "45", "72", "humidity digits flipped by KV noise"),
    (
        "sync",
        ',"wind":"NW 12km/h","forecast":[{"day":"Mon","high_c":20,"low_c":12},'
        '{"day":"Tue","high_c":19,"low_c":11}],"source":"open-meteo","valid":',
    ),
    ("sync", "true}"),
)

FULL_TEXT = "".join(seg[1] if seg[0] == "sync" else seg[2] for seg in TIMELINE)
LOSSY_TEXT = "".join(seg[1] if seg[0] == "sync" else seg[1] for seg in TIMELINE)
STREAM_TOTAL = len(FULL_TEXT)

LOGO = [
    " ███████╗██╗  ██╗ █████╗  ██████╗████████╗██╗  ██╗██╗   ██╗",
    " ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝╚══██╔══╝██║ ██╔╝██║   ██║",
    " █████╗   ╚███╔╝ ███████║██║        ██║   █████╔╝ ██║   ██║",
    " ██╔══╝   ██╔██╗ ██╔══██║██║        ██║   ██╔═██╗ ╚██╗ ██╔╝",
    " ███████╗██╔╝ ██╗██║  ██║╚██████╗   ██║   ██║  ██╗ ╚████╔╝ ",
    " ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝  ╚═══╝  ",
]

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _panel_width() -> int:
    cols = shutil.get_terminal_size((110, 24)).columns
    return min(104, max(92, cols - 4))


def _col_width(inner: int) -> int:
    return max(20, (inner - 8) // 3)


def _pause(seconds: float, *, no_delay: bool) -> None:
    if not no_delay and seconds > 0:
        time.sleep(seconds)


def _wrap_snippet(text: str, width: int, *, max_lines: int = SNIPPET_LINES) -> list[str]:
    """Wrap on punctuation/word boundaries; never split mid-word when avoidable."""
    one = text.replace("\n", "\\n").replace("\r", "")
    if width <= 0:
        return [""] * max_lines
    if not one:
        return [""] * max_lines

    lines: list[str] = []
    i = 0
    n = len(one)
    while i < n:
        end = min(i + width, n)
        if end < n:
            segment = one[i:end]
            split = -1
            for sep in (" ", ",", '"', ":", "}", "{"):
                pos = segment.rfind(sep)
                if pos > 0:
                    split = pos + (1 if sep != " " else 1)
                    break
            if split <= 0 and one[end - 1].isalnum() and one[end].isalnum():
                j = end - 1
                while j > i and one[j - 1].isalnum():
                    j -= 1
                if j > i:
                    split = j
            if split > 0:
                end = i + split
        lines.append(one[i:end].rstrip())
        i = end
        while i < n and one[i] == " ":
            i += 1

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    while len(lines) < max_lines:
        lines.append("")
    return lines


def _fit_cell(plain: str, width: int) -> str:
    if len(plain) <= width:
        return plain.ljust(width)
    line = _wrap_snippet(plain, width, max_lines=1)[0]
    if len(line) < len(plain) and width >= 2:
        if len(line) >= width:
            line = line[: width - 1] + "…"
        else:
            line = line + "…"
    return line.ljust(width)


def _box_line(inner: str, width: int, left: str = "┃", right: str = "┃") -> str:
    plain = _strip_ansi(inner)
    if len(plain) > width:
        inner = _fit_cell(plain, width)
    return f"{left} {inner.ljust(width)} {right}"


def _intro_frame(style: TerminalStyle, *, step: int) -> list[str]:
    inner = _panel_width()
    sep = "━" * (inner + 2)
    lines = [f"┏{sep}┓", _box_line("", inner)]
    for row in LOGO:
        lines.append(_box_line(row.center(inner), inner))
    lines.append(_box_line("", inner))
    title = "CRASH TEST  ·  LIVE VERIFIER REPLAY"
    if not style.plain:
        title = style.bold(style.cyan(title))
    lines.append(_box_line(title.center(inner), inner))
    if step >= 1:
        sub = "Every token: compressed draft vs full KV  ·  drift blocked before it ships"
        if not style.plain:
            sub = style.white(sub.center(inner))
        else:
            sub = sub.center(inner)
        lines.append(_box_line(sub, inner))
    if step >= 2:
        tag = PUBLIC_TAGLINE.replace("\n", "  ·  ")
        lines.append(_box_line("", inner))
        if not style.plain:
            lines.append(_box_line(style.bold(style.white(tag.center(inner))), inner))
        else:
            lines.append(_box_line(tag.center(inner), inner))
    lines.extend([_box_line("", inner), f"┗{sep}┛"])
    return lines


def _value_prop(style: TerminalStyle) -> list[str]:
    lines = [
        "PROBLEM   Compressed KV silently picks wrong tokens — benchmarks miss when",
        "EXACTKV   Crash-tests every token vs full KV  ·  records first-divergence + acceptance",
        "PANELS    MBPP code  ·  BFCL tool calls  ·  HF LongBench reading  ·  8,132 cells",
        "RESULT    exactkv_failures = 0 across headline GPU panels (claim-safe release)",
    ]
    if style.plain:
        return lines + [""]
    return [
        style.red(style.bold(lines[0])),
        style.green(style.bold(lines[1])),
        style.white(lines[2]),
        style.white(lines[3]),
        "",
    ]


def _context_block(style: TerminalStyle) -> list[str]:
    rows = [
        f"PANEL   {META['panel']}",
        "KV      2,048-token compressed cache loaded  ·  full-KV reference locked for verify",
        f"TASK    {META['prompt']}",
    ]
    if style.plain:
        return rows + [""]
    return [style.white(r) for r in rows] + [""]


def _top_rail(style: TerminalStyle) -> list[str]:
    inner = _panel_width()
    sep = "━" * (inner + 2)
    badge = "verifier LIVE"
    if not style.plain:
        badge = style.green(style.bold("verifier LIVE"))
    title = style.bold("EXACTKV") if not style.plain else "EXACTKV"
    meta = f"{META['compressor']}  ·  {META['prompt_label']}"
    if not style.plain:
        meta = style.white(meta)
    return [
        f"┏{sep}┓",
        _box_line(f"{title}   {META['model']:<28}{badge:>{inner - 44}}", inner),
        _box_line(meta, inner),
        f"┗{sep}┛",
    ]


def _hud(
    style: TerminalStyle,
    *,
    stream_pos: int,
    drifts_caught: int,
    phase: str,
) -> list[str]:
    pct = int(100 * stream_pos / STREAM_TOTAL) if STREAM_TOTAL else 0
    bar = progress_bar(stream_pos, STREAM_TOTAL, width=20)
    drift_txt = str(drifts_caught)
    if drifts_caught and not style.plain:
        drift_txt = style.red(style.bold(str(drifts_caught)))
    if phase == "drift":
        verifier = style.red(style.bold("BLOCKING")) if not style.plain else "BLOCKING"
    elif phase == "commit":
        verifier = style.green(style.bold("CORRECTING")) if not style.plain else "CORRECTING"
    elif phase == "victory":
        verifier = style.green(style.bold("MATCH")) if not style.plain else "MATCH"
    else:
        verifier = style.cyan("armed") if not style.plain else "armed"
    line1 = (
        f"DECODE {bar} {pct:>3}%   token {stream_pos}/{STREAM_TOTAL}   "
        f"drifts {drift_txt:<3}   verifier {verifier}"
    )
    line2 = "draft / full KV / ExactKV output compared every greedy step"
    if not style.plain:
        line2 = style.white(line2)
    return [line1, line2, ""]


def _drift_alert(
    style: TerminalStyle,
    *,
    note: str,
    wrong: str,
    right: str,
    flash: bool = False,
) -> list[str]:
    inner = _panel_width()
    sep = "━" * (inner + 2)
    headline = "DRIFT  lossy draft ≠ full KV  ·  verifier stops the bad token"
    draft = f"draft wrote:  {wrong}"
    truth = f"full KV wants:  {right}"
    why = note
    if not style.plain:
        headline = style.red(style.bold(headline))
        draft = style.white(style.bold(draft) if flash else draft)
        truth = style.green(truth)
        why = style.white(f"({why})")
    return [
        f"┏{sep}┓",
        _box_line(headline, inner),
        _box_line(draft, inner),
        _box_line(truth, inner),
        _box_line(why, inner),
        f"┗{sep}┛",
        "",
    ]


def _verifier_card(style: TerminalStyle, *, wrong: str, right: str, drift_num: int) -> list[str]:
    inner = _panel_width()
    sep = "═" * inner
    reject = f"REJECT  «{wrong}»  — never reaches output"
    commit = f"COMMIT  «{right}»  — from full-KV reference"
    foot = f"drift #{drift_num} corrected  ·  greedy path preserved"
    if not style.plain:
        reject = style.red(style.bold(reject))
        commit = style.green(style.bold(commit))
        foot = style.white(foot)
    return [
        "╔══ VERIFIER ══════════════════════════════════════════════════════════════════╗",
        _box_line(reject, inner, left="║", right="║"),
        _box_line(commit, inner, left="║", right="║"),
        _box_line(foot, inner, left="║", right="║"),
        f"╚{sep}╝",
        "",
    ]


def _flag_label(style: TerminalStyle, flag: str) -> str:
    if style.plain:
        return flag
    if flag == "DRIFT":
        return style.red(style.bold(flag))
    if flag == "MATCH":
        return style.green(style.bold(flag))
    if flag == "HOLD":
        return style.cyan("HOLD")
    if flag == "FIX":
        return style.cyan(style.bold("FIX"))
    if flag == "ok":
        return style.white("streaming")
    return style.white(flag)


def _column_header(style: TerminalStyle, label: str, *, active: bool, hot: bool) -> str:
    if style.plain:
        return ("▶ " if active else "  ") + label
    text = ("▶ " if active else "  ") + label
    if hot:
        return style.red(style.bold(text))
    if active:
        return style.bold(style.white(text))
    return style.white(text)


def _comparison_columns(
    *,
    style: TerminalStyle,
    full_vis: str,
    lossy_vis: str,
    exactkv_vis: str,
    active: int,
    lossy_flag: str,
    exactkv_flag: str,
    cursor_path: int = -1,
    hot_lossy: bool = False,
) -> list[str]:
    cw = _col_width(_panel_width())
    div = "─" * (cw + 2)
    top = f"┌{div}┬{div}┬{div}┐"
    mid = f"├{div}┼{div}┼{div}┤"
    bot = f"└{div}┴{div}┴{div}┘"

    headers = [
        _column_header(style, "FULL KV", active=(active == 0), hot=False),
        _column_header(style, "LOSSY DRAFT", active=(active == 1), hot=hot_lossy),
        _column_header(style, "EXACTKV OUT", active=(active == 2), hot=False),
    ]
    flags = [
        _flag_label(style, "truth"),
        _flag_label(style, lossy_flag),
        _flag_label(style, exactkv_flag),
    ]

    full_lines = _wrap_snippet(full_vis, cw, max_lines=SNIPPET_LINES)
    lossy_lines = _wrap_snippet(lossy_vis, cw, max_lines=SNIPPET_LINES)
    exact_lines = _wrap_snippet(exactkv_vis, cw, max_lines=SNIPPET_LINES)

    if cursor_path == 1 and lossy_lines:
        lossy_lines = list(lossy_lines)
        lossy_lines[-1] = lossy_lines[-1] + ("▌" if style.plain else style.cyan("▌"))
    if cursor_path == 2 and exact_lines:
        exact_lines = list(exact_lines)
        exact_lines[-1] = exact_lines[-1] + ("▌" if style.plain else style.cyan("▌"))

    def _styled_cell(plain: str, *, hot: bool) -> str:
        fitted = _fit_cell(plain, cw)
        if style.plain:
            return fitted
        stripped = fitted.rstrip()
        pad = " " * max(0, cw - len(stripped))
        if hot and stripped:
            return style.red(style.bold(stripped)) + pad
        if stripped:
            return style.white(stripped) + pad
        return fitted

    title = "SIDE-BY-SIDE TOKEN PATHS"
    if not style.plain:
        title = style.bold(style.white(title))

    lines: list[str] = [title, top]
    lines.append("│" + "│".join(f" {_fit_cell(_strip_ansi(h), cw)} " for h in headers) + "│")
    lines.append(mid)
    lines.append("│" + "│".join(f" {_fit_cell(_strip_ansi(f), cw)} " for f in flags) + "│")
    lines.append(mid)
    for i in range(SNIPPET_LINES):
        c1 = _styled_cell(full_lines[i], hot=False)
        c2 = _styled_cell(lossy_lines[i], hot=hot_lossy and bool(lossy_lines[i].strip()))
        c3 = _styled_cell(exact_lines[i], hot=False)
        lines.append("│" + f" {c1} " + "│" + f" {c2} " + "│" + f" {c3} " + "│")
    lines.append(bot)
    return lines


def _ship_comparison(style: TerminalStyle) -> list[str]:
    if style.plain:
        return [
            "",
            "WITHOUT EXACTKV (compressed KV only)     WITH EXACTKV (verifier crash-test)",
            '  "units":"imperial"                         "units":"metric"',
            '  "temp_c":22                               "temp_c":18',
            '  "conditions":"overcast"                   "conditions":"clear skies"',
            '  "humidity_pct":45                         "humidity_pct":72',
            "  wrong tool JSON ships silently             full-KV greedy path preserved",
            "  downstream agent gets bad facts            8,132-cell panels: 0 exactness failures",
            "",
        ]
    return [
        "",
        style.bold(style.white("WITHOUT EXACTKV (compressed KV only)")),
        style.red('  "units":"imperial"'),
        style.red('  "temp_c":22'),
        style.red('  "conditions":"overcast"'),
        style.red('  "humidity_pct":45'),
        style.red("  wrong tool JSON ships silently"),
        "",
        style.bold(style.white("WITH EXACTKV (verifier crash-test)")),
        style.green('  "units":"metric"'),
        style.green('  "temp_c":18'),
        style.green('  "conditions":"clear skies"'),
        style.green('  "humidity_pct":72'),
        style.green("  full-KV greedy path preserved"),
        style.white("  8,132-cell panels · MBPP + BFCL + LongBench · exactkv_failures: 0"),
        "",
    ]


def _victory_banner(style: TerminalStyle, *, drifts_caught: int) -> list[str]:
    headline = "EXACTKV MATCH  ·  shipped output ≡ full precision KV"
    stats = (
        f"drifts caught & corrected: {drifts_caught}  ·  acceptance restored  ·  "
        f"exactkv_failures: 0"
    )
    scope = "measures: first-divergence index · token acceptance · verifier agreement per cell"
    if not style.plain:
        headline = style.green(style.bold(headline))
        stats = style.white(stats)
        scope = style.white(scope)
    return ["", headline, stats, scope, ""]


def _frame(
    *,
    style: TerminalStyle,
    full_vis: str,
    lossy_vis: str,
    exactkv_vis: str,
    active: int,
    lossy_flag: str,
    exactkv_flag: str,
    stream_pos: int,
    drifts_caught: int,
    phase: str,
    status: str = "",
    drift_note: str = "",
    drift_wrong: str = "",
    drift_right: str = "",
    drift_flash: bool = False,
    verifier_card: tuple[str, str, int] | None = None,
    show_ship: bool = False,
    cursor_path: int = -1,
    hot_lossy: bool = False,
) -> list[str]:
    body: list[str] = []
    body.extend(_value_prop(style))
    body.extend(_top_rail(style))
    body.extend(_context_block(style))
    body.append("")
    body.extend(_hud(style, stream_pos=stream_pos, drifts_caught=drifts_caught, phase=phase))
    if status:
        body.append(status)
        body.append("")
    if phase == "drift" and drift_wrong:
        body.extend(
            _drift_alert(
                style,
                note=drift_note,
                wrong=drift_wrong,
                right=drift_right,
                flash=drift_flash,
            )
        )
    body.extend(
        _comparison_columns(
            style=style,
            full_vis=full_vis,
            lossy_vis=lossy_vis,
            exactkv_vis=exactkv_vis,
            active=active,
            lossy_flag=lossy_flag,
            exactkv_flag=exactkv_flag,
            cursor_path=cursor_path,
            hot_lossy=hot_lossy,
        )
    )
    if verifier_card:
        wrong, right, num = verifier_card
        body.extend(_verifier_card(style, wrong=wrong, right=right, drift_num=num))
    if show_ship:
        body.extend(_ship_comparison(style))
        body.extend(_victory_banner(style, drifts_caught=drifts_caught))
    return body


def _draw_frame(live: LiveFrame, style: TerminalStyle, captured: list[str], **kwargs: object) -> None:
    lines = _frame(style=style, **kwargs)  # type: ignore[arg-type]
    captured.extend(lines)
    live.draw(lines, delay=0.0, no_delay=True)


def _stream_chars(
    text: str,
    *,
    style: TerminalStyle,
    live: LiveFrame,
    no_delay: bool,
    char_delay: float,
    full_vis: str,
    lossy_vis: str,
    exactkv_vis: str,
    stream_pos: int,
    drifts_caught: int,
    decision_card: tuple[str, str, int] | None,
    captured: list[str],
) -> tuple[str, str, str, int]:
    full_acc, lossy_acc, exact_acc = full_vis, lossy_vis, exactkv_vis
    pos = stream_pos
    status = style.white("streaming tokens…") if not style.plain else "streaming…"
    for ch in text:
        full_acc += ch
        lossy_acc += ch
        exact_acc += ch
        pos += 1
        _draw_frame(
            live,
            style,
            captured,
            full_vis=full_acc,
            lossy_vis=lossy_acc,
            exactkv_vis=exact_acc,
            active=2,
            lossy_flag="ok",
            exactkv_flag="ok",
            stream_pos=pos,
            drifts_caught=drifts_caught,
            phase="stream",
            status=status,
            verifier_card=decision_card,
            cursor_path=2,
        )
        _pause(char_delay, no_delay=no_delay)
    return full_acc, lossy_acc, exact_acc, pos


def run_streaming_demo(
    *,
    out: TextIO | None = None,
    no_delay: bool = False,
    plain: bool = False,
    speed: str = "launch",
) -> str:
    """Stream tool JSON; dramatic drift alerts; verifier reject/commit; victory beat."""
    if out is None:
        out = sys.stdout
    profile = SPEED_PROFILES.get(speed, SPEED_PROFILES["launch"])
    drift_pause = float(profile.get("drift_pause", profile["dramatic"]))
    char_delay = float(profile.get("typing", 0.05))
    style = TerminalStyle(plain=plain)
    live = LiveFrame(out=out, plain=plain)
    captured: list[str] = []

    if no_delay or plain:
        lines = _frame(
            style=style,
            full_vis=FULL_TEXT,
            lossy_vis=LOSSY_TEXT,
            exactkv_vis=FULL_TEXT,
            active=2,
            lossy_flag="DRIFT",
            exactkv_flag="MATCH",
            stream_pos=STREAM_TOTAL,
            drifts_caught=4,
            phase="victory",
            verifier_card=("imperial", "metric", 1),
            show_ship=True,
        )
        for line in lines:
            _emit(style, line, out=out, delay=0.0, no_delay=True)
        for line in CLOSING_LINES.splitlines():
            _emit(style, style.bold(line), out=out, delay=0.0, no_delay=True)
        return "\n".join(captured)

    for step in (0, 1, 2):
        live.draw(_intro_frame(style, step=step), delay=profile["section"], no_delay=no_delay)
    live.commit()
    _pause(profile["dramatic"] * 0.5, no_delay=no_delay)

    full_vis = lossy_vis = exactkv_vis = ""
    pos = 0
    drifts_caught = 0
    last_card: tuple[str, str, int] | None = None

    for seg in TIMELINE:
        if seg[0] == "sync":
            full_vis, lossy_vis, exactkv_vis, pos = _stream_chars(
                seg[1],
                style=style,
                live=live,
                no_delay=no_delay,
                char_delay=char_delay,
                full_vis=full_vis,
                lossy_vis=lossy_vis,
                exactkv_vis=exactkv_vis,
                stream_pos=pos,
                drifts_caught=drifts_caught,
                decision_card=last_card,
                captured=captured,
            )
            continue

        _, wrong, right, note = seg

        for j in range(1, len(wrong) + 1):
            lossy_vis = lossy_vis + wrong[j - 1]
            _draw_frame(
                live,
                style,
                captured,
                full_vis=full_vis + right,
                lossy_vis=lossy_vis,
                exactkv_vis=exactkv_vis,
                active=1,
                lossy_flag="DRIFT",
                exactkv_flag="HOLD",
                stream_pos=pos + j,
                drifts_caught=drifts_caught,
                phase="drift",
                drift_note=note,
                drift_wrong=wrong[:j],
                drift_right=right,
                hot_lossy=True,
                cursor_path=1,
            )
            _pause(char_delay * 1.4, no_delay=no_delay)

        flashes = 2
        for f in range(flashes):
            _draw_frame(
                live,
                style,
                captured,
                full_vis=full_vis + right,
                lossy_vis=lossy_vis,
                exactkv_vis=exactkv_vis,
                active=1,
                lossy_flag="DRIFT",
                exactkv_flag="HOLD",
                stream_pos=pos + len(wrong),
                drifts_caught=drifts_caught,
                phase="drift",
                drift_note=note,
                drift_wrong=wrong,
                drift_right=right,
                drift_flash=(f % 2 == 0),
                hot_lossy=True,
            )
            _pause(drift_pause / flashes, no_delay=no_delay)

        drifts_caught += 1
        last_card = (wrong, right, drifts_caught)
        _draw_frame(
            live,
            style,
            captured,
            full_vis=full_vis + right,
            lossy_vis=lossy_vis,
            exactkv_vis=exactkv_vis,
            active=2,
            lossy_flag="DRIFT",
            exactkv_flag="FIX",
            stream_pos=pos + len(wrong),
            drifts_caught=drifts_caught,
            phase="commit",
            verifier_card=last_card,
            hot_lossy=True,
        )
        _pause(drift_pause * 0.15, no_delay=no_delay)

        for k in range(1, len(right) + 1):
            exactkv_vis += right[k - 1]
            full_vis += right[k - 1]
            _draw_frame(
                live,
                style,
                captured,
                full_vis=full_vis,
                lossy_vis=lossy_vis,
                exactkv_vis=exactkv_vis,
                active=2,
                lossy_flag="DRIFT",
                exactkv_flag="FIX",
                stream_pos=pos + k,
                drifts_caught=drifts_caught,
                phase="commit",
                status=style.green(f"committing full-KV token… drift #{drifts_caught}")
                if not style.plain
                else f"committing… drift #{drifts_caught}",
                verifier_card=last_card,
                cursor_path=2,
                hot_lossy=True,
            )
            _pause(char_delay, no_delay=no_delay)
        pos += len(right)
        _pause(profile["section"] * 0.4, no_delay=no_delay)

    _draw_frame(
        live,
        style,
        captured,
        full_vis=FULL_TEXT,
        lossy_vis=LOSSY_TEXT,
        exactkv_vis=FULL_TEXT,
        active=2,
        lossy_flag="DRIFT",
        exactkv_flag="MATCH",
        stream_pos=STREAM_TOTAL,
        drifts_caught=drifts_caught,
        phase="victory",
        show_ship=True,
    )
    _pause(profile["dramatic"], no_delay=no_delay)
    live.commit()

    for line in CLOSING_LINES.splitlines():
        _emit(style, style.bold(line), out=out, delay=profile["section"], no_delay=no_delay)

    return "\n".join(captured)
