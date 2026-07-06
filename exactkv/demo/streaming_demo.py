"""Single-act streaming terminal demo — cinematic crash-test replay."""
from __future__ import annotations

import sys
import time
from typing import TextIO

from exactkv.demo.case_study_loader import CLOSING_LINES, PUBLIC_TAGLINE
from exactkv.demo.live_terminal import SPEED_PROFILES, LiveFrame, TerminalStyle, _emit, progress_bar

INNER = 78

META = {
    "prompt_label": "structured tool JSON · int4_sim 4× · continuous greedy decode",
    "prompt": 'Emit weather tool result: {"tool":"get_weather","city":"Paris",',
    "compressor": "int4_sim · 4× quant",
    "model": "Mistral-7B-Instruct",
}

TIMELINE: tuple[tuple[str, ...], ...] = (
    ("sync", '{"tool":"get_weather","city":"Paris","units":"'),
    ("drift", "imperial", "metric", "wrong units field"),
    ("sync", '","temp_c":18,"conditions":"'),
    ("drift", "overcast", "clear skies", "wrong conditions field"),
    ("sync", '","wind":"NW 12km/h","valid":true}'),
)

FULL_TEXT = "".join(seg[1] if seg[0] == "sync" else seg[2] for seg in TIMELINE)
LOSSY_TEXT = "".join(seg[1] if seg[0] == "sync" else seg[1] for seg in TIMELINE)
STREAM_TOTAL = len(FULL_TEXT)

SNIPPET_LINES = 3

LOGO = [
    " ███████╗██╗  ██╗ █████╗  ██████╗████████╗██╗  ██╗██╗   ██╗",
    " ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝╚══██╔══╝██║ ██╔╝██║   ██║",
    " █████╗   ╚███╔╝ ███████║██║        ██║   █████╔╝ ██║   ██║",
    " ██╔══╝   ██╔██╗ ██╔══██║██║        ██║   ██╔═██╗ ╚██╗ ██╔╝",
    " ███████╗██╔╝ ██╗██║  ██║╚██████╗   ██║   ██║  ██╗ ╚████╔╝ ",
    " ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝  ╚═══╝  ",
]


def _pause(seconds: float, *, no_delay: bool) -> None:
    if not no_delay and seconds > 0:
        time.sleep(seconds)


def _wrap_snippet(text: str, width: int, *, max_lines: int = SNIPPET_LINES) -> list[str]:
    """Wrap snippet text on word/punctuation boundaries; never split mid-word when avoidable."""
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


def _box_line(inner: str, left: str = "┃", right: str = "┃") -> str:
    return f"{left} {inner[:INNER].ljust(INNER)} {right}"


def _intro_frame(style: TerminalStyle, *, step: int) -> list[str]:
    sep = "━" * (INNER + 2)
    lines = [f"┏{sep}┓", _box_line("")]
    for row in LOGO:
        lines.append(_box_line(row.center(INNER)))
    lines.append(_box_line(""))
    title = "▶  CRASH TEST  ·  LIVE VERIFIER REPLAY"
    if not style.plain:
        title = style.bold(style.cyan(title))
    lines.append(_box_line(title.center(INNER)))
    if step >= 1:
        sub = "compressed KV drafts  ·  full KV verifies  ·  drift never ships"
        lines.append(_box_line(style.dim(sub.center(INNER)) if not style.plain else sub.center(INNER)))
    if step >= 2:
        tag = PUBLIC_TAGLINE.replace("\n", "  ·  ")
        lines.append(_box_line(""))
        if not style.plain:
            lines.append(_box_line(style.bold(tag.center(INNER))))
        else:
            lines.append(_box_line(tag.center(INNER)))
    lines.extend([_box_line(""), f"┗{sep}┛"])
    return lines


def _top_rail(style: TerminalStyle) -> list[str]:
    sep = "━" * (INNER + 2)
    badge = "verifier ● LIVE"
    if not style.plain:
        badge = style.green("verifier") + style.dim(" ● ") + style.bold("LIVE")
    title = style.bold("EXACTKV") if not style.plain else "EXACTKV"
    return [
        f"┏{sep}┓",
        _box_line(f"{title:<20}{META['model']:<34}{badge:>{INNER - 54}}"),
        _box_line(style.dim(META["compressor"] + "  ·  " + META["prompt_label"]) if not style.plain
                  else META["compressor"] + "  ·  " + META["prompt_label"]),
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
        drift_txt = style.red(f"● {drifts_caught}")
    if phase == "drift":
        verifier = style.red("ENGAGED") if not style.plain else "[ENGAGED]"
    elif phase == "commit":
        verifier = style.green("FIXING") if not style.plain else "[FIXING]"
    elif phase == "victory":
        verifier = style.green("MATCH") if not style.plain else "MATCH"
    else:
        verifier = style.cyan("armed") if not style.plain else "armed"
    line1 = f"DECODE {bar} {pct:>3}%   DRIFTS {drift_txt:<6}   VERIFIER {verifier}"
    line2 = style.dim("full KV reference locked · lossy draft streams · verifier compares every token") if not style.plain else "full KV reference · lossy draft · verifier compares"
    return [line1[:INNER + 6], line2[:INNER + 6], ""]


def _drift_alert(
    style: TerminalStyle,
    *,
    note: str,
    wrong: str,
    right: str,
    flash: bool = False,
) -> list[str]:
    sep = "━" * (INNER + 2)
    headline = "⛔  DIVERGENCE DETECTED  ·  lossy draft ≠ full KV  ·  verifier blocking output"
    detail = f"{note} — draft proposed «{wrong}»  ·  full KV expects «{right}»"
    if not style.plain:
        headline = style.red(style.bold(headline))
        detail = style.yellow(style.bold(detail) if flash else detail)
    return [
        f"┏{sep}┓",
        _box_line(headline),
        _box_line(detail),
        f"┗{sep}┛",
        "",
    ]


def _verifier_card(style: TerminalStyle, *, wrong: str, right: str, drift_num: int) -> list[str]:
    sep = "═" * INNER
    reject = f"✗  REJECT   «{wrong}»"
    commit = f"✓  COMMIT   «{right}»"
    foot = f"draft never reaches output buffer  ·  drift #{drift_num} corrected"
    if not style.plain:
        reject = style.red(style.bold(reject))
        commit = style.green(style.bold(commit))
        foot = style.dim(foot)
    return [
        f"╔══ VERIFIER ACTION ═══════════════════════════════════════════════════════════╗",
        _box_line(reject, left="║", right="║"),
        _box_line(commit, left="║", right="║"),
        _box_line(foot, left="║", right="║"),
        f"╚{sep}╝",
        "",
    ]


def _path_block_rows(
    *,
    style: TerminalStyle,
    idx: int,
    active: int,
    label: str,
    flag: str,
    vis: str,
    label_w: int,
    flag_w: int,
    snippet_w: int,
    cursor: bool = False,
    hot: bool = False,
) -> list[str]:
    prefix = "▶" if idx == active else " "
    flag_out = flag
    if not style.plain:
        if flag == "DRIFT":
            flag_out = style.red(style.bold(flag))
        elif flag == "MATCH":
            flag_out = style.green(style.bold(flag))
        elif flag == "hold":
            flag_out = style.yellow("HOLD")
        elif flag == "fix":
            flag_out = style.cyan("FIX")
        elif flag == "ok":
            flag_out = style.dim("ok")
    flag_pad = flag_w if style.plain else max(flag_w, len(flag))
    wrapped = _wrap_snippet(vis, snippet_w)
    if cursor and wrapped:
        last = wrapped[-1]
        wrapped[-1] = last + (style.cyan("▌") if not style.plain else "|")
    cont_pad = 2 + (label_w - 1) + 1 + flag_w + 1
    label_out = label
    if hot and not style.plain:
        label_out = style.red(style.bold(label))
    elif idx == 2 and not style.plain:
        label_out = style.green(label)
    elif idx == 0 and not style.plain:
        label_out = style.dim(label)
    rows: list[str] = []
    for line_no, snip in enumerate(wrapped):
        if line_no == 0:
            row = f"{prefix} {label_out:<{label_w - 1}} {flag_out:<{flag_pad}} {snip}"
        else:
            row = f"{' ' * cont_pad}{snip}"
        border = "│"
        rows.append(f"{border} " + row[:INNER].ljust(INNER - 1) + " │")
    return rows


def _comparison_block(
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
    label_w, flag_w = 14, 8
    snippet_w = INNER - label_w - flag_w - 7
    sep = "─" * INNER
    paths = [
        (0, "Full KV ref", "—", full_vis, False),
        (1, "Lossy draft", lossy_flag, lossy_vis, hot_lossy),
        (2, "ExactKV out", exactkv_flag, exactkv_vis, False),
    ]
    hdr = "LIVE TOKEN PATHS  ·  compressed draft vs verified output"
    if not style.plain:
        hdr = style.bold(hdr)
    lines = [
        f"┌{sep}┐",
        f"│ {hdr[:INNER].ljust(INNER)} │",
        f"├{sep}┤",
    ]
    for pidx, (idx, label, flag, vis, hot) in enumerate(paths):
        if pidx:
            lines.append(f"├{sep}┤")
        lines.extend(
            _path_block_rows(
                style=style,
                idx=idx,
                active=active,
                label=label,
                flag=flag,
                vis=vis,
                label_w=label_w,
                flag_w=flag_w,
                snippet_w=snippet_w,
                cursor=(idx == cursor_path),
                hot=hot,
            )
        )
    lines.append(f"└{sep}┘")
    return lines


def _ship_comparison(style: TerminalStyle) -> list[str]:
    if style.plain:
        return [
            "",
            "WHAT WOULD HAVE SHIPPED",
            "",
            "  WITHOUT EXACTKV              WITH EXACTKV",
            '  "units":"imperial"           "units":"metric"',
            '  "conditions":"overcast"      "conditions":"clear skies"',
            "  silent wrong tool call       exact greedy path preserved",
            "",
        ]
    return [
        "",
        style.bold("WHAT WOULD HAVE SHIPPED"),
        "",
        f"  {style.red('WITHOUT EXACTKV'):<28}  {style.green('WITH EXACTKV')}",
        f"  {style.red('\"units\":\"imperial\"'):<28}  {style.green('\"units\":\"metric\"')}",
        f"  {style.red('\"conditions\":\"overcast\"'):<28}  {style.green('\"conditions\":\"clear skies\"')}",
        f"  {style.red('✗ silent wrong tool call'):<28}  {style.green('✓ exact greedy path')}",
        "",
    ]


def _victory_banner(style: TerminalStyle, *, drifts_caught: int) -> list[str]:
    bar = "█" * min(INNER, 60)
    headline = "EXACTKV MATCH  ·  final output ≡ full precision KV"
    stats = f"drifts caught & corrected: {drifts_caught}  ·  exactkv_failures: 0"
    if not style.plain:
        headline = style.green(style.bold(headline))
        stats = style.bold(stats)
    return ["", bar, headline, stats, bar, ""]


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
    body.extend(_top_rail(style))
    body.append("")
    body.append(META["prompt"])
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
        _comparison_block(
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
    status = style.dim("streaming greedy decode…") if not style.plain else "streaming…"
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
            drifts_caught=2,
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
                exactkv_flag="hold",
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

        flashes = 3 if drift_pause >= 2.0 else 2
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
                exactkv_flag="hold",
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
            exactkv_flag="fix",
            stream_pos=pos + len(wrong),
            drifts_caught=drifts_caught,
            phase="commit",
            verifier_card=last_card,
            hot_lossy=True,
        )
        _pause(drift_pause * 0.35, no_delay=no_delay)

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
                exactkv_flag="fix",
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
