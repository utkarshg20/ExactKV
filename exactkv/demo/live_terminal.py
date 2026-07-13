"""Shard/TurboQuant-style terminal UI for ExactKV case-study replay."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO

from exactkv.demo.case_study_loader import (
    CLOSING_LINES,
    PUBLIC_TAGLINE,
    CaseStudy,
)

SPEED_PROFILES = {
    "instant": {"pause": 0.0, "typing": 0.0, "dramatic": 0.0, "section": 0.0, "row": 0.0},
    "fast": {"pause": 0.08, "typing": 0.02, "dramatic": 0.25, "section": 0.06, "row": 0.12},
    "cinematic": {"pause": 0.55, "typing": 0.045, "dramatic": 1.8, "section": 0.45, "row": 0.35},
    "launch": {"pause": 0.55, "typing": 0.082, "dramatic": 1.6, "section": 0.4, "row": 0.3, "drift_pause": 2.0},
    "social": {"pause": 0.4, "typing": 0.038, "dramatic": 1.6, "section": 0.35, "row": 0.3, "drift_pause": 2.0},
    # Hero cut: ~20–28s terminal segment after optional Sora cold-open.
    "hero": {"pause": 0.28, "typing": 0.028, "dramatic": 1.35, "section": 0.22, "row": 0.2, "drift_pause": 1.55},
    "default": {"pause": 0.35, "typing": 0.03, "dramatic": 1.0, "section": 0.25, "row": 0.2},
}


@dataclass
class TerminalStyle:
    plain: bool = False

    def wrap(self, text: str, code: str) -> str:
        if self.plain:
            return text
        return f"\033[{code}m{text}\033[0m"

    def red(self, text: str) -> str:
        return self.wrap(text, "31")

    def green(self, text: str) -> str:
        return self.wrap(text, "32")

    def yellow(self, text: str) -> str:
        return self.wrap(text, "33")

    def bold(self, text: str) -> str:
        return self.wrap(text, "1")

    def dim(self, text: str) -> str:
        return self.wrap(text, "2")

    def white(self, text: str) -> str:
        return self.wrap(text, "37")

    def cyan(self, text: str) -> str:
        return self.wrap(text, "36")

    def magenta(self, text: str) -> str:
        return self.wrap(text, "35")

    def inverse(self, text: str) -> str:
        return self.wrap(text, "7")

    def on_red(self, text: str) -> str:
        return self.wrap(text, "41;37;1")

    def on_green(self, text: str) -> str:
        return self.wrap(text, "42;30;1")

    def print(self, text: str = "", *, file: TextIO | None = None) -> None:
        print(text, file=file or sys.stdout)


def progress_bar(filled: int, total: int, width: int = 20) -> str:
    filled = max(0, min(filled, total))
    if total <= 0:
        return "[" + " " * width + "]"
    n = int(width * filled / total)
    return "[" + "█" * n + "░" * (width - n) + "]"


def _header_lines() -> list[str]:
    return [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║ EXACTKV LIVE CASE STUDIES                                        ║",
        "║ Full KV · Lossy draft · ExactKV verifier · verified GPU panels   ║",
        "╚══════════════════════════════════════════════════════════════════╝",
    ]



def _first_diff_suffix(full: str, lossy: str) -> str:
    limit = min(len(full), len(lossy))
    for i in range(limit):
        if full[i] != lossy[i]:
            start = max(0, i - 12)
            return f"first diff near: …{lossy[start:i + 8]}…"
    if len(lossy) != len(full):
        return "paths diverge in length"
    return ""


def _pause(seconds: float, *, no_delay: bool) -> None:
    if not no_delay and seconds > 0:
        time.sleep(seconds)


def _snippet_one_line(text: str, width: int) -> str:
    one = text.replace("\n", "\\n").replace("\r", "")
    if len(one) <= width:
        return one
    return one[: width - 1] + "…"


class LiveFrame:
    """Redraw a fixed-height block in place (Shard-style measuring updates)."""

    def __init__(self, *, out: TextIO, plain: bool) -> None:
        self.out = out
        self.plain = plain
        self._line_count = 0

    def draw(self, lines: list[str], *, delay: float = 0.0, no_delay: bool = False) -> None:
        if self._line_count and not self.plain:
            self.out.write(f"\033[{self._line_count}A")
        for line in lines:
            if self._line_count and not self.plain:
                self.out.write("\033[2K\r")
            print(line, file=self.out)
        self._line_count = len(lines)
        self.out.flush()
        _pause(delay, no_delay=no_delay)

    def commit(self) -> None:
        """Stop in-place updates; following output appends below."""
        self._line_count = 0


def _emit(style: TerminalStyle, text: str, *, out: TextIO, delay: float, no_delay: bool) -> None:
    style.print(text, file=out)
    _pause(delay, no_delay=no_delay)


def _comparison_table(
    case: CaseStudy,
    *,
    style: TerminalStyle,
    active_row: int,
    reveal_chars: dict[str, int],
    inner_width: int = 66,
) -> list[str]:
    """Three-path table with optional partial reveal per path key."""
    label_w = 14
    flag_w = 8
    snippet_w = inner_width - label_w - flag_w - 7

    rows: list[tuple[str, str, str, str]] = [
        ("Full KV", "—", case.full_snippet, "full"),
        ("Lossy draft", "DRIFT" if case.has_drift else "same", case.lossy_snippet, "lossy"),
        ("ExactKV out", "MATCH" if case.exactkv_matches_full else "diff", case.exactkv_snippet, "exactkv"),
    ]

    lines: list[str] = []
    sep = "─" * inner_width
    lines.append(f"┌{sep}┐")
    header = f"│ {'path':<{label_w}} {'drift?':<{flag_w}} snippet"
    lines.append(header.ljust(inner_width + 1) + "│")
    lines.append(f"├{'─' * inner_width}┤")

    for idx, (label, flag, snippet, key) in enumerate(rows):
        prefix = "▶" if idx == active_row else " "
        n = reveal_chars.get(key, len(snippet))
        visible = _snippet_one_line(snippet[:n], snippet_w)
        flag_styled = flag
        if not style.plain:
            if flag == "DRIFT":
                flag_styled = style.red(flag)
            elif flag == "MATCH":
                flag_styled = style.green(flag)

        row = f"{prefix} {label:<{label_w - 1}} {flag_styled:<{flag_w if style.plain else len(flag)}} {visible}"
        if len(row) > inner_width + 1:
            row = row[: inner_width]
        lines.append("│ " + row.ljust(inner_width - 1) + "│")

    lines.append(f"└{sep}┘")
    return lines


def _live_panel(
    case: CaseStudy,
    *,
    style: TerminalStyle,
    active_row: int,
    reveal_chars: dict[str, int],
    paths_filled: int,
    paths_total: int,
    status: str,
) -> list[str]:
    table = _comparison_table(case, style=style, active_row=active_row, reveal_chars=reveal_chars)
    bar = progress_bar(paths_filled, paths_total)
    return [f"paths {bar}  {status}", ""] + table


def render_case_study(
    case: CaseStudy,
    *,
    case_num: int,
    case_total: int,
    out: TextIO | None = None,
    no_delay: bool = False,
    plain: bool = False,
    speed: str = "default",
) -> str:
    if out is None:
        out = sys.stdout
    profile = SPEED_PROFILES.get(speed, SPEED_PROFILES["default"])
    style = TerminalStyle(plain=plain)
    captured: list[str] = []

    def step(text: str = "", *, delay: float | None = None) -> None:
        captured.append(text)
        _emit(style, text, out=out, delay=delay or profile["pause"], no_delay=no_delay)

    def dramatic(text: str = "") -> None:
        captured.append(text)
        _emit(style, text, out=out, delay=profile["dramatic"], no_delay=no_delay)

    # Case banner (Shard-style section header)
    step(style.bold(f"{'═' * 70}"))
    step(style.bold(f"CASE {case_num}/{case_total}  {case.title}"))
    step(style.dim(case.meta_line))
    step(style.bold(f"{'═' * 70}"))
    step("")

    step(style.dim("comparing token paths from headline GPU panel…"))
    step("")

    paths_total = 3
    full_len = len(case.full_snippet)
    lossy_len = len(case.lossy_snippet)
    exact_len = len(case.exactkv_snippet)

    if no_delay:
        step(f"paths {progress_bar(paths_total, paths_total)}  {paths_total} / {paths_total}  all paths measured")
        block = _comparison_table(
            case,
            style=style,
            active_row=1 if case.has_drift else 2,
            reveal_chars={"full": full_len, "lossy": lossy_len, "exactkv": exact_len},
        )
        for line in block:
            step(line, delay=0.0)
        step("")
        if case.has_drift:
            dramatic(style.bold(style.red("DRIFT DETECTED — lossy draft ≠ full KV")))
        if case.exactkv_matches_full:
            dramatic(style.bold(style.green("EXACTKV MATCH — verifier output = full KV")))
        step("")
        failures = 1 if case.exactkv_failure else 0
        score = [
            f"exactkv_failures: {failures}",
            f"final matches full KV: {'YES' if case.exactkv_matches_full else 'NO'}",
            f"lossy drifted: {'YES' if case.has_drift else 'NO'}",
        ]
        step("  ·  ".join(score), delay=0.0)
        step("")
        return "\n".join(captured)

    reveal: dict[str, int] = {"full": 0, "lossy": 0, "exactkv": 0}
    live = LiveFrame(out=out, plain=plain)

    def _animate_path(
        key: str,
        active_row: int,
        path_len: int,
        paths_filled: int,
        status: str,
        *,
        steps: int = 24,
        typing_scale: float = 1.0,
    ) -> None:
        chunk = max(1, path_len // steps or 1)
        for filled in range(0, path_len + 1, chunk):
            reveal[key] = min(filled, path_len)
            frame = _live_panel(
                case,
                style=style,
                active_row=active_row,
                reveal_chars=reveal,
                paths_filled=paths_filled,
                paths_total=paths_total,
                status=status,
            )
            live.draw(frame, delay=profile["typing"] * typing_scale, no_delay=no_delay)
        reveal[key] = path_len

    # Path 1 — Full KV
    _animate_path("full", 0, full_len, 0, "measuring Full KV…", typing_scale=2.0)
    live.draw(
        _live_panel(
            case,
            style=style,
            active_row=0,
            reveal_chars=reveal,
            paths_filled=1,
            paths_total=paths_total,
            status=f"1 / {paths_total}  Full KV locked",
        ),
        delay=profile["section"],
        no_delay=no_delay,
    )
    live.commit()
    step("")

    # Path 2 — Lossy draft
    _animate_path("lossy", 1, lossy_len, 1, "measuring Lossy draft…", typing_scale=3.0)
    live.commit()

    if case.has_drift:
        dramatic(style.bold(style.red("DRIFT DETECTED — lossy draft ≠ full KV")))
        hint = _first_diff_suffix(case.full_snippet, case.lossy_snippet)
        if hint:
            step(style.dim(hint))
    else:
        step(style.dim("lossy path matches full on this snippet window"))
    step("")

    live.draw(
        _live_panel(
            case,
            style=style,
            active_row=1,
            reveal_chars=reveal,
            paths_filled=2,
            paths_total=paths_total,
            status=f"2 / {paths_total}  Lossy draft measured",
        ),
        delay=profile["section"],
        no_delay=no_delay,
    )
    live.commit()
    step("")

    # Path 3 — ExactKV
    _animate_path("exactkv", 2, exact_len, 2, "measuring ExactKV verifier…", typing_scale=2.0)
    live.draw(
        _live_panel(
            case,
            style=style,
            active_row=2,
            reveal_chars=reveal,
            paths_filled=paths_total,
            paths_total=paths_total,
            status=f"{paths_total} / {paths_total}  all paths measured",
        ),
        delay=profile["section"],
        no_delay=no_delay,
    )
    live.commit()

    if case.exactkv_matches_full:
        dramatic(style.bold(style.green("EXACTKV MATCH — verifier output = full KV")))
    step("")
    failures = 1 if case.exactkv_failure else 0
    score = [
        f"exactkv_failures: {failures}",
        f"final matches full KV: {'YES' if case.exactkv_matches_full else 'NO'}",
        f"lossy drifted: {'YES' if case.has_drift else 'NO'}",
    ]
    step("  ·  ".join(score), delay=profile["section"])
    step("")

    return "\n".join(captured)


def run_live_demo(
    cases: list[CaseStudy],
    *,
    out: TextIO | None = None,
    no_delay: bool = False,
    plain: bool = False,
    speed: str = "default",
) -> str:
    if out is None:
        out = sys.stdout
    profile = SPEED_PROFILES.get(speed, SPEED_PROFILES["default"])
    style = TerminalStyle(plain=plain)
    parts: list[str] = []

    for line in _header_lines():
        _emit(style, line, out=out, delay=profile["section"], no_delay=no_delay)
        parts.append(line)

    _emit(style, "", out=out, delay=profile["pause"], no_delay=no_delay)
    for tag in PUBLIC_TAGLINE.splitlines():
        _emit(style, style.bold(tag), out=out, delay=profile["section"], no_delay=no_delay)
        parts.append(tag)
    _emit(style, "", out=out, delay=profile["pause"], no_delay=no_delay)

    total = len(cases)
    for i, case in enumerate(cases, start=1):
        body = render_case_study(
            case,
            case_num=i,
            case_total=total,
            out=out,
            no_delay=no_delay,
            plain=plain,
            speed=speed,
        )
        parts.append(body)
        if i < total:
            _emit(style, "", out=out, delay=profile["section"], no_delay=no_delay)

    for line in CLOSING_LINES.splitlines():
        _emit(style, style.bold(line), out=out, delay=profile["section"], no_delay=no_delay)
        parts.append(line)

    return "\n".join(parts)
