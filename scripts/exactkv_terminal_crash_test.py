#!/usr/bin/env python3
"""ExactKV terminal-native crash-test demo (V13 Phase 8e).

Replays a verified semantic correction trace (Exp 034b pharm_001 preferred;
Exp 034 tj_002 fallback). No model inference in default replay mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_034B_JSON = _ROOT / "reports" / "experiment_034b_semantic_correction_search.json"
_DEFAULT_034_JSON = _ROOT / "reports" / "experiment_034_killer_correction_demo.json"

PUBLIC_TAGLINE = (
    "Everyone is racing to shrink KV caches.\n"
    "ExactKV tells you when they start lying."
)

CLOSING_LINES = (
    "KV compression should not be trusted.\n"
    "It should be crash-tested.\n"
    "\n"
    "ExactKV: token-level crash tests for KV-cache compression."
)

# Verified Exp 034b winner: pharm_001 × k8_v4_sim (drop → pickup).
SEMANTIC_TRACE_FIXTURE: dict[str, Any] = {
    "source": "experiment_034b_semantic_correction_search (pharm_001)",
    "trace_kind": "semantic",
    "prompt_id": "pharm_001",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "v10_suite": "crafted_pharmacy",
    "compressor_name": "k8_v4_sim",
    "draft_len": 4,
    "prompt": (
        'JSON tool call only: {"tool":"refill_prescription","drug":"ibuprofen",'
        '"quantity":30,"pickup":'
    ),
    "prompt_label": "PHARMACY TOOL CALL PROMPT",
    "full_output_text": (
        'true,"pickup_date":"2022-01-01","pickup_time":"10:00:00","dropoff":true'
    ),
    "lossy_output_text": (
        'true,"dropoff":true,"date":"2019-01-01","time":"10:00:00","notes'
    ),
    "exactkv_output_text": (
        'true,"pickup_date":"2022-01-01","pickup_time":"10:00:00","dropoff":true'
    ),
    "lossy_first_divergence_idx": 2,
    "exactkv_exact_match": True,
    "exactkv_failures": 0,
    "drift_message": "compressed KV tried to use dropoff instead of pickup",
    "highlight_round": {
        "round_idx": 0,
        "first_rejected_token": 6719,
        "correction_token": 84684,
        "first_rejected_text": "drop",
        "correction_text": "pickup",
        "lossy_draft_fragment": 'true,"drop',
        "draft_tokens_text": ["true", ',"', "drop", "off"],
    },
}

# Exp 034 punctuation fallback (tj_002 × int4_sim).
EXP034_TRACE_FIXTURE: dict[str, Any] = {
    "source": "experiment_034_killer_correction_demo (tj_002 fallback)",
    "trace_kind": "punctuation_fallback",
    "prompt_id": "tj_002",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "v10_suite": "tool_json",
    "compressor_name": "int4_sim",
    "draft_len": 4,
    "prompt": (
        'Complete this tool call JSON: {"name": "get_weather", "arguments": '
        '{"city": "Paris", "units":'
    ),
    "prompt_label": "STRUCTURED TOOL CALL PROMPT",
    "full_output_text": (
        ' "metric"}} To complete this tool call JSON, you would need to define a '
        "function that takes in the necessary parameters and returns the weather "
        "data for the specified city"
    ),
    "lossy_output_text": (
        ' "}}}\n\n{"name": "get_weather", "arguments": {"city": "Paris", '
        '"units": "metric"}}}\n\n{"name": "get_weather'
    ),
    "exactkv_output_text": (
        ' "metric"}} To complete this tool call JSON, you would need to define a '
        "function that takes in the necessary parameters and returns the weather "
        "data for the specified city"
    ),
    "lossy_first_divergence_idx": 1,
    "exactkv_exact_match": True,
    "exactkv_failures": 0,
    "drift_message": "compressed KV tried to close the JSON before the units value",
    "highlight_round": {
        "round_idx": 0,
        "first_rejected_token": 3417,
        "correction_token": 15903,
        "first_rejected_text": "}}",
        "correction_text": "metric",
        "lossy_draft_fragment": '"}}',
        "draft_tokens_text": [' "', "}}", "}", "\n"],
    },
}

SPEED_PROFILES = {
    "default": {"pause": 0.55, "typing": 0.12, "dramatic": 1.4, "section": 0.35},
    "fast": {"pause": 0.12, "typing": 0.04, "dramatic": 0.35, "section": 0.08},
    "cinematic": {"pause": 0.85, "typing": 0.16, "dramatic": 2.2, "section": 0.55},
}


@dataclass(frozen=True)
class CrashTestTrace:
    prompt_id: str
    model_name: str
    v10_suite: str
    compressor_name: str
    draft_len: int
    prompt: str
    prompt_label: str
    full_output_text: str
    lossy_output_text: str
    exactkv_output_text: str
    lossy_first_divergence_idx: int
    exactkv_exact_match: bool
    exactkv_failures: int
    rejected_token_text: str
    correction_token_text: str
    lossy_draft_fragment: str
    drift_message: str
    trace_kind: str
    trace_source: str
    draft_token_count: int = 4


class Style:
    """ANSI styling with plain fallback."""

    def __init__(self, *, plain: bool) -> None:
        self.plain = plain
        self._rich_console = None
        if not plain:
            try:
                from rich.console import Console

                self._rich_console = Console()
            except ImportError:
                pass

    @property
    def has_rich(self) -> bool:
        return self._rich_console is not None

    def _wrap(self, text: str, code: str) -> str:
        if self.plain:
            return text
        return f"\033[{code}m{text}\033[0m"

    def red(self, text: str) -> str:
        return self._wrap(text, "31")

    def green(self, text: str) -> str:
        return self._wrap(text, "32")

    def yellow(self, text: str) -> str:
        return self._wrap(text, "33")

    def bold(self, text: str) -> str:
        return self._wrap(text, "1")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def print(self, text: str = "", file: TextIO | None = None) -> None:
        if file is None:
            file = sys.stdout
        print(text, file=file)


def _progress_bar(filled: int, total: int, width: int = 16) -> str:
    filled = max(0, min(filled, total))
    if total <= 0:
        return "[" + " " * width + "]"
    n = int(width * filled / total)
    return "[" + "█" * n + "░" * (width - n) + "]"


def _header_box() -> list[str]:
    return [
        "╔════════════════════════════════════════════════════════════════╗",
        "║ EXACTKV CRASH TEST                                            ║",
        "║ Compressed KV drafts. Full KV verifies.                       ║",
        "╚════════════════════════════════════════════════════════════════╝",
    ]


def _panel(title: str, lines: list[str], width: int = 64) -> list[str]:
    inner = max(4, width - 4)
    out = [f"┌─ {title} " + "─" * max(0, inner - len(title) - 3) + "┐"]
    for line in lines:
        if len(line) > inner:
            line = line[: inner - 1] + "…"
        out.append("│ " + line.ljust(inner) + " │")
    out.append("└" + "─" * inner + "┘")
    return out


def _infer_drift_message(rejected: str, correction: str, suite: str) -> str:
    rej = rejected.strip().strip('"').lower()
    corr = correction.strip().strip('"').lower()
    if "drop" in rej and "pickup" in corr:
        return "compressed KV tried to use dropoff instead of pickup"
    if rej == "}}" and corr == "metric":
        return "compressed KV tried to close the JSON before the units value"
    if "south" in rej and "north" in corr:
        return "compressed KV corrupted the entity name (SOUTH instead of NORTH)"
    if suite.startswith("crafted_") or suite == "tool_json":
        return "compressed KV changed the tool call"
    return "compressed KV drifted from full-KV greedy output"


def _lossy_fragment(demo: dict[str, Any], rejected: str) -> str:
    hr = demo.get("highlight_round") or {}
    if hr.get("lossy_draft_fragment"):
        return str(hr["lossy_draft_fragment"])
    lossy = str(demo.get("lossy_output_text", ""))
    if rejected and rejected in lossy:
        return lossy[: lossy.index(rejected) + len(rejected)]
    texts = hr.get("draft_tokens_text") or []
    if texts:
        return "".join(texts)
    return lossy[: min(24, len(lossy))]


def _prompt_label(demo: dict[str, Any]) -> str:
    if demo.get("prompt_label"):
        return str(demo["prompt_label"])
    suite = str(demo.get("v10_suite", ""))
    if suite == "crafted_pharmacy":
        return "PHARMACY TOOL CALL PROMPT"
    if suite in ("crafted_order", "tool_json"):
        return "STRUCTURED TOOL CALL PROMPT"
    return "STRUCTURED OUTPUT PROMPT"


def _demo_from_dict(demo: dict[str, Any], source: str) -> CrashTestTrace:
    hr = demo.get("highlight_round") or {}
    rejected = hr.get("first_rejected_text", "")
    correction = hr.get("correction_text", "")
    fragment = _lossy_fragment(demo, rejected)
    draft_count = len(hr.get("draft_tokens_text") or hr.get("draft_tokens") or [1, 2, 3, 4])
    return CrashTestTrace(
        prompt_id=str(demo.get("prompt_id", "?")),
        model_name=str(demo.get("model_name", "Qwen/Qwen2.5-0.5B")),
        v10_suite=str(demo.get("v10_suite", "")),
        compressor_name=str(demo.get("compressor_name", "")),
        draft_len=int(demo.get("draft_len", 4)),
        prompt=str(demo.get("prompt", "")),
        prompt_label=_prompt_label(demo),
        full_output_text=str(demo.get("full_output_text", "")),
        lossy_output_text=str(demo.get("lossy_output_text", "")),
        exactkv_output_text=str(demo.get("exactkv_output_text", "")),
        lossy_first_divergence_idx=int(demo.get("lossy_first_divergence_idx", 0)),
        exactkv_exact_match=bool(demo.get("exactkv_exact_match", True)),
        exactkv_failures=0 if demo.get("exactkv_exact_match", True) else 1,
        rejected_token_text=rejected,
        correction_token_text=correction,
        lossy_draft_fragment=str(fragment),
        drift_message=str(
            demo.get("drift_message")
            or _infer_drift_message(rejected, correction, str(demo.get("v10_suite", "")))
        ),
        trace_kind=str(demo.get("trace_kind", "semantic")),
        trace_source=source,
        draft_token_count=max(4, draft_count),
    )


def load_trace(source_json: Path | None) -> CrashTestTrace:
    if source_json and source_json.is_file():
        report = json.loads(source_json.read_text(encoding="utf-8"))
        selected = report.get("selected_demo")
        if selected:
            demo = dict(selected)
            if "034b" not in str(source_json) and report.get("experiment") == "034":
                demo.setdefault("trace_kind", "punctuation_fallback")
                demo.setdefault(
                    "drift_message",
                    "compressed KV tried to close the JSON before the units value",
                )
                demo.setdefault("prompt_label", "STRUCTURED TOOL CALL PROMPT")
            return _demo_from_dict(demo, str(source_json))

    if _DEFAULT_034B_JSON.is_file():
        report = json.loads(_DEFAULT_034B_JSON.read_text(encoding="utf-8"))
        if report.get("better_than_exp034") and report.get("selected_demo"):
            return _demo_from_dict(report["selected_demo"], str(_DEFAULT_034B_JSON))

    return _demo_from_dict(SEMANTIC_TRACE_FIXTURE, "embedded Exp 034b semantic fixture")


def _pause(seconds: float, *, no_delay: bool) -> None:
    if not no_delay:
        time.sleep(seconds)


def _emit(
    lines: list[str],
    text: str = "",
    *,
    out: TextIO,
    style: Style,
    delay: float,
    no_delay: bool,
) -> None:
    lines.append(text)
    style.print(text, file=out)
    _pause(delay, no_delay=no_delay)


def _emit_block(
    lines: list[str],
    block: list[str],
    *,
    out: TextIO,
    style: Style,
    delay: float,
    no_delay: bool,
) -> None:
    for row in block:
        _emit(lines, row, out=out, style=style, delay=delay, no_delay=no_delay)


def run_crash_test(
    trace: CrashTestTrace,
    *,
    out: TextIO | None = None,
    no_delay: bool = False,
    plain: bool = False,
    speed: str = "default",
) -> str:
    if out is None:
        out = sys.stdout
    profile = SPEED_PROFILES.get(speed, SPEED_PROFILES["default"])
    style = Style(plain=plain)
    lines: list[str] = []

    def step(text: str = "", *, delay: float | None = None) -> None:
        _emit(lines, text, out=out, style=style, delay=delay or profile["pause"], no_delay=no_delay)

    def dramatic(text: str = "") -> None:
        _emit(lines, text, out=out, style=style, delay=profile["dramatic"], no_delay=no_delay)

    # 1. Header + tagline
    _emit_block(lines, _header_box(), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")
    for tag in PUBLIC_TAGLINE.splitlines():
        step(style.bold(tag), delay=profile["section"])
    step("")

    # 2. Prompt panel
    prompt_lines = [trace.prompt_label, "", trace.prompt]
    _emit_block(lines, _panel("PROMPT", prompt_lines), out=out, style=style, delay=profile["section"] * 0.5, no_delay=no_delay)
    step("")

    # 3. Drafter panel + progress
    drafter = [
        f"DRAFTER: {trace.compressor_name} compressed KV",
        "mode: lossy draft only",
        "status: proposing tokens...",
        "",
        f"draft tokens  {_progress_bar(0, trace.draft_token_count)}  0 / {trace.draft_token_count}",
    ]
    _emit_block(lines, _panel("CACHE / DRAFTER", drafter), out=out, style=style, delay=profile["section"], no_delay=no_delay)

    # 4. Verifier panel
    verifier = [
        "VERIFIER: full FP KV",
        "status: checking draft span...",
    ]
    _emit_block(lines, _panel("VERIFIER", verifier), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")

    # 5. Token stream animation
    fragment = trace.lossy_draft_fragment
    rejected = trace.rejected_token_text
    prefix = "Lossy draft:  "
    partial = ""
    freeze_at = -1
    if rejected and rejected in fragment:
        freeze_at = fragment.index(rejected) + len(rejected)

    for i, ch in enumerate(fragment):
        partial += ch
        display = partial
        if rejected and rejected in partial:
            display = partial.replace(rejected, style.red(rejected), 1)
        step(f"{prefix}{display}", delay=profile["typing"])
        if freeze_at > 0 and len(partial) >= freeze_at:
            break

    step("")
    dramatic(style.yellow(style.bold("DRIFT DETECTED")))
    dramatic(trace.drift_message)
    step("")

    # Fill progress bar at divergence moment
    div = min(trace.lossy_first_divergence_idx + 1, trace.draft_token_count)
    step(
        f"draft tokens  {_progress_bar(div, trace.draft_token_count)}  {div} / {trace.draft_token_count}",
        delay=profile["pause"],
    )
    step("")

    # 6. Crash-test decision
    decision = [
        f"draft token      {style.red(trace.rejected_token_text):<12} REJECTED",
        f"verifier token   {style.green(trace.correction_token_text):<12} COMMITTED",
    ]
    if plain:
        decision = [
            f"draft token      {trace.rejected_token_text:<12} REJECTED",
            f"verifier token   {trace.correction_token_text:<12} COMMITTED",
        ]
    _emit_block(lines, _panel("CRASH-TEST DECISION", decision), out=out, style=style, delay=profile["dramatic"] * 0.5, no_delay=no_delay)
    step("")

    # 7. Final output comparison
    full_lane = f'FULL KV\n"{trace.full_output_text[:56]}..."' if len(trace.full_output_text) > 56 else f"FULL KV\n{trace.full_output_text}"
    lossy_lane = (
        f'LOSSY KV ONLY\n"{trace.lossy_output_text[:56]}..."'
        if len(trace.lossy_output_text) > 56
        else f"LOSSY KV ONLY\n{trace.lossy_output_text}"
    )
    exact_lane = (
        f'EXACTKV\n"{trace.exactkv_output_text[:56]}..."'
        if len(trace.exactkv_output_text) > 56
        else f"EXACTKV\n{trace.exactkv_output_text}"
    )
    _emit_block(
        lines,
        _panel("FINAL OUTPUT COMPARISON", [full_lane, "", lossy_lane, "", exact_lane]),
        out=out,
        style=style,
        delay=profile["section"] * 0.4,
        no_delay=no_delay,
    )
    step("")

    # 8. Scoreboard
    scoreboard = [
        f"ExactKV failures          {trace.exactkv_failures}",
        f"Final output match        {str(trace.exactkv_exact_match).upper()}",
        "Rejected token committed  FALSE",
        "Verifier authoritative    TRUE",
    ]
    _emit_block(lines, _panel("SCOREBOARD", scoreboard), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")

    # 9. V13 proof strip
    proof = [
        "Span grid: 600 cells, 0 failures",
        "Llama-3.1-8B: 48 cells, 0 failures",
        "SnapKV smoke: 8 cells, 0 failures",
        "Timing: slower today, no speedup claim",
        "Memory: no active VRAM savings claim",
    ]
    _emit_block(lines, _panel("V13 PROOF STRIP", proof), out=out, style=style, delay=profile["section"] * 0.35, no_delay=no_delay)
    step("")

    # 10. Closing
    for closing in CLOSING_LINES.splitlines():
        if closing:
            step(style.bold(closing), delay=profile["section"])
        else:
            step("")

    if not plain:
        step(style.dim(f"[trace: {trace.prompt_id} × {trace.compressor_name} · {trace.trace_source}]"), delay=0.05)

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ExactKV terminal-native crash-test demo (replay mode)",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=None,
        help="Load trace from Exp 034b or Exp 034 JSON report",
    )
    parser.add_argument("--no-delay", action="store_true", help="Instant output for tests")
    parser.add_argument("--plain", action="store_true", help="Disable ANSI colors")
    parser.add_argument(
        "--speed",
        choices=["default", "fast", "cinematic"],
        default="default",
        help="Pacing profile (default ~75-120s, fast ~20-30s, cinematic ~90-120s)",
    )
    args = parser.parse_args()

    trace = load_trace(args.source_json)
    run_crash_test(
        trace,
        no_delay=args.no_delay,
        plain=args.plain,
        speed=args.speed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
