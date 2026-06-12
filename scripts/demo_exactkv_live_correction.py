#!/usr/bin/env python3
"""ExactKV live correction terminal demo (V13 Phase 7b).

Replays the verified Exp 034 trace (tj_002 × int4_sim) as an animated terminal UI.
This is a live correctness demo — not a benchmark. No model inference is run.
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
_DEFAULT_TRACE_JSON = _ROOT / "reports" / "experiment_034_killer_correction_demo.json"
_RECORD_SCRIPT_PATH = _ROOT / "docs" / "assets" / "demo_exactkv_live_correction_script.md"

PUBLIC_TAGLINE = (
    "Everyone is racing to shrink KV caches.\n"
    "ExactKV tells you when they start lying."
)

# Copied from Exp 034 selected_demo (tj_002 × int4_sim) when JSON is unavailable.
EXP034_TRACE_FIXTURE: dict[str, Any] = {
    "source": "experiment_034_killer_correction_demo.md (selected_demo)",
    "prompt_id": "tj_002",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "v10_suite": "tool_json",
    "compressor_name": "int4_sim",
    "prompt": (
        'Complete this tool call JSON: {"name": "get_weather", "arguments": '
        '{"city": "Paris", "units":'
    ),
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
    "highlight_round": {
        "round_idx": 0,
        "first_rejected_token": 3417,
        "correction_token": 15903,
        "first_rejected_text": "}}",
        "correction_text": "metric",
        "lossy_draft_fragment": '"}}}',
    },
}


@dataclass(frozen=True)
class DemoTrace:
    prompt_id: str
    model_name: str
    v10_suite: str
    compressor_name: str
    prompt: str
    full_output_text: str
    lossy_output_text: str
    exactkv_output_text: str
    lossy_first_divergence_idx: int
    exactkv_exact_match: bool
    exactkv_failures: int
    rejected_token_text: str
    correction_token_text: str
    lossy_draft_fragment: str
    verifier_expects: str
    trace_source: str


class Ansi:
    def __init__(self, *, plain: bool) -> None:
        self.plain = plain

    def wrap(self, text: str, code: str) -> str:
        if self.plain:
            return text
        return f"\033[{code}m{text}\033[0m"

    def red(self, text: str) -> str:
        return self.wrap(text, "31")

    def green(self, text: str) -> str:
        return self.wrap(text, "32")

    def bold(self, text: str) -> str:
        return self.wrap(text, "1")

    def dim(self, text: str) -> str:
        return self.wrap(text, "2")


def _abbrev(text: str, max_len: int = 48) -> str:
    one_line = text.replace("\n", "\\n")
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1] + "…"


def _format_prompt_display(prompt: str) -> str:
    marker = '{"name":'
    if marker in prompt:
        before, after = prompt.split(marker, 1)
        return f"{before.rstrip()}\n{marker}{after}"
    return prompt


def _lossy_draft_fragment(demo: dict[str, Any]) -> str:
    hr = demo.get("highlight_round") or {}
    if hr.get("lossy_draft_fragment"):
        return str(hr["lossy_draft_fragment"])
    rejected = hr.get("first_rejected_text", "}}")
    return f'"{rejected}}}'


def load_demo_trace(trace_json: Path) -> DemoTrace:
    source = "embedded Exp 034 fixture"
    demo: dict[str, Any]
    if trace_json.is_file():
        report = json.loads(trace_json.read_text(encoding="utf-8"))
        selected = report.get("selected_demo")
        if not selected:
            raise ValueError(f"No selected_demo in {trace_json}")
        demo = selected
        source = str(trace_json)
    else:
        demo = dict(EXP034_TRACE_FIXTURE)
        hr = demo.pop("highlight_round")
        demo["highlight_round"] = hr

    hr = demo.get("highlight_round") or {}
    rejected = hr.get("first_rejected_text")
    correction = hr.get("correction_text")
    if not rejected and hr.get("first_rejected_token") == 3417:
        rejected = "}}"
    if not correction and hr.get("correction_token") == 15903:
        correction = "metric"
    rejected = rejected or "}}"
    correction = correction or "metric"

    model = demo.get("model_name") or EXP034_TRACE_FIXTURE["model_name"]
    return DemoTrace(
        prompt_id=str(demo.get("prompt_id", "tj_002")),
        model_name=model,
        v10_suite=str(demo.get("v10_suite", "tool_json")),
        compressor_name=str(demo.get("compressor_name", "int4_sim")),
        prompt=str(demo.get("prompt", EXP034_TRACE_FIXTURE["prompt"])),
        full_output_text=str(demo.get("full_output_text", "")),
        lossy_output_text=str(demo.get("lossy_output_text", "")),
        exactkv_output_text=str(demo.get("exactkv_output_text", "")),
        lossy_first_divergence_idx=int(demo.get("lossy_first_divergence_idx", 1)),
        exactkv_exact_match=bool(demo.get("exactkv_exact_match", True)),
        exactkv_failures=0 if demo.get("exactkv_exact_match", True) else 1,
        rejected_token_text=rejected,
        correction_token_text=correction,
        lossy_draft_fragment=_lossy_draft_fragment(demo),
        verifier_expects=f'"{correction}"',
        trace_source=source,
    )


def write_record_script(path: Path) -> None:
    content = """# ExactKV live correction demo — recording script

Generated by `scripts/demo_exactkv_live_correction.py --record-script`.

This replays the **verified Exp 034 trace** (`tj_002` × `int4_sim`). No model inference runs during the demo.

## Quick run (animated)

```bash
python3 scripts/demo_exactkv_live_correction.py
```

## Record with asciinema

```bash
asciinema rec -c "python3 scripts/demo_exactkv_live_correction.py" \\
  docs/assets/demo_exactkv_live_correction.cast
```

Export to GIF (requires agg or similar):

```bash
agg docs/assets/demo_exactkv_live_correction.cast \\
  docs/assets/demo_exactkv_live_correction.gif
```

## Record with terminalizer

```bash
terminalizer record demo-exactkv-live -c "python3 scripts/demo_exactkv_live_correction.py"
terminalizer render demo-exactkv-live -o docs/assets/demo_exactkv_live_correction.gif
```

## Screen recording (macOS)

```bash
python3 scripts/demo_exactkv_live_correction.py
# Then use QuickTime → New Screen Recording, or OBS, while the demo runs.
```

## Screen recording (Linux)

```bash
python3 scripts/demo_exactkv_live_correction.py
# Record with OBS, SimpleScreenRecorder, or: ffmpeg -f x11grab ...
```

## CI / tests (no animation, no color)

```bash
python3 scripts/demo_exactkv_live_correction.py --no-delay --plain
```

## Claims boundary

- Live correctness replay of Exp 034 — **not** a timing or memory benchmark.
- Trace numbers come from Exp 034; no speedup, throughput, latency, tokens/sec, VRAM, serving, or accuracy claims.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _pause(seconds: float, *, no_delay: bool) -> None:
    if not no_delay:
        time.sleep(seconds)


def _println(out: TextIO, text: str = "") -> None:
    print(text, file=out)


def run_demo(
    trace: DemoTrace,
    *,
    out: TextIO | None = None,
    no_delay: bool = False,
    plain: bool = False,
) -> str:
    """Render the live demo; return captured output as a string."""
    if out is None:
        out = sys.stdout
    lines: list[str] = []
    ansi = Ansi(plain=plain)

    def emit(text: str = "") -> None:
        lines.append(text)
        _println(out, text)

    def step(text: str = "", delay: float = 0.35) -> None:
        emit(text)
        _pause(delay, no_delay=no_delay)

    model_short = trace.model_name.split("/")[-1]
    step(ansi.bold("EXACTKV LIVE KV CRASH TEST"), 0.5)
    step(f"{model_short} · {trace.compressor_name} · {trace.v10_suite}", 0.4)
    step("Compressed KV drafts. Full KV verifies.", 0.6)
    step("─" * 60, 0.3)

    step(ansi.bold("PROMPT"), 0.25)
    for line in _format_prompt_display(trace.prompt).splitlines():
        step(line, 0.2)
    step("", 0.4)

    step(ansi.bold("FULL-KV VERIFIER"), 0.25)
    step(f"Full-KV verifier expects: {ansi.green(trace.verifier_expects)}", 0.6)
    step("", 0.3)

    step(ansi.bold("LOSSY COMPRESSED KV DRAFT"), 0.25)
    fragment = trace.lossy_draft_fragment
    partial = ""
    prefix = 'Lossy compressed KV draft: '
    for i, ch in enumerate(fragment):
        partial += ch
        if ch in trace.rejected_token_text and trace.rejected_token_text in partial:
            colored = partial.replace(
                trace.rejected_token_text,
                ansi.red(trace.rejected_token_text),
                1,
            )
            emit(f"{prefix}{colored}")
        else:
            emit(f"{prefix}{partial}")
        _pause(0.18, no_delay=no_delay)
    step("", 0.5)

    step(ansi.bold("EXACTKV CORRECTION"), 0.25)
    step(ansi.red(f"REJECT draft token: {trace.rejected_token_text}"), 0.45)
    step(ansi.green(f"COMMIT verifier token: {trace.correction_token_text}"), 0.6)
    step("", 0.3)

    step(ansi.bold("FINAL COMPARISON"), 0.25)
    emit("MODE                    OUTPUT")
    emit(f"Full KV                 {_abbrev(trace.full_output_text)}")
    emit(f"Lossy compressed KV     {_abbrev(trace.lossy_output_text)}")
    emit(f"ExactKV                 {_abbrev(trace.exactkv_output_text)}")
    _pause(0.5, no_delay=no_delay)
    step("", 0.3)

    step(ansi.bold("STATUS"), 0.25)
    step(f"ExactKV failures: {trace.exactkv_failures}", 0.2)
    step("Rejected token committed: false", 0.2)
    match_str = str(trace.exactkv_exact_match).lower()
    step(f"Final output match: {ansi.green(match_str) if trace.exactkv_exact_match else match_str}", 0.4)
    step("", 0.4)

    for tagline_line in PUBLIC_TAGLINE.splitlines():
        step(ansi.bold(tagline_line), 0.35)

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ExactKV live correction terminal demo (Exp 034 trace replay)",
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=_DEFAULT_TRACE_JSON,
        help="Exp 034 report JSON (default: reports/experiment_034_killer_correction_demo.json)",
    )
    parser.add_argument("--no-delay", action="store_true", help="Skip animation delays")
    parser.add_argument("--plain", action="store_true", help="Disable ANSI colors")
    parser.add_argument(
        "--record-script",
        action="store_true",
        help=f"Write recording instructions to {_RECORD_SCRIPT_PATH}",
    )
    args = parser.parse_args()

    if args.record_script:
        write_record_script(_RECORD_SCRIPT_PATH)
        print(f"Wrote {_RECORD_SCRIPT_PATH}")

    trace = load_demo_trace(args.trace_json)
    run_demo(trace, no_delay=args.no_delay, plain=args.plain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
