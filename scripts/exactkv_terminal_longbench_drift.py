#!/usr/bin/env python3
"""ExactKV terminal-native LongBench-style drift demo (V13 Phase 10A / Exp 037).

Replays a verified score-preserving drift trace where task heuristics stay green
but compressed KV changed the model's token path. No model inference in replay mode.
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
_DEFAULT_037_JSON = _ROOT / "reports" / "experiment_037_longbench_style_drift_candidates.json"

COLD_OPEN = (
    "The answer can still score.\n"
    "The compressed cache can still drift."
)

CLOSING_LINES = (
    "Outcome benchmarks ask whether the answer scored well.\n"
    "ExactKV asks whether compression changed the model's behavior.\n"
    "\n"
    "KV compression should not be trusted.\n"
    "It should be crash-tested."
)

# Verified Exp 037 winner: lb_md_001 × int4_sim (billing → answer).
LONGBENCH_TRACE_FIXTURE: dict[str, Any] = {
    "source": "experiment_037_longbench_style_drift_search (lb_md_001)",
    "trace_kind": "longbench_style_score_preserving",
    "prompt_id": "lb_md_001",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "v10_suite": "longbench_style_v1",
    "prompt_category": "multi_doc_qa",
    "compressor_name": "int4_sim",
    "draft_len": 4,
    "prompt": (
        "Context document 1: Friday follow-up on SSO is assigned to Maya.\n"
        "Context document 2: Billing migration checkpoint remains open.\n"
        "Context document 3: Launch communications go to Priya.\n"
        "\n"
        "Use the context documents to answer exactly who owns the Friday follow-up.\n"
    ),
    "prompt_label": "LONGBENCH-STYLE MULTI-DOC QA",
    "full_output_text": "The answer is: Maya",
    "lossy_output_text": (
        "The billing migration checkpoint is assigned to Maya.\n"
        "The billing migration checkpoint is assigned to Maya."
    ),
    "exactkv_output_text": "The answer is: Maya",
    "lossy_first_divergence_idx": 1,
    "exactkv_exact_match": True,
    "exactkv_failures": 0,
    "task_heuristic_summary": "full=pass lossy=pass exactkv=pass (reference: Maya)",
    "drift_message": "compressed KV opened with billing context instead of answer",
    "highlight_round": {
        "round_idx": 0,
        "first_rejected_token": 0,
        "correction_token": 0,
        "first_rejected_text": "billing",
        "correction_text": "answer",
        "lossy_draft_fragment": "The billing",
        "draft_tokens_text": ["The", " billing", " migration", " checkpoint"],
    },
}

SPEED_PROFILES = {
    "default": {"pause": 0.55, "typing": 0.12, "dramatic": 1.4, "section": 0.35},
    "fast": {"pause": 0.12, "typing": 0.04, "dramatic": 0.35, "section": 0.08},
    "cinematic": {"pause": 0.85, "typing": 0.16, "dramatic": 2.2, "section": 0.55},
}


@dataclass(frozen=True)
class LongBenchDriftTrace:
    prompt_id: str
    model_name: str
    v10_suite: str
    prompt_category: str
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
    task_heuristic_summary: str
    trace_source: str


class Style:
    def __init__(self, *, plain: bool) -> None:
        self.plain = plain

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


def _panel(title: str, lines: list[str], width: int = 64) -> list[str]:
    inner = max(4, width - 4)
    out = [f"┌─ {title} " + "─" * max(0, inner - len(title) - 3) + "┐"]
    for line in lines:
        if len(line) > inner:
            line = line[: inner - 1] + "…"
        out.append("│ " + line.ljust(inner) + " │")
    out.append("└" + "─" * inner + "┘")
    return out


def _header_box() -> list[str]:
    return [
        "╔════════════════════════════════════════════════════════════════╗",
        "║ EXACTKV LONGBENCH-STYLE DRIFT DEMO                            ║",
        "║ Outcome green. Token path changed.                            ║",
        "╚════════════════════════════════════════════════════════════════╝",
    ]


def _lossy_fragment(demo: dict[str, Any], rejected: str) -> str:
    hr = demo.get("highlight_round") or {}
    if hr.get("lossy_draft_fragment"):
        return str(hr["lossy_draft_fragment"])
    lossy = str(demo.get("lossy_output_text", ""))
    if rejected and rejected in lossy:
        return lossy[: lossy.index(rejected) + len(rejected)]
    texts = hr.get("draft_tokens_text") or []
    if texts:
        return "".join(texts[:3])
    return lossy[: min(24, len(lossy))]


def _infer_drift_message(rejected: str, correction: str, category: str) -> str:
    rej, corr = rejected.strip().lower(), correction.strip().lower()
    if "billing" in rej and "answer" in corr:
        return "compressed KV opened with billing context instead of answer"
    if "priya" in rej and "maya" in corr:
        return "compressed KV named Priya instead of Maya"
    if category.endswith("_qa"):
        return "compressed KV changed the answer opening while keeping a compatible reference"
    return "compressed KV drifted from full-KV greedy output"


def _display_prompt(demo: dict[str, Any]) -> str:
    raw = str(demo.get("prompt", ""))
    for marker in (
        "Context document 1:",
        "Summarize this",
        "According to the policy",
        "Support policy excerpt:",
        "Weekly operations log:",
        "Long-context retrieval task:",
    ):
        if marker in raw:
            return raw[raw.index(marker) :].strip()
    if len(raw) > 400:
        return raw[-400:].strip()
    return raw.strip()


def _trace_from_dict(demo: dict[str, Any], source: str) -> LongBenchDriftTrace:
    hr = demo.get("highlight_round") or {}
    rejected = str(hr.get("first_rejected_text", "")).strip()
    correction = str(hr.get("correction_text", "")).strip()
    th = demo.get("task_heuristic") or {}
    summary = demo.get("task_heuristic_summary")
    if not summary and th:
        summary = (
            f"full={'pass' if th.get('full', {}).get('pass') else 'fail'} "
            f"lossy={'pass' if th.get('lossy', {}).get('pass') else 'fail'} "
            f"exactkv={'pass' if th.get('exactkv', {}).get('pass') else 'fail'}"
        )
    return LongBenchDriftTrace(
        prompt_id=str(demo.get("prompt_id", "?")),
        model_name=str(demo.get("model_name", "Qwen/Qwen2.5-0.5B")),
        v10_suite=str(demo.get("v10_suite", "")),
        prompt_category=str(demo.get("prompt_category", demo.get("category", ""))),
        compressor_name=str(demo.get("compressor_name", "")),
        draft_len=int(demo.get("draft_len", 4)),
        prompt=_display_prompt(demo),
        prompt_label=str(demo.get("prompt_label", "LONGBENCH-STYLE TASK")),
        full_output_text=str(demo.get("full_output_text", "")),
        lossy_output_text=str(demo.get("lossy_output_text", "")),
        exactkv_output_text=str(demo.get("exactkv_output_text", "")),
        lossy_first_divergence_idx=int(demo.get("lossy_first_divergence_idx", 0)),
        exactkv_exact_match=bool(demo.get("exactkv_exact_match", True)),
        exactkv_failures=0 if demo.get("exactkv_exact_match", True) else 1,
        rejected_token_text=rejected,
        correction_token_text=correction,
        lossy_draft_fragment=_lossy_fragment(demo, rejected),
        drift_message=str(
            demo.get("drift_message")
            or _infer_drift_message(rejected, correction, str(demo.get("prompt_category", "")))
        ),
        task_heuristic_summary=str(summary or "heuristic pass on all lanes"),
        trace_source=source,
    )


def load_trace(source_json: Path | None) -> LongBenchDriftTrace:
    if source_json and source_json.is_file():
        report = json.loads(source_json.read_text(encoding="utf-8"))
        selected = report.get("selected_demo")
        if selected and selected.get("candidate_score", 0) >= 200:
            return _trace_from_dict(selected, str(source_json))

    if _DEFAULT_037_JSON.is_file():
        report = json.loads(_DEFAULT_037_JSON.read_text(encoding="utf-8"))
        selected = report.get("selected_demo")
        if selected:
            return _trace_from_dict(selected, str(_DEFAULT_037_JSON))
        for cand in report.get("top_candidates") or []:
            if cand.get("candidate_score", 0) >= 200:
                return _trace_from_dict(cand, str(_DEFAULT_037_JSON))

    return _trace_from_dict(LONGBENCH_TRACE_FIXTURE, "embedded Exp 037 fixture")


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


def _truncate_path(text: str, limit: int = 72) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1] + "…"


def run_longbench_drift_demo(
    trace: LongBenchDriftTrace,
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

    _emit_block(lines, _header_box(), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")
    for line in COLD_OPEN.splitlines():
        step(style.bold(line), delay=profile["section"])
    step("")

    task_lines = [trace.prompt_label, f"category: {trace.prompt_category}", "", trace.prompt.strip()]
    _emit_block(lines, _panel("TASK", task_lines), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")

    outcome = [
        "Outcome panel (transparent heuristic, not official LongBench):",
        "Full KV:      acceptable",
        "Lossy KV:     acceptable",
        "ExactKV:      acceptable",
        f"detail: {trace.task_heuristic_summary}",
    ]
    _emit_block(lines, _panel("OUTCOME", outcome), out=out, style=style, delay=profile["dramatic"] * 0.45, no_delay=no_delay)
    step("")

    behavior = [
        f"Full KV path:  {_truncate_path(trace.full_output_text)}",
        f"Lossy KV path: {_truncate_path(trace.lossy_output_text)}",
    ]
    _emit_block(lines, _panel("BEHAVIOR (token path = exact words generated)", behavior), out=out, style=style, delay=profile["section"], no_delay=no_delay)
    step("")

    dramatic(style.yellow(style.bold("DRIFT DETECTED")))
    dramatic("Outcome stayed green.")
    dramatic("The exact words changed.")
    dramatic(trace.drift_message)
    step("")

    fragment = trace.lossy_draft_fragment
    rejected = trace.rejected_token_text
    prefix = "Lossy draft:  "
    partial = ""
    freeze_at = -1
    if rejected and rejected in fragment:
        freeze_at = fragment.index(rejected) + len(rejected)

    for ch in fragment:
        partial += ch
        display = partial
        if rejected and rejected in partial:
            display = partial.replace(rejected, style.red(rejected), 1)
        step(f"{prefix}{display}", delay=profile["typing"])
        if freeze_at > 0 and len(partial) >= freeze_at:
            break

    step("")
    decision = [
        "reject lossy draft",
        f"  rejected token: {trace.rejected_token_text!r}",
        "commit full-KV correction",
        f"  verifier token: {trace.correction_token_text!r}",
    ]
    _emit_block(lines, _panel("EXACTKV DECISION", decision), out=out, style=style, delay=profile["dramatic"] * 0.5, no_delay=no_delay)
    step("")

    scoreboard = [
        "Task score changed:              no",
        "Compressed KV behavior changed:  yes",
        f"ExactKV failures:                {trace.exactkv_failures}",
        f"Final output match:              {str(trace.exactkv_exact_match).lower()}",
        "Rejected token committed:        false",
    ]
    _emit_block(lines, _panel("FINAL SCOREBOARD", scoreboard), out=out, style=style, delay=profile["dramatic"] * 0.45, no_delay=no_delay)
    step("")

    for closing in CLOSING_LINES.splitlines():
        if closing:
            step(style.bold(closing), delay=profile["section"])
        else:
            step("")

    if not plain:
        step(
            style.dim(f"[trace: {trace.prompt_id} × {trace.compressor_name} · {trace.trace_source}]"),
            delay=0.05,
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="ExactKV LongBench-style drift terminal demo")
    parser.add_argument("--source-json", type=Path, default=None)
    parser.add_argument("--no-delay", action="store_true")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--speed", choices=["default", "fast", "cinematic"], default="default")
    args = parser.parse_args()
    trace = load_trace(args.source_json)
    run_longbench_drift_demo(trace, no_delay=args.no_delay, plain=args.plain, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
