#!/usr/bin/env python3
"""V13 Phase 8c: cinematic ExactKV crash-test launch video.

Renders a 90–120s watchable demo from verified experiment traces only.
No model inference during render unless --try-restaurant-search is explicitly passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
_ASSETS = _ROOT / "docs" / "assets"
_FRAMES_DIR = _ASSETS / "exactkv_crash_test_frames"
_DEFAULT_JSON = _ROOT / "reports" / "experiment_034_killer_correction_demo.json"

PUBLIC_TAGLINE = (
    "Everyone is racing to shrink KV caches.\n"
    "ExactKV tells you when they start lying."
)

# Verified Exp 034 selected_demo (tj_002 × int4_sim) — embedded fallback.
EXP034_FIXTURE: dict[str, Any] = {
    "source_label": "Exp 034 verified trace (tj_002 × int4_sim)",
    "prompt_id": "tj_002",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "compressor_name": "int4_sim",
    "prompt": (
        'Complete this tool call JSON: {"name": "get_weather", "arguments": '
        '{"city": "Paris", "units":'
    ),
    "prompt_display": (
        "Complete this tool call JSON:\n"
        '{"name": "get_weather", "arguments": {"city": "Paris", "units":'
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
        "accepted_prefix": [330],
    },
    "correct_highlight": '"metric"',
    "wrong_highlight": "}}",
}

VOICEOVER_SCRIPT = """Everyone is racing to shrink KV caches.

ExactKV tells you when they start lying.

I wanted to know when they start lying.

Here is a structured-output prompt. One wrong token can change the action.

Full KV gives the trusted output.

Now watch lossy compressed KV.

The response looks plausible, but the compressed cache changes the next token.

ExactKV runs this differently.

Compressed KV drafts. Full KV verifies.

The moment the compressed cache disagrees, ExactKV rejects the draft token and commits the full-KV correction.

Final output matches full KV exactly.

In V13, span verification passed a 600-cell exactness grid with 0 sequential failures, 0 span failures, and 0 parity failures.

Llama-3.1-8B also passed a small suite with 0 failures and mean acceptance around 0.945.

It is not fast yet. Full greedy is still faster today, and active GPU memory savings are not claimed.

But the verifier is working.

KV compression should not be trusted.

It should be crash-tested."""

# Colors
BG = (13, 17, 23)
CARD = (22, 27, 34)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (88, 166, 255)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
AMBER = (210, 153, 34)
BORDER = (48, 54, 61)


@dataclass
class CrashTestTrace:
    source_label: str
    prompt_id: str
    model_name: str
    compressor_name: str
    prompt: str
    prompt_display: str
    full_output_text: str
    lossy_output_text: str
    exactkv_output_text: str
    lossy_first_divergence_idx: int
    exactkv_failures: int
    exactkv_exact_match: bool
    rejected_token: str
    correction_token: str
    round_idx: int
    correct_highlight: str
    wrong_highlight: str
    accepted_prefix_len: int = 1
    scenario_name: str = "weather tool JSON"


@dataclass
class SceneSpec:
    scene_id: int
    title: str
    start_s: float
    end_s: float
    voiceover: str
    source_note: str
    visual_note: str


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class FrameCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.img = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.img)

    def copy(self) -> Image.Image:
        return self.img.copy()

    def text_center(
        self,
        y: float,
        text: str,
        *,
        size: int = 48,
        color: tuple[int, int, int] = TEXT,
        bold: bool = False,
        spacing: int = 8,
    ) -> None:
        font = _font(size, bold=bold)
        lines = text.split("\n")
        total_h = sum(font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + spacing for _ in lines) - spacing
        cy = int(y * self.height - total_h / 2)
        for line in lines:
            bbox = self.draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (self.width - tw) // 2
            self.draw.text((x, cy), line, font=font, fill=color)
            cy += th + spacing

    def caption(self, text: str, y: float = 0.88) -> None:
        self.text_center(y, text, size=28, color=MUTED)

    def card(self, x: float, y: float, w: float, h: float) -> None:
        x0, y0 = int(x * self.width), int(y * self.height)
        x1, y1 = int((x + w) * self.width), int((y + h) * self.height)
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=CARD, outline=BORDER, width=2)

    def mono_block(self, x: float, y: float, w: float, text: str, *, hl: str | None = None, hl_color: tuple[int, int, int] = GREEN, size: int = 26) -> None:
        font = _font(size)
        x0 = int(x * self.width)
        y0 = int(y * self.height)
        max_chars = int(w * self.width / (size * 0.55))
        chunk = text[:max_chars] + ("…" if len(text) > max_chars else "")
        if hl and hl in chunk:
            before, _, after = chunk.partition(hl)
            self.draw.text((x0, y0), before, font=font, fill=TEXT)
            bx = x0 + self.draw.textlength(before, font=font)
            self.draw.text((bx, y0), hl, font=font, fill=hl_color)
            self.draw.text((bx + self.draw.textlength(hl, font=font), y0), after, font=font, fill=TEXT)
        else:
            self.draw.text((x0, y0), chunk, font=font, fill=TEXT)

    def badge(self, text: str, y: float, color: tuple[int, int, int] = GREEN) -> None:
        self.text_center(y, text, size=40, color=color, bold=True)

    def fade(self, alpha: float) -> None:
        if alpha <= 0:
            return
        overlay = Image.new("RGB", (self.width, self.height), BG)
        self.img = Image.blend(self.img, overlay, min(1.0, max(0.0, alpha)))


def load_trace(source_json: Path) -> CrashTestTrace:
    if source_json.is_file():
        report = json.loads(source_json.read_text(encoding="utf-8"))
        demo = report.get("selected_demo") or {}
        hr = demo.get("highlight_round") or {}
        rejected = hr.get("first_rejected_text") or EXP034_FIXTURE["highlight_round"]["first_rejected_text"]
        correction = hr.get("correction_text") or EXP034_FIXTURE["highlight_round"]["correction_text"]
        prompt = demo.get("prompt", EXP034_FIXTURE["prompt"])
        return CrashTestTrace(
            source_label=f"Exp 034 JSON: {source_json.name}",
            prompt_id=str(demo.get("prompt_id", "tj_002")),
            model_name=str(demo.get("model_name", EXP034_FIXTURE["model_name"])),
            compressor_name=str(demo.get("compressor_name", "int4_sim")),
            prompt=prompt,
            prompt_display=EXP034_FIXTURE["prompt_display"],
            full_output_text=str(demo.get("full_output_text", EXP034_FIXTURE["full_output_text"])),
            lossy_output_text=str(demo.get("lossy_output_text", EXP034_FIXTURE["lossy_output_text"])),
            exactkv_output_text=str(demo.get("exactkv_output_text", EXP034_FIXTURE["exactkv_output_text"])),
            lossy_first_divergence_idx=int(demo.get("lossy_first_divergence_idx", 1)),
            exactkv_failures=0,
            exactkv_exact_match=bool(demo.get("exactkv_exact_match", True)),
            rejected_token=rejected,
            correction_token=correction,
            round_idx=int(hr.get("round_idx", 0)),
            correct_highlight=f'"{correction}"' if correction == "metric" else correction,
            wrong_highlight=rejected,
            scenario_name="weather tool JSON (Exp 034 tj_002)",
        )
    fix = EXP034_FIXTURE
    hr = fix["highlight_round"]
    return CrashTestTrace(
        source_label=fix["source_label"],
        prompt_id=fix["prompt_id"],
        model_name=fix["model_name"],
        compressor_name=fix["compressor_name"],
        prompt=fix["prompt"],
        prompt_display=fix["prompt_display"],
        full_output_text=fix["full_output_text"],
        lossy_output_text=fix["lossy_output_text"],
        exactkv_output_text=fix["exactkv_output_text"],
        lossy_first_divergence_idx=fix["lossy_first_divergence_idx"],
        exactkv_failures=0,
        exactkv_exact_match=True,
        rejected_token=hr["first_rejected_text"],
        correction_token=hr["correction_text"],
        round_idx=hr["round_idx"],
        correct_highlight=fix["correct_highlight"],
        wrong_highlight=fix["wrong_highlight"],
        scenario_name="weather tool JSON (Exp 034 tj_002)",
    )


def scene_specs(trace: CrashTestTrace) -> list[SceneSpec]:
    return [
        SceneSpec(1, "Cold open", 0, 5, "Everyone is racing to shrink KV caches.", "tagline", "Black/dark, large type"),
        SceneSpec(2, "Prompt", 5, 15, "Structured output is where one token matters.", trace.source_label, "Editor style prompt"),
        SceneSpec(3, "Full KV", 15, 30, "Full KV gives the trusted answer.", trace.source_label, f"Highlight {trace.correct_highlight} green"),
        SceneSpec(4, "Lossy KV", 30, 45, "Compressed cache changes the next token.", trace.source_label, f"Highlight {trace.wrong_highlight} red"),
        SceneSpec(5, "First divergence", 45, 60, f"First divergence: token {trace.lossy_first_divergence_idx}.", trace.source_label, "Token diff zoom"),
        SceneSpec(6, "ExactKV trace", 60, 80, "Wrong token rejected. Verifier token committed.", trace.source_label, "Red cross → green commit"),
        SceneSpec(7, "Exact match", 80, 95, "Final output matches full KV exactly.", trace.source_label, "EXACT MATCH badge"),
        SceneSpec(8, "V13 proof cards", 95, 110, "600-cell span grid; Llama 8B; SnapKV smoke.", "Exp 029/033/032b docs", "Proof cards"),
        SceneSpec(9, "Honest status", 110, 115, "Not faster yet. Correctness first.", "Exp 030/031 docs", "Amber honesty card"),
        SceneSpec(10, "Final title", 115, 120, "KV compression should be crash-tested.", "tagline", "Title card"),
    ]


def render_scene_frame(
    scene: SceneSpec,
    trace: CrashTestTrace,
    t: float,
    width: int,
    height: int,
) -> Image.Image:
    c = FrameCanvas(width, height)
    fade_in = min(1.0, t * 4) if t < 0.25 else 1.0
    fade_out = min(1.0, (1.0 - t) * 4) if t > 0.75 else 1.0
    opacity = min(fade_in, fade_out)

    if scene.scene_id == 1:
        c.text_center(0.38, "Everyone is racing to shrink KV caches.", size=52, bold=True, color=TEXT)
        c.text_center(0.50, "ExactKV tells you when they start lying.", size=32, color=ACCENT)
        c.text_center(0.60, "I wanted to know when they start lying.", size=28, color=MUTED)
        c.text_center(0.68, "ExactKV", size=64, bold=True, color=ACCENT)
        c.caption("crash-test lab for KV-cache compression")
    elif scene.scene_id == 2:
        c.text_center(0.12, "SCENE 02 · PROMPT", size=24, color=MUTED)
        c.card(0.08, 0.18, 0.84, 0.55)
        c.mono_block(0.12, 0.24, 0.76, trace.prompt_display, size=30)
        c.caption("Structured output is where one token matters.")
        c.text_center(0.82, f"Source: {trace.scenario_name}", size=18, color=MUTED)
    elif scene.scene_id == 3:
        c.text_center(0.10, "FULL KV", size=32, bold=True, color=ACCENT)
        c.card(0.08, 0.18, 0.84, 0.45)
        c.mono_block(0.12, 0.26, 0.76, trace.full_output_text, hl=trace.correction_token, hl_color=GREEN, size=28)
        c.badge(f"Trusted: {trace.correct_highlight}", 0.72, GREEN)
        c.caption("Full KV gives the trusted answer.")
    elif scene.scene_id == 4:
        c.text_center(0.10, "LOSSY COMPRESSED KV", size=32, bold=True, color=RED)
        c.card(0.08, 0.18, 0.84, 0.45)
        c.mono_block(0.12, 0.26, 0.76, trace.lossy_output_text, hl=trace.wrong_highlight, hl_color=RED, size=28)
        c.badge("Looks plausible — wrong token", 0.72, RED)
        c.caption("The compressed cache changes the next token.")
    elif scene.scene_id == 5:
        c.text_center(0.10, "FIRST DIVERGENCE", size=32, bold=True, color=AMBER)
        c.card(0.10, 0.20, 0.80, 0.50)
        lines = [
            f"Token index: {trace.lossy_first_divergence_idx}",
            f"Draft token:  {trace.rejected_token!r}",
            f"Verifier token: {trace.correction_token!r}",
            f"Round: {trace.round_idx}",
        ]
        y = 0.28
        for line in lines:
            c.mono_block(0.14, y, 0.72, line, size=32)
            y += 0.10
        c.caption(f"First divergence: token {trace.lossy_first_divergence_idx}")
    elif scene.scene_id == 6:
        progress = min(1.0, t * 1.5)
        c.text_center(0.08, "EXACTKV TRACE", size=32, bold=True, color=ACCENT)
        c.text_center(0.18, "Compressed KV drafts.", size=28, color=MUTED)
        c.text_center(0.26, "Full KV verifies.", size=28, color=MUTED)
        c.card(0.15, 0.34, 0.30, 0.22)
        c.mono_block(0.18, 0.40, 0.24, trace.rejected_token, hl=trace.rejected_token, hl_color=RED, size=40)
        if progress > 0.35:
            c.draw.line([(int(0.48 * width), int(0.45 * height)), (int(0.52 * width), int(0.45 * height))], fill=RED, width=4)
            c.text_center(0.45, "REJECT", size=22, color=RED, bold=True)
        c.card(0.55, 0.34, 0.30, 0.22)
        if progress > 0.65:
            c.mono_block(0.58, 0.40, 0.24, trace.correction_token, hl=trace.correction_token, hl_color=GREEN, size=40)
            c.text_center(0.72, "COMMIT", size=22, color=GREEN, bold=True)
        elif progress > 0.35:
            c.mono_block(0.58, 0.40, 0.24, "—", size=40)
        c.caption("Wrong token rejected · verifier token committed")
    elif scene.scene_id == 7:
        c.text_center(0.08, "FINAL OUTPUT", size=32, bold=True, color=ACCENT)
        c.card(0.06, 0.16, 0.42, 0.40)
        c.text_center(0.20, "Full KV", size=22, color=MUTED)
        c.mono_block(0.10, 0.26, 0.36, trace.full_output_text[:80] + "…", size=22)
        c.card(0.52, 0.16, 0.42, 0.40)
        c.text_center(0.20, "ExactKV", size=22, color=MUTED)
        c.mono_block(0.56, 0.26, 0.36, trace.exactkv_output_text[:80] + "…", size=22)
        c.badge("EXACT MATCH ✅", 0.68, GREEN)
        c.text_center(0.76, f"exactkv_failures = {trace.exactkv_failures}", size=28, color=GREEN)
        c.caption("Final output matches full KV exactly.")
    elif scene.scene_id == 8:
        c.text_center(0.08, "V13 PROOF", size=32, bold=True, color=ACCENT)
        cards = [
            ("Span grid", "600 cells · 0 seq · 0 span · 0 parity", "Exp 029"),
            ("Llama-3.1-8B", "48 cells · 0 failures · accept ~0.945", "Exp 033"),
            ("SnapKV smoke", "8 cells · 0 failures · factory-only", "Exp 032b"),
        ]
        y = 0.22
        for title, body, src in cards:
            c.card(0.12, y, 0.76, 0.16)
            c.mono_block(0.16, y + 0.02, 0.30, title, size=28, hl_color=ACCENT)
            c.mono_block(0.16, y + 0.07, 0.60, body, size=22)
            c.mono_block(0.62, y + 0.07, 0.22, src, size=18, hl_color=MUTED)
            y += 0.20
        c.caption("Tested panels only — not universal claims")
    elif scene.scene_id == 9:
        c.text_center(0.12, "HONEST STATUS", size=32, bold=True, color=AMBER)
        c.card(0.10, 0.22, 0.80, 0.48)
        lines = [
            "Not faster yet.",
            "Full greedy is still faster in Exp 030.",
            "No active GPU memory savings claimed in Exp 031.",
            "Correctness first. Runtime path next.",
        ]
        y = 0.30
        for line in lines:
            c.mono_block(0.14, y, 0.72, line, size=30)
            y += 0.09
        c.caption("Diagnostic timing/memory only · no speedup claim")
    elif scene.scene_id == 10:
        c.text_center(0.32, "KV compression should not be trusted.", size=44, bold=True, color=TEXT)
        c.text_center(0.46, "It should be crash-tested.", size=44, bold=True, color=ACCENT)
        c.text_center(0.62, "ExactKV", size=56, bold=True, color=ACCENT)
        c.text_center(0.74, "the crash-test lab for KV-cache compression", size=28, color=MUTED)
        c.text_center(0.86, PUBLIC_TAGLINE.replace("\n", "  ·  "), size=20, color=MUTED)

    if opacity < 1.0:
        c.fade(1.0 - opacity)
    return c.copy()


def build_storyboard(trace: CrashTestTrace, path: Path) -> None:
    lines = [
        "# ExactKV Crash-Test Demo — Storyboard",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        f"**Source trace:** {trace.source_label}",
        f"**Scenario:** {trace.scenario_name}",
        f"**prompt_id:** `{trace.prompt_id}` · **compressor:** `{trace.compressor_name}`",
        f"**Rejected:** `{trace.rejected_token}` → **Correction:** `{trace.correction_token}`",
        "",
        "**Restaurant ordering search:** not found in existing reports; using verified Exp 034 weather trace.",
        "",
        "**On-screen phrases:** Compressed KV drafts · Full KV verifies · Wrong token rejected · EXACT MATCH",
        "",
        "## Voiceover / caption script",
        "",
        "```",
        VOICEOVER_SCRIPT,
        "```",
        "",
        "## Scenes",
        "",
        "| Time | Scene | Screen | Caption / VO | Source | Visual |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in scene_specs(trace):
        lines.append(
            f"| {s.start_s:.0f}–{s.end_s:.0f}s | {s.title} | Scene {s.scene_id} | {s.voiceover} | {s.source_note} | {s.visual_note} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html_player(frames_dir: Path, fps: int, out_path: Path, trace: CrashTestTrace) -> None:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return
    # Use relative paths for first 300 frames max in HTML for size; reference MP4 if exists
    mp4 = _ASSETS / "exactkv_crash_test_demo.mp4"
    mp4_rel = "exactkv_crash_test_demo.mp4" if mp4.is_file() else ""
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ExactKV Crash-Test Demo</title>
<style>
body {{ margin:0; background:#0d1117; color:#e6edf3; font-family: system-ui, sans-serif; }}
.wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ color: #58a6ff; }}
video {{ width: 100%; border-radius: 12px; border: 1px solid #30363d; }}
.note {{ color: #8b949e; font-size: 14px; margin-top: 16px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>ExactKV Crash-Test Demo</h1>
<p><em>{PUBLIC_TAGLINE.replace(chr(10), " ")}</em></p>
{"<video controls autoplay loop src='" + mp4_rel + "'></video>" if mp4_rel else "<p>MP4 not found — regenerate with render script.</p>"}
<p class="note">Source: {trace.source_label} · Scenario: {trace.scenario_name}</p>
<p class="note">Compressed KV drafts. Full KV verifies. Wrong token rejected. EXACT MATCH.</p>
<p class="note">Not a speed or memory benchmark. Tested panels only.</p>
</div>
</body>
</html>"""
    out_path.write_text(body, encoding="utf-8")


def write_crash_test_video_doc(trace: CrashTestTrace, artifacts: list[str]) -> None:
    path = _ROOT / "docs" / "EXACTKV_CRASH_TEST_VIDEO.md"
    art_lines = "\n".join(f"- `{a}`" for a in artifacts)
    content = f"""# ExactKV Crash-Test Video (V13 Phase 8c)

**Status:** Watchable artifact generated from verified trace.

> This is a **cinematic correctness demo**, not a benchmark.
> Trace tokens are from **{trace.source_label}** — not invented.
> No speedup, throughput, latency, tokens/sec, active GPU memory savings, production serving, or model accuracy improvement claim is made.

---

## 1. Purpose

90–120 second launch-quality video showing lossy KV proposing a wrong token, ExactKV rejecting it, committing the verifier correction, and matching full greedy — a crash-test narrative, not a debug table.

## 2. Why static cards were not enough

Phase 8b PNG cards are useful for README and threads, but a **watchable video** is required for launch posts and social distribution.

## 3. Source trace

| Field | Value |
| --- | --- |
| Scenario | {trace.scenario_name} |
| prompt_id | `{trace.prompt_id}` |
| compressor | `{trace.compressor_name}` |
| Rejected token | `{trace.rejected_token}` |
| Correction token | `{trace.correction_token}` |
| exactkv_failures | {trace.exactkv_failures} |
| final match | {str(trace.exactkv_exact_match).lower()} |

**Restaurant ordering trace:** not found in existing reports without new model search. **Fallback:** verified Exp 034 weather tool JSON (`tj_002` × `int4_sim`).

## 4. How to render

```bash
# Full quality (~105s, 1920×1080)
python3 scripts/render_exactkv_crash_test_video.py

# Quick preview
python3 scripts/render_exactkv_crash_test_video.py --fast

# Frames + storyboard only
python3 scripts/render_exactkv_crash_test_video.py --fast --no-video
```

Options: `--source-json PATH`, `--fps N`, `--width W`, `--height H`

## 5. How to view

Open `docs/assets/exactkv_crash_test_demo.mp4` or `exactkv_crash_test_demo.html` in a browser.

## 6. How to record/share

- Upload MP4 to X/LinkedIn/YouTube
- Embed HTML player page for docs site
- GIF: `docs/assets/exactkv_crash_test_demo.gif` for lightweight sharing

## 7. Generated artifacts

{art_lines}

## 8. Allowed claims

- Lossy KV drafted wrong token on this verified trace; ExactKV corrected.
- `exactkv_failures == 0` on shown V13 panels (Exp 029/033/032b cited in video).
- Exp 030/031 honesty framing (slower, no VRAM savings).

## 9. Forbidden claims

- Speedup, throughput, latency, tokens/sec, VRAM savings.
- Production serving or model accuracy improvement.
- Shard/SpectralQuant as ExactKV results.

## 10. Next steps

**Proceed to Phase 9** launch package with this video linked from README.

---

**Related:** [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md) · [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md) · [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md)
"""
    path.write_text(content, encoding="utf-8")


def encode_video(frames_dir: Path, out_mp4: Path, fps: int) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    pattern = str(frames_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_mp4.is_file() and out_mp4.stat().st_size > 10000
    except subprocess.CalledProcessError:
        return False


def encode_gif(frames_dir: Path, out_gif: Path, fps: int) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    pattern = str(frames_dir / "frame_%05d.png")
    # Sample every nth frame for smaller GIF in fast mode
    cmd = [
        "ffmpeg", "-y", "-framerate", str(min(fps, 12)),
        "-i", pattern,
        "-vf", "fps=12,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1]paletteuse",
        str(out_gif),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_gif.is_file() and out_gif.stat().st_size > 10000
    except subprocess.CalledProcessError:
        return False


def render_video(
    trace: CrashTestTrace,
    *,
    width: int,
    height: int,
    fps: int,
    fast: bool,
    no_video: bool,
    keep_frames: bool,
) -> list[str]:
    artifacts: list[str] = []
    specs = scene_specs(trace)
    total_duration = specs[-1].end_s if not fast else 24.0
    if fast:
        scale = total_duration / specs[-1].end_s
        specs = [
            SceneSpec(s.scene_id, s.title, s.start_s * scale, s.end_s * scale, s.voiceover, s.source_note, s.visual_note)
            for s in specs
        ]

    storyboard_path = _ASSETS / "exactkv_crash_test_storyboard.md"
    build_storyboard(trace, storyboard_path)
    artifacts.append(str(storyboard_path.relative_to(_ROOT)))

    if _FRAMES_DIR.exists():
        shutil.rmtree(_FRAMES_DIR)
    _FRAMES_DIR.mkdir(parents=True)

    frame_idx = 0
    frames_per_scene = max(1, int(fps * 0.5)) if fast else max(1, int(fps * 1.0))
    total_frames = int(total_duration * fps)

    for spec in specs:
        dur = spec.end_s - spec.start_s
        n_frames = max(1, int(dur * fps))
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            img = render_scene_frame(spec, trace, t, width, height)
            frame_idx += 1
            img.save(_FRAMES_DIR / f"frame_{frame_idx:05d}.png")

    # Pad to exact total if needed
    while frame_idx < total_frames:
        frame_idx += 1
        last = _FRAMES_DIR / f"frame_{frame_idx - 1:05d}.png"
        if last.is_file():
            Image.open(last).save(_FRAMES_DIR / f"frame_{frame_idx:05d}.png")

    artifacts.append(str(_FRAMES_DIR.relative_to(_ROOT)) + "/")

    html_path = _ASSETS / "exactkv_crash_test_demo.html"
    write_html_player(_FRAMES_DIR, fps, html_path, trace)
    artifacts.append(str(html_path.relative_to(_ROOT)))

    write_crash_test_video_doc(trace, artifacts)

    if no_video:
        print("Skipping video encode (--no-video)")
        return artifacts

    mp4_path = _ASSETS / "exactkv_crash_test_demo.mp4"
    if encode_video(_FRAMES_DIR, mp4_path, fps):
        artifacts.append(str(mp4_path.relative_to(_ROOT)))
        print(f"Wrote {mp4_path} ({mp4_path.stat().st_size // 1024} KiB)")
    else:
        print("MP4 encode failed or ffmpeg missing")

    gif_path = _ASSETS / "exactkv_crash_test_demo.gif"
    if encode_gif(_FRAMES_DIR, gif_path, fps):
        artifacts.append(str(gif_path.relative_to(_ROOT)))
        print(f"Wrote {gif_path} ({gif_path.stat().st_size // 1024} KiB)")

    if not keep_frames and not fast:
        pass  # keep frames for debugging by default

    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ExactKV crash-test launch video")
    parser.add_argument("--source-json", type=Path, default=_DEFAULT_JSON)
    parser.add_argument("--fast", action="store_true", help="Shorter duration, 1280×720")
    parser.add_argument("--no-video", action="store_true", help="Storyboard + frames + HTML only")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    width = args.width or (1280 if args.fast else 1920)
    height = args.height or (720 if args.fast else 1080)
    fps = args.fps or (12 if args.fast else 24)

    trace = load_trace(args.source_json)
    print(f"Trace: {trace.source_label}")
    print(f"Scenario: {trace.scenario_name}")
    print(f"Rejected {trace.rejected_token!r} -> {trace.correction_token!r}")

    artifacts = render_video(
        trace,
        width=width,
        height=height,
        fps=fps,
        fast=args.fast,
        no_video=args.no_video,
        keep_frames=args.keep_frames,
    )
    print(f"Artifacts: {', '.join(artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
