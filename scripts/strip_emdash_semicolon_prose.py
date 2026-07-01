#!/usr/bin/env python3
"""Remove em dashes and semicolons from blog/paper prose (not code)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EM_DASH = "\u2014"  # —


def _cap_after_period(text: str) -> str:
    """Capitalize letter after '. a' patterns from semicolon splits."""
    return re.sub(
        r"\. ([a-z])",
        lambda m: ". " + m.group(1).upper(),
        text,
    )


def clean_prose_chunk(chunk: str) -> str:
    if not chunk:
        return chunk
    s = chunk
    # Em dash: ordered specific patterns first
    patterns = [
        (f"{EM_DASH}and ", ", and "),
        (f"{EM_DASH}or ", ", or "),
        (f" {EM_DASH} if ", ". If "),
        (f" {EM_DASH} not ", ", not "),
        (f" {EM_DASH} the ", ". The "),
        (f"</strong> {EM_DASH} ", "</strong>: "),
        (f"</h2>.*{EM_DASH} ", None),  # handled below
        (f" {EM_DASH} ", ", "),
        (f"{EM_DASH} ", ": "),
        (f" {EM_DASH}", ","),
        (EM_DASH, ", "),
    ]
    for old, new in patterns:
        if new is None:
            continue
        s = s.replace(old, new)

    # Heading-style "surprise, same" -> "surprise: same" for h2/h3 numeric sections
    s = re.sub(
        r"(<h2>\d+\. [^<]+), (same|three|logit|downstream|H2O|bit-width|faithful)",
        r"\1: \2",
        s,
        flags=re.I,
    )

    # Table / stat placeholders: lone em dash in cells
    s = re.sub(r">\s*" + re.escape(EM_DASH) + r"\s*<", ">n/a<", s)
    s = re.sub(r"\|\s*" + re.escape(EM_DASH) + r"\s*\|", "| n/a |", s)

    # Semicolons in prose (not inside HTML tags)
    parts = re.split(r"(<[^>]+>)", s)
    out: list[str] = []
    for i, part in enumerate(parts):
        if part.startswith("<") and part.endswith(">"):
            out.append(part)
            continue
        p = part
        # List-style a; b; c -> a. b. c
        p = re.sub(r";(\s+)", r".\1", p)
        p = _cap_after_period(p)
        out.append(p)
    return "".join(out)


def process_markdown(text: str) -> str:
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]+`)", text)
    return "".join(
        p if (p.startswith("```") or (p.startswith("`") and p.endswith("`"))) else clean_prose_chunk(p)
        for p in parts
    )


def process_html(text: str) -> str:
    protected: list[str] = []

    def protect(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f"\x00P{len(protected) - 1}\x00"

    for pat in (
        r"<script[\s\S]*?</script>",
        r"<style[\s\S]*?</style>",
        r"<link[^>]*>",
        r"<code[\s\S]*?</code>",
        r"<pre[\s\S]*?</pre>",
        r'style="[^"]*"',
    ):
        text = re.sub(pat, protect, text, flags=re.I)

    text = clean_prose_chunk(text)

    def restore(m: re.Match[str]) -> str:
        return protected[int(m.group(1))]

    text = re.sub(r"\x00P(\d+)\x00", restore, text)
    return text


def main() -> None:
    targets = [
        ROOT / "site" / "index.html",
        ROOT / "paper" / "ExactKV_Technical_Report.md",
    ]
    for path in targets:
        raw = path.read_text()
        if path.suffix == ".html":
            out = process_html(raw)
        else:
            out = process_markdown(raw)
        path.write_text(out)
        em = out.count(EM_DASH)
        # semicolons outside code-ish regions (rough count in file)
        print(f"{path.name}: em dashes remaining={em}, semicolons remaining={out.count(';')}")


if __name__ == "__main__":
    main()
