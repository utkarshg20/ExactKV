"""Prompt loading utilities for ExactKV benchmarks.

Prompts are stored as JSONL files (one JSON object per line).

Required fields per prompt entry:
  * ``prompt_id``  — unique identifier string
  * ``category``   — coarse category string (e.g. "natural_language", "code")
  * ``prompt``     — plain-text prompt string

Optional fields (ignored by the loader):
  * ``notes``      — human-readable annotation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Default location for bundled prompt files (relative to this source file).
_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
SMOKE_JSONL = _BENCHMARKS_DIR / "prompts" / "smoke.jsonl"


def load_prompts(path: Optional[Path | str] = None) -> list[dict]:
    """Load a JSONL prompt file and return a list of prompt dicts.

    Args:
        path: Path to a .jsonl file.  Defaults to the bundled smoke suite.

    Returns:
        List of dicts, each with at least ``prompt_id``, ``category``,
        and ``prompt`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If a line is missing a required field.
    """
    if path is None:
        path = SMOKE_JSONL
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = json.loads(line)
            for field in ("prompt_id", "category", "prompt"):
                if field not in entry:
                    raise ValueError(
                        f"{path}:{lineno}: missing required field {field!r}"
                    )
            prompts.append(entry)
    return prompts


def load_smoke_prompts() -> list[dict]:
    """Load the bundled smoke prompt suite."""
    return load_prompts(SMOKE_JSONL)
