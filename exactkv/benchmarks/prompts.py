"""Prompt loading utilities for ExactKV benchmarks.

Prompts are stored as JSONL files (one JSON object per line).

Required fields per prompt entry:
  * ``prompt_id``  — unique identifier string
  * ``category``   — coarse category string (e.g. "natural_language", "code")
  * ``prompt``     — plain-text prompt string

Optional fields (ignored by the loader):
  * ``notes``      — human-readable annotation

Named suites
------------
Each suite is a bundled JSONL file under ``benchmarks/prompts/``.

+-------------+-------------------------------------------------------------------+
| Name        | Purpose                                                           |
+=============+===================================================================+
| smoke       | 16 prompts, fast CI / quick-sanity suite.  Unchanged since V1.   |
+-------------+-------------------------------------------------------------------+
| core        | ~34 prompts, broad category coverage.  Default for documented     |
|             | experiments.                                                      |
+-------------+-------------------------------------------------------------------+
| structured  | ~28 prompts focused on JSON, tables, schemas, function-calls.     |
|             | Tests acceptance on highly-templated / low-entropy output.        |
+-------------+-------------------------------------------------------------------+
| code        | ~30 code-generation and completion prompts.  Tests acceptance on  |
|             | syntax-sensitive, structured continuations.                       |
+-------------+-------------------------------------------------------------------+
| stress      | ~25 longer, harder, higher-entropy prompts.  Designed to produce  |
|             | lower acceptance rates and surface more lossy divergence.         |
+-------------+-------------------------------------------------------------------+
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Root of the bundled prompt file directory.
_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
_PROMPTS_DIR = _BENCHMARKS_DIR / "prompts"

# ---------------------------------------------------------------------------
# Named-suite registry: name -> JSONL path
# ---------------------------------------------------------------------------

SUITE_REGISTRY: dict[str, Path] = {
    "smoke":      _PROMPTS_DIR / "smoke.jsonl",
    "core":       _PROMPTS_DIR / "core.jsonl",
    "structured": _PROMPTS_DIR / "structured.jsonl",
    "code":       _PROMPTS_DIR / "code.jsonl",
    "stress":     _PROMPTS_DIR / "stress.jsonl",
}

# Convenience alias kept for backward compatibility.
SMOKE_JSONL: Path = SUITE_REGISTRY["smoke"]


def list_suites() -> list[str]:
    """Return a sorted list of all registered named-suite names."""
    return sorted(SUITE_REGISTRY)


def resolve_suite(name: str) -> Path:
    """Return the JSONL path for a registered named suite.

    Args:
        name: A key from :data:`SUITE_REGISTRY`.

    Returns:
        Absolute ``Path`` to the ``.jsonl`` file.

    Raises:
        ValueError: If ``name`` is not a registered suite.
    """
    if name not in SUITE_REGISTRY:
        raise ValueError(
            f"Unknown prompt suite {name!r}. "
            f"Available named suites: {list_suites()}. "
            "Use load_prompts(path) to load a custom JSONL file."
        )
    return SUITE_REGISTRY[name]


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_prompts(path: Optional[Path | str] = None) -> list[dict]:
    """Load a JSONL prompt file and return a list of prompt dicts.

    Args:
        path: Path to a ``.jsonl`` file.  Defaults to the bundled ``smoke``
              suite.

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


def load_suite(name: str) -> list[dict]:
    """Load a registered named suite by name.

    Args:
        name: A key from :data:`SUITE_REGISTRY` (e.g. ``"smoke"``,
              ``"core"``, ``"code"``).

    Returns:
        List of prompt dicts.

    Raises:
        ValueError:        If ``name`` is not registered.
        FileNotFoundError: If the backing JSONL file does not exist.
    """
    return load_prompts(resolve_suite(name))


# ---------------------------------------------------------------------------
# Named convenience helpers (backward-compatible)
# ---------------------------------------------------------------------------

def load_smoke_prompts() -> list[dict]:
    """Load the bundled ``smoke`` prompt suite (16 prompts, fast CI)."""
    return load_suite("smoke")


def load_core_prompts() -> list[dict]:
    """Load the bundled ``core`` prompt suite (~34 broad prompts)."""
    return load_suite("core")


def load_structured_prompts() -> list[dict]:
    """Load the bundled ``structured`` prompt suite (~28 prompts)."""
    return load_suite("structured")


def load_code_prompts() -> list[dict]:
    """Load the bundled ``code`` prompt suite (~30 prompts)."""
    return load_suite("code")


def load_stress_prompts() -> list[dict]:
    """Load the bundled ``stress`` prompt suite (~25 prompts)."""
    return load_suite("stress")
