"""V10 evaluation prompt suite loader.

Loads versioned JSONL suites authored in V10 Phase 1.  Legacy suites
(``smoke``, ``core``, …) remain in :mod:`exactkv.benchmarks.prompts`.

V10 rows use ``id``, ``primary_category``, ``suite_version``.  This module
normalizes them to the runner shape (``prompt_id``, ``category``, ``prompt``)
and preserves V10 metadata for experiment analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
_PROMPTS_DIR = _BENCHMARKS_DIR / "prompts"

V10_SUITE_REGISTRY: dict[str, Path] = {
    "core_v2": _PROMPTS_DIR / "core_v2.jsonl",
    "code_structured": _PROMPTS_DIR / "code_structured.jsonl",
    "long_context": _PROMPTS_DIR / "long_context.jsonl",
    "reasoning_math": _PROMPTS_DIR / "reasoning_math.jsonl",
    "multilingual": _PROMPTS_DIR / "multilingual.jsonl",
    "retrieval_copy": _PROMPTS_DIR / "retrieval_copy.jsonl",
    "tool_json": _PROMPTS_DIR / "tool_json.jsonl",
}


def list_v10_suites() -> list[str]:
    """Return sorted V10 suite names."""
    return sorted(V10_SUITE_REGISTRY)


def load_v10_suite(name: str) -> list[dict[str, Any]]:
    """Load one V10 suite and return runner-compatible prompt dicts.

    Each dict includes ``prompt_id``, ``category``, ``prompt`` for
    :func:`exactkv.benchmarks.runner.run_one`, plus V10 metadata keys:
    ``v10_id``, ``v10_suite``, ``v10_suite_version``, ``v10_primary_category``,
    ``v10_secondary_tags``.
    """
    if name not in V10_SUITE_REGISTRY:
        raise ValueError(
            f"Unknown V10 suite {name!r}. Available: {list_v10_suites()}"
        )
    path = V10_SUITE_REGISTRY[name]
    if not path.exists():
        raise FileNotFoundError(f"V10 suite file not found: {path}")

    prompts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            for field in ("id", "prompt", "primary_category", "suite_version"):
                if field not in row:
                    raise ValueError(
                        f"{path}:{lineno}: missing required field {field!r}"
                    )
            prompts.append({
                "prompt_id": row["id"],
                "category": row["primary_category"],
                "prompt": row["prompt"],
                "v10_id": row["id"],
                "v10_suite": name,
                "v10_suite_version": row["suite_version"],
                "v10_primary_category": row["primary_category"],
                "v10_secondary_tags": row.get("secondary_tags"),
                "v10_source_note": row.get("source_note"),
            })
    return prompts


def load_all_v10_prompts(
    suite_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and concatenate V10 suites in stable order."""
    names = suite_names if suite_names is not None else list_v10_suites()
    out: list[dict[str, Any]] = []
    for name in names:
        out.extend(load_v10_suite(name))
    return out
