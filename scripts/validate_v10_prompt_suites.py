#!/usr/bin/env python3
"""Validate V10 evaluation prompt suites under benchmarks/prompts/.

Checks required files, JSONL schema, global ID uniqueness, category taxonomy,
minimum counts, and forbidden performance field names.

Exit 0 on success; exit 1 with stderr details on failure.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "benchmarks" / "prompts"

V10_SUITES: dict[str, dict] = {
    "core_v2": {
        "path": PROMPTS_DIR / "core_v2.jsonl",
        "min_count": 40,
        "suite_version_prefix": "core_v2",
    },
    "code_structured": {
        "path": PROMPTS_DIR / "code_structured.jsonl",
        "min_count": 20,
        "suite_version_prefix": "code_structured",
    },
    "long_context": {
        "path": PROMPTS_DIR / "long_context.jsonl",
        "min_count": 15,
        "suite_version_prefix": "long_context",
    },
    "reasoning_math": {
        "path": PROMPTS_DIR / "reasoning_math.jsonl",
        "min_count": 15,
        "suite_version_prefix": "reasoning_math",
    },
    "multilingual": {
        "path": PROMPTS_DIR / "multilingual.jsonl",
        "min_count": 15,
        "suite_version_prefix": "multilingual",
    },
    "retrieval_copy": {
        "path": PROMPTS_DIR / "retrieval_copy.jsonl",
        "min_count": 10,
        "suite_version_prefix": "retrieval_copy",
    },
    "tool_json": {
        "path": PROMPTS_DIR / "tool_json.jsonl",
        "min_count": 10,
        "suite_version_prefix": "tool_json",
    },
}

ALLOWED_PRIMARY_CATEGORIES = frozenset({
    "natural_language",
    "code",
    "structured_json",
    "long_context",
    "reasoning_math",
    "multilingual",
    "retrieval_copy",
    "qa_factual",
    "tool_schema",
})

ALLOWED_SECONDARY_TAGS = frozenset({
    "short_prefill",
    "medium_prefill",
    "long_prefill",
    "repetition_heavy",
    "symbol_heavy",
    "whitespace_sensitive",
    "numeric_heavy",
})

REQUIRED_FIELDS = ("id", "prompt", "primary_category", "suite_version")

FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

# Intentional duplicate prompt texts (regression anchors from original core suite).
DOCUMENTED_DUPLICATE_PROMPTS = frozenset({
    "The capital of France is",
})


def load_suite_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{lineno}: row must be a JSON object")
            rows.append(entry)
    return rows


def validate(*, strict_duplicates: bool = True) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    all_rows: list[dict] = []
    global_ids: dict[str, str] = {}
    prompt_text_locations: dict[str, list[str]] = defaultdict(list)

    for suite_name, spec in V10_SUITES.items():
        path: Path = spec["path"]
        if not path.exists():
            errors.append(f"Missing suite file: {path}")
            continue

        try:
            rows = load_suite_rows(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if len(rows) < spec["min_count"]:
            errors.append(
                f"{suite_name}: count {len(rows)} < minimum {spec['min_count']}"
            )

        prefix = spec["suite_version_prefix"]
        for i, row in enumerate(rows, start=1):
            loc = f"{suite_name}:{path.name}:row{i}"
            for field in REQUIRED_FIELDS:
                if field not in row:
                    errors.append(f"{loc}: missing required field {field!r}")
            if errors and any(f"missing required field" in e for e in errors[-4:]):
                continue

            row_id = row["id"]
            if not isinstance(row_id, str) or not row_id.strip():
                errors.append(f"{loc}: id must be a non-empty string")
            elif row_id in global_ids:
                errors.append(
                    f"{loc}: duplicate global id {row_id!r} "
                    f"(first seen in {global_ids[row_id]})"
                )
            else:
                global_ids[row_id] = loc

            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{loc}: prompt must be a non-empty string")
            else:
                prompt_text_locations[prompt].append(row_id)

            cat = row["primary_category"]
            if cat not in ALLOWED_PRIMARY_CATEGORIES:
                errors.append(
                    f"{loc}: invalid primary_category {cat!r}; "
                    f"allowed: {sorted(ALLOWED_PRIMARY_CATEGORIES)}"
                )

            sv = row["suite_version"]
            if not isinstance(sv, str) or not sv.strip():
                errors.append(f"{loc}: suite_version must be a non-empty string")
            elif not sv.startswith(prefix):
                errors.append(
                    f"{loc}: suite_version {sv!r} must start with {prefix!r}"
                )

            tags = row.get("secondary_tags")
            if tags is not None:
                if not isinstance(tags, list):
                    errors.append(f"{loc}: secondary_tags must be a list when present")
                else:
                    for tag in tags:
                        if tag not in ALLOWED_SECONDARY_TAGS:
                            errors.append(
                                f"{loc}: invalid secondary_tag {tag!r}"
                            )

            for forbidden in FORBIDDEN_FIELDS:
                if forbidden in row:
                    errors.append(f"{loc}: forbidden field {forbidden!r}")

            row_copy = dict(row)
            row_copy["_suite_name"] = suite_name
            all_rows.append(row_copy)

    if strict_duplicates:
        for prompt_text, ids in prompt_text_locations.items():
            if len(ids) > 1 and prompt_text not in DOCUMENTED_DUPLICATE_PROMPTS:
                errors.append(
                    f"Duplicate prompt text across ids {ids!r} "
                    f"(not in documented duplicate allowlist)"
                )

    return all_rows, errors


def summary_table(rows: list[dict]) -> str:
    by_suite: dict[str, int] = Counter()
    by_suite_cat: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        suite = row["_suite_name"]
        by_suite[suite] += 1
        by_suite_cat[suite][row["primary_category"]] += 1

    lines = ["V10 prompt suite summary", "=" * 60]
    total = 0
    for suite_name in V10_SUITES:
        count = by_suite.get(suite_name, 0)
        total += count
        min_count = V10_SUITES[suite_name]["min_count"]
        lines.append(f"{suite_name:20s} {count:4d}  (min {min_count})")
        for cat, n in sorted(by_suite_cat[suite_name].items()):
            lines.append(f"  {cat:22s} {n:4d}")
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':20s} {total:4d}")
    return "\n".join(lines)


def main() -> int:
    rows, errors = validate()
    if errors:
        print("V10 prompt suite validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(summary_table(rows))
    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
