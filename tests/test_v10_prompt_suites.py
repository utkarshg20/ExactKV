"""Tests for V10 Phase 1: V10 prompt suite files and validator.

Gates
-----
* Validator passes on committed suites.
* All suite files exist and parse.
* Minimum counts met per suite.
* Global ID uniqueness.
* Categories and metadata valid.
* No empty prompts.
* No forbidden performance field names in suite metadata.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "benchmarks" / "prompts"
VALIDATOR = ROOT / "scripts" / "validate_v10_prompt_suites.py"

V10_SUITE_FILES = [
    "core_v2.jsonl",
    "code_structured.jsonl",
    "long_context.jsonl",
    "reasoning_math.jsonl",
    "multilingual.jsonl",
    "retrieval_copy.jsonl",
    "tool_json.jsonl",
]

MIN_COUNTS = {
    "core_v2": 40,
    "code_structured": 20,
    "long_context": 15,
    "reasoning_math": 15,
    "multilingual": 15,
    "retrieval_copy": 10,
    "tool_json": 10,
}

ALLOWED_PRIMARY = {
    "natural_language",
    "code",
    "structured_json",
    "long_context",
    "reasoning_math",
    "multilingual",
    "retrieval_copy",
    "qa_factual",
    "tool_schema",
}

_FORBIDDEN = {
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
}


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_v10", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_all_v10_rows() -> list[dict]:
    mod = _load_validator_module()
    rows, errors = mod.validate()
    assert not errors, f"validator errors: {errors}"
    return rows


def _load_suite_file(name: str) -> list[dict]:
    rows = []
    path = PROMPTS_DIR / name
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def test_validator_script_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Validation PASSED" in proc.stdout


def test_all_suite_files_exist():
    for name in V10_SUITE_FILES:
        assert (PROMPTS_DIR / name).exists(), f"missing {name}"


@pytest.mark.parametrize("suite_file", V10_SUITE_FILES)
def test_suite_file_parses(suite_file):
    rows = _load_suite_file(suite_file)
    assert len(rows) > 0


@pytest.mark.parametrize("suite_name,min_count", list(MIN_COUNTS.items()))
def test_minimum_counts_met(suite_name, min_count):
    rows = _load_suite_file(f"{suite_name}.jsonl")
    assert len(rows) >= min_count


def test_ids_globally_unique():
    seen: set[str] = set()
    for suite_file in V10_SUITE_FILES:
        for row in _load_suite_file(suite_file):
            row_id = row["id"]
            assert row_id not in seen, f"duplicate id {row_id!r}"
            seen.add(row_id)


def test_required_metadata_present():
    for suite_file in V10_SUITE_FILES:
        for i, row in enumerate(_load_suite_file(suite_file)):
            for field in ("id", "prompt", "primary_category", "suite_version"):
                assert field in row, f"{suite_file} row {i}: missing {field!r}"


def test_primary_categories_valid():
    for row in _load_all_v10_rows():
        assert row["primary_category"] in ALLOWED_PRIMARY


def test_secondary_tags_are_lists_when_present():
    for suite_file in V10_SUITE_FILES:
        for row in _load_suite_file(suite_file):
            tags = row.get("secondary_tags")
            if tags is not None:
                assert isinstance(tags, list)


def test_no_empty_prompts():
    for row in _load_all_v10_rows():
        assert row["prompt"].strip(), f"empty prompt for id {row['id']!r}"


def test_suite_version_matches_suite_family():
    mod = _load_validator_module()
    for suite_name, spec in mod.V10_SUITES.items():
        prefix = spec["suite_version_prefix"]
        for row in _load_suite_file(f"{suite_name}.jsonl"):
            assert row["suite_version"].startswith(prefix), (
                f"{row['id']}: suite_version {row['suite_version']!r} "
                f"does not start with {prefix!r}"
            )


@pytest.mark.parametrize("suite_file", V10_SUITE_FILES)
def test_no_forbidden_fields_in_metadata(suite_file):
    for row in _load_suite_file(suite_file):
        for field in _FORBIDDEN:
            assert field not in row, (
                f"{suite_file} id {row['id']!r}: forbidden field {field!r}"
            )


def test_total_prompt_count_at_least_125():
    total = sum(len(_load_suite_file(f)) for f in V10_SUITE_FILES)
    assert total >= 125
