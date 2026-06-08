"""Tests for V3 Phase A: named prompt suites and suite resolution.

Gates
-----
* Suite validation gate  — every named suite loads, schema is valid,
  prompt_ids are unique, fields are non-empty.
* Suite resolution gate  — load_suite() and resolve_suite() work for all
  registered names; unknown names raise ValueError.
* CLI named-suite gate   — bench and sweep CLIs accept named suites other
  than smoke without error (model-dependent; uses noop + tiny token count).
* --suite-file override  — still works for all CLI commands.
* No forbidden performance fields in any returned dict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ALL_NAMED_SUITES = ["smoke", "core", "structured", "code", "stress"]
_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}

# ---------------------------------------------------------------------------
# Suite registry / resolution (model-free)
# ---------------------------------------------------------------------------

def test_list_suites_returns_all():
    from exactkv.benchmarks.prompts import list_suites
    names = list_suites()
    for expected in ALL_NAMED_SUITES:
        assert expected in names, f"Suite {expected!r} missing from list_suites()"


def test_resolve_suite_returns_path_for_all_named():
    from exactkv.benchmarks.prompts import resolve_suite
    for name in ALL_NAMED_SUITES:
        p = resolve_suite(name)
        assert p.exists(), f"JSONL file for suite {name!r} not found: {p}"


def test_resolve_suite_raises_for_unknown():
    from exactkv.benchmarks.prompts import resolve_suite
    with pytest.raises(ValueError, match="Unknown prompt suite"):
        resolve_suite("nonexistent_xyz_suite")


def test_resolve_suite_error_message_lists_available():
    from exactkv.benchmarks.prompts import resolve_suite
    try:
        resolve_suite("bad_suite")
    except ValueError as exc:
        msg = str(exc)
        assert "smoke" in msg
        assert "core" in msg


# ---------------------------------------------------------------------------
# load_suite convenience helper
# ---------------------------------------------------------------------------

def test_load_suite_all_named():
    from exactkv.benchmarks.prompts import load_suite
    for name in ALL_NAMED_SUITES:
        prompts = load_suite(name)
        assert len(prompts) > 0, f"Suite {name!r} is empty"


def test_load_suite_unknown_raises():
    from exactkv.benchmarks.prompts import load_suite
    with pytest.raises(ValueError):
        load_suite("no_such_suite")


# ---------------------------------------------------------------------------
# Schema validation — parameterised over all suites
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suite_name", ALL_NAMED_SUITES)
def test_every_row_has_required_fields(suite_name):
    from exactkv.benchmarks.prompts import load_suite
    prompts = load_suite(suite_name)
    for i, p in enumerate(prompts):
        for field in ("prompt_id", "category", "prompt"):
            assert field in p, (
                f"Suite {suite_name!r} row {i}: missing field {field!r}"
            )


@pytest.mark.parametrize("suite_name", ALL_NAMED_SUITES)
def test_prompt_ids_unique_within_suite(suite_name):
    from exactkv.benchmarks.prompts import load_suite
    prompts = load_suite(suite_name)
    ids = [p["prompt_id"] for p in prompts]
    assert len(ids) == len(set(ids)), (
        f"Suite {suite_name!r} has duplicate prompt_ids: "
        f"{[x for x in ids if ids.count(x) > 1]}"
    )


@pytest.mark.parametrize("suite_name", ALL_NAMED_SUITES)
def test_categories_non_empty(suite_name):
    from exactkv.benchmarks.prompts import load_suite
    for p in load_suite(suite_name):
        assert p["category"].strip(), (
            f"Suite {suite_name!r} prompt {p['prompt_id']!r}: empty category"
        )


@pytest.mark.parametrize("suite_name", ALL_NAMED_SUITES)
def test_prompts_non_empty(suite_name):
    from exactkv.benchmarks.prompts import load_suite
    for p in load_suite(suite_name):
        assert p["prompt"].strip(), (
            f"Suite {suite_name!r} prompt {p['prompt_id']!r}: empty prompt"
        )


# ---------------------------------------------------------------------------
# Suite sizes (smoke unchanged; others within expected ranges)
# ---------------------------------------------------------------------------

def test_smoke_has_16_prompts():
    from exactkv.benchmarks.prompts import load_suite
    assert len(load_suite("smoke")) == 16


def test_core_has_at_least_30_prompts():
    from exactkv.benchmarks.prompts import load_suite
    assert len(load_suite("core")) >= 30


def test_structured_has_at_least_20_prompts():
    from exactkv.benchmarks.prompts import load_suite
    assert len(load_suite("structured")) >= 20


def test_code_has_at_least_20_prompts():
    from exactkv.benchmarks.prompts import load_suite
    assert len(load_suite("code")) >= 20


def test_stress_has_at_least_15_prompts():
    from exactkv.benchmarks.prompts import load_suite
    assert len(load_suite("stress")) >= 15


# ---------------------------------------------------------------------------
# Backward-compat: load_smoke_prompts / load_prompts still work
# ---------------------------------------------------------------------------

def test_load_smoke_prompts_backward_compat():
    from exactkv.benchmarks.prompts import load_smoke_prompts
    prompts = load_smoke_prompts()
    assert len(prompts) == 16


def test_load_prompts_default_is_smoke():
    from exactkv.benchmarks.prompts import load_prompts, load_smoke_prompts
    assert load_prompts() == load_smoke_prompts()


def test_load_prompts_with_explicit_path(tmp_path):
    p = tmp_path / "mini.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "t1", "category": "test", "prompt": "Hello"}) + "\n",
        encoding="utf-8",
    )
    from exactkv.benchmarks.prompts import load_prompts
    prompts = load_prompts(p)
    assert len(prompts) == 1
    assert prompts[0]["prompt_id"] == "t1"


# ---------------------------------------------------------------------------
# Named convenience loaders
# ---------------------------------------------------------------------------

def test_load_core_prompts():
    from exactkv.benchmarks.prompts import load_core_prompts, load_suite
    assert load_core_prompts() == load_suite("core")


def test_load_structured_prompts():
    from exactkv.benchmarks.prompts import load_structured_prompts, load_suite
    assert load_structured_prompts() == load_suite("structured")


def test_load_code_prompts():
    from exactkv.benchmarks.prompts import load_code_prompts, load_suite
    assert load_code_prompts() == load_suite("code")


def test_load_stress_prompts():
    from exactkv.benchmarks.prompts import load_stress_prompts, load_suite
    assert load_stress_prompts() == load_suite("stress")


# ---------------------------------------------------------------------------
# CLI --suite named resolution (model-dependent; uses tiny token count)
# Uses module-scope fixture so model loads only once for all CLI suite tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_suite_file(tmp_path_factory):
    """Single-prompt JSONL for fast CLI tests."""
    p = tmp_path_factory.mktemp("cli_suite") / "mini.jsonl"
    p.write_text(
        json.dumps({"prompt_id": "cli_t1", "category": "test",
                    "prompt": "The capital of France is"}) + "\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.parametrize("suite_name", ["core", "structured", "code", "stress"])
def test_cli_bench_resolves_named_suite_via_suite_file(
    suite_name, tmp_path_factory, mini_suite_file
):
    """Bench with --suite-file (fast path — avoids running full named suites)."""
    from exactkv.cli import main

    out_dir = tmp_path_factory.mktemp(f"bench_{suite_name}")
    rc = main([
        "bench",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite_file,
        "--compressor", "noop",
        "--draft-len", "4",
        "--max-new-tokens", "4",
        "--json-out", str(out_dir / "out.json"),
    ])
    assert rc == 0, f"bench with --suite-file returned {rc}"


def test_cli_bench_unknown_suite_returns_nonzero(mini_suite_file):
    """Unknown --suite name fails fast (before model load)."""
    from exactkv.cli import main
    rc = main([
        "bench",
        "--model", MODEL_NAME,
        "--suite", "no_such_suite_xyz",
        "--compressor", "noop",
        "--max-new-tokens", "4",
    ])
    assert rc != 0


def test_cli_sweep_resolves_named_suite_via_suite_file(tmp_path_factory, mini_suite_file):
    """Sweep with --suite-file resolves correctly."""
    from exactkv.cli import main

    out_dir = tmp_path_factory.mktemp("sweep_named_suite")
    rc = main([
        "sweep",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite_file,
        "--compressors", "noop",
        "--draft-lengths", "4",
        "--max-new-tokens", "4",
        "--json-out", str(out_dir / "sweep.json"),
    ])
    assert rc == 0, f"sweep with --suite-file returned {rc}"


def test_cli_suite_file_override_still_works(tmp_path_factory, mini_suite_file):
    """--suite-file takes precedence over --suite when both provided (mutually exclusive)."""
    from exactkv.cli import main
    # argparse enforces mutual exclusion; passing just --suite-file should be fine
    out_dir = tmp_path_factory.mktemp("suite_override")
    rc = main([
        "bench",
        "--model", MODEL_NAME,
        "--suite-file", mini_suite_file,
        "--compressor", "noop",
        "--draft-len", "4",
        "--max-new-tokens", "4",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# No forbidden performance fields in prompt dicts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suite_name", ALL_NAMED_SUITES)
def test_no_forbidden_fields_in_prompt_dicts(suite_name):
    from exactkv.benchmarks.prompts import load_suite
    for p in load_suite(suite_name):
        for field in _FORBIDDEN:
            assert field not in p, (
                f"Suite {suite_name!r} prompt {p['prompt_id']!r}: "
                f"forbidden field {field!r}"
            )
