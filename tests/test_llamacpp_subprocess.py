"""Unit tests for llama.cpp subprocess parsing helpers (Exp 022)."""
from __future__ import annotations

import pytest

from exactkv.research.llamacpp_subprocess import (
    extract_continuation_text,
    parse_llama_token_ids,
    strip_llama_output,
    tokenizer_ids_match,
)


def test_parse_llama_token_ids_from_list_line() -> None:
    stdout = "noise\n[785, 6722, 315, 9625, 374]\n"
    assert parse_llama_token_ids(stdout) == [785, 6722, 315, 9625, 374]


def test_strip_llama_output_removes_perf_lines() -> None:
    raw = (
        "The capital of France is Paris.\n"
        "0.02.993.609 I common_perf_print: prompt eval time = 79 ms\n"
    )
    assert "perf_print" not in strip_llama_output(raw)
    assert "Paris" in strip_llama_output(raw)


def test_extract_continuation_text() -> None:
    prompt = "The capital of France is"
    full = "The capital of France is Paris."
    assert extract_continuation_text(full, prompt) == " Paris."


def test_tokenizer_ids_match() -> None:
    assert tokenizer_ids_match([1, 2, 3], [1, 2, 3])
    assert not tokenizer_ids_match([1, 2], [1, 3])


def test_parse_llama_token_ids_raises_on_missing_list() -> None:
    with pytest.raises(ValueError):
        parse_llama_token_ids("no ids here")
