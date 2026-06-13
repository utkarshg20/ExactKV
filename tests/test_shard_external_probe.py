"""Tests for Experiment 038 Shard external-drafter feasibility probe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "probe_shard_external_drafter.py"
_DOC = _ROOT / "docs" / "EXPERIMENT_038_SHARD_EXTERNAL_DRAFTER_PROBE.md"


class _Tok:
    def __init__(self, prefix: int) -> None:
        self._prefix = prefix

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._prefix] + [ord(c) % 97 for c in text[:8]]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(97 + (i % 26)) for i in ids)


def _run_probe(
    *,
    env: dict[str, str] | None = None,
    extra: tuple[str, ...] = (),
    json_out: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    import os

    merged = {k: v for k, v in os.environ.items() if k != "SHARD_REPO_PATH"}
    if env:
        merged.update(env)
    cmd = [sys.executable, str(_SCRIPT), *extra]
    if json_out is not None:
        cmd.extend(["--json-out", str(json_out)])
    return subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=merged,
        check=False,
    )


def test_script_exits_cleanly_without_shard_repo(tmp_path: Path) -> None:
    out_path = tmp_path / "probe.json"
    result = _run_probe(json_out=out_path)
    assert result.returncode == 0
    assert "blocked" in result.stdout.lower()
    assert "Shard repo not provided" in result.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["probe_status"] == "blocked"
    assert report["shard_repo_path_present"] is False


def test_blocked_report_shape(tmp_path: Path) -> None:
    from exactkv.external.shard_probe import blocked_report, validate_report_shape

    report = blocked_report(
        reason="blocked: test",
        shard_repo_path_present=False,
    )
    validate_report_shape(report)
    assert report["claims_forbidden"]
    assert "not integrated" in report["claims_forbidden"][0].lower()


def test_token_alignment_rejects_mismatched_tokenizers() -> None:
    from exactkv.external.shard_probe import check_tokenizer_alignment

    a = _Tok(prefix=1)
    b = _Tok(prefix=9)
    result = check_tokenizer_alignment(a, b, "hello probe")
    assert result["prompt_aligned"] is False
    assert result["alignment_pass"] is False


def test_token_alignment_accepts_matching_tokenizers() -> None:
    from exactkv.external.shard_probe import check_tokenizer_alignment

    tok = _Tok(prefix=3)
    result = check_tokenizer_alignment(tok, tok, "aligned prompt")
    assert result["prompt_aligned"] is True
    assert result["alignment_pass"] is True


def test_compare_token_sequences_first_divergence() -> None:
    from exactkv.external.shard_probe import compare_token_sequences

    cmp = compare_token_sequences([10, 11, 12, 13], [10, 11, 99, 13])
    assert cmp["accepted_prefix_length"] == 2
    assert cmp["first_divergence_index"] == 2
    assert cmp["draft_token_id"] == 99
    assert cmp["verifier_token_id"] == 12
    assert cmp["correction_needed"] is True
    assert cmp["exact_match"] is False


def test_docs_mention_restricted_feasibility() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "external-drafter feasibility" in text.lower()
    assert "not integrated as a default ExactKV compressor" in text
    assert "External Shard claims are not ExactKV results" in text
    assert "restricted_no_go" in text


@pytest.mark.parametrize(
    "forbidden",
    [
        "2x speedup",
        "10× memory",
        "production serving ready",
        "accuracy improvement over",
    ],
)
def test_docs_avoid_positive_forbidden_claims(forbidden: str) -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert forbidden.lower() not in text


def test_prompt_ids_comparable_bos_prefix() -> None:
    from exactkv.external.shard_probe import prompt_ids_comparable

    class _Tok:
        bos_token_id = 128000

    tok = _Tok()
    assert prompt_ids_comparable([1, 2, 3], [128000, 1, 2, 3], tok) is True
    assert prompt_ids_comparable([1, 2, 3], [1, 2, 3], tok) is True
    assert prompt_ids_comparable([1, 2, 3], [9, 1, 2, 3], tok) is False


def test_script_blocked_with_missing_repo_dir(tmp_path: Path) -> None:
    missing = tmp_path / "no_shard_here"
    out_path = tmp_path / "probe.json"
    result = _run_probe(
        env={"SHARD_REPO_PATH": str(missing)},
        json_out=out_path,
    )
    assert result.returncode == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["probe_status"] == "blocked"
    assert report["shard_repo_path_present"] is True
    assert report["shard_import_success"] is False
