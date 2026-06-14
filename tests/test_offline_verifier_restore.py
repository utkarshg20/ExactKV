"""Tests for Phase 12C offline verifier restore (no model download by default)."""
from __future__ import annotations

from pathlib import Path

from exactkv.cache.offline_verifier import (
    DRAFT_SOURCE_TYPE,
    EXPERIMENT_048_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_VERIFIER_CLAIM_NOTE,
    VERIFIER_SOURCE,
    propose_controlled_draft,
    reconstruct_output_from_acceptances,
    truncate_at_eos,
    validate_exp048_report,
)
from exactkv.verification.acceptance import AcceptanceResult, compute_acceptance

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_048_OFFLINE_VERIFIER_RESTORE_SMOKE.md"
)


def _acceptance(
    draft: list[int],
    verifier: list[int],
) -> AcceptanceResult:
    return compute_acceptance(draft, verifier)


def _synthetic_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "offline_001",
        "prompt": "test",
        "backend_name": "in_memory_kv_storage",
        "cache_format": "dynamic_v5",
        "draft_source_type": DRAFT_SOURCE_TYPE,
        "verifier_source": VERIFIER_SOURCE,
        "live_reference_token_ids": [1, 2, 3, 4],
        "offline_output_token_ids": [1, 2, 3, 4],
        "token_exact_match": True,
        "exactkv_failures": 0,
        "accepted_prefix_lengths": [2, 2],
        "first_divergence_idx": None,
        "restore_blocker": "",
        "verification_blocker": "",
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    cells = [
        _synthetic_cell(),
        _synthetic_cell(
            prompt_id="offline_002",
            backend_name="file_kv_storage",
        ),
    ]
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_048_ID,
        "model": "Qwen/Qwen2.5-0.5B",
        "device": "cpu",
        "dtype": "torch.float32",
        "prompt_count": 2,
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "cache_format": "dynamic_v5",
        "draft_source_type": DRAFT_SOURCE_TYPE,
        "verifier_source": VERIFIER_SOURCE,
        "cells": cells,
        "exactkv_failures": 0,
        "token_exact_match_count": 2,
        "accepted_prefix_lengths": [[2, 2], [1, 3]],
        "first_divergences": [],
        "restore_blockers": [],
        "verification_blockers": [],
        "claim_note": OFFLINE_VERIFIER_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_validate_exp048_report_schema() -> None:
    assert validate_exp048_report(_synthetic_report()) == []


def test_report_counts_reconcile() -> None:
    report = _synthetic_report(
        cells=[
            _synthetic_cell(),
            _synthetic_cell(
                prompt_id="offline_002",
                token_exact_match=False,
                exactkv_failures=1,
                first_divergence_idx=2,
            ),
        ],
        exactkv_failures=1,
        token_exact_match_count=1,
        first_divergences=[{"prompt_id": "offline_002", "first_divergence_idx": 2}],
    )
    assert validate_exp048_report(report) == []


def test_controlled_draft_injects_mismatch_on_odd_rounds() -> None:
    reference = [10, 20, 30, 40, 50]
    even = propose_controlled_draft(reference, 0, 4, round_idx=0, vocab_size=100)
    odd = propose_controlled_draft(reference, 0, 4, round_idx=1, vocab_size=100)
    assert even == [10, 20, 30, 40]
    assert odd[0] == 10
    assert odd[1] != 20
    assert odd[2:] == [30, 40]


def test_accept_correct_reconstructs_reference() -> None:
    reference = [10, 20, 30, 40]
    draft1 = [10, 99, 30, 40]
    verifier1 = [10, 20]
    acc1 = _acceptance(draft1, verifier1)
    assert acc1.accepted_tokens == [10]
    assert acc1.correction_token == 20

    draft2 = [30, 40]
    verifier2 = [30, 40]
    acc2 = _acceptance(draft2, verifier2)
    output = reconstruct_output_from_acceptances([(draft1, acc1), (draft2, acc2)])
    assert output == reference


def test_reconstruct_output_truncates_at_eos() -> None:
    acc = _acceptance([1, 2, 3], [1, 2, 3])
    output = reconstruct_output_from_acceptances([( [1, 2, 3], acc)], eos_token_id=2)
    assert output == [1, 2]


def test_restore_blocker_schema_in_cell() -> None:
    report = _synthetic_report(
        cells=[
            _synthetic_cell(
                restore_blocker="Unsupported HF past_key_values format",
                token_exact_match=False,
                exactkv_failures=1,
            )
        ],
        exactkv_failures=1,
        token_exact_match_count=0,
        restore_blockers=["in_memory/offline_001: Unsupported"],
    )
    assert validate_exp048_report(report) == []


def test_verification_blocker_schema_in_cell() -> None:
    report = _synthetic_report(
        cells=[
            _synthetic_cell(
                verification_blocker="round 0: RuntimeError: boom",
                token_exact_match=False,
                exactkv_failures=1,
            )
        ],
        exactkv_failures=1,
        token_exact_match_count=0,
        verification_blockers=["in_memory/offline_001: round 0: RuntimeError: boom"],
    )
    assert validate_exp048_report(report) == []


def test_first_divergence_schema() -> None:
    cell = _synthetic_cell(
        token_exact_match=False,
        exactkv_failures=1,
        first_divergence_idx=1,
        offline_output_token_ids=[1, 99, 3, 4],
    )
    report = _synthetic_report(
        cells=[cell],
        exactkv_failures=1,
        token_exact_match_count=0,
        first_divergences=[{"prompt_id": "offline_001", "first_divergence_idx": 1}],
    )
    assert validate_exp048_report(report) == []


def test_truncate_at_eos() -> None:
    tokens, found = truncate_at_eos([1, 2, 3, 4], eos_token_id=3)
    assert tokens == [1, 2, 3]
    assert found is True


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline verifier restore smoke",
        "not default runtime integration",
        "isolated experiment path",
        "vllm",
        "lmcache",
        "remote prefix",
        "generation and verification behavior is unchanged",
        "throughput",
        "vericache",
        "active memory savings",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text
