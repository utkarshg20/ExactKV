"""V11 Phase 3: ServingSidecarProbe gate tests."""
from __future__ import annotations

import copy
import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import exactkv.compressors  # noqa: F401

from exactkv.compressors import get_compressor
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.serving.cache_lifecycle import AUTHORITATIVE_FULL, COMPRESSED_DRAFT
from exactkv.serving.sidecar_probe import (
    PROBE_INVARIANTS,
    ServingSidecarProbe,
    run_exactkv_with_sidecar_probe,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden_fields(obj: object) -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        assert not hits, f"Forbidden fields {hits}"
        for v in obj.values():
            _assert_no_forbidden_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_forbidden_fields(item)


def _make_fake_full_state(seq_len: int = 6) -> "FullKVState":
    from exactkv.cache.full_state import FullKVState

    pkv = tuple(
        (
            torch.randn(1, 1, seq_len, 4, dtype=torch.float32),
            torch.randn(1, 1, seq_len, 4, dtype=torch.float32),
        )
        for _ in range(2)
    )
    return FullKVState(
        past_key_values=pkv,
        prompt_ids=torch.zeros(1, seq_len, dtype=torch.long),
        generated_ids=torch.zeros(1, 0, dtype=torch.long),
        full_sequence_ids=torch.zeros(1, seq_len, dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float32,
        metadata={"next_token_id": 1},
    )


class TestSidecarProbeUnit:
    def test_probe_invariants_list_is_stable(self):
        assert "verification_uses_authoritative_full" in PROBE_INVARIANTS
        assert len(PROBE_INVARIANTS) >= 5

    def test_attach_does_not_mutate_states(self):
        full = _make_fake_full_state()
        snap = copy.deepcopy(full)
        compressor = get_compressor("noop")
        compressed = compressor.compress(full)
        comp_snap = copy.deepcopy(compressed)

        probe = ServingSidecarProbe()
        probe.attach_prefill(full, compressed, compressor=compressor)

        assert full.seq_len == snap.seq_len
        assert compressed.logical_seq_len == comp_snap.logical_seq_len

    def test_observe_commit_round_advances_logical_lengths(self):
        full = _make_fake_full_state(seq_len=4)
        compressor = get_compressor("noop")
        compressed = compressor.compress(full)

        probe = ServingSidecarProbe()
        probe.attach_prefill(full, compressed, compressor=compressor)
        probe.observe_commit_round(0, 2)
        probe.observe_commit_round(1, 1)

        summary = probe.finalize()
        assert summary["round_count"] == 2
        assert summary["verification_uses"] == AUTHORITATIVE_FULL
        assert summary["owners_separate"] is True
        assert summary["all_invariants_pass"] is True
        _assert_no_forbidden_fields(summary)

    def test_finalize_reports_compressed_draft_separate(self):
        full = _make_fake_full_state(seq_len=5)
        compressor = get_compressor("int8")
        compressed = compressor.compress(full)

        probe = ServingSidecarProbe()
        probe.attach_prefill(full, compressed, compressor=compressor)
        summary = probe.finalize()

        checks = summary["invariant_checks"]
        assert checks["compressed_draft_separate"] is True
        assert checks["sidecar_observational_only"] is True
        assert summary["probe_role"] == "metadata_only_sidecar"

    def test_observe_before_attach_raises(self):
        probe = ServingSidecarProbe()
        with pytest.raises(RuntimeError, match="not attached"):
            probe.observe_commit_round(0, 1)


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime

    return ModelRuntime(MODEL_NAME, device="cpu", dtype=DTYPE)


class TestSidecarProbeIntegration:
    @pytest.mark.parametrize(
        "compressor_name",
        ["noop", "int8", "k8_v4_boundary4_v8_sim", "backend_passthrough"],
    )
    def test_run_exactkv_with_sidecar_probe(self, runtime, compressor_name):
        compressor = get_compressor(compressor_name)
        prompt = "The capital of France is"

        result, probe_summary = run_exactkv_with_sidecar_probe(
            runtime,
            prompt,
            compressor,
            draft_len=4,
            max_new_tokens=8,
        )

        full_out = generate_full_greedy(runtime, prompt, max_new_tokens=8)
        assert token_exact_match(full_out.generated_ids, result.output_ids)
        assert probe_summary["probe_outcome"] == "sidecar_probe_pass"
        assert probe_summary["exactkv_token_match"] is True
        assert probe_summary["verification_uses"] == AUTHORITATIVE_FULL
        assert probe_summary["round_count"] == len(result.traces)
        _assert_no_forbidden_fields(probe_summary)

    def test_probe_matches_harness_exactness_smoke(self, runtime):
        prompt = "In machine learning, a neural network is"
        compressor = get_compressor("int8")
        probe = ServingSidecarProbe()

        full_state = prefill_to_full_state(runtime, prompt)
        compressed = compressor.compress(full_state)
        probe.attach_prefill(full_state, compressed, compressor=compressor)

        gen = ExactKVGenerator(runtime, compressor, draft_len=4)
        result = gen.generate(prompt, max_new_tokens=8)
        for idx, trace in enumerate(result.traces):
            committed = trace.acceptance.num_accepted
            if trace.acceptance.correction_token is not None:
                committed += 1
            probe.observe_commit_round(idx, committed)

        summary = probe.finalize()
        full_out = generate_full_greedy(runtime, prompt, max_new_tokens=8)
        assert token_exact_match(full_out.generated_ids, result.output_ids)
        assert summary["all_invariants_pass"] is True
