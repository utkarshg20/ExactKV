"""V6 Phase C validation gate for KVPressKnormAdapter (``.venv-kvpress`` only).

Runs the full ``core`` prompt suite (34 prompts) with:
  * Qwen/Qwen2.5-0.5B, float32
  * KVPressKnormAdapter, compression_ratio=0.5
  * draft_len=4, max_new_tokens=16

Hard gates:
  * exactkv_output_ids == full_output_ids (exactkv_failures == 0)
  * acceptance / rejection / correction counts reconcile per round
  * logical_seq_len == full_state.seq_len; physical kv_seq_len < logical when pruned
  * verification model hook count == 0 throughout
  * compression model hooks return to baseline after compress
  * workspace fields reconcile; no forbidden performance fields

Run: ``.venv-kvpress/bin/pytest tests/test_kvpress_knorm_validation.py -v``
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import json
import os
import sys
from typing import Any

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_KVPRESS_INSTALLED = importlib.util.find_spec("kvpress") is not None

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
DRAFT_LEN = 4
MAX_NEW_TOKENS = 16
COMPRESSION_RATIO = 0.5
SUITE_NAME = "core"

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
})


def _assert_no_forbidden_fields(d: dict, context: str = "") -> None:
    hits = _FORBIDDEN_FIELDS & d.keys()
    assert not hits, f"Forbidden performance fields {hits} in {context or 'dict'}"


def _env_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("transformers", "kvpress", "torch"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def collect_kvpress_knorm_core_validation_summary() -> dict[str, Any]:
    """Run core-suite validation and return a JSON-serialisable summary."""
    from exactkv.benchmarks.prompts import load_core_prompts
    from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, kv_total_bytes
    from exactkv.compressors.kvpress_knorm import (
        count_attention_forward_hooks,
        create_kvpress_knorm_adapter,
    )
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy
    from exactkv.runtime.model_runtime import ModelRuntime
    from exactkv.runtime.prefill import prefill_to_full_state

    runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)
    adapter = create_kvpress_knorm_adapter(
        runtime, compression_ratio=COMPRESSION_RATIO
    )
    generator = ExactKVGenerator(runtime, adapter, draft_len=DRAFT_LEN)
    prompts = load_core_prompts()

    exactkv_failures = 0
    failure_ids: list[str] = []
    total_accepted = 0
    total_rejected = 0
    total_corrections = 0
    prompts_with_rejections = 0
    prompts_with_corrections = 0
    pruning_observed = 0
    pruning_equal = 0
    hook_safety_ok = True
    workspace_ok = True
    per_prompt: list[dict[str, Any]] = []

    for entry in prompts:
        prompt_id = entry["prompt_id"]
        prompt = entry["prompt"]
        pid_record: dict[str, Any] = {"prompt_id": prompt_id}

        if count_attention_forward_hooks(runtime.model) != 0:
            hook_safety_ok = False
            pid_record["verify_hook_error"] = "hooks active before run"

        full_result = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
        ekv_result = generator.generate(prompt, MAX_NEW_TOKENS)

        full_ids = full_result.generated_ids.squeeze(0).tolist()
        ekv_ids = ekv_result.output_ids.squeeze(0).tolist()
        if ekv_ids != full_ids:
            exactkv_failures += 1
            failure_ids.append(prompt_id)
            pid_record["exactkv_mismatch"] = True
        else:
            pid_record["exactkv_mismatch"] = False

        total_accepted += ekv_result.total_accepted
        total_rejected += ekv_result.total_rejected
        total_corrections += ekv_result.total_corrections
        if ekv_result.total_rejected > 0:
            prompts_with_rejections += 1
        if ekv_result.total_corrections > 0:
            prompts_with_corrections += 1

        pid_record["total_accepted"] = ekv_result.total_accepted
        pid_record["total_rejected"] = ekv_result.total_rejected
        pid_record["total_corrections"] = ekv_result.total_corrections
        pid_record["acceptance_rate"] = ekv_result.acceptance_rate
        pid_record["num_rounds"] = ekv_result.num_rounds

        for i, trace in enumerate(ekv_result.traces):
            drafted = len(trace.draft_tokens)
            acc = trace.acceptance.num_accepted
            rej = len(trace.acceptance.rejected_tokens)
            if acc + rej != drafted:
                raise AssertionError(
                    f"{prompt_id} round {i}: accepted({acc}) + rejected({rej}) "
                    f"!= drafted({drafted})"
                )
            if trace.full_seq_len_after != trace.compressed_seq_len_after:
                raise AssertionError(
                    f"{prompt_id} round {i}: alignment broken "
                    f"full={trace.full_seq_len_after} "
                    f"logical={trace.compressed_seq_len_after}"
                )

        if count_attention_forward_hooks(runtime.model) != 0:
            hook_safety_ok = False
            pid_record["verify_hook_error"] = "hooks active after generate"

        state = prefill_to_full_state(runtime, prompt)
        compressed = adapter.compress(state)
        cache = adapter.materialize_for_draft(compressed)
        physical = kv_seq_len(cache)
        logical = state.seq_len

        if compressed.logical_seq_len != logical:
            raise AssertionError(
                f"{prompt_id}: logical_seq_len={compressed.logical_seq_len} "
                f"!= full prefill len={logical}"
            )
        if physical < logical:
            pruning_observed += 1
        elif physical == logical:
            pruning_equal += 1

        pid_record["logical_seq_len"] = logical
        pid_record["physical_seq_len"] = physical

        hooks_before = compressed.data["__hook_count_before__"]
        hooks_during = compressed.data["__hook_count_during__"]
        hooks_after = compressed.data["__hook_count_after__"]
        if hooks_after != hooks_before:
            hook_safety_ok = False
            pid_record["compress_hook_error"] = (
                f"before={hooks_before}, during={hooks_during}, after={hooks_after}"
            )

        stats = adapter.stats(compressed)
        stats_dict = dataclasses.asdict(stats)
        _assert_no_forbidden_fields(stats_dict, f"stats({prompt_id})")
        if stats.materialized_working_kv_bytes != stats.stored_kv_bytes:
            workspace_ok = False
        expected_total = (
            stats.stored_kv_bytes
            + stats.materialized_working_kv_bytes
            + stats.metadata_bytes
            + stats.temporary_workspace_bytes
        )
        if stats.total_kv_footprint_bytes != expected_total:
            workspace_ok = False

        pid_record["stored_kv_bytes"] = stats.stored_kv_bytes
        pid_record["full_bytes"] = stats.full_bytes
        pid_record["supports_real_bytes_claim"] = (
            adapter.capabilities.supports_real_bytes_claim
        )

        per_prompt.append(pid_record)

    denom = total_accepted + total_rejected
    aggregate_acceptance_rate = total_accepted / denom if denom > 0 else 1.0

    return {
        "environment": ".venv-kvpress",
        "versions": _env_versions(),
        "model": MODEL_NAME,
        "compressor": "kvpress_knorm",
        "compression_ratio": COMPRESSION_RATIO,
        "draft_len": DRAFT_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_suite": SUITE_NAME,
        "prompt_count": len(prompts),
        "exactkv_failures": exactkv_failures,
        "failure_prompt_ids": failure_ids,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_corrections": total_corrections,
        "aggregate_acceptance_rate": aggregate_acceptance_rate,
        "prompts_with_rejections": prompts_with_rejections,
        "prompts_with_corrections": prompts_with_corrections,
        "pruning_observed_count": pruning_observed,
        "pruning_equal_count": pruning_equal,
        "hook_safety_ok": hook_safety_ok,
        "workspace_ok": workspace_ok,
        "per_prompt": per_prompt,
    }


@pytest.mark.skipif(not _KVPRESS_INSTALLED, reason="kvpress optional extra not installed")
class TestKVPressKnormCoreValidation:
    @pytest.fixture(scope="module")
    def validation_stack(self):
        from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter
        from exactkv.runtime.exactkv_generator import ExactKVGenerator
        from exactkv.runtime.model_runtime import ModelRuntime

        runtime = ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)
        adapter = create_kvpress_knorm_adapter(
            runtime, compression_ratio=COMPRESSION_RATIO
        )
        generator = ExactKVGenerator(runtime, adapter, draft_len=DRAFT_LEN)
        return runtime, adapter, generator

    def test_core_suite_exactness_and_invariants(self):
        summary = collect_kvpress_knorm_core_validation_summary()
        assert summary["exactkv_failures"] == 0, (
            f"exactkv_failures={summary['exactkv_failures']}: "
            f"{summary['failure_prompt_ids']}"
        )
        assert summary["hook_safety_ok"] is True
        assert summary["workspace_ok"] is True
        assert summary["pruning_observed_count"] > 0, (
            "Expected KnormPress pruning on at least one core prompt"
        )

    def test_full_state_unchanged_during_verification(self, validation_stack):
        from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, kv_total_bytes
        from exactkv.compressors.kvpress_knorm import count_attention_forward_hooks
        from exactkv.runtime.prefill import prefill_to_full_state
        from exactkv.verification.engine import VerificationEngine

        runtime, adapter, _generator = validation_stack
        engine = VerificationEngine(runtime)
        state = prefill_to_full_state(runtime, "The capital of France is Paris")

        full_bytes_before = kv_total_bytes(state.past_key_values)
        seq_before = state.seq_len
        k_before, v_before, _ = extract_kv_tensors(state.past_key_values)
        k_snap = [t.clone() for t in k_before]
        v_snap = [t.clone() for t in v_before]

        adapter.compress(state)
        draft = [state.next_token_id]

        assert count_attention_forward_hooks(runtime.model) == 0
        with adapter.verification_mode():
            assert count_attention_forward_hooks(runtime.model) == 0
            engine.verify_sequential(state, draft)
            assert count_attention_forward_hooks(runtime.model) == 0

        assert kv_total_bytes(state.past_key_values) == full_bytes_before
        assert state.seq_len == seq_before
        assert kv_seq_len(state.past_key_values) == seq_before
        k_after, v_after, _ = extract_kv_tensors(state.past_key_values)
        for orig_k, cur_k in zip(k_snap, k_after):
            assert torch.equal(orig_k, cur_k)
        for orig_v, cur_v in zip(v_snap, v_after):
            assert torch.equal(orig_v, cur_v)

    def test_supports_real_bytes_claim_documents_pruned_cache(self, validation_stack):
        from exactkv.cache.utils import kv_total_bytes
        from exactkv.runtime.prefill import prefill_to_full_state

        _runtime, adapter, _generator = validation_stack
        assert adapter.capabilities.supports_real_bytes_claim is True

        state = prefill_to_full_state(
            _runtime, "The capital of France is Paris and the river"
        )
        compressed = adapter.compress(state)
        stats = adapter.stats(compressed)
        pruned_bytes = kv_total_bytes(compressed.data["dynamic_cache"])

        assert stats.stored_kv_bytes == pruned_bytes
        assert stats.metadata_bytes == 0
        assert stats.stored_kv_bytes < stats.full_bytes


if __name__ == "__main__":
    if not _KVPRESS_INSTALLED:
        print("kvpress not installed", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(collect_kvpress_knorm_core_validation_summary(), indent=2))
