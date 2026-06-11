"""Metadata-only serving sidecar probe for ExactKV compatibility evaluation.

Observes cache lifecycle, ownership, and verification authority during ExactKV
runs.  Wraps :class:`ServingCacheLifecycleHarness` without mutating
authoritative full KV or integrating vLLM, LMCache, or PagedAttention.

This is not production serving infrastructure.  It does not measure throughput,
latency, speedup, runtime, tokens/sec, or active GPU memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.serving.cache_lifecycle import (
    AUTHORITATIVE_FULL,
    COMPRESSED_DRAFT,
    SERVING_HARNESS,
    ServingCacheLifecycleHarness,
)

ProbeOutcome = Literal["sidecar_probe_pass", "sidecar_probe_fail"]

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

PROBE_INVARIANTS = (
    "verification_uses_authoritative_full",
    "owners_separate",
    "sidecar_observational_only",
    "compressed_draft_separate",
    "logical_alignment_maintained",
)


@dataclass
class ProbeRoundObservation:
    """One post-commit observation from the sidecar probe."""

    round_idx: int
    committed_tokens: int
    authoritative_logical_seq_len: int
    compressed_logical_seq_len: int | None
    verification_uses: str
    owners_separate: bool
    invariants_valid: bool


@dataclass
class ServingSidecarProbe:
    """Metadata-only sidecar observing ExactKV cache lifecycle.

    The probe registers authoritative and compressed caches via the harness,
    records per-round observations after commit, and validates that verification
    authority remains on ``authoritative_full``.  It never replaces or mutates
    authoritative full KV storage.
    """

    harness: ServingCacheLifecycleHarness = field(
        default_factory=lambda: ServingCacheLifecycleHarness(block_size=16)
    )
    round_observations: list[ProbeRoundObservation] = field(default_factory=list)
    _attached: bool = False

    def attach_prefill(
        self,
        full_state: FullKVState,
        compressed_state: CompressedKVState,
        *,
        compressor: Any | None = None,
    ) -> None:
        """Register prefill caches for observation without mutating states."""
        self.harness.register_authoritative_full(full_state)
        self.harness.register_compressed_cache(
            compressed_state, compressor=compressor
        )
        self.harness.validate_invariants()
        self._check_verification_authority()
        self._attached = True

    def observe_commit_round(self, round_idx: int, committed_tokens: int) -> None:
        """Record post-commit lifecycle state; advance harness logical lengths."""
        if not self._attached:
            raise RuntimeError("probe not attached; call attach_prefill first")
        if committed_tokens < 0:
            raise ValueError(f"committed_tokens must be non-negative, got {committed_tokens}")

        if committed_tokens > 0:
            self.harness.append_committed_tokens(committed_tokens)
        self.harness.validate_invariants()
        self._check_verification_authority()

        summary = self.harness.summarize()
        obs = ProbeRoundObservation(
            round_idx=round_idx,
            committed_tokens=committed_tokens,
            authoritative_logical_seq_len=summary["authoritative_logical_seq_len"] or 0,
            compressed_logical_seq_len=summary["compressed_logical_seq_len"],
            verification_uses=summary["verification_uses"],
            owners_separate=(
                summary["authoritative_cache_id"] != summary["compressed_cache_id"]
            ),
            invariants_valid=summary["invariants_valid"],
        )
        self.round_observations.append(obs)

    def _check_verification_authority(self) -> None:
        summary = self.harness.summarize()
        if summary["verification_uses"] != AUTHORITATIVE_FULL:
            raise ValueError(
                f"verification must use {AUTHORITATIVE_FULL!r}, "
                f"got {summary['verification_uses']!r}"
            )
        if summary["authoritative_cache_id"] == summary["compressed_cache_id"]:
            raise ValueError(
                "sidecar probe: authoritative and compressed entries must be separate"
            )

    def finalize(self) -> dict[str, Any]:
        """Return JSON-serialisable probe summary without forbidden fields."""
        summary = self.harness.summarize()
        invariant_checks = _collect_invariant_checks(summary)
        all_pass = all(invariant_checks.values())

        result = {
            "probe_outcome": "sidecar_probe_pass" if all_pass else "sidecar_probe_fail",
            "probe_role": "metadata_only_sidecar",
            "harness_owner": SERVING_HARNESS,
            "verification_uses": summary["verification_uses"],
            "authoritative_cache_id": summary["authoritative_cache_id"],
            "compressed_cache_id": summary["compressed_cache_id"],
            "owners_separate": (
                summary["authoritative_cache_id"] != summary["compressed_cache_id"]
            ),
            "round_count": len(self.round_observations),
            "invariant_checks": invariant_checks,
            "all_invariants_pass": all_pass,
            "round_observations": [
                {
                    "round_idx": o.round_idx,
                    "committed_tokens": o.committed_tokens,
                    "authoritative_logical_seq_len": o.authoritative_logical_seq_len,
                    "compressed_logical_seq_len": o.compressed_logical_seq_len,
                    "verification_uses": o.verification_uses,
                    "owners_separate": o.owners_separate,
                    "invariants_valid": o.invariants_valid,
                }
                for o in self.round_observations
            ],
            "note": (
                "Metadata-only sidecar probe; not vLLM or LMCache integration. "
                "Authoritative full-KV verifier remains separate. "
                "No throughput, latency, speedup, runtime, tokens/sec, or "
                "active GPU memory fields."
            ),
        }
        _assert_no_forbidden_fields(result)
        return result


def _assert_no_forbidden_fields(obj: Any, path: str = "probe") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_fields(item, f"{path}[{i}]")


def _collect_invariant_checks(summary: dict[str, Any]) -> dict[str, bool]:
    auth_logical = summary.get("authoritative_logical_seq_len")
    comp_logical = summary.get("compressed_logical_seq_len")
    return {
        "verification_uses_authoritative_full": (
            summary.get("verification_uses") == AUTHORITATIVE_FULL
        ),
        "owners_separate": (
            summary.get("authoritative_cache_id") is not None
            and summary.get("compressed_cache_id") is not None
            and summary["authoritative_cache_id"] != summary["compressed_cache_id"]
        ),
        "sidecar_observational_only": summary.get("harness_owner") == SERVING_HARNESS,
        "compressed_draft_separate": any(
            e.get("owner") == COMPRESSED_DRAFT for e in summary.get("entries", [])
        ),
        "logical_alignment_maintained": (
            auth_logical is not None
            and comp_logical is not None
            and auth_logical == comp_logical
        ),
    }


def run_exactkv_with_sidecar_probe(
    runtime: ModelRuntime,
    prompt: str,
    compressor: Any,
    *,
    draft_len: int = 4,
    max_new_tokens: int = 16,
    block_size: int = 16,
) -> tuple[Any, dict[str, Any]]:
    """Run ExactKV generation with a metadata-only sidecar probe attached.

    Returns ``(ExactKVResult, probe_summary)``.  The probe observes commit
    rounds but does not alter generation or verification behaviour.
    """
    probe = ServingSidecarProbe(
        harness=ServingCacheLifecycleHarness(block_size=block_size)
    )

    full_state = prefill_to_full_state(runtime, prompt)
    compressed = compressor.compress(full_state)
    probe.attach_prefill(full_state, compressed, compressor=compressor)

    gen = ExactKVGenerator(runtime, compressor, draft_len=draft_len)
    result = gen.generate(prompt, max_new_tokens)

    for round_idx, trace in enumerate(result.traces):
        committed = trace.acceptance.num_accepted
        if trace.acceptance.correction_token is not None:
            committed += 1
        probe.observe_commit_round(round_idx, committed)

    probe_summary = probe.finalize()
    probe_summary["exactkv_token_match"] = token_exact_match(
        generate_full_greedy(runtime, prompt, max_new_tokens).generated_ids,
        result.output_ids,
    )
    return result, probe_summary
