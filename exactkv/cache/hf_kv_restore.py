"""HF full-KV capture, storage, reload, and continuation smoke (Phase 12A).

Real ``past_key_values`` round-trip through ``KVStorageBackend``. **Not** wired into
``ExactKVGenerator`` or default runtime.

This is a full-KV restore smoke, not a serving runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.storage import (
    KVStorageBackend,
    KVStorageHandle,
    KVStorageMetadata,
    build_verifier_storage_metadata,
    iter_tensors,
    summarize_tensor_payload,
)
from exactkv.cache.utils import (
    _detect_format,
    extract_kv_tensors,
    kv_seq_len,
    rebuild_cache,
)
from exactkv.runtime.model_runtime import ModelRuntime

EXPERIMENT_046_ID = "exp046_full_kv_restore_smoke"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_NEW_TOKENS = 8

CLAIM_NOTE = (
    "Full-KV restore smoke (Phase 12A). Storage round-trip on real HF past_key_values "
    "only — not vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, memory, or production-serving claim."
)

FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "production serving",
    "vericache throughput reproduced",
    "full vericache reproduction",
)


class HfKvRestoreError(Exception):
    """Raised when HF KV capture/restore cannot proceed."""


@dataclass
class HfCacheSummary:
    """Shape/dtype accounting for a HF ``past_key_values`` object."""

    cache_format: str
    layer_count: int
    seq_len: int
    dtype_summary: str
    shape_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrefillKVCapture:
    """Prefill snapshot before any decode continuation."""

    model_name: str
    device: str
    dtype: str
    prompt_ids: torch.Tensor
    next_token_id: int
    past_key_values: Any
    cache_summary: HfCacheSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "prompt_ids": self.prompt_ids.detach().cpu().tolist(),
            "next_token_id": self.next_token_id,
            "cache_summary": self.cache_summary.to_dict(),
        }


@dataclass
class ContinuationResult:
    """Greedy continuation from a KV cache state."""

    token_ids: list[int]
    decoded_text: str
    stopped_on_eos: bool
    final_past_key_values: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "decoded_text": self.decoded_text,
            "stopped_on_eos": self.stopped_on_eos,
        }


@dataclass
class RestorePromptResult:
    """Per-prompt restore equivalence result."""

    prompt_id: str
    prompt: str
    backend_name: str
    cache_format: str
    token_exact_match: bool
    live_token_ids: list[int]
    restored_token_ids: list[int]
    live_decoded: str
    restored_decoded: str
    first_divergence_idx: int | None = None
    restore_blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hf_cache_format(past_key_values: Any) -> str:
    """Detect supported HF cache format or raise ``HfKvRestoreError``."""
    try:
        return _detect_format(past_key_values)
    except TypeError as exc:
        raise HfKvRestoreError(
            f"Unsupported HF past_key_values format: {type(past_key_values).__name__}. "
            "Supported: tuple legacy, DynamicCache v4/v5."
        ) from exc


def summarize_hf_cache(past_key_values: Any) -> HfCacheSummary:
    """Summarize layer count, sequence length, dtypes, and shapes."""
    cache_format = detect_hf_cache_format(past_key_values)
    k_tensors, v_tensors, _ = extract_kv_tensors(past_key_values)
    layer_count = len(k_tensors)
    seq_len = kv_seq_len(past_key_values)
    count, _, dtype_summary, shape_summary = summarize_tensor_payload(
        {"k": k_tensors, "v": v_tensors}
    )
    if count != layer_count * 2:
        dtype_summary = ",".join(sorted({str(t.dtype) for t in k_tensors + v_tensors}))
        shapes = [tuple(t.shape) for t in k_tensors[:4]]
        shape_summary = repr(shapes)
    return HfCacheSummary(
        cache_format=cache_format,
        layer_count=layer_count,
        seq_len=seq_len,
        dtype_summary=dtype_summary,
        shape_summary=shape_summary,
    )


def build_storage_payload_from_cache(
    capture: PrefillKVCapture,
) -> dict[str, Any]:
    """Build a storable payload with cloned KV tensors and restore metadata."""
    k_tensors, v_tensors, cache_format = extract_kv_tensors(capture.past_key_values)
    return {
        "schema_version": "1",
        "cache_format": cache_format,
        "k_tensors": [t.detach().clone() for t in k_tensors],
        "v_tensors": [t.detach().clone() for t in v_tensors],
        "seq_len": capture.cache_summary.seq_len,
        "prompt_ids": capture.prompt_ids.detach().clone(),
        "next_token_id": capture.next_token_id,
        "model_name": capture.model_name,
        "device": capture.device,
        "dtype": capture.dtype,
    }


def restore_cache_from_storage_payload(
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> tuple[Any, int]:
    """Rebuild ``past_key_values`` and ``next_token_id`` from a stored payload."""
    cache_format = str(payload["cache_format"])
    k_tensors = [t.to(device) if device is not None else t for t in payload["k_tensors"]]
    v_tensors = [t.to(device) if device is not None else t for t in payload["v_tensors"]]
    seq_len = int(payload.get("seq_len", kv_seq_len(rebuild_cache(k_tensors, v_tensors, cache_format, 0))))
    cache = rebuild_cache(k_tensors, v_tensors, cache_format, seq_len)
    next_token_id = int(payload["next_token_id"])
    return cache, next_token_id


@torch.no_grad()
def capture_prefill_kv(runtime: ModelRuntime, prompt: str) -> PrefillKVCapture:
    """Tokenize prompt, run prefill with ``use_cache=True``, capture real KV."""
    prompt_ids = runtime.encode(prompt)
    out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    if out.past_key_values is None:
        raise HfKvRestoreError("Prefill forward returned past_key_values=None")
    summary = summarize_hf_cache(out.past_key_values)
    next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
    return PrefillKVCapture(
        model_name=runtime.model_name,
        device=str(runtime.device),
        dtype=str(runtime.dtype),
        prompt_ids=prompt_ids,
        next_token_id=next_token_id,
        past_key_values=out.past_key_values,
        cache_summary=summary,
    )


@torch.no_grad()
def continue_greedy_from_cache(
    runtime: ModelRuntime,
    past_key_values: Any,
    next_token_id: int,
    max_new_tokens: int,
) -> ContinuationResult:
    """Generate up to ``max_new_tokens`` greedily from an existing KV cache."""
    generated: list[int] = []
    cache = past_key_values
    token = next_token_id
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        generated.append(token)
        if token == runtime.eos_token_id:
            stopped_on_eos = True
            break
        tok_tensor = torch.tensor([[token]], dtype=torch.long, device=runtime.device)
        step_out = runtime.forward(
            input_ids=tok_tensor,
            past_key_values=cache,
            use_cache=True,
        )
        cache = step_out.past_key_values
        token = int(step_out.logits[:, -1, :].argmax(dim=-1).item())

    gen_tensor = torch.tensor([generated], dtype=torch.long, device=runtime.device)
    return ContinuationResult(
        token_ids=generated,
        decoded_text=runtime.decode(gen_tensor),
        stopped_on_eos=stopped_on_eos,
        final_past_key_values=cache,
    )


def first_divergence_index(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def store_prefill_payload(
    backend: KVStorageBackend,
    handle: KVStorageHandle,
    payload: dict[str, Any],
    *,
    residency: CacheResidency,
) -> None:
    """Store cloned prefill payload via ``KVStorageBackend``."""
    metadata = build_verifier_storage_metadata(
        payload,
        residency=residency,
        backend_name=getattr(backend, "_backend_name", "full_kv_storage"),
        claim_note=CLAIM_NOTE,
    )
    backend.put(handle, payload, metadata)


def run_restore_equivalence_for_prompt(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    backend: KVStorageBackend,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> RestorePromptResult:
    """Capture, store, reload, and compare greedy continuations."""
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    try:
        capture = capture_prefill_kv(runtime, prompt)
        payload = build_storage_payload_from_cache(capture)
        live_cache, live_next = restore_cache_from_storage_payload(
            payload, device=runtime.device
        )
        live = continue_greedy_from_cache(
            runtime, live_cache, live_next, max_new_tokens
        )

        handle = KVStorageHandle(
            namespace=f"exp046/{capture.model_name}",
            key=prompt_id,
            version="1",
        )
        residency = (
            CacheResidency.CPU
            if backend_name == "in_memory_kv_storage"
            else CacheResidency.DISK
        )
        store_prefill_payload(backend, handle, payload, residency=residency)
        loaded = backend.get(handle).payload
        restored_cache, restored_next = restore_cache_from_storage_payload(
            loaded, device=runtime.device
        )
        restored = continue_greedy_from_cache(
            runtime, restored_cache, restored_next, max_new_tokens
        )

        div = first_divergence_index(live.token_ids, restored.token_ids)
        return RestorePromptResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            cache_format=capture.cache_summary.cache_format,
            token_exact_match=div is None and live.token_ids == restored.token_ids,
            live_token_ids=live.token_ids,
            restored_token_ids=restored.token_ids,
            live_decoded=live.decoded_text,
            restored_decoded=restored.decoded_text,
            first_divergence_idx=div,
        )
    except HfKvRestoreError as exc:
        return RestorePromptResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            cache_format="unknown",
            token_exact_match=False,
            live_token_ids=[],
            restored_token_ids=[],
            live_decoded="",
            restored_decoded="",
            restore_blocker=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — smoke must report blockers
        return RestorePromptResult(
            prompt_id=prompt_id,
            prompt=prompt,
            backend_name=backend_name,
            cache_format="unknown",
            token_exact_match=False,
            live_token_ids=[],
            restored_token_ids=[],
            live_decoded="",
            restored_decoded="",
            restore_blocker=f"{type(exc).__name__}: {exc}",
        )


def validate_exp046_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 046 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends_tested",
        "cache_format_detected",
        "token_exact_match_count",
        "failures_count",
        "per_prompt",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_046_ID:
        errors.append("experiment_id must be exp046_full_kv_restore_smoke")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    return errors


def default_smoke_prompts() -> list[dict[str, str]]:
    """Tiny deterministic prompt panel (2–4 prompts)."""
    return [
        {"prompt_id": "restore_001", "prompt": "What is 2+2? Answer in one word."},
        {"prompt_id": "restore_002", "prompt": "Name the capital of France."},
        {"prompt_id": "restore_003", "prompt": "List three primary colors."},
        {"prompt_id": "restore_004", "prompt": "Say hello in one short sentence."},
    ]
