"""HF full-KV capture, storage, reload, and continuation smoke (Phase 12A–12B).

Real ``past_key_values`` round-trip through ``KVStorageBackend``. **Not** wired into
``ExactKVGenerator`` or default runtime.

This is a full-KV restore smoke/panel, not a serving runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import transformers

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
EXPERIMENT_047_ID = "exp047_full_kv_restore_panel"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_PANEL_MAX_NEW_TOKENS = 12

CLAIM_NOTE = (
    "Full-KV restore smoke (Phase 12A). Storage round-trip on real HF past_key_values "
    "only — not vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, memory, or production-serving claim."
)

PANEL_CLAIM_NOTE = (
    "Full-KV restore panel (Phase 12B). Multi-prompt storage round-trip on real HF "
    "past_key_values only — not vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

FORBIDDEN_CLAIMS = (
    "speedup",
    "latency improvement",
    "throughput improvement",
    "memory savings",
    "active memory savings",
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


@dataclass
class DeviceDtypeConfig:
    """Device/dtype matrix entry for panel runs."""

    device: str
    dtype: str
    required: bool
    status: str = "pending"  # pending | tested | skipped
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestorePanelCellResult:
    """Per-cell restore equivalence result with hardened metadata."""

    prompt_id: str
    prompt: str
    category: str
    backend_name: str
    device: str
    dtype: str
    cache_format: str
    prompt_length: int
    continuation_token_count: int
    layer_count: int
    shape_summary: str
    dtype_summary: str
    payload_byte_summary: int
    token_exact_match: bool
    live_token_ids: list[int]
    restored_token_ids: list[int]
    live_decoded: str
    restored_decoded: str
    first_divergence_idx: int | None = None
    restore_blocker: str = ""
    cell_status: str = "passed"  # passed | failed | skipped

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


def payload_byte_summary(payload: dict[str, Any]) -> int:
    """Return total tensor payload bytes for a storage payload."""
    _, total_bytes, _, _ = summarize_tensor_payload(payload)
    return total_bytes


def resolve_panel_device_dtype_configs() -> list[DeviceDtypeConfig]:
    """Return required CPU float32 plus optional CUDA configs when available."""
    configs = [
        DeviceDtypeConfig(device="cpu", dtype="float32", required=True, status="pending"),
    ]
    if torch.cuda.is_available():
        configs.append(
            DeviceDtypeConfig(
                device="cuda", dtype="float16", required=False, status="pending"
            )
        )
        if torch.cuda.is_bf16_supported():
            configs.append(
                DeviceDtypeConfig(
                    device="cuda", dtype="bfloat16", required=False, status="pending"
                )
            )
    else:
        configs.extend(
            [
                DeviceDtypeConfig(
                    device="cuda",
                    dtype="float16",
                    required=False,
                    status="skipped",
                    skip_reason="CUDA unavailable",
                ),
                DeviceDtypeConfig(
                    device="cuda",
                    dtype="bfloat16",
                    required=False,
                    status="skipped",
                    skip_reason="CUDA unavailable",
                ),
            ]
        )
    return configs


def default_panel_prompts() -> list[dict[str, str]]:
    """Deterministic 8–16 prompt panel covering common continuation styles."""
    return [
        {
            "prompt_id": "panel_001",
            "category": "short_natural",
            "prompt": "The weather today is sunny and",
        },
        {
            "prompt_id": "panel_002",
            "category": "short_natural",
            "prompt": "What is the square root of 64? Answer:",
        },
        {
            "prompt_id": "panel_003",
            "category": "structured_json",
            "prompt": 'Return JSON only: {"name": "Ada", "role":',
        },
        {
            "prompt_id": "panel_004",
            "category": "structured_json",
            "prompt": 'List as JSON array: ["red", "green",',
        },
        {
            "prompt_id": "panel_005",
            "category": "retrieval_copy",
            "prompt": (
                "Passage: The capital of France is Paris. "
                "Repeat exactly: The capital is"
            ),
        },
        {
            "prompt_id": "panel_006",
            "category": "retrieval_copy",
            "prompt": "Source: Mount Everest is 8849 meters tall. Quote the height:",
        },
        {
            "prompt_id": "panel_007",
            "category": "code_like",
            "prompt": "def factorial(n):\n    if n <= 1:\n        return",
        },
        {
            "prompt_id": "panel_008",
            "category": "code_like",
            "prompt": "import json\n\ndata = {'key': 'value'}\nprint(",
        },
        {
            "prompt_id": "panel_009",
            "category": "long_context_summary",
            "prompt": (
                "Background: ExactKV uses a full-KV verifier to correct lossy draft "
                "KV drift while preserving greedy output on tested panels. "
                "In one sentence, summarize the verifier role:"
            ),
        },
        {
            "prompt_id": "panel_010",
            "category": "long_context_summary",
            "prompt": (
                "Notes: Phase 12 stores real HF past_key_values through pluggable "
                "backends and reloads them for continuation equivalence checks. "
                "Summarize the storage step in one short phrase:"
            ),
        },
        {
            "prompt_id": "panel_011",
            "category": "tool_call_style",
            "prompt": '<tool_call>{"name": "search", "args": {"query":',
        },
        {
            "prompt_id": "panel_012",
            "category": "tool_call_style",
            "prompt": 'Use function get_weather(city="London") to fetch',
        },
    ]


def run_restore_equivalence_for_prompt(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    backend: KVStorageBackend,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    namespace_prefix: str = "exp046",
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
            namespace=f"{namespace_prefix}/{capture.model_name}",
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


@torch.no_grad()
def run_restore_panel_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt: str,
    category: str,
    backend: KVStorageBackend,
    max_new_tokens: int = DEFAULT_PANEL_MAX_NEW_TOKENS,
    namespace_prefix: str = "exp047",
) -> RestorePanelCellResult:
    """Capture, store, reload, and compare greedy continuations with panel metadata."""
    backend_name = getattr(backend, "_backend_name", type(backend).__name__)
    device = str(runtime.device)
    dtype = str(runtime.dtype)
    try:
        capture = capture_prefill_kv(runtime, prompt)
        payload = build_storage_payload_from_cache(capture)
        prompt_length = int(capture.prompt_ids.shape[-1])
        layer_count = capture.cache_summary.layer_count
        shape_summary = capture.cache_summary.shape_summary
        dtype_summary = capture.cache_summary.dtype_summary
        byte_summary = payload_byte_summary(payload)

        live_cache, live_next = restore_cache_from_storage_payload(
            payload, device=runtime.device
        )
        live = continue_greedy_from_cache(
            runtime, live_cache, live_next, max_new_tokens
        )

        handle = KVStorageHandle(
            namespace=f"{namespace_prefix}/{capture.model_name}/{device}/{dtype}",
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
        exact = div is None and live.token_ids == restored.token_ids
        return RestorePanelCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            device=device,
            dtype=dtype,
            cache_format=capture.cache_summary.cache_format,
            prompt_length=prompt_length,
            continuation_token_count=len(live.token_ids),
            layer_count=layer_count,
            shape_summary=shape_summary,
            dtype_summary=dtype_summary,
            payload_byte_summary=byte_summary,
            token_exact_match=exact,
            live_token_ids=live.token_ids,
            restored_token_ids=restored.token_ids,
            live_decoded=live.decoded_text,
            restored_decoded=restored.decoded_text,
            first_divergence_idx=div,
            cell_status="passed" if exact else "failed",
        )
    except HfKvRestoreError as exc:
        return RestorePanelCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            device=device,
            dtype=dtype,
            cache_format="unknown",
            prompt_length=0,
            continuation_token_count=0,
            layer_count=0,
            shape_summary="",
            dtype_summary="",
            payload_byte_summary=0,
            token_exact_match=False,
            live_token_ids=[],
            restored_token_ids=[],
            live_decoded="",
            restored_decoded="",
            restore_blocker=str(exc),
            cell_status="failed",
        )
    except Exception as exc:  # noqa: BLE001 — panel must report blockers
        return RestorePanelCellResult(
            prompt_id=prompt_id,
            prompt=prompt,
            category=category,
            backend_name=backend_name,
            device=device,
            dtype=dtype,
            cache_format="unknown",
            prompt_length=0,
            continuation_token_count=0,
            layer_count=0,
            shape_summary="",
            dtype_summary="",
            payload_byte_summary=0,
            token_exact_match=False,
            live_token_ids=[],
            restored_token_ids=[],
            live_decoded="",
            restored_decoded="",
            restore_blocker=f"{type(exc).__name__}: {exc}",
            cell_status="failed",
        )


def reconcile_panel_cell_counts(report: dict[str, Any]) -> list[str]:
    """Verify passed + failed + skipped == total_cells."""
    errors: list[str] = []
    per_cell = report.get("per_cell", [])
    passed = int(report.get("passed_cells", 0))
    failed = int(report.get("failed_cells", 0))
    skipped = int(report.get("skipped_cells", 0))
    total = int(report.get("total_cells", 0))
    if passed + failed + skipped != total:
        errors.append(
            f"cell count mismatch: passed({passed})+failed({failed})+skipped({skipped})"
            f" != total_cells({total})"
        )
    status_counts = {"passed": 0, "failed": 0, "skipped": 0}
    for cell in per_cell:
        status = cell.get("cell_status", "")
        if status in status_counts:
            status_counts[status] += 1
    if status_counts["passed"] != passed:
        errors.append("passed_cells does not match per_cell passed count")
    if status_counts["failed"] != failed:
        errors.append("failed_cells does not match per_cell failed count")
    if status_counts["skipped"] != skipped:
        errors.append("skipped_cells does not match per_cell skipped count")
    return errors


def validate_exp047_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 047 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "transformers_version",
        "total_cells",
        "passed_cells",
        "failed_cells",
        "skipped_cells",
        "storage_backends_tested",
        "device_dtype_configs_tested",
        "cache_formats_detected",
        "aggregate_exactness",
        "per_cell",
        "restore_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_047_ID:
        errors.append("experiment_id must be exp047_full_kv_restore_panel")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    errors.extend(reconcile_panel_cell_counts(report))
    agg = report.get("aggregate_exactness", {})
    if isinstance(agg, dict):
        if int(agg.get("token_exact_match_count", -1)) < 0:
            errors.append("aggregate_exactness.token_exact_match_count required")
    else:
        errors.append("aggregate_exactness must be a dict")
    for cell in report.get("per_cell", []):
        for field in (
            "prompt_id",
            "backend_name",
            "device",
            "dtype",
            "cache_format",
            "cell_status",
        ):
            if field not in cell:
                errors.append(f"per_cell missing field: {field}")
        div = cell.get("first_divergence_idx")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence_idx must be int or null")
    return errors


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
