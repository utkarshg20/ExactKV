"""Restricted experimental KVPress SnapKVPress adapter (isolated [kvpress] env only).

NOT registered in the default compressor registry.  kvpress is imported lazily
inside this module only when ``KVPressSnapKVAdapter`` is constructed.

Scope: SnapKVPress only via kvpress replay prefill.  This is a **restricted
experimental** adapter — not a claim of paper-exact SnapKV or production SnapKV.
External kvpress/SnapKV results are not ExactKV results.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import torch

from exactkv.cache.utils import kv_seq_len, kv_total_bytes
from exactkv.compressors.backend_adapter import BackendAdapter
from exactkv.compressors.base import CompressorCapabilities
from exactkv.cache.full_state import FullKVState
from exactkv.compressors.kvpress_knorm import (
    assert_no_attention_forward_hooks,
    count_attention_forward_hooks,
    _iter_attention_modules,
)
from exactkv.runtime.model_runtime import ModelRuntime


class KVPressSnapKVAdapter(BackendAdapter):
    """Experimental SnapKVPress adapter using replay prefill under ``with press(model):``.

    Mirrors ``KVPressKnormAdapter`` safety isolation: compressed/pruned KV for
    drafting only; full-KV verifier remains authoritative.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        compression_ratio: float = 0.5,
        *,
        window_size: int = 64,
        kernel_size: int = 5,
        isolate_compression_model: bool = True,
    ) -> None:
        import importlib.metadata

        self._runtime = runtime
        self._compression_ratio = compression_ratio
        self._window_size = window_size
        self._kernel_size = kernel_size
        if isolate_compression_model:
            self._compression_model = copy.deepcopy(runtime.model)
            self._compression_model.eval()
        else:
            self._compression_model = runtime.model

        from kvpress import SnapKVPress  # noqa: PLC0415

        self._SnapKVPress = SnapKVPress
        self.name = "snapkv_experimental"
        try:
            kvpress_version = importlib.metadata.version("kvpress")
        except importlib.metadata.PackageNotFoundError:
            kvpress_version = "unknown"
        self.capabilities = CompressorCapabilities(
            name="snapkv_experimental",
            compressor_type="token_dropping",
            is_simulated=False,
            supports_real_bytes_claim=True,
            supports_token_dropping=True,
            supports_quantization=False,
            key_bit_width=None,
            value_bit_width=None,
            asymmetric=False,
            backend_name="kvpress",
            backend_version=kvpress_version,
            adapter_name="KVPressSnapKVAdapter",
            adapter_version="0.1.0",
            notes=(
                "Restricted experimental SnapKVPress adapter ([kvpress] env only). "
                "Replay prefill under with press(model):. NOT paper-exact SnapKV "
                "unless verified against reference behavior. Factory-only; not "
                "production-ready. Draft may diverge from full KV; verification "
                "uses authoritative full state only. window_size is clamped to "
                "seq_len-1 when shorter than configured (kvpress API). No active "
                "GPU memory savings claim."
            ),
        )

    def _compresses_via_full_state(self) -> bool:
        return True

    @contextmanager
    def verification_mode(self):
        """Assert no kvpress forward hooks are active during verification."""
        assert_no_attention_forward_hooks(self._runtime.model)
        try:
            yield
        finally:
            assert_no_attention_forward_hooks(self._runtime.model)

    def _effective_window_size(self, seq_len: int) -> int:
        """SnapKVPress requires query length > window_size (kvpress API constraint)."""
        if seq_len <= 1:
            raise ValueError(
                f"{self.name}: sequence length {seq_len} too short for SnapKVPress"
            )
        return min(self._window_size, seq_len - 1)

    def _make_press(self, seq_len: int) -> Any:
        window = self._effective_window_size(seq_len)
        return self._SnapKVPress(
            compression_ratio=self._compression_ratio,
            window_size=window,
            kernel_size=self._kernel_size,
        )

    def _backend_compress_from_full_state(self, state: FullKVState) -> dict:
        assert_no_attention_forward_hooks(self._runtime.model)

        from transformers import DynamicCache  # noqa: PLC0415

        effective_window = self._effective_window_size(state.seq_len)
        press = self._make_press(state.seq_len)
        input_ids = state.full_sequence_ids.to(self._runtime.device)
        cache = DynamicCache()
        compress_model = self._compression_model

        hooks_before = count_attention_forward_hooks(compress_model)

        with torch.no_grad():
            with press(compress_model):
                hooks_during = count_attention_forward_hooks(compress_model)
                out = compress_model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                )

        hooks_after = count_attention_forward_hooks(compress_model)
        if hooks_after != hooks_before:
            raise RuntimeError(
                f"kvpress hooks not cleaned up after SnapKV compression replay: "
                f"before={hooks_before}, after={hooks_after}"
            )

        compressed_cache = out.past_key_values
        stored_bytes = kv_total_bytes(compressed_cache)
        physical_seq = kv_seq_len(compressed_cache)
        next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())

        return {
            "cache_format": "dynamic_v5",
            "dynamic_cache": compressed_cache,
            "__stored_kv_bytes__": stored_bytes,
            "__physical_seq_len__": physical_seq,
            "__compressed_next_token_id__": next_token_id,
            "__num_layers__": len(list(_iter_attention_modules(compress_model))),
            "__hook_count_before__": hooks_before,
            "__hook_count_during__": hooks_during,
            "__hook_count_after__": hooks_after,
            "__snapkv_window_size__": self._window_size,
            "__snapkv_effective_window_size__": effective_window,
            "__snapkv_kernel_size__": self._kernel_size,
            "__snapkv_compression_ratio__": self._compression_ratio,
        }

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> dict:
        raise RuntimeError(
            f"{self.name} uses replay compression only; "
            "_backend_compress is not supported"
        )

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        return backend_data["dynamic_cache"]

    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,  # noqa: ARG002
        backend_data: dict,
    ) -> dict:
        stored = backend_data.get(
            "__stored_kv_bytes__",
            kv_total_bytes(backend_data["dynamic_cache"]),
        )
        materialized = stored
        metadata = 0
        temporary = 0
        total = stored + materialized + metadata + temporary
        return {
            "stored_kv_bytes": stored,
            "materialized_working_kv_bytes": materialized,
            "metadata_bytes": metadata,
            "temporary_workspace_bytes": temporary,
            "total_kv_footprint_bytes": total,
        }

    def _get_next_token_id(self, state: FullKVState, backend_data: dict) -> int:
        return backend_data.get("__compressed_next_token_id__", state.next_token_id)


def create_snapkv_experimental_adapter(
    runtime: ModelRuntime,
    compression_ratio: float = 0.5,
    *,
    window_size: int = 64,
    kernel_size: int = 5,
    isolate_compression_model: bool = True,
) -> KVPressSnapKVAdapter:
    """Factory for the restricted SnapKV experimental adapter (not in default registry)."""
    return KVPressSnapKVAdapter(
        runtime=runtime,
        compression_ratio=compression_ratio,
        window_size=window_size,
        kernel_size=kernel_size,
        isolate_compression_model=isolate_compression_model,
    )
