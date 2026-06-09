"""Restricted experimental KVPress KnormPress adapter (isolated [kvpress] env only).

NOT registered in the default compressor registry.  kvpress is imported lazily
inside this module only when ``KVPressKnormAdapter`` is constructed.

Scope: KnormPress only.  No KVPressTextGenerationPipeline, DecodingPress,
AdaKVPress, or ComposedPress.
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
from exactkv.runtime.model_runtime import ModelRuntime


def _iter_attention_modules(model: torch.nn.Module):
    """Yield attention submodules (Qwen-style model.model.layers[].self_attn)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            yield attn


def count_attention_forward_hooks(model: torch.nn.Module) -> int:
    """Count active ``_forward_hooks`` on attention modules."""
    return sum(len(attn._forward_hooks) for attn in _iter_attention_modules(model))


def assert_no_attention_forward_hooks(model: torch.nn.Module) -> None:
    """Fail loudly if kvpress (or any) forward hooks are active on attention."""
    count = count_attention_forward_hooks(model)
    if count != 0:
        raise RuntimeError(
            f"kvpress forward hooks must not be active during verification "
            f"(found {count} hook(s) on attention modules)"
        )


def snapshot_attention_model_state(model: torch.nn.Module) -> dict[str, Any]:
    """Snapshot hook counts and module identities for mutation regression tests."""
    attns = list(_iter_attention_modules(model))
    return {
        "hook_counts": [len(a._forward_hooks) for a in attns],
        "attn_module_ids": [id(a) for a in attns],
        "rotary_emb_ids": [id(getattr(a, "rotary_emb", None)) for a in attns],
        "num_layers": len(attns),
        "config_vocab_size": getattr(getattr(model, "config", None), "vocab_size", None),
    }


class KVPressKnormAdapter(BackendAdapter):
    """Experimental KnormPress adapter using replay prefill under ``with press(model):``.

    Compression replays the full prefill sequence with kvpress hooks active,
    producing a pruned DynamicCache.  Logical sequence length is preserved
    separately from the shorter physical KV length.

    Verification and commit always use the authoritative full KV state; hooks
    must never be active during ``verification_mode()``.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        compression_ratio: float = 0.5,
        *,
        isolate_compression_model: bool = True,
    ) -> None:
        import importlib.metadata

        self._runtime = runtime
        self._compression_ratio = compression_ratio
        if isolate_compression_model:
            # kvpress may leave rotary_emb assignments mutated after press();
            # never run hooked replay on the model used for verification.
            self._compression_model = copy.deepcopy(runtime.model)
            self._compression_model.eval()
        else:
            self._compression_model = runtime.model
        # Lazy import — kvpress must not load on default ExactKV imports.
        from kvpress import KnormPress  # noqa: PLC0415

        self._KnormPress = KnormPress
        self.name = "kvpress_knorm"
        try:
            kvpress_version = importlib.metadata.version("kvpress")
        except importlib.metadata.PackageNotFoundError:
            kvpress_version = "unknown"
        self.capabilities = CompressorCapabilities(
            name="kvpress_knorm",
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
            adapter_name="KVPressKnormAdapter",
            adapter_version="0.1.0",
            notes=(
                "Experimental restricted KnormPress adapter ([kvpress] env only). "
                "Replay prefill under with press(model):. Not production-ready. "
                "Draft may diverge from full KV; verification uses authoritative "
                "full state only."
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

    def _backend_compress_from_full_state(self, state: FullKVState) -> dict:
        assert_no_attention_forward_hooks(self._runtime.model)

        from transformers import DynamicCache  # noqa: PLC0415

        press = self._KnormPress(compression_ratio=self._compression_ratio)
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
                f"kvpress hooks not cleaned up after compression replay: "
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
        # Token-dropping: materialized working cache is the pruned DynamicCache.
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
        # Draft from compressed cache may diverge from full-KV prediction.
        return backend_data.get("__compressed_next_token_id__", state.next_token_id)


def create_kvpress_knorm_adapter(
    runtime: ModelRuntime,
    compression_ratio: float = 0.5,
    *,
    isolate_compression_model: bool = True,
) -> KVPressKnormAdapter:
    """Factory for the restricted KnormPress adapter (not in default registry)."""
    return KVPressKnormAdapter(
        runtime=runtime,
        compression_ratio=compression_ratio,
        isolate_compression_model=isolate_compression_model,
    )
