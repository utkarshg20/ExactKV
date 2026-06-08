"""Cache utility helpers shared across ExactKV.

Supports three past_key_values formats:
  * "tuple" — legacy tuple-of-tuples: ((k0, v0), (k1, v1), ...)
  * "dynamic_v5" — transformers >=5 DynamicCache with .layers[i].keys/.values
  * "dynamic_v4" — transformers 4.36–4.x DynamicCache with .key_cache/.value_cache

The helpers detect the format automatically and are tested against
Qwen/Qwen2.5-0.5B on transformers 5.8.x.
"""
from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Helpers to detect cache format
# ---------------------------------------------------------------------------

def _detect_format(past_key_values: Any) -> str:
    if isinstance(past_key_values, tuple):
        return "tuple"
    if hasattr(past_key_values, "layers"):
        return "dynamic_v5"
    if hasattr(past_key_values, "key_cache"):
        return "dynamic_v4"
    raise TypeError(
        f"Cannot detect cache format for type {type(past_key_values).__name__}"
    )


# ---------------------------------------------------------------------------
# Seq length and byte size
# ---------------------------------------------------------------------------

def kv_seq_len(past_key_values: Any) -> int:
    """Return the sequence length encoded in a HF past_key_values object."""
    if past_key_values is None:
        return 0
    fmt = _detect_format(past_key_values)
    if fmt == "tuple":
        return int(past_key_values[0][0].shape[2])
    if fmt == "dynamic_v5":
        layers = past_key_values.layers
        if layers and layers[0].keys is not None:
            return int(layers[0].keys.shape[-2])
        return 0
    if fmt == "dynamic_v4":
        cache = past_key_values.key_cache
        if cache:
            return int(cache[0].shape[2])
        return 0
    return 0


def kv_total_bytes(past_key_values: Any) -> int:
    """Return the total bytes occupied by all KV tensors."""
    if past_key_values is None:
        return 0
    tensors: list[torch.Tensor] = []
    fmt = _detect_format(past_key_values)
    if fmt == "tuple":
        for layer_kv in past_key_values:
            tensors.extend(layer_kv)
    elif fmt == "dynamic_v5":
        for layer in past_key_values.layers:
            if layer.keys is not None:
                tensors.append(layer.keys)
            if layer.values is not None:
                tensors.append(layer.values)
    elif fmt == "dynamic_v4":
        tensors.extend(past_key_values.key_cache)
        tensors.extend(past_key_values.value_cache)
    return sum(int(t.nelement()) * int(t.element_size()) for t in tensors)


# ---------------------------------------------------------------------------
# Cache extraction / reconstruction  (used by compressors)
# ---------------------------------------------------------------------------

def extract_kv_tensors(
    past_key_values: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor], str]:
    """Extract (key_tensors, value_tensors, format_str) from a cache object.

    format_str is "tuple", "dynamic_v5", or "dynamic_v4".  Pass it unchanged
    to ``rebuild_cache`` to reconstruct the same type.

    The returned lists contain *references* to the cache tensors — callers
    must not modify them.  To store safely, call ``.clone()`` on each tensor.
    """
    fmt = _detect_format(past_key_values)
    if fmt == "tuple":
        k = [layer[0] for layer in past_key_values]
        v = [layer[1] for layer in past_key_values]
        return k, v, "tuple"
    if fmt == "dynamic_v5":
        k = [layer.keys for layer in past_key_values.layers]
        v = [layer.values for layer in past_key_values.layers]
        return k, v, "dynamic_v5"
    if fmt == "dynamic_v4":
        k = list(past_key_values.key_cache)
        v = list(past_key_values.value_cache)
        return k, v, "dynamic_v4"
    raise TypeError(f"Unknown format: {fmt}")


def rebuild_cache(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
    cache_format: str,
    seq_len: int,  # noqa: ARG001 — kept for API compatibility
) -> Any:
    """Rebuild a past_key_values object from extracted tensor lists.

    The rebuilt cache is independent: each tensor is NOT cloned here (callers
    that need isolation should clone before passing in).
    """
    if cache_format == "tuple":
        return tuple((k, v) for k, v in zip(k_tensors, v_tensors))

    if cache_format == "dynamic_v5":
        try:
            from transformers import DynamicCache  # type: ignore[import]
            from transformers.cache_utils import DynamicLayer  # type: ignore[import]

            new_cache = DynamicCache()
            for k, v in zip(k_tensors, v_tensors):
                layer = DynamicLayer()
                layer.is_initialized = True
                layer.dtype = k.dtype
                layer.device = k.device
                layer.keys = k
                layer.values = v
                new_cache.layers.append(layer)
            return new_cache
        except Exception as exc:
            raise RuntimeError(
                f"Failed to rebuild dynamic_v5 DynamicCache: {exc}"
            ) from exc

    if cache_format == "dynamic_v4":
        try:
            from transformers import DynamicCache  # type: ignore[import]

            if hasattr(DynamicCache, "from_legacy_cache"):
                legacy = tuple((k, v) for k, v in zip(k_tensors, v_tensors))
                return DynamicCache.from_legacy_cache(legacy)

            new_cache = DynamicCache()
            new_cache.key_cache = list(k_tensors)
            new_cache.value_cache = list(v_tensors)
            try:
                new_cache._seen_tokens = seq_len
            except (AttributeError, TypeError):
                pass
            return new_cache
        except Exception as exc:
            # Final fallback: legacy tuple format
            return tuple((k, v) for k, v in zip(k_tensors, v_tensors))

    raise ValueError(f"Unknown cache_format: {cache_format!r}")
