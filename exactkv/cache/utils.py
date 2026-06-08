from __future__ import annotations

from typing import Any


def kv_seq_len(past_key_values: Any) -> int:
    """Return the sequence length encoded in a HF past_key_values object.

    Handles both the legacy tuple-of-tuples format and the newer Cache objects
    introduced in transformers >= 4.36.
    """
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, tuple):
        # (layer_0_kv, layer_1_kv, ...) where layer_i_kv = (key, value)
        # key shape: [batch, num_kv_heads, seq_len, head_dim]
        return int(past_key_values[0][0].shape[2])
    if hasattr(past_key_values, "key_cache"):
        # DynamicCache / Cache subclasses
        cache = past_key_values.key_cache
        if cache:
            return int(cache[0].shape[2])
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    raise TypeError(
        f"Cannot determine seq_len from past_key_values of type "
        f"{type(past_key_values).__name__}"
    )


def kv_total_bytes(past_key_values: Any) -> int:
    """Return the total number of bytes occupied by all KV tensors."""
    if past_key_values is None:
        return 0
    tensors = []
    if isinstance(past_key_values, tuple):
        for layer_kv in past_key_values:
            tensors.extend(layer_kv)
    elif hasattr(past_key_values, "key_cache"):
        tensors.extend(past_key_values.key_cache)
        tensors.extend(past_key_values.value_cache)
    else:
        return 0
    return sum(int(t.nelement()) * int(t.element_size()) for t in tensors)
