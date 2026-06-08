"""Shared prefill helper for ExactKV.

Centralises the repeated pattern:
    encode prompt → single forward pass → wrap result into FullKVState

Previously duplicated in:
  * runtime/generation.py      (generate_lossy_greedy)
  * runtime/exactkv_generator.py (ExactKVGenerator.generate)
  * metrics/memory.py           (estimate_kv_memory)

``generate_full_greedy`` uses the same prefill step but proceeds directly to
the decode loop with ``past_key_values`` and ``next_token_id``; it has been
refactored to call this helper too.
"""
from __future__ import annotations

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.runtime.model_runtime import ModelRuntime


@torch.no_grad()
def prefill_to_full_state(runtime: ModelRuntime, prompt: str) -> FullKVState:
    """Run a single prefill forward pass and return an authoritative FullKVState.

    Args:
        runtime: Loaded ModelRuntime.
        prompt:  Plain-text prompt string.

    Returns:
        FullKVState after prefill:
          * ``past_key_values`` — KV cache covering all prompt tokens.
          * ``prompt_ids``      — shape ``[1, prompt_len]``.
          * ``generated_ids``   — empty tensor ``[1, 0]``.
          * ``full_sequence_ids`` — equals ``prompt_ids`` (nothing generated yet).
          * ``device``, ``dtype`` — from the runtime.
          * ``metadata["next_token_id"]`` — argmax of the last-position prefill
            logits; the full model's greedy prediction for the first generated token.

    DynamicCache note: ``past_key_values`` may be a mutable ``DynamicCache``
    object.  Callers that need an independent copy must deep-copy before any
    forward pass.
    """
    prompt_ids: torch.Tensor = runtime.encode(prompt)  # [1, L]
    out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    next_token_id: int = int(out.logits[:, -1, :].argmax(dim=-1).item())

    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    return FullKVState(
        past_key_values=out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_token_id},
    )
