from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from exactkv.runtime.model_runtime import ModelRuntime


@dataclass
class FullGreedyResult:
    """Result of generate_full_greedy.

    Attributes:
        prompt_ids:      Token IDs of the input prompt. Shape [1, prompt_len].
        generated_ids:   Token IDs produced *after* the prompt. Shape [1, gen_len].
        full_sequence_ids: Prompt + generated IDs concatenated. Shape [1, prompt_len + gen_len].
        output_text:     Decoded string of the generated tokens only.
        past_key_values: Full KV cache after generation ends (includes prompt + generated).
        stopped_on_eos:  True if generation stopped because EOS was produced.
    """

    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    full_sequence_ids: torch.Tensor
    output_text: str
    past_key_values: Any
    stopped_on_eos: bool


def generate_full_greedy(
    runtime: ModelRuntime,
    prompt: str,
    max_new_tokens: int,
) -> FullGreedyResult:
    """Greedy autoregressive generation using the full (uncompressed) KV cache.

    This is the ground-truth baseline. It must exactly reproduce the output of
    ``model.generate(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)``.

    Algorithm:
      1. Tokenise the prompt.
      2. Prefill: one forward pass over the full prompt, capture past_key_values.
      3. Decode: at each step, feed the single last token, argmax the logits,
         append the result, update past_key_values. Stop on EOS or budget.

    Args:
        runtime:          Loaded ModelRuntime.
        prompt:           Plain-text prompt string.
        max_new_tokens:   Maximum number of tokens to generate.

    Returns:
        FullGreedyResult with generated_ids, full_sequence_ids, output_text, and
        the final past_key_values.
    """
    prompt_ids: torch.Tensor = runtime.encode(prompt)  # [1, prompt_len]

    # --- Prefill -----------------------------------------------------------
    prefill_out = runtime.forward(
        input_ids=prompt_ids,
        past_key_values=None,
        use_cache=True,
    )
    # The logits at the last prompt position tell us the first generated token.
    past_key_values: Any = prefill_out.past_key_values
    next_logits: torch.Tensor = prefill_out.logits[:, -1, :]  # [1, vocab]

    # --- Decode ------------------------------------------------------------
    generated_ids: list[int] = []
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        next_token_id: int = int(next_logits.argmax(dim=-1).item())
        generated_ids.append(next_token_id)

        if next_token_id == runtime.eos_token_id:
            stopped_on_eos = True
            break

        # Feed only the new token; KV cache carries the history.
        next_token_tensor = torch.tensor(
            [[next_token_id]], dtype=torch.long, device=runtime.device
        )
        step_out = runtime.forward(
            input_ids=next_token_tensor,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = step_out.past_key_values
        next_logits = step_out.logits[:, -1, :]  # [1, vocab]

    gen_tensor = torch.tensor(
        [generated_ids], dtype=torch.long, device=runtime.device
    )
    full_sequence = torch.cat([prompt_ids, gen_tensor], dim=1)
    output_text = runtime.decode(gen_tensor)

    return FullGreedyResult(
        prompt_ids=prompt_ids,
        generated_ids=gen_tensor,
        full_sequence_ids=full_sequence,
        output_text=output_text,
        past_key_values=past_key_values,
        stopped_on_eos=stopped_on_eos,
    )


# ---------------------------------------------------------------------------
# Lossy greedy generation (no verification)
# ---------------------------------------------------------------------------

@dataclass
class LossyGreedyResult:
    """Result of generate_lossy_greedy.

    No exactness guarantee — generated_ids may differ from full-KV output.
    """
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor       # [1, gen_len]
    full_sequence_ids: torch.Tensor   # [1, prompt_len + gen_len]
    output_text: str
    stopped_on_eos: bool


@torch.no_grad()
def generate_lossy_greedy(
    runtime: ModelRuntime,
    prompt: str,
    compressor: Any,
    max_new_tokens: int,
) -> LossyGreedyResult:
    """Greedy generation using a lossy (compressed) KV cache — no verification.

    The initial prompt KV is compressed then materialised back to full-precision.
    New token KVs are appended at full precision by the model during generation.
    Depending on the compressor, the output may diverge from ``generate_full_greedy``.

    This function does NOT implement the ExactKV verification loop; it exists
    solely to characterise lossy mode and confirm that divergence is possible.

    DynamicCache safety: the materialised cache is deep-copied before the
    generation loop so that ``compressor.data`` (which may alias the full-state
    cache for NoOp) is not extended in-place.

    Args:
        runtime:          Loaded ModelRuntime.
        prompt:           Plain-text prompt string.
        compressor:       Any KVCompressor-compatible object.
        max_new_tokens:   Maximum number of tokens to generate.

    Returns:
        LossyGreedyResult (no exactness guarantee).
    """
    from exactkv.cache.full_state import FullKVState

    prompt_ids = runtime.encode(prompt)  # [1, L]

    # Prefill to get the authoritative initial KV
    prefill_out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    next_token_id = int(prefill_out.logits[:, -1, :].argmax(dim=-1).item())

    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    full_state = FullKVState(
        past_key_values=prefill_out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_token_id},
    )

    # Compress and materialise (deep-copy guards against aliasing with full_state)
    compressed = compressor.compress(full_state)
    lossy_kv = copy.deepcopy(compressor.materialize_for_draft(compressed))

    # Generate using the lossy historical KV; new-token KVs are full-precision
    generated_ids: list[int] = []
    next_tok: int = compressed.next_token_id
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        generated_ids.append(next_tok)

        if next_tok == runtime.eos_token_id:
            stopped_on_eos = True
            break

        tok_tensor = torch.tensor(
            [[next_tok]], dtype=torch.long, device=runtime.device
        )
        step_out = runtime.forward(tok_tensor, past_key_values=lossy_kv)
        lossy_kv = step_out.past_key_values   # DynamicCache mutated in-place; that's fine here
        next_tok = int(step_out.logits[:, -1, :].argmax(dim=-1).item())

    gen_tensor = torch.tensor(
        [generated_ids], dtype=torch.long, device=runtime.device
    )
    full_seq = torch.cat([prompt_ids, gen_tensor], dim=1)

    return LossyGreedyResult(
        prompt_ids=prompt_ids,
        generated_ids=gen_tensor,
        full_sequence_ids=full_seq,
        output_text=runtime.decode(gen_tensor),
        stopped_on_eos=stopped_on_eos,
    )
