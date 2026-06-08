from __future__ import annotations

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
