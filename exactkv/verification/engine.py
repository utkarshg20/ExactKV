"""Sequential VerificationEngine for ExactKV V1.

Design notes
------------
* ``verify_sequential`` is the only verification method in V1.
* It verifies draft tokens one position at a time using the full KV state.
* The authoritative FullKVState is NEVER mutated.  In transformers >= 4.36 the
  default cache is a ``DynamicCache`` that IS mutated in-place when passed to a
  forward call.  To guard against this, ``verify_sequential`` deep-copies the
  cache before starting the temp verification chain.
* Bonus-token acceptance is disabled (V1 decision).
* The engine does NOT commit tokens; the ExactKVGenerator does that.

Sequential verification algorithm
----------------------------------
State at start of round:
  full_state.past_key_values  — covers ALL committed tokens (prompt + gen)
  full_state.next_token_id    — full model's cached prediction for next token
                                (from the last forward pass; no extra call needed)

For each position i in draft_tokens:
  1. v_i = stored_prediction  (first i uses cached; subsequent use forward output)
  2. Compare d_i with v_i.
  3. If mismatch → stop; collect verifier_tokens = [v_0, ..., v_i].
  4. If match and i < n-1 → feed d_i to full model to get v_{i+1}.
  5. If all n tokens match → verifier_tokens has length n.

Cost: (n-1) full-model forward passes when all tokens match; fewer on mismatch.
DynamicCache deep-copy cost for Qwen2.5-0.5B in fp32: ~200 KB per round —
acceptable for V1.
"""
from __future__ import annotations

import copy

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.acceptance import AcceptanceResult, compute_acceptance


class VerificationEngine:
    """Verifies draft tokens against the authoritative full KV state."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime

    @torch.no_grad()
    def verify_sequential(
        self,
        full_state: FullKVState,
        draft_tokens: list[int],
    ) -> AcceptanceResult:
        """Verify ``draft_tokens`` against the full model.

        The authoritative ``full_state`` is read but never modified.
        A local ``temp_kv`` reference is advanced each step; because
        ``runtime.forward`` returns a NEW past_key_values object each call,
        ``full_state.past_key_values`` is never mutated in place.

        Args:
            full_state:    Authoritative KV state after the last commit.
            draft_tokens:  Tokens produced by the draft model this round.

        Returns:
            AcceptanceResult from ``compute_acceptance``.
        """
        if not draft_tokens:
            return compute_acceptance([], [])

        # Cache-alignment pre-condition assertion.
        kv_len_before = kv_seq_len(full_state.past_key_values)

        verifier_tokens: list[int] = []
        # Deep-copy the cache so that DynamicCache in-place mutations (transformers
        # >= 4.36 default behaviour) do not touch full_state.past_key_values.
        temp_kv = copy.deepcopy(full_state.past_key_values)
        v_next: int = full_state.next_token_id   # no forward pass for v_0

        for i, draft_tok in enumerate(draft_tokens):
            v: int = v_next
            verifier_tokens.append(v)

            if draft_tok != v:
                # Mismatch: stop collecting verifier tokens.
                break

            # All tokens matched so far; advance if more remain to check.
            if i < len(draft_tokens) - 1:
                tok_tensor = torch.tensor(
                    [[draft_tok]], dtype=torch.long, device=self.runtime.device
                )
                out = self.runtime.forward(tok_tensor, past_key_values=temp_kv)
                temp_kv = out.past_key_values   # new object, full_state unchanged
                v_next = int(out.logits[:, -1, :].argmax(dim=-1).item())

        # Post-condition: authoritative full_state must be identical to before.
        assert kv_seq_len(full_state.past_key_values) == kv_len_before, (
            "VerificationEngine mutated full_state.past_key_values — this is a bug."
        )

        return compute_acceptance(draft_tokens, verifier_tokens)
