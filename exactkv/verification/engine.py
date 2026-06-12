"""Sequential VerificationEngine for ExactKV V1.

Design notes
------------
* ``verify_sequential`` verifies draft tokens one position at a time.
* ``verify_span`` (V13 opt-in) verifies an entire draft span in one teacher-forced
  full-KV forward; see ``SPAN_VERIFICATION_DESIGN.md``.
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
from typing import Any

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.acceptance import AcceptanceResult, compute_acceptance


class VerificationEngine:
    """Verifies draft tokens against the authoritative full KV state."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime

    def _span_verify_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        **kwargs: Any,
    ) -> Any:
        """Forward for batched span verify.

        On fp16 CUDA, default SDPA batched forwards can tie-break argmax differently
        than single-step sequential forwards when logits are nearly equal (Exp 030b).
        Math-only SDPA matches eager parity on the lc_003 blocker cell.
        """
        if (
            self.runtime.dtype == torch.float16
            and self.runtime.device.type == "cuda"
        ):
            try:
                from torch.backends.cuda import sdp_kernel

                with sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    return self.runtime.forward(
                        input_ids,
                        past_key_values=past_key_values,
                        **kwargs,
                    )
            except Exception:
                pass
        return self.runtime.forward(
            input_ids, past_key_values=past_key_values, **kwargs
        )

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

    @torch.no_grad()
    def _verify_span_batched(
        self,
        full_state: FullKVState,
        draft_tokens: list[int],
    ) -> AcceptanceResult:
        """Teacher-forced batched span verify (fast path when parity holds)."""
        if not draft_tokens:
            return compute_acceptance([], [])

        kv_len_before = kv_seq_len(full_state.past_key_values)
        next_before = full_state.next_token_id
        k = len(draft_tokens)

        verifier_tokens: list[int] = [full_state.next_token_id]

        if k >= 2:
            temp_kv = copy.deepcopy(full_state.past_key_values)
            teacher_tokens = draft_tokens[:-1]
            past_len = kv_seq_len(temp_kv)
            L = len(teacher_tokens)
            input_ids = torch.tensor(
                [teacher_tokens], dtype=torch.long, device=self.runtime.device
            )
            fwd_kwargs: dict[str, Any] = {
                "cache_position": torch.arange(
                    past_len,
                    past_len + L,
                    device=self.runtime.device,
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    1,
                    past_len + L,
                    device=self.runtime.device,
                    dtype=torch.long,
                ),
            }
            out = self._span_verify_forward(
                input_ids, temp_kv, **fwd_kwargs
            )
            for i in range(1, k):
                v_i = int(
                    out.logits[:, i - 1, :].float().argmax(dim=-1).item()
                )
                verifier_tokens.append(v_i)

        for i, d_i in enumerate(draft_tokens):
            if d_i != verifier_tokens[i]:
                verifier_tokens = verifier_tokens[: i + 1]
                break

        assert kv_seq_len(full_state.past_key_values) == kv_len_before
        assert full_state.next_token_id == next_before

        return compute_acceptance(draft_tokens, verifier_tokens)

    @torch.no_grad()
    def verify_span(
        self,
        full_state: FullKVState,
        draft_tokens: list[int],
    ) -> AcceptanceResult:
        """Verify ``draft_tokens`` via batched teacher-forced forward when possible.

        HF causal LM logits shift (see ``SPAN_VERIFICATION_DESIGN.md``).  When the
        batched path disagrees with ``verify_sequential`` (observed on fp16 Qwen2.5-0.5B
        long-context, Exp 030), fall back to sequential for exactness parity.
        """
        batched = self._verify_span_batched(full_state, draft_tokens)
        sequential = self.verify_sequential(full_state, draft_tokens)
        if batched != sequential:
            return sequential
        return batched
