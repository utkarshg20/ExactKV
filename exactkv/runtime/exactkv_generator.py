"""ExactKVGenerator — ExactKV generation loop for Phase 1 (V1).

V1 constraints (hard)
---------------------
* Greedy decoding only.
* Single request, single device.
* Sequential verification only (default); span verification opt-in (V13).
* Bonus-token acceptance disabled.
* NoOpCompressor only (INT8 in later steps).
* Recompress from authoritative full KV after every commit round.

Loop overview (one round)
-------------------------
1. DRAFT  : generate ``draft_len`` tokens using the materialized compressed KV.
2. VERIFY : compare draft tokens to full-KV predictions (sequential).
3. COMMIT : accept matching prefix + correction token (if mismatch) into the
            authoritative FullKVState.
4. ALIGN  : update CompressedKVState from new FullKVState (V1: recompress).
5. STOP   : if EOS committed or ``max_new_tokens`` reached.

Cache alignment invariant (asserted every round)
------------------------------------------------
  full_state.seq_len == compressed_state.logical_seq_len

DynamicCache note
-----------------
Transformers >= 4.36 uses ``DynamicCache`` objects that are mutated in-place
when passed to a model forward call.  For NoOpCompressor, ``compressed.data``
and ``full_state.past_key_values`` refer to the SAME DynamicCache object.

We deep-copy the cache in ``_draft`` (to avoid corrupting the authoritative
state during drafting) and in ``VerificationEngine.verify_sequential`` (same
reason).  Only ``_commit`` intentionally mutates the cache — that mutation is
immediately captured in the new FullKVState that replaces the old one.
"""
from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.compressors.base import KVCompressor
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.verification.acceptance import (
    AcceptanceResult,
    DraftResult,
    ExactKVResult,
    VerificationTrace,
)
from exactkv.verification.engine import VerificationEngine

VerificationMethod = Literal["sequential", "span"]
_VALID_VERIFICATION_METHODS = frozenset({"sequential", "span"})


class ExactKVGenerator:
    """Runs the ExactKV draft-verify-commit loop.

    Args:
        runtime:    Loaded ModelRuntime (Hugging Face model + tokenizer).
        compressor: A KVCompressor instance (NoOpCompressor for V1).
        draft_len:  Maximum number of tokens to draft per round.
        verification_method: ``"sequential"`` (default) or ``"span"`` (V13 opt-in).
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        compressor: KVCompressor,
        draft_len: int = 8,
        verification_method: VerificationMethod = "sequential",
    ) -> None:
        if verification_method not in _VALID_VERIFICATION_METHODS:
            raise ValueError(
                f"Invalid verification_method {verification_method!r}; "
                f"expected one of {sorted(_VALID_VERIFICATION_METHODS)}"
            )
        self.runtime = runtime
        self.compressor = compressor
        self.draft_len = draft_len
        self.verification_method: VerificationMethod = verification_method
        self.engine = VerificationEngine(runtime)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int) -> ExactKVResult:
        """Run the ExactKV loop and return an ExactKVResult.

        Args:
            prompt:         Plain-text prompt string.
            max_new_tokens: Hard budget for generated tokens.

        Returns:
            ExactKVResult with output_ids matching generate_full_greedy exactly
            when NoOpCompressor is used.
        """
        # ── Prefill (via shared helper) ───────────────────────────────────
        full_state = prefill_to_full_state(self.runtime, prompt)

        compressed = self.compressor.compress(full_state)
        self._assert_alignment(full_state, compressed, round_idx=-1)

        # ── Round loop ───────────────────────────────────────────────────
        all_generated: list[int] = []
        traces: list[VerificationTrace] = []
        total_accepted = 0
        total_rejected = 0
        total_corrections = 0
        done = False
        round_idx = 0

        while not done and len(all_generated) < max_new_tokens:
            seq_len_before = full_state.seq_len

            # 1. Draft
            remaining = max_new_tokens - len(all_generated)
            n = min(self.draft_len, remaining)
            draft_result = self._draft(compressed, n)

            # 2. Verify (inside verification_mode when compressor provides it)
            acceptance = self._verify_draft_tokens(
                full_state, draft_result.token_ids
            )

            # 3. Determine tokens to commit
            committed = list(acceptance.accepted_tokens)
            if acceptance.correction_token is not None:
                committed.append(acceptance.correction_token)
                total_corrections += 1

            # Truncate committed at first EOS
            committed, eos_found = self._truncate_at_eos(committed)
            if eos_found:
                done = True

            if not committed:
                # Safety: should never happen (at minimum a correction is committed)
                done = True
                break

            # 4. Commit: update authoritative full state
            full_state = self._commit(full_state, committed)

            # 5. Align: recompress from new full state (V1 strategy)
            compressed = self.compressor.update_after_commit(compressed, full_state)
            self._assert_alignment(full_state, compressed, round_idx=round_idx)

            # Accumulate bookkeeping
            all_generated.extend(committed)
            total_accepted += acceptance.num_accepted
            total_rejected += acceptance.num_rejected

            traces.append(
                VerificationTrace(
                    round_idx=round_idx,
                    draft_tokens=list(draft_result.token_ids),
                    acceptance=acceptance,
                    full_seq_len_before=seq_len_before,
                    full_seq_len_after=full_state.seq_len,
                    compressed_seq_len_after=compressed.logical_seq_len,
                )
            )
            round_idx += 1

            if len(all_generated) >= max_new_tokens:
                done = True

        # ── Package result ────────────────────────────────────────────────
        output_ids = torch.tensor(
            [all_generated], dtype=torch.long, device=self.runtime.device
        )
        prompt_ids = full_state.prompt_ids
        full_seq = torch.cat([prompt_ids, output_ids], dim=1)
        output_text = self.runtime.decode(output_ids)

        denom = total_accepted + total_rejected
        acceptance_rate = total_accepted / denom if denom > 0 else 1.0

        return ExactKVResult(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            full_sequence_ids=full_seq,
            output_text=output_text,
            stopped_on_eos=done and bool(all_generated) and all_generated[-1] == self.runtime.eos_token_id,
            traces=traces,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            total_corrections=total_corrections,
            acceptance_rate=acceptance_rate,
            num_rounds=round_idx,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_draft_tokens(
        self,
        full_state: FullKVState,
        draft_tokens: list[int],
    ) -> AcceptanceResult:
        """Run verification (sequential or span), inside ``verification_mode`` when available."""
        if self.verification_method == "span":
            verify = self.engine.verify_span
        else:
            verify = self.engine.verify_sequential
        mode = getattr(self.compressor, "verification_mode", None)
        if callable(mode):
            with mode():
                return verify(full_state, draft_tokens)
        return verify(full_state, draft_tokens)

    @torch.no_grad()
    def _draft(self, compressed: CompressedKVState, n: int) -> DraftResult:
        """Generate up to ``n`` draft tokens using the compressed KV cache.

        The first draft token comes from ``compressed.next_token_id`` (cached
        from the last compress/update_after_commit call) — no extra forward
        pass.  Subsequent tokens are produced by feeding each draft token
        through the draft model.
        """
        # Deep-copy the materialized cache so that forward passes during drafting
        # do not mutate compressed.data / full_state.past_key_values (they share
        # the same DynamicCache object for NoOpCompressor).
        draft_kv: Any = copy.deepcopy(self.compressor.materialize_for_draft(compressed))
        d_current: int = compressed.next_token_id
        draft_tokens: list[int] = []
        stopped_on_eos = False

        for i in range(n):
            draft_tokens.append(d_current)

            if d_current == self.runtime.eos_token_id:
                stopped_on_eos = True
                break

            if i < n - 1:
                tok_tensor = torch.tensor(
                    [[d_current]], dtype=torch.long, device=self.runtime.device
                )
                out = self.runtime.forward(tok_tensor, past_key_values=draft_kv)
                draft_kv = out.past_key_values
                d_current = int(out.logits[:, -1, :].argmax(dim=-1).item())

        return DraftResult(token_ids=draft_tokens, stopped_on_eos=stopped_on_eos)

    @torch.no_grad()
    def _commit(self, full_state: FullKVState, committed_tokens: list[int]) -> FullKVState:
        """Run the full model on committed tokens and return the updated state.

        V1 strategy: feed each committed token through the full model in order
        to advance past_key_values and capture the next prediction.
        We stop before running a forward pass on EOS (matching generate_full_greedy).
        """
        past_kv = full_state.past_key_values
        current_next_token_id = full_state.next_token_id
        new_gen_ids: list[int] = full_state.generated_ids.squeeze(0).tolist()

        for token_id in committed_tokens:
            new_gen_ids.append(token_id)

            if token_id == self.runtime.eos_token_id:
                # Do not run a forward pass after EOS (matches greedy baseline).
                current_next_token_id = self.runtime.eos_token_id
                break

            tok_tensor = torch.tensor(
                [[token_id]], dtype=torch.long, device=self.runtime.device
            )
            out = self.runtime.forward(tok_tensor, past_key_values=past_kv)
            past_kv = out.past_key_values
            current_next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())

        gen_tensor = torch.tensor(
            [new_gen_ids], dtype=torch.long, device=self.runtime.device
        )
        full_seq = torch.cat([full_state.prompt_ids, gen_tensor], dim=1)

        return FullKVState(
            past_key_values=past_kv,
            prompt_ids=full_state.prompt_ids,
            generated_ids=gen_tensor,
            full_sequence_ids=full_seq,
            device=full_state.device,
            dtype=full_state.dtype,
            metadata={"next_token_id": current_next_token_id},
        )

    def _truncate_at_eos(self, tokens: list[int]) -> tuple[list[int], bool]:
        """Return tokens up to and including the first EOS (if any)."""
        result: list[int] = []
        eos_found = False
        for t in tokens:
            result.append(t)
            if t == self.runtime.eos_token_id:
                eos_found = True
                break
        return result, eos_found

    def _assert_alignment(
        self,
        full_state: FullKVState,
        compressed: CompressedKVState,
        round_idx: int,
    ) -> None:
        """Assert the cache alignment invariant at round boundaries."""
        full_len = full_state.seq_len
        comp_len = compressed.logical_seq_len
        assert full_len == comp_len, (
            f"Cache alignment broken after round {round_idx}: "
            f"full_state.seq_len={full_len}, "
            f"compressed.logical_seq_len={comp_len}"
        )
