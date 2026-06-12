"""Span verification parity debugging (Exp 030b).

Research-only helpers comparing sequential vs batched verifier token extraction.
No timing, throughput, or production claims.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.acceptance import AcceptanceResult, compute_acceptance


@dataclass
class VerifierTokenComparison:
    draft_tokens: list[int]
    sequential_tokens: list[int]
    batched_tokens: list[int]
    parity_pass: bool
    first_mismatch_index: int | None = None
    argmax_flip_indices: list[int] = field(default_factory=list)
    logit_stats: list[dict[str, Any]] = field(default_factory=list)
    sequential_acceptance: dict[str, Any] = field(default_factory=dict)
    batched_acceptance: dict[str, Any] = field(default_factory=dict)
    forward_mode: str = "default"


def _top_k_logits(logits_1d: torch.Tensor, k: int = 5) -> list[dict[str, int | float]]:
    vals, idx = torch.topk(logits_1d.float(), k)
    return [
        {"token_id": int(i.item()), "logit": float(v.item())}
        for v, i in zip(vals, idx, strict=True)
    ]


def collect_sequential_verifier_tokens(
    runtime: ModelRuntime,
    full_state: FullKVState,
    draft_tokens: list[int],
) -> list[int]:
    """Mirror ``VerificationEngine.verify_sequential`` token collection."""
    if not draft_tokens:
        return []
    temp_kv = copy.deepcopy(full_state.past_key_values)
    verifier_tokens: list[int] = []
    v_next = full_state.next_token_id
    for i, draft_tok in enumerate(draft_tokens):
        verifier_tokens.append(v_next)
        if draft_tok != v_next:
            break
        if i < len(draft_tokens) - 1:
            tok_tensor = torch.tensor(
                [[draft_tok]], dtype=torch.long, device=runtime.device
            )
            out = runtime.forward(tok_tensor, past_key_values=temp_kv)
            temp_kv = out.past_key_values
            v_next = int(out.logits[:, -1, :].argmax(dim=-1).item())
    return verifier_tokens


def _build_span_forward_kwargs(
    runtime: ModelRuntime,
    full_state: FullKVState,
    teacher_tokens: list[int],
    mode: str,
) -> dict[str, Any]:
    """Build optional HF forward kwargs for batched span verify experiments."""
    if mode == "default":
        return {}

    past_len = kv_seq_len(full_state.past_key_values)
    L = len(teacher_tokens)
    device = runtime.device

    if mode == "cache_position":
        return {
            "cache_position": torch.arange(
                past_len, past_len + L, device=device, dtype=torch.long
            ),
        }

    if mode == "cache_position_and_mask":
        return {
            "cache_position": torch.arange(
                past_len, past_len + L, device=device, dtype=torch.long
            ),
            "attention_mask": torch.ones(
                1, past_len + L, device=device, dtype=torch.long
            ),
        }

    if mode == "position_ids":
        return {
            "position_ids": torch.arange(
                past_len, past_len + L, device=device, dtype=torch.long
            ).unsqueeze(0),
        }

    if mode == "position_ids_and_mask":
        past_len = kv_seq_len(full_state.past_key_values)
        return {
            "position_ids": torch.arange(
                past_len, past_len + L, device=device, dtype=torch.long
            ).unsqueeze(0),
            "attention_mask": torch.ones(
                1, past_len + L, device=device, dtype=torch.long
            ),
        }

    raise ValueError(f"Unknown forward mode: {mode!r}")


def collect_batched_verifier_tokens(
    runtime: ModelRuntime,
    full_state: FullKVState,
    draft_tokens: list[int],
    *,
    forward_mode: str = "default",
    teacher_slice: str = "minus_last",
    logits_argmax_dtype: str = "float32",
) -> list[int]:
    """Mirror ``VerificationEngine._verify_span_batched`` token collection."""
    if not draft_tokens:
        return []
    k = len(draft_tokens)
    verifier_tokens: list[int] = [full_state.next_token_id]
    if k < 2:
        return verifier_tokens

    if teacher_slice == "minus_last":
        teacher_tokens = draft_tokens[:-1]
    elif teacher_slice == "all":
        teacher_tokens = list(draft_tokens)
    else:
        raise ValueError(f"Unknown teacher_slice: {teacher_slice!r}")

    temp_kv = copy.deepcopy(full_state.past_key_values)
    input_ids = torch.tensor([teacher_tokens], dtype=torch.long, device=runtime.device)
    fwd_kwargs = _build_span_forward_kwargs(
        runtime, full_state, teacher_tokens, forward_mode
    )
    out = runtime.forward(input_ids, past_key_values=temp_kv, **fwd_kwargs)

    for i in range(1, k):
        logit_row = out.logits[:, i - 1, :]
        if logits_argmax_dtype == "float32":
            v_i = int(logit_row.float().argmax(dim=-1).item())
        else:
            v_i = int(logit_row.argmax(dim=-1).item())
        verifier_tokens.append(v_i)

    for i, d_i in enumerate(draft_tokens):
        if d_i != verifier_tokens[i]:
            verifier_tokens = verifier_tokens[: i + 1]
            break

    return verifier_tokens


def _logit_position_stats(
    runtime: ModelRuntime,
    full_state: FullKVState,
    draft_tokens: list[int],
    position: int,
    forward_mode: str,
) -> dict[str, Any]:
    """Compare sequential vs batched logits at verifier index ``position``."""
    teacher_tokens = draft_tokens[:-1]
    k = len(draft_tokens)
    if position < 1 or position >= k:
        return {}

    # Sequential logit at position (after draft_tokens[position-1] fed)
    temp_kv = copy.deepcopy(full_state.past_key_values)
    v_next = full_state.next_token_id
    seq_logit = None
    for i in range(position):
        if i > 0:
            tok = torch.tensor(
                [[draft_tokens[i - 1]]], dtype=torch.long, device=runtime.device
            )
            out = runtime.forward(tok, past_key_values=temp_kv)
            temp_kv = out.past_key_values
            v_next = int(out.logits[:, -1, :].argmax(dim=-1).item())
        if i == position - 1:
            tok = torch.tensor(
                [[draft_tokens[i]]], dtype=torch.long, device=runtime.device
            )
            out = runtime.forward(tok, past_key_values=temp_kv)
            seq_logit = out.logits[:, -1, :].squeeze(0)

    temp_kv2 = copy.deepcopy(full_state.past_key_values)
    input_ids = torch.tensor([teacher_tokens], dtype=torch.long, device=runtime.device)
    fwd_kwargs = _build_span_forward_kwargs(
        runtime, full_state, teacher_tokens, forward_mode
    )
    out_b = runtime.forward(input_ids, past_key_values=temp_kv2, **fwd_kwargs)
    batched_logit = out_b.logits[:, position - 1, :].squeeze(0)

    if seq_logit is None:
        return {}

    diff = (seq_logit.float() - batched_logit.float()).abs()
    seq_argmax = int(seq_logit.argmax(dim=-1).item())
    bat_argmax = int(batched_logit.argmax(dim=-1).item())
    return {
        "position": position,
        "sequential_argmax": seq_argmax,
        "batched_argmax": bat_argmax,
        "argmax_match": seq_argmax == bat_argmax,
        "mean_abs_logit_diff": float(diff.mean().item()),
        "max_abs_logit_diff": float(diff.max().item()),
        "sequential_top5": _top_k_logits(seq_logit),
        "batched_top5": _top_k_logits(batched_logit),
    }


def compare_verifier_tokens(
    runtime: ModelRuntime,
    full_state: FullKVState,
    draft_tokens: list[int],
    *,
    forward_mode: str = "default",
    teacher_slice: str = "minus_last",
    logits_argmax_dtype: str = "float32",
    logit_positions: list[int] | None = None,
) -> VerifierTokenComparison:
    """Compare sequential vs batched verifier token lists."""
    seq_tokens = collect_sequential_verifier_tokens(
        runtime, full_state, draft_tokens
    )
    bat_tokens = collect_batched_verifier_tokens(
        runtime,
        full_state,
        draft_tokens,
        forward_mode=forward_mode,
        teacher_slice=teacher_slice,
        logits_argmax_dtype=logits_argmax_dtype,
    )
    parity = seq_tokens == bat_tokens
    first_mismatch: int | None = None
    flips: list[int] = []
    for i, (s, b) in enumerate(zip(seq_tokens, bat_tokens, strict=False)):
        if s != b:
            if first_mismatch is None:
                first_mismatch = i
            flips.append(i)
        if len(seq_tokens) != len(bat_tokens) and first_mismatch is None:
            first_mismatch = min(len(seq_tokens), len(bat_tokens))

    if len(seq_tokens) != len(bat_tokens) and first_mismatch is None:
        first_mismatch = min(len(seq_tokens), len(bat_tokens))

    logit_stats: list[dict[str, Any]] = []
    if logit_positions:
        for pos in logit_positions:
            stats = _logit_position_stats(
                runtime, full_state, draft_tokens, pos, forward_mode
            )
            if stats:
                logit_stats.append(stats)

    seq_acc = compute_acceptance(draft_tokens, seq_tokens)
    bat_acc = compute_acceptance(draft_tokens, bat_tokens)

    return VerifierTokenComparison(
        draft_tokens=draft_tokens,
        sequential_tokens=seq_tokens,
        batched_tokens=bat_tokens,
        parity_pass=parity,
        first_mismatch_index=first_mismatch,
        argmax_flip_indices=flips,
        logit_stats=logit_stats,
        sequential_acceptance={
            "accepted": seq_acc.accepted_tokens,
            "correction": seq_acc.correction_token,
            "all_matched": seq_acc.all_matched,
        },
        batched_acceptance={
            "accepted": bat_acc.accepted_tokens,
            "correction": bat_acc.correction_token,
            "all_matched": bat_acc.all_matched,
        },
        forward_mode=forward_mode,
    )


def acceptance_result_to_dict(acc: AcceptanceResult) -> dict[str, Any]:
    return {
        "accepted_tokens": acc.accepted_tokens,
        "correction_token": acc.correction_token,
        "all_matched": acc.all_matched,
        "verifier_tokens": acc.verifier_tokens,
    }


def state_and_draft_at_round(
    runtime: ModelRuntime,
    compressor: Any,
    prompt: str,
    *,
    draft_len: int,
    round_idx: int,
) -> tuple[FullKVState, list[int]]:
    """Replay ExactKV sequential loop to state/draft at ``round_idx``."""
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.prefill import prefill_to_full_state

    gen = ExactKVGenerator(
        runtime, compressor, draft_len=draft_len, verification_method="sequential"
    )
    full_state = prefill_to_full_state(runtime, prompt)
    compressed = compressor.compress(full_state)
    draft_at_round: list[int] = []

    for r in range(round_idx + 1):
        draft_res = gen._draft(compressed, draft_len)
        if r == round_idx:
            draft_at_round = list(draft_res.token_ids)
            break
        acc = gen._verify_draft_tokens(full_state, draft_res.token_ids)
        committed = list(acc.accepted_tokens)
        if acc.correction_token is not None:
            committed.append(acc.correction_token)
        committed, _ = gen._truncate_at_eos(committed)
        if not committed:
            break
        full_state = gen._commit(full_state, committed)
        compressed = compressor.update_after_commit(compressed, full_state)

    return full_state, draft_at_round
