#!/usr/bin/env python3
"""Compare sequential vs span verifier token extraction at failing round."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.model_runtime import ModelRuntime

MODEL = "Qwen/Qwen2.5-0.5B"
COMPRESSOR = "k8_v4_sim"
DRAFT_LEN = 8
MAX_NEW = 32
DRAFT = [2550, 1969, 2432, 92382, 44378, 6896, 13, 576]


def _state_after_rounds(runtime, prompt, n_rounds):
    comp = get_compressor(COMPRESSOR)
    gen = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="sequential"
    )
    # Run n_rounds then capture full_state via internal path
    from exactkv.runtime.prefill import prefill_to_full_state

    full_state = prefill_to_full_state(runtime, prompt)
    compressed = comp.compress(full_state)
    for _ in range(n_rounds):
        draft = gen._draft(compressed, DRAFT_LEN)
        acc = gen._verify_draft_tokens(full_state, draft.token_ids)
        committed = list(acc.accepted_tokens)
        if acc.correction_token is not None:
            committed.append(acc.correction_token)
        committed, _ = gen._truncate_at_eos(committed)
        full_state = gen._commit(full_state, committed)
        compressed = comp.update_after_commit(full_state, compressed)
    return full_state


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    runtime = ModelRuntime(model_name=MODEL, device=device, dtype=dtype)
    prompt = load_v10_suite("long_context")[2]["prompt"]
    full_state = _state_after_rounds(runtime, prompt, 2)

    # Sequential v_7 path
    temp_kv = copy.deepcopy(full_state.past_key_values)
    v_next = full_state.next_token_id
    seq_verifier = [v_next]
    for i, d in enumerate(DRAFT):
        if d != v_next:
            break
        if i < len(DRAFT) - 1:
            out = runtime.forward(
                torch.tensor([[d]], dtype=torch.long, device=runtime.device),
                past_key_values=temp_kv,
            )
            temp_kv = out.past_key_values
            v_next = int(out.logits[:, -1, :].argmax(dim=-1).item())
            seq_verifier.append(v_next)

    # Span batched path
    temp_kv2 = copy.deepcopy(full_state.past_key_values)
    input_ids = torch.tensor([DRAFT], dtype=torch.long, device=runtime.device)
    out = runtime.forward(input_ids, past_key_values=temp_kv2)
    span_verifier = [full_state.next_token_id]
    for i in range(1, len(DRAFT)):
        logit = out.logits[:, i - 1, :]
        span_verifier.append(int(logit.argmax(dim=-1).item()))
        span_verifier_fp32 = int(logit.float().argmax(dim=-1).item())

    print("v_0 cached", full_state.next_token_id)
    print("sequential verifier", seq_verifier)
    print("span verifier", span_verifier)
    print("span v_7 fp16 argmax", span_verifier[-1])
    print("span v_7 fp32 argmax", int(out.logits[:, 6, :].float().argmax(dim=-1).item()))
    print("sequential v_7", seq_verifier[-1] if len(seq_verifier) > 7 else "n/a")

    # Single-token forward of d_6 only (what sequential uses for v_7)
    temp_kv3 = copy.deepcopy(full_state.past_key_values)
    for d in DRAFT[:6]:
        o = runtime.forward(
            torch.tensor([[d]], dtype=torch.long, device=runtime.device),
            past_key_values=temp_kv3,
        )
        temp_kv3 = o.past_key_values
    o6 = runtime.forward(
        torch.tensor([[DRAFT[6]]], dtype=torch.long, device=runtime.device),
        past_key_values=temp_kv3,
    )
    v7_single = int(o6.logits[:, -1, :].argmax(dim=-1).item())
    v7_single_f = int(o6.logits[:, -1, :].float().argmax(dim=-1).item())
    print("v_7 from single forward d_6 fp16", v7_single)
    print("v_7 from single forward d_6 fp32", v7_single_f)
    print("batched logits[:,6] top token fp16", int(out.logits[:, 6, :].argmax(dim=-1).item()))
    print("batched logits[:,6] top token fp32", int(out.logits[:, 6, :].float().argmax(dim=-1).item()))


if __name__ == "__main__":
    main()
