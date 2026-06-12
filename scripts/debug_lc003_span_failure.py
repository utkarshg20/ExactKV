#!/usr/bin/env python3
"""Debug lc_003 span vs sequential divergence (Exp 030 blocker)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.engine import VerificationEngine

MODEL = "Qwen/Qwen2.5-0.5B"
COMPRESSOR = "k8_v4_sim"
DRAFT_LEN = 8
MAX_NEW = 32


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    runtime = ModelRuntime(model_name=MODEL, device=device, dtype=dtype)
    comp = get_compressor(COMPRESSOR)
    prompt = load_v10_suite("long_context")[2]["prompt"]

    full = generate_full_greedy(runtime, prompt, MAX_NEW)
    full_ids = full.generated_ids.squeeze(0).tolist()

    seq_gen = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="sequential"
    )
    span_gen = ExactKVGenerator(
        runtime, comp, draft_len=DRAFT_LEN, verification_method="span"
    )
    seq = seq_gen.generate(prompt, MAX_NEW)
    span = span_gen.generate(prompt, MAX_NEW)
    seq_ids = seq.output_ids.squeeze(0).tolist()
    span_ids = span.output_ids.squeeze(0).tolist()

    print(f"device={device} dtype={dtype}")
    print(f"full_greedy match seq: {seq_ids == full_ids}")
    print(f"full_greedy match span: {span_ids == full_ids}")
    print(f"seq match span: {seq_ids == span_ids}")

    if seq_ids != span_ids:
        for i, (a, b) in enumerate(zip(seq_ids, span_ids)):
            if a != b:
                print(f"first seq/span diff at {i}: seq={a} span={b}")
                break
        if len(seq_ids) != len(span_ids):
            print(f"lengths seq={len(seq_ids)} span={len(span_ids)}")

    # Round-by-round trace comparison
    engine = VerificationEngine(runtime)
    for round_idx, (st, sp) in enumerate(zip(seq.traces, span.traces)):
        sd, sdd = st.draft_tokens, sp.draft_tokens
        if sd != sdd:
            print(f"round {round_idx}: draft mismatch seq={sd} span={sdd}")
            break
        sa, spa = st.acceptance, sp.acceptance
        if (
            sa.accepted_tokens != spa.accepted_tokens
            or sa.correction_token != spa.correction_token
        ):
            print(f"round {round_idx}: acceptance mismatch")
            print(f"  draft={sd}")
            print(f"  seq accepted={sa.accepted_tokens} corr={sa.correction_token}")
            print(f"  span accepted={spa.accepted_tokens} corr={spa.correction_token}")
            print(f"  seq verifier={sa.verifier_tokens}")
            print(f"  span verifier={spa.verifier_tokens}")
            # Re-verify this round's draft with both methods if we can reconstruct state
            break
    else:
        if len(seq.traces) != len(span.traces):
            print(
                f"trace count mismatch seq={len(seq.traces)} span={len(span.traces)}"
            )

    return 0 if seq_ids == full_ids and span_ids == full_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
