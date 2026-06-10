#!/usr/bin/env python3
"""Synthetic KVQuant calibration for Qwen2.5-1.5B (Phase E optional artifact)."""
from __future__ import annotations

import pickle
import sys
import types

import torch
from transformers import AutoTokenizer

import llama_simquant as ls

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SEQLEN = 128
NSAMPLES = 4
ABITS = 4
PICKLE_PATH = "/workspace/kvquant_d4/quantizers_qwen15b.pickle"
DEV = torch.device("cuda:0")


def main() -> int:
    ls.args = types.SimpleNamespace(
        nsamples=NSAMPLES,
        perchannel=["k_proj"],
        pertoken=["v_proj"],
        abits=ABITS,
        include_sparse=False,
        sparsity_threshold=1.0,
        nuq=False,
        norm=False,
        cap_outliers=-1,
        first_few_fp16=-1,
        fisher=None,
        quantizer_path=PICKLE_PATH,
    )

    print("loading_model", MODEL_ID, flush=True)
    model = ls.get_model(MODEL_ID, SEQLEN, SEQLEN)
    model.eval()
    model.seqlen = SEQLEN
    model = model.half()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = ("The capital of France is Paris. " * 80) + ("Machine learning is fun. " * 80)
    enc = tok(text, return_tensors="pt")
    trainloader = []
    for i in range(NSAMPLES):
        start = i * 17
        inp = enc.input_ids[:, start : start + SEQLEN]
        if inp.shape[1] < SEQLEN:
            pad = SEQLEN - inp.shape[1]
            inp = torch.nn.functional.pad(inp, (0, pad), value=tok.pad_token_id or 0)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    print("synthetic_samples", len(trainloader), "seqlen", SEQLEN, flush=True)
    quantizers = ls.llama_calibration(
        model,
        trainloader,
        DEV,
        ls.args.perchannel,
        ls.args.pertoken,
        ls.args.abits,
        include_sparse=ls.args.include_sparse,
        sparsity_threshold=ls.args.sparsity_threshold,
        nuq=ls.args.nuq,
        fisher=None,
        norm=ls.args.norm,
        cap_outliers=ls.args.cap_outliers,
        first_few_fp16=ls.args.first_few_fp16,
    )

    with open(PICKLE_PATH, "wb") as handle:
        pickle.dump(quantizers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("quantizer_keys_count", len(quantizers), flush=True)
    print("pickle_written", PICKLE_PATH, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
