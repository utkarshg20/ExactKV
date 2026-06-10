#!/usr/bin/env python3
"""KVQuant QuantLinearSim forward + draft/verify isolation check."""
from __future__ import annotations

import copy
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant.simquant_module_quantizer import QuantLinearSim, make_quant_sim

MODEL_ID = "Qwen/Qwen2.5-0.5B"
PICKLE_PATH = "/workspace/kvquant_d4/quantizers_qwen05b.pickle"

with open(PICKLE_PATH, "rb") as f:
    quantizers = pickle.load(f)

perchannel = {k: v for k, v in quantizers.items() if "k_proj" in k}
pertoken = {k: v for k, v in quantizers.items() if "v_proj" in k}
print("quantizer_keys", len(quantizers), "k", len(perchannel), "v", len(pertoken))

draft = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa"
)
verify = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
)

draft_clone = copy.deepcopy(draft)
make_quant_sim(draft_clone, perchannel, 4, perchannel=True, include_sparse=False)
make_quant_sim(draft_clone, pertoken, 4, perchannel=False, dynamicquantization=True)

draft_has_quant = any(type(m).__name__ == "QuantLinearSim" for m in draft.modules())
clone_has_quant = any(type(m).__name__ == "QuantLinearSim" for m in draft_clone.modules())
print("draft_unmutated_after_deepcopy", not draft_has_quant)
print("clone_has_quant", clone_has_quant)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
inp = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = draft_clone(input_ids=inp, use_cache=True)
print("draft_forward_ok", tuple(out.logits.shape), "past", out.past_key_values is not None)
if out.past_key_values is not None:
    print("past_len", len(out.past_key_values))

verify_clean = not any(type(m).__name__ == "QuantLinearSim" for m in verify.modules())
print("verify_model_clean", verify_clean)
