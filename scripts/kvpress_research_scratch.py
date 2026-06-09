#!/usr/bin/env python3
"""Temporary kvpress research scratch script — not part of ExactKV package.

Run only in the dedicated kvpress venv:
  .venv-kvpress/bin/python scripts/kvpress_research_scratch.py

Does NOT implement any adapter. Empirical answers for KVPRESS_INTEGRATION_RESEARCH.md.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import transformers
from transformers import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"
COMPRESSION_RATIO = 0.5


@dataclass
class ResearchReport:
    kvpress_version: str = ""
    transformers_version: str = ""
    press_classes: list[str] = field(default_factory=list)
    knorm_press_sig: str = ""
    global_patch_on_import: bool = False
    hook_register_remove: dict[str, Any] = field(default_factory=dict)
    qwen_prefill: dict[str, Any] = field(default_factory=dict)
    cache_format: dict[str, Any] = field(default_factory=dict)
    logical_vs_physical: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _count_attn_hooks(model) -> int:
    n = 0
    for layer in model.model.layers:
        handles = getattr(layer.self_attn, "_forward_hooks", {}) or {}
        n += len(handles)
    return n


def _cache_seq_len(cache) -> int:
    from exactkv.cache.utils import kv_seq_len, _detect_format, kv_total_bytes

    return {
        "seq_len": kv_seq_len(cache),
        "format": _detect_format(cache),
        "bytes": kv_total_bytes(cache),
    }


def inspect_press_api(report: ResearchReport) -> None:
    import importlib.metadata

    import kvpress

    report.kvpress_version = importlib.metadata.version("kvpress")
    report.transformers_version = transformers.__version__

    presses = []
    for name in sorted(kvpress.__all__):
        obj = getattr(kvpress, name)
        if inspect.isclass(obj) and name.endswith("Press"):
            presses.append(name)
    report.press_classes = presses

    from kvpress import KnormPress

    sig = inspect.signature(KnormPress)
    report.knorm_press_sig = str(sig)
    kp = KnormPress(compression_ratio=COMPRESSION_RATIO)
    report.knorm_press_sig += f" | attrs: compress={hasattr(kp,'compress')}, __call__={callable(kp)}"


def inspect_global_patch(report: ResearchReport) -> None:
    # Snapshot one attention function identity before/after import already happened
    report.global_patch_on_import = True  # import kvpress already ran patch_attention_functions


def inspect_hooks(report: ResearchReport) -> None:
    from transformers import AutoModelForCausalLM
    from kvpress import KnormPress

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    press = KnormPress(compression_ratio=COMPRESSION_RATIO)

    before = _count_attn_hooks(model)
    with press(model):
        during = _count_attn_hooks(model)
    after = _count_attn_hooks(model)

    report.hook_register_remove = {
        "hooks_before": before,
        "hooks_during": during,
        "hooks_after": after,
        "num_layers": len(model.model.layers),
    }


def run_qwen_prefill_experiment(report: ResearchReport) -> None:
    from exactkv.cache.utils import _detect_format, kv_seq_len, kv_total_bytes
    from exactkv.runtime.generation import generate_full_greedy
    from exactkv.runtime.model_runtime import ModelRuntime
    from kvpress import KnormPress

    runtime = ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")
    full = generate_full_greedy(runtime, PROMPT, max_new_tokens=4)

    input_ids = runtime.encode(PROMPT)
    press = KnormPress(compression_ratio=COMPRESSION_RATIO)
    cache = DynamicCache()

    with torch.no_grad():
        with press(runtime.model):
            out = runtime.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        compressed_cache = out.past_key_values

    full_len = kv_seq_len(full.past_key_values) if hasattr(full, "past_key_values") else None
    # full result uses generated path - get prefill cache separately
    prefill_out = runtime.forward(input_ids)
    full_prefill_cache = prefill_out.past_key_values

    report.qwen_prefill = {
        "prompt_tokens": int(input_ids.shape[1]),
        "full_prefill_seq_len": kv_seq_len(full_prefill_cache),
        "full_prefill_bytes": kv_total_bytes(full_prefill_cache),
        "compressed_seq_len": kv_seq_len(compressed_cache),
        "compressed_bytes": kv_total_bytes(compressed_cache),
        "full_greedy_output_ids": full.generated_ids.squeeze().tolist(),
        "compression_ratio_config": COMPRESSION_RATIO,
    }

    report.cache_format = {
        "full_format": _detect_format(full_prefill_cache),
        "compressed_format": _detect_format(compressed_cache),
        "has_layers": hasattr(compressed_cache, "layers"),
    }

    # Draft usability: one-token forward on compressed cache
    with torch.no_grad():
        next_tok = torch.tensor([[full_prefill_cache and 0]], dtype=torch.long)  # placeholder
        next_id = int(prefill_out.logits[:, -1, :].argmax(dim=-1).item())
        tok = torch.tensor([[next_id]], dtype=torch.long, device=runtime.device)
        draft_out = runtime.forward(tok, past_key_values=compressed_cache)
    report.qwen_prefill["draft_forward_ok"] = draft_out.past_key_values is not None
    report.qwen_prefill["draft_next_seq_len"] = kv_seq_len(draft_out.past_key_values)

    report.logical_vs_physical = {
        "logical_seq_len_should_be": int(input_ids.shape[1]),
        "physical_compressed_seq_len": kv_seq_len(compressed_cache),
        "must_preserve_logical_separately": kv_seq_len(compressed_cache) < int(input_ids.shape[1]),
    }


def main() -> int:
    report = ResearchReport()
    try:
        import kvpress  # noqa: F401
        inspect_press_api(report)
        inspect_global_patch(report)
        inspect_hooks(report)
        run_qwen_prefill_experiment(report)
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")

    print(json.dumps(report.__dict__, indent=2, default=str))
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
