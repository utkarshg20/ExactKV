"""ExactKV benchmark runner.

Runs a single prompt under three modes and reports a JSON-compatible dict:
  1. ``full``    — generate_full_greedy (ground truth)
  2. ``lossy``   — generate_lossy_greedy (no verification)
  3. ``exactkv`` — ExactKVGenerator draft-verify-commit loop

The runner is single-threaded and correctness-first.  It does NOT report
timing, throughput, latency, or speedup numbers.

V2 addition: ``run_one`` now includes ``compressor_capabilities`` in the
returned dict so that reports.py can enrich JSON/CSV output with compressor
metadata (``is_simulated``, ``supports_real_bytes_claim``, etc.).

V2.4.2 addition: ``capture_divergence_topk`` captures the top-k full and
lossy logit distributions at the first divergence point for forensic analysis.

Supported compressor names: ``"noop"``, ``"int8"``, ``"int4_sim"``, ``"debug_noise"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from exactkv.benchmarks.prompts import load_prompts
from exactkv.compressors import get_compressor
from exactkv.metrics.acceptance import summarize_acceptance
from exactkv.metrics.exactness import first_divergence_idx, token_exact_match
from exactkv.metrics.memory import estimate_kv_memory
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy, generate_lossy_greedy
from exactkv.runtime.model_runtime import ModelRuntime


@dataclass
class RunConfig:
    """Configuration for one benchmark run."""
    compressor_name: str = "int8"
    draft_len: int = 4
    max_new_tokens: int = 32


def run_one(
    runtime: ModelRuntime,
    prompt_entry: dict,
    config: RunConfig,
) -> dict:
    """Benchmark a single prompt and return a JSON-compatible report dict.

    Args:
        runtime:      Loaded ModelRuntime.
        prompt_entry: Dict with at least ``prompt_id``, ``category``, ``prompt``.
        config:       RunConfig specifying compressor, draft_len, max_new_tokens.

    Returns:
        Dict with keys: prompt_id, prompt, category, model_name,
        compressor_name, draft_len, max_new_tokens, full, lossy, exactkv,
        memory, exactkv_failure.
    """
    compressor = get_compressor(config.compressor_name)
    # Capture capabilities for report enrichment (V2); safe for compressors
    # that predate the capabilities attribute.
    caps_dict: dict = {}
    if hasattr(compressor, "capabilities"):
        from dataclasses import asdict
        caps_dict = asdict(compressor.capabilities)

    prompt = prompt_entry["prompt"]
    max_new = config.max_new_tokens

    # 1. Full greedy (ground truth)
    full_res = generate_full_greedy(runtime, prompt, max_new)
    full_ids = full_res.generated_ids.squeeze(0).tolist()

    # 2. Lossy greedy (no verification)
    lossy_res = generate_lossy_greedy(runtime, prompt, compressor, max_new)
    lossy_ids = lossy_res.generated_ids.squeeze(0).tolist()

    lossy_exact = token_exact_match(full_res.generated_ids, lossy_res.generated_ids)
    lossy_div = first_divergence_idx(full_res.generated_ids, lossy_res.generated_ids)

    # 3. ExactKV
    ekv_res = ExactKVGenerator(runtime, compressor, draft_len=config.draft_len).generate(
        prompt, max_new
    )
    ekv_ids = ekv_res.output_ids.squeeze(0).tolist()

    ekv_exact = token_exact_match(full_res.generated_ids, ekv_res.output_ids)
    acceptance = summarize_acceptance(ekv_res.traces)

    # 4. Memory estimate (fresh prefill — independent of generation)
    mem = estimate_kv_memory(runtime, prompt, compressor)

    return {
        "prompt_id": prompt_entry["prompt_id"],
        "prompt": prompt,
        "category": prompt_entry.get("category", "unknown"),
        "model_name": runtime.model_name,
        "compressor_name": config.compressor_name,
        "compressor_capabilities": caps_dict,   # V2: enables report honesty fields
        "draft_len": config.draft_len,
        "max_new_tokens": max_new,
        "full": {
            "output_ids": full_ids,
            "output_text": full_res.output_text,
        },
        "lossy": {
            "output_ids": lossy_ids,
            "output_text": lossy_res.output_text,
            "token_exact_match": lossy_exact,
            "first_divergence_idx": lossy_div,
        },
        "exactkv": {
            "output_ids": ekv_ids,
            "output_text": ekv_res.output_text,
            "token_exact_match": ekv_exact,
            "acceptance": acceptance.to_dict(),
        },
        "memory": mem.to_dict(),
        "exactkv_failure": not ekv_exact,
    }


def _materialize_lossy_kv_for_topk(compressor: object, full_state: Any) -> Any:
    """Build lossy past_key_values for divergence top-k forensics."""
    import copy

    if not hasattr(compressor, "compress") or not hasattr(compressor, "materialize_for_draft"):
        raise TypeError(
            "capture_divergence_topk requires a KVCompressor with "
            "compress(FullKVState) and materialize_for_draft()"
        )
    compressed = compressor.compress(full_state)
    return copy.deepcopy(compressor.materialize_for_draft(compressed))


def capture_divergence_topk(
    runtime: ModelRuntime,
    prompt: str,
    compressor: object,
    full_ids: list[int],
    lossy_ids: list[int],
    *,
    k: int = 5,
) -> dict | None:
    """Capture the top-k logit distributions at the first divergence token.

    Runs one extra forward pass each for the full and lossy KV path up to the
    divergence point, then returns the softmax probabilities for the top-k tokens
    from each path alongside the argmax gap (logit margin).

    Returns None if the sequences are identical (no divergence) or if the
    divergence point is at token 0 (cannot prefill to that position cheaply).

    The returned dict has the shape::

        {
          "divergence_token_idx": int,
          "full_top_k": [{"token_id": int, "token_str": str, "prob": float}, ...],
          "lossy_top_k": [{"token_id": int, "token_str": str, "prob": float}, ...],
          "full_top1_logit_margin": float,   # prob[0] - prob[1]
          "lossy_top1_logit_margin": float,
          "kl_div_approx": float,            # KL(full || lossy) at divergence step
        }

    This function requires ``torch.nn.functional`` and triggers two extra GPU
    forward passes up to the divergence step. Use for post-hoc forensic analysis
    of the most diagnostic divergent cells only; do not apply to all cells.
    """
    try:
        import torch
        import torch.nn.functional as F
        from exactkv.runtime.generation import prefill_to_full_state
        from exactkv.metrics.exactness import first_divergence_idx
    except ImportError:
        return None

    full_tensor = torch.tensor([full_ids], dtype=torch.long, device=runtime.device)
    lossy_tensor = torch.tensor([lossy_ids], dtype=torch.long, device=runtime.device)
    div_idx = first_divergence_idx(full_tensor, lossy_tensor)
    if div_idx is None or div_idx < 1:
        return None

    def topk_at_divergence(token_sequence: list[int], apply_compression: bool) -> tuple[list[dict], list[dict]]:
        """Re-run the prefix up to div_idx and extract top-k probabilities."""
        full_state = prefill_to_full_state(runtime, prompt)
        past_kv = full_state.past_key_values

        if apply_compression:
            past_kv = _materialize_lossy_kv_for_topk(compressor, full_state)

        # Step through tokens 0..div_idx-2 to arrive at the state before div_idx
        for tok_id in token_sequence[1:div_idx]:
            tok_tensor = torch.tensor([[tok_id]], dtype=torch.long, device=runtime.device)
            out = runtime.forward(input_ids=tok_tensor, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

        # One more step to get logits at position div_idx
        tok_tensor = torch.tensor(
            [[token_sequence[div_idx - 1]]], dtype=torch.long, device=runtime.device
        )
        out = runtime.forward(input_ids=tok_tensor, past_key_values=past_kv, use_cache=True)
        logits_vec = out.logits[:, -1, :].squeeze(0)
        probs = F.softmax(logits_vec, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k)
        entries = [
            {
                "token_id": int(topk_ids[i].item()),
                "token_str": runtime.tokenizer.decode([int(topk_ids[i].item())]),
                "prob": round(float(topk_probs[i].item()), 6),
            }
            for i in range(k)
        ]
        return probs, entries

    try:
        full_probs, full_entries = topk_at_divergence(full_ids, apply_compression=False)
        lossy_probs, lossy_entries = topk_at_divergence(lossy_ids, apply_compression=True)

        margin_full = (
            full_entries[0]["prob"] - full_entries[1]["prob"] if len(full_entries) > 1 else 0.0
        )
        margin_lossy = (
            lossy_entries[0]["prob"] - lossy_entries[1]["prob"] if len(lossy_entries) > 1 else 0.0
        )

        eps = 1e-9
        p = full_probs.clamp(min=eps)
        q = lossy_probs.clamp(min=eps)
        kl = float((p * (p / q).log()).sum().item())

        return {
            "divergence_token_idx": div_idx,
            "full_top_k": full_entries,
            "lossy_top_k": lossy_entries,
            "full_top1_logit_margin": round(margin_full, 6),
            "lossy_top1_logit_margin": round(margin_lossy, 6),
            "kl_div_approx": round(kl, 6),
        }
    except (TypeError, ValueError):
        raise
    except Exception:
        return None


def run_suite(
    runtime: ModelRuntime,
    prompts: list[dict],
    config: RunConfig,
) -> dict:
    """Run the benchmark on a list of prompts and aggregate results.

    Returns:
        Dict with ``results`` (per-prompt dicts) and ``aggregate`` summary.
    """
    results: list[dict] = []
    failures = 0

    for entry in prompts:
        report = run_one(runtime, entry, config)
        results.append(report)
        if report["exactkv_failure"]:
            failures += 1

    total = len(results)
    return {
        "results": results,
        "aggregate": {
            "total_prompts": total,
            "compressor_name": config.compressor_name,
            "exactkv_failures": failures,
            "exactkv_pass_rate": (total - failures) / max(total, 1),
        },
    }
