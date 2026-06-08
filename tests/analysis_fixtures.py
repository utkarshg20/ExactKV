"""Shared synthetic report fixtures for analysis tests.

These helpers build minimal report dicts that exercise the analysis layer
without requiring a model forward pass.  Each function returns a plain dict
matching the structure produced by ``runner.run_one`` / ``run_suite``.
"""
from __future__ import annotations

from typing import Any


def make_result(
    prompt_id: str = "p1",
    category: str = "test",
    compressor_name: str = "int8",
    draft_len: int = 4,
    exactkv_failure: bool = False,
    lossy_exact: bool = True,
    first_div_idx: int | None = None,
    total_drafted: int = 8,
    total_accepted: int = 7,
    total_rejected: int = 1,
    total_corrections: int = 1,
    acceptance_rate: float = 0.875,
    avg_accepted_per_round: float = 3.5,
    is_simulated: bool = False,
    supports_real_bytes_claim: bool = True,
) -> dict[str, Any]:
    """Return a synthetic run-one result dict."""
    return {
        "prompt_id": prompt_id,
        "category": category,
        "model_name": "Qwen/Qwen2.5-0.5B",
        "compressor_name": compressor_name,
        "compressor_capabilities": {
            "name": compressor_name,
            "compressor_type": "quantization",
            "is_simulated": is_simulated,
            "supports_real_bytes_claim": supports_real_bytes_claim,
            "supports_token_dropping": False,
            "supports_quantization": True,
            "notes": "",
        },
        "draft_len": draft_len,
        "max_new_tokens": 16,
        "full": {"output_ids": [10, 20, 30], "output_text": "abc"},
        "lossy": {
            "output_ids": [10, 20, 30] if lossy_exact else [10, 99, 30],
            "output_text": "abc" if lossy_exact else "aXc",
            "token_exact_match": lossy_exact,
            "first_divergence_idx": first_div_idx,
        },
        "exactkv": {
            "output_ids": [10, 99, 30] if exactkv_failure else [10, 20, 30],
            "output_text": "aXc" if exactkv_failure else "abc",
            "token_exact_match": not exactkv_failure,
            "acceptance": {
                "total_rounds": 2,
                "total_drafted": total_drafted,
                "total_accepted": total_accepted,
                "total_rejected": total_rejected,
                "total_corrections": total_corrections,
                "acceptance_rate": acceptance_rate,
                "avg_accepted_per_round": avg_accepted_per_round,
                "avg_drafted_per_round": total_drafted / 2,
            },
        },
        "memory": {
            "full_bytes": 4000,
            "compressed_bytes": 1000,
            "compression_ratio": 0.25,
            "memory_reduction_factor": 4.0,
        },
        "exactkv_failure": exactkv_failure,
    }


def make_report(*results: dict[str, Any]) -> dict[str, Any]:
    """Wrap results into a minimal report dict."""
    return {"results": list(results)}
