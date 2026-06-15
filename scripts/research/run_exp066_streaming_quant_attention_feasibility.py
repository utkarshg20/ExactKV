#!/usr/bin/env python3
"""Experiment 066: streaming quantized-KV attention feasibility (Phase 16A).

Tensor-level reference probe only — not model inference or ExactKV integration.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.streaming_quant_attention import (  # noqa: E402
    EXPERIMENT_066_ID,
    EXP066_CLAIM_NOTE,
    FORBIDDEN_ATTENTION_CLAIMS,
    DEFAULT_EXP066_REPORT,
    run_attention_feasibility_cell,
    validate_exp066_report,
)

LIMITATIONS = [
    "Reference symmetric int8 KV only; not a production compressor.",
    "PyTorch tensor reference; no CUDA/Triton kernels.",
    "Theoretical memory accounting only; not measured active GPU memory.",
    "Streaming-vs-materialized pass does not imply model output preservation.",
    "No vLLM, LMCache, or ExactKV default-generation integration.",
    "Causal mode assumes queries occupy the last Q positions in the sequence.",
]


def _dtype_list() -> list[torch.dtype]:
    dtypes = [torch.float32]
    if hasattr(torch, "float16"):
        dtypes.append(torch.float16)
    return dtypes


def _sweep_cells() -> list[dict[str, Any]]:
    dtypes = _dtype_list()
    heads = (2, 4)
    queries = (1, 4)
    seq_lens = (32, 128, 512)
    dims = (32, 64)
    chunk_sizes = (16, 32, 64)
    cells: list[dict[str, Any]] = []
    seed = 6600

    for dtype in dtypes:
        for h in heads:
            for q_len in queries:
                for t_len in seq_lens:
                    for d in dims:
                        for chunk_size in chunk_sizes:
                            gen = torch.Generator(device="cpu")
                            gen.manual_seed(seed)
                            seed += 1
                            b = 1
                            k = torch.randn(b, h, t_len, d, generator=gen, dtype=dtype)
                            v = torch.randn(b, h, t_len, d, generator=gen, dtype=dtype)
                            q = torch.randn(b, h, q_len, d, generator=gen, dtype=dtype)

                            result = run_attention_feasibility_cell(
                                q=q,
                                k=k,
                                v=v,
                                chunk_size=chunk_size,
                                causal=False,
                            )
                            cell = {
                                "dtype": str(dtype).replace("torch.", ""),
                                "B": b,
                                "H": h,
                                "Q": q_len,
                                "T": t_len,
                                "D": d,
                                "chunk_size": chunk_size,
                                "causal": False,
                                "passed": result.passed,
                                "tolerance": result.tolerance,
                                "max_abs_streaming_vs_materialized": result.max_abs_streaming_vs_materialized,
                                "max_abs_full_vs_materialized": result.max_abs_full_vs_materialized,
                                "max_abs_full_vs_streaming": result.max_abs_full_vs_streaming,
                                "memory_accounting": result.memory_accounting.to_dict(),
                            }
                            cells.append(cell)
    return cells


def _summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    pass_cells = sum(1 for c in cells if c.get("passed"))
    failed_cells = len(cells) - pass_cells
    max_stream = max((c["max_abs_streaming_vs_materialized"] for c in cells), default=0.0)
    max_full_stream = max((c["max_abs_full_vs_streaming"] for c in cells), default=0.0)
    reductions = [
        c["memory_accounting"]["theoretical_streaming_working_reduction_vs_materialized"]
        for c in cells
        if c.get("memory_accounting")
    ]
    best_reduction = max(reductions) if reductions else 0.0
    worst_reduction = min(reductions) if reductions else 0.0
    status = "pass" if failed_cells == 0 and cells else "failed"
    return {
        "experiment_id": EXPERIMENT_066_ID,
        "status": status,
        "total_cells": len(cells),
        "pass_cells": pass_cells,
        "failed_cells": failed_cells,
        "max_streaming_vs_materialized_error": max_stream,
        "max_full_vs_streaming_error": max_full_stream,
        "best_theoretical_streaming_working_reduction": best_reduction,
        "worst_theoretical_streaming_working_reduction": worst_reduction,
        "cells": cells,
        "claim_note": EXP066_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": LIMITATIONS,
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made by this experiment."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 066 streaming quant attention feasibility")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP066_REPORT)
    args = parser.parse_args()

    cells = _sweep_cells()
    report = _summarize(cells)
    errors = validate_exp066_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 066: {report['status']} cells={report['total_cells']} "
        f"pass={report['pass_cells']} fail={report['failed_cells']}"
    )
    print(f"  max_streaming_vs_materialized={report['max_streaming_vs_materialized_error']:.6g}")
    print(f"  max_full_vs_streaming={report['max_full_vs_streaming_error']:.6g}")
    print(
        "  theoretical_reduction="
        f"{report['best_theoretical_streaming_working_reduction']:.4f}.."
        f"{report['worst_theoretical_streaming_working_reduction']:.4f}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
