"""Tensor-level attention feasibility utilities (research only).

Not wired into ExactKV default generation or model inference.
"""

from exactkv.attention.streaming_quant_attention import (
    EXPERIMENT_066_ID,
    EXP066_CLAIM_NOTE,
    FORBIDDEN_ATTENTION_CLAIMS,
    AttentionFeasibilityResult,
    MemoryAccounting,
    QuantizedKV,
    attention_full,
    attention_materialized_compressed,
    attention_streaming_compressed,
    dequantize_kv_materialized,
    estimate_attention_memory_bytes,
    quantize_kv_int8_reference,
    run_attention_feasibility_cell,
    validate_exp066_report,
)

from exactkv.attention.hf_single_layer_probe import (
    DEFAULT_MODEL_ID,
    EXPERIMENT_067_ID,
    EXP067_CLAIM_NOTE,
    compute_drift_metrics,
    extract_qkv_from_qwen2_layer,
    run_exp067_probe,
    run_hf_attention_drift_cell,
    validate_exp067_report,
)

__all__ = [
    "EXPERIMENT_066_ID",
    "EXP066_CLAIM_NOTE",
    "FORBIDDEN_ATTENTION_CLAIMS",
    "AttentionFeasibilityResult",
    "MemoryAccounting",
    "QuantizedKV",
    "attention_full",
    "attention_materialized_compressed",
    "attention_streaming_compressed",
    "dequantize_kv_materialized",
    "estimate_attention_memory_bytes",
    "quantize_kv_int8_reference",
    "run_attention_feasibility_cell",
    "validate_exp066_report",
    "DEFAULT_MODEL_ID",
    "EXPERIMENT_067_ID",
    "EXP067_CLAIM_NOTE",
    "compute_drift_metrics",
    "extract_qkv_from_qwen2_layer",
    "run_exp067_probe",
    "run_hf_attention_drift_cell",
    "validate_exp067_report",
]
