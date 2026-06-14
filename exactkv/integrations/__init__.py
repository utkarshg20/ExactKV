"""Integration contract layers (Phase 11F+). No runtime wiring."""
from exactkv.integrations.vllm_contract import (
    VLLMCacheCapability,
    VLLMIntegrationStatus,
    VLLMPrototypeGate,
    VLLMPrototypePlan,
    assert_vllm_not_required,
    build_default_vllm_prototype_plan,
    validate_vllm_prototype_plan,
)

__all__ = [
    "VLLMCacheCapability",
    "VLLMIntegrationStatus",
    "VLLMPrototypeGate",
    "VLLMPrototypePlan",
    "assert_vllm_not_required",
    "build_default_vllm_prototype_plan",
    "validate_vllm_prototype_plan",
]
