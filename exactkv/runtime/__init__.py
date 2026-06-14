"""ExactKV runtime — generation and experimental entry points."""
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

_EXPERIMENTAL_EXPORTS = {
    "EXPERIMENT_054_ID",
    "EXP054_CLAIM_NOTE",
    "EXPERIMENTAL_RUNTIME_CLAIM_NOTE",
    "ExperimentalRestoredVerifierConfig",
    "ExperimentalRuntimeMode",
    "ExperimentalRuntimeResult",
    "default_experimental_smoke_config",
    "report_to_exp054_json",
    "run_experimental_restored_verifier",
    "validate_exp054_report",
    "validate_experimental_config",
}

__all__ = [
    "EXPERIMENT_054_ID",
    "EXP054_CLAIM_NOTE",
    "EXPERIMENTAL_RUNTIME_CLAIM_NOTE",
    "ExactKVGenerator",
    "ExperimentalRestoredVerifierConfig",
    "ExperimentalRuntimeMode",
    "ExperimentalRuntimeResult",
    "ModelRuntime",
    "default_experimental_smoke_config",
    "generate_full_greedy",
    "report_to_exp054_json",
    "run_experimental_restored_verifier",
    "validate_exp054_report",
    "validate_experimental_config",
]


def __getattr__(name: str):
    if name in _EXPERIMENTAL_EXPORTS:
        from exactkv.runtime import experimental

        return getattr(experimental, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
