"""ExactKV runtime — generation and experimental entry points."""
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

_EXPERIMENTAL_EXPORTS = {
    "EXPERIMENT_054_ID",
    "EXPERIMENT_056_ID",
    "EXP054_CLAIM_NOTE",
    "EXP056_CLAIM_NOTE",
    "EXPERIMENTAL_RUNTIME_CLAIM_NOTE",
    "EXPERIMENTAL_CUDA_GATE_CLAIM_NOTE",
    "CLI_OPT_IN_REQUIRED",
    "RUNTIME_PATH_EXPERIMENTAL",
    "DEFAULT_CUDA_GATE_DRAFT_LEN",
    "DEFAULT_CUDA_GATE_MAX_NEW_TOKENS",
    "CudaRuntimeGateDtypeResult",
    "CudaRuntimeGateResult",
    "ExperimentalRestoredVerifierConfig",
    "ExperimentalRuntimeMode",
    "ExperimentalRuntimeResult",
    "default_cuda_gate_experimental_config",
    "default_experimental_smoke_config",
    "report_to_exp054_json",
    "report_to_exp056_json",
    "run_cuda_restored_verifier_runtime_gate",
    "run_experimental_restored_verifier",
    "validate_exp054_report",
    "validate_exp056_report",
    "validate_experimental_config",
}

_EXPERIMENTAL_CLI_EXPORTS = {
    "EXPERIMENT_055_ID",
    "EXP055_CLAIM_NOTE",
    "EXPERIMENTAL_CLI_CLAIM_NOTE",
    "CLI_FLAG_OPTION",
    "CLI_FLAG_DEST",
    "ExperimentalCliResolution",
    "add_experimental_restored_verifier_cli_args",
    "format_cli_summary",
    "report_to_exp055_json",
    "resolve_experimental_cli_args",
    "run_experimental_restored_verifier_from_cli",
    "validate_exp055_report",
}

__all__ = [
    "CLI_FLAG_DEST",
    "CLI_FLAG_OPTION",
    "CLI_OPT_IN_REQUIRED",
    "DEFAULT_CUDA_GATE_DRAFT_LEN",
    "DEFAULT_CUDA_GATE_MAX_NEW_TOKENS",
    "EXPERIMENT_054_ID",
    "EXPERIMENT_055_ID",
    "EXPERIMENT_056_ID",
    "EXP054_CLAIM_NOTE",
    "EXP055_CLAIM_NOTE",
    "EXP056_CLAIM_NOTE",
    "EXPERIMENTAL_CLI_CLAIM_NOTE",
    "EXPERIMENTAL_CUDA_GATE_CLAIM_NOTE",
    "EXPERIMENTAL_RUNTIME_CLAIM_NOTE",
    "RUNTIME_PATH_EXPERIMENTAL",
    "CudaRuntimeGateDtypeResult",
    "CudaRuntimeGateResult",
    "ExactKVGenerator",
    "ExperimentalCliResolution",
    "ExperimentalRestoredVerifierConfig",
    "ExperimentalRuntimeMode",
    "ExperimentalRuntimeResult",
    "ModelRuntime",
    "add_experimental_restored_verifier_cli_args",
    "default_cuda_gate_experimental_config",
    "default_experimental_smoke_config",
    "format_cli_summary",
    "generate_full_greedy",
    "report_to_exp054_json",
    "report_to_exp055_json",
    "report_to_exp056_json",
    "resolve_experimental_cli_args",
    "run_cuda_restored_verifier_runtime_gate",
    "run_experimental_restored_verifier",
    "run_experimental_restored_verifier_from_cli",
    "validate_exp054_report",
    "validate_exp055_report",
    "validate_exp056_report",
    "validate_experimental_config",
]


def __getattr__(name: str):
    if name in _EXPERIMENTAL_EXPORTS:
        from exactkv.runtime import experimental

        return getattr(experimental, name)
    if name in _EXPERIMENTAL_CLI_EXPORTS:
        from exactkv.runtime import experimental_cli

        return getattr(experimental_cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
