"""Restricted KVQuant simquant adapter (isolated KVQuant env only).

NOT registered in the default compressor registry.  The upstream ``kvquant``
package (KVQuant ``quant/``) is imported lazily when ``KVQuantSimAdapter`` is
constructed.

Scope: faithful simquant replay via draft-model clone + pre-RoPE k_proj/v_proj
QuantLinearSim.  No deployment CUDA, no forked transformers, no post-RoPE
tensor bridge.
"""

from __future__ import annotations

import copy
import os
import pickle
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import (
    _detect_format,
    extract_kv_tensors,
    kv_seq_len,
    kv_total_bytes,
    rebuild_cache,
)
from exactkv.compressors.backend_adapter import BackendAdapter
from exactkv.compressors.base import CompressorCapabilities
from exactkv.runtime.model_runtime import ModelRuntime


def _import_kvquant() -> Any:
    """Lazy import of the KVQuant quant package."""
    try:
        import kvquant  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "kvquant is not importable. Clone https://github.com/SqueezeAILab/KVQuant "
            "and pip install -e quant/ in an isolated venv with transformers~=4.44. "
            "Set EXACTKV_KVQUANT_QUANTIZERS to a quantizers.pickle path."
        ) from exc
    return kvquant


def _discover_backend_version() -> str:
    try:
        import kvquant.simquant_module_quantizer as smq  # noqa: PLC0415

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(smq.__file__))))
        git_head = os.path.join(root, ".git", "HEAD")
        if os.path.isfile(git_head):
            with open(git_head, encoding="utf-8") as fh:
                ref = fh.read().strip()
            if ref.startswith("ref: "):
                ref_path = os.path.join(root, ".git", ref[5:])
                if os.path.isfile(ref_path):
                    with open(ref_path, encoding="utf-8") as fh:
                        return fh.read().strip()[:12]
            return ref[:12]
        return f"dev:{os.path.basename(root)}"
    except Exception:
        return "unknown"


def _has_quant_linear_sim(model: torch.nn.Module) -> bool:
    try:
        from kvquant.simquant_module_quantizer import QuantLinearSim  # noqa: PLC0415
    except ImportError:
        return False
    return any(isinstance(m, QuantLinearSim) for m in model.modules())


@contextmanager
def _scoped_kvquant_qwen_bias_fix() -> Iterator[Any]:
    """Adapter-scoped patch for Qwen2 k_proj/v_proj bias (D4b finding).

    Upstream ``make_quant_sim`` passes ``tmp.bias is not None`` (bool) and
    ``QuantLinearSim.__init__`` uses ``if bias:``, which breaks when bias is a
    tensor.  Patches are restored on exit — no permanent KVQuant mutation.
    """
    import kvquant.simquant_module_quantizer as smq  # noqa: PLC0415

    orig_make = smq.make_quant_sim
    orig_init = smq.QuantLinearSim.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        """Route tensor bias through orig_init with bias=None, then restore."""
        args_list = list(args)
        saved_bias = None
        if len(args_list) >= 7 and isinstance(args_list[6], torch.Tensor):
            saved_bias = args_list[6]
            args_list[6] = None
        orig_init(self, *args_list, **kwargs)
        if saved_bias is not None:
            self.bias = saved_bias.detach().cpu()

    def _patched_make_quant_sim(
        module,
        quantizers,
        bits,
        name="",
        perchannel=True,
        include_sparse=False,
        sparsity_threshold=0.999,
        dynamicquantization=False,
        nuq=False,
        nf_nuq=0,
        norm=False,
        cap_outliers=-1,
        first_few_fp16=-1,
        clamp=False,
    ):
        QuantLinearSim = smq.QuantLinearSim
        if isinstance(module, QuantLinearSim):
            return
        for attr in dir(module):
            tmp = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in quantizers.keys():
                delattr(module, attr)
                setattr(
                    module,
                    attr,
                    QuantLinearSim(
                        name1,
                        bits,
                        quantizers[name1],
                        tmp.in_features,
                        tmp.out_features,
                        tmp.weight,
                        tmp.bias,
                        perchannel=perchannel,
                        include_sparse=include_sparse,
                        sparsity_threshold=sparsity_threshold,
                        dynamicquantization=dynamicquantization,
                        nuq=nuq,
                        nf_nuq=nf_nuq,
                        norm=norm,
                        cap_outliers=cap_outliers,
                        first_few_fp16=first_few_fp16,
                        clamp=clamp,
                    ),
                )
            del tmp
        for name1, child in module.named_children():
            _patched_make_quant_sim(
                child,
                quantizers,
                bits,
                name + "." + name1 if name else name1,
                perchannel=perchannel,
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                dynamicquantization=dynamicquantization,
                nuq=nuq,
                nf_nuq=nf_nuq,
                norm=norm,
                cap_outliers=cap_outliers,
                first_few_fp16=first_few_fp16,
                clamp=clamp,
            )

    smq.make_quant_sim = _patched_make_quant_sim
    smq.QuantLinearSim.__init__ = _patched_init
    try:
        yield smq
    finally:
        smq.make_quant_sim = orig_make
        smq.QuantLinearSim.__init__ = orig_init


def _split_quantizers(quantizers: dict) -> tuple[dict, dict]:
    perchannel = {k: v for k, v in quantizers.items() if "k_proj" in k}
    pertoken = {k: v for k, v in quantizers.items() if "v_proj" in k}
    return perchannel, pertoken


def _move_cache_to_runtime(
    past_key_values: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Move materialized cache tensors to the runtime device/dtype."""
    cache_format = _detect_format(past_key_values)
    k_tensors, v_tensors, fmt = extract_kv_tensors(past_key_values)
    seq_len = kv_seq_len(past_key_values)
    k_out = [t.to(device=device, dtype=dtype) for t in k_tensors]
    v_out = [t.to(device=device, dtype=dtype) for t in v_tensors]
    return rebuild_cache(k_out, v_out, fmt or cache_format, seq_len)


class KVQuantSimAdapter(BackendAdapter):
    """Faithful restricted KVQuant simquant adapter.

    Replays prefill through a deep-copied draft model with QuantLinearSim on
    k_proj/v_proj only.  The authoritative ``runtime.model`` is never mutated.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        quantizers_path: str,
        *,
        abits: int = 4,
        isolate_draft_model: bool = True,
    ) -> None:
        _import_kvquant()

        if not os.path.isfile(quantizers_path):
            raise FileNotFoundError(f"KVQuant quantizers pickle not found: {quantizers_path}")

        self._runtime = runtime
        self._quantizers_path = quantizers_path
        self._abits = abits
        self._quantizer_pickle_bytes = os.path.getsize(quantizers_path)

        with open(quantizers_path, "rb") as handle:
            self._quantizers = pickle.load(handle)

        if isolate_draft_model:
            self._draft_model = copy.deepcopy(runtime.model)
        else:
            self._draft_model = runtime.model
        self._draft_model.eval()

        if torch.cuda.is_available():
            self._draft_model = self._draft_model.to(device="cuda", dtype=torch.float16)
            self._draft_device = torch.device("cuda")
        else:
            self._draft_device = next(self._draft_model.parameters()).device

        self._apply_quant_to_draft()

        if _has_quant_linear_sim(runtime.model):
            raise RuntimeError(
                "runtime.model must not contain QuantLinearSim modules; "
                "use isolate_draft_model=True"
            )

        self.name = "kvquant_sim_qwen05b"
        self.capabilities = CompressorCapabilities(
            name=self.name,
            compressor_type="quantization",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=True,
            key_bit_width=abits,
            value_bit_width=abits,
            key_bit_width_label="kvquant_sim_k",
            value_bit_width_label="kvquant_sim_v",
            asymmetric=False,
            backend_name="kvquant",
            backend_version=_discover_backend_version(),
            adapter_name="KVQuantSimAdapter",
            adapter_version="0.1.0",
            notes=(
                "Restricted V9 Phase D5 faithful KVQuant simquant adapter (isolated "
                "KVQuant venv only). Uses pre-RoPE k_proj/v_proj QuantLinearSim on a "
                "deep-copied draft model; replays prefill via _compresses_via_full_state. "
                "Not post-RoPE tensor approximation. Not KVQuant deployment CUDA. "
                "Not forked transformers deployment. Requires external quantizers.pickle "
                "(not committed). Not in the default compressor registry. "
                "supports_real_bytes_claim=False: stored bytes count quantizer pickle "
                "and metadata only, not packed KV bytes. "
                "Draft may diverge from full KV; verification uses authoritative full "
                "state only. ExactKV does not claim upstream KVQuant paper results. "
                "No throughput, latency, speedup, runtime, tokens/sec, active GPU "
                "memory, or production-readiness claims."
            ),
        )

    def _apply_quant_to_draft(self) -> None:
        perchannel, pertoken = _split_quantizers(self._quantizers)
        with _scoped_kvquant_qwen_bias_fix() as smq:
            smq.make_quant_sim(
                self._draft_model,
                perchannel,
                self._abits,
                perchannel=True,
                include_sparse=False,
            )
            smq.make_quant_sim(
                self._draft_model,
                pertoken,
                self._abits,
                perchannel=False,
                dynamicquantization=True,
            )

    def _compresses_via_full_state(self) -> bool:
        return True

    @torch.no_grad()
    def _backend_compress_from_full_state(self, state: FullKVState) -> dict:
        if _has_quant_linear_sim(self._runtime.model):
            raise RuntimeError("runtime.model must remain free of QuantLinearSim")

        input_ids = state.full_sequence_ids.to(self._draft_device)
        out = self._draft_model(input_ids=input_ids, use_cache=True)

        past_key_values = out.past_key_values
        if past_key_values is None:
            raise RuntimeError("KVQuant draft replay did not return past_key_values")

        cache_format = _detect_format(past_key_values)
        next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        num_layers = len(self._quantizers) // 2

        metadata_bytes = len(self._quantizers) * 64

        return {
            "cache_format": cache_format,
            "past_key_values": past_key_values,
            "__stored_kv_bytes__": self._quantizer_pickle_bytes,
            "__metadata_bytes_fixed__": metadata_bytes,
            "__compressed_next_token_id__": next_token_id,
            "__num_layers__": num_layers,
            "__quantizers_path__": self._quantizers_path,
            "__abits__": self._abits,
        }

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> dict:
        raise RuntimeError(
            f"{self.name} uses replay compression only; _backend_compress is not supported"
        )

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        pkv = backend_data["past_key_values"]
        return _move_cache_to_runtime(
            pkv,
            device=self._runtime.device,
            dtype=self._runtime.dtype,
        )

    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,
        backend_data: dict,
    ) -> dict:
        stored = int(backend_data.get("__stored_kv_bytes__", self._quantizer_pickle_bytes))
        metadata = int(backend_data.get("__metadata_bytes_fixed__", 0))
        materialized = full_kv_bytes
        temporary = max(full_kv_bytes // 4, 0)
        total = stored + materialized + metadata + temporary
        return {
            "stored_kv_bytes": stored,
            "materialized_working_kv_bytes": materialized,
            "metadata_bytes": metadata,
            "temporary_workspace_bytes": temporary,
            "total_kv_footprint_bytes": total,
        }

    def _get_next_token_id(self, state: FullKVState, backend_data: dict) -> int:
        return backend_data.get("__compressed_next_token_id__", state.next_token_id)


def create_kvquant_sim_adapter(
    runtime: ModelRuntime,
    quantizers_path: str | None = None,
    *,
    abits: int = 4,
    isolate_draft_model: bool = True,
) -> KVQuantSimAdapter:
    """Factory for the restricted KVQuant simquant adapter (not in default registry)."""
    path = quantizers_path or os.environ.get("EXACTKV_KVQUANT_QUANTIZERS")
    if not path:
        raise ValueError(
            "quantizers_path is required, or set EXACTKV_KVQUANT_QUANTIZERS "
            "to a quantizers.pickle file path"
        )
    return KVQuantSimAdapter(
        runtime=runtime,
        quantizers_path=path,
        abits=abits,
        isolate_draft_model=isolate_draft_model,
    )
