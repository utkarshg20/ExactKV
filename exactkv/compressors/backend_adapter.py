"""BackendAdapter base class and PassThroughBackendAdapter proof-of-concept.

V6 Phase B — real-backend adapter boundary.

BackendAdapter is an abstract base class that satisfies the KVCompressor protocol
via sealed public methods.  Each public method enforces ExactKV correctness
invariants (no mutation of FullKVState, determinism, workspace-memory honesty)
before and after delegating to a small set of abstract protected hooks that
subclasses implement to integrate a real backend.

PassThroughBackendAdapter is a zero-dependency PoC:
  - stores full-precision KV tensors unchanged (no quantisation, no dropping)
  - materialises back to a HF-compatible cache using cache/utils.rebuild_cache
  - behaves token-for-token identically to NoOpCompressor
  - exercises every code path in BackendAdapter without wrapping any real library

No external backend dependencies.  No throughput, latency, speedup, tokens/sec,
or runtime_seconds fields.  total_kv_footprint_bytes is a conservative accounting
sum, not a measured peak GPU memory value.
"""
from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import Any, Generator

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, kv_total_bytes, rebuild_cache
from exactkv.compressors.base import CompressorCapabilities, CompressionStats


class BackendAdapter(abc.ABC):
    """Abstract base class for real-backend KVCompressor adapters.

    Implements the KVCompressor protocol via sealed public methods that enforce
    common correctness invariants.  Subclasses integrate a specific backend by
    overriding the three protected abstract methods.

    Subclass obligations
    --------------------
    Class attributes (set at class level, not in __init__):
        name            str  — registry key (lower-case, underscore-separated).
        capabilities    CompressorCapabilities — populated with backend identity.

    Protected abstract methods:
        _backend_compress(k_tensors, v_tensors, cache_format) -> dict
            Compress the (already-cloned) KV tensors.  Return a dict
            (backend_data) that will be stored in CompressedKVState.data.
            Must include "cache_format".  The base class injects
            "__full_bytes__" and "__num_layers__" automatically.

        _backend_materialize(backend_data, cache_format) -> past_key_values
            Reconstruct a HF-compatible cache from backend_data.
            Must be deterministic.  Must NOT mutate backend_data.

        _backend_workspace_bytes(full_kv_bytes, backend_data) -> dict
            Return workspace-aware accounting with keys:
              stored_kv_bytes, materialized_working_kv_bytes,
              metadata_bytes, temporary_workspace_bytes,
              total_kv_footprint_bytes.
            total_kv_footprint_bytes is a conservative accounting sum,
            NOT a measured peak GPU memory value.

    Optional override:
        _get_next_token_id(state, backend_data) -> int
            Return the compressed model's next-token prediction.  Default
            returns state.next_token_id, which is correct for lossless adapters.
            Lossy backends must override to run a materialize + forward pass.

    Correctness policies
    --------------------
    * compress() clones every KV tensor before passing to _backend_compress so
      that the authoritative FullKVState is never mutated.
    * update_after_commit() always re-compresses from the new full state.
    * stats() requires no extra forward pass — all byte counts come from values
      stored in backend_data at compress() time.
    * No performance claims in any returned data structure.
    """

    # Subclasses must set these at class level.
    name: str
    capabilities: CompressorCapabilities

    # ── Public sealed protocol methods ──────────────────────────────────────

    def _compresses_via_full_state(self) -> bool:
        """True when compression replays from FullKVState (e.g. kvpress hooks).

        Replay backends implement ``_backend_compress_from_full_state`` instead
        of the tensor-only ``_backend_compress`` path.
        """
        return False

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Create the backend's compressed representation from full_state.

        Does NOT mutate state or its past_key_values.
        Invariant: result.logical_seq_len == state.seq_len.
        """
        full_kv_bytes = kv_total_bytes(state.past_key_values)

        if self._compresses_via_full_state():
            backend_data = self._backend_compress_from_full_state(state)
            num_layers = backend_data.get("__num_layers__", 0)
        else:
            k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
            num_layers = len(k_tensors)

            # Clone before forwarding so _backend_compress cannot accidentally
            # mutate the authoritative full-state tensors.
            k_clones = [t.clone() for t in k_tensors]
            v_clones = [t.clone() for t in v_tensors]

            backend_data = self._backend_compress(k_clones, v_clones, cache_format)

        # Inject bookkeeping metadata so stats() can operate without an extra
        # forward pass.  setdefault so subclasses may provide their own values.
        backend_data.setdefault("__full_bytes__", full_kv_bytes)
        backend_data.setdefault("__num_layers__", num_layers)

        next_token_id = self._get_next_token_id(state, backend_data)

        result = CompressedKVState(
            data=backend_data,
            metadata={"next_token_id": next_token_id},
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

        if result.logical_seq_len != state.seq_len:
            raise RuntimeError(
                f"{self.name}.compress: alignment violation: "
                f"result.logical_seq_len={result.logical_seq_len} "
                f"!= state.seq_len={state.seq_len}"
            )
        return result

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Return past_key_values suitable for the draft model.

        Must NOT mutate compressed.data.
        Returns a HF-compatible cache (tuple, dynamic_v4, or dynamic_v5).
        """
        bd = compressed.data
        cache_format = bd["cache_format"]
        return self._backend_materialize(bd, cache_format)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Re-synchronize the compressed state after a commit (V1 strategy).

        Default: recompress from scratch from new_full_state.
        Subclasses may override for incremental update, but must preserve
        result.logical_seq_len == new_full_state.seq_len.
        """
        result = self.compress(new_full_state)
        if result.logical_seq_len != new_full_state.seq_len:
            raise RuntimeError(
                f"{self.name}.update_after_commit: alignment violation after commit"
            )
        return result

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        """Return CompressionStats with all V5 workspace-aware fields.

        All byte counts come from backend_data injected during compress() —
        no additional forward pass is needed.

        No performance fields are returned.  total_kv_footprint_bytes is
        a conservative accounting sum, not a measured peak GPU memory value.
        """
        bd = compressed.data
        full_bytes = bd.get("__full_bytes__", 0)
        ws = self._backend_workspace_bytes(full_bytes, bd)

        stored = ws["stored_kv_bytes"]
        materialized = ws["materialized_working_kv_bytes"]
        metadata = ws["metadata_bytes"]
        temporary = ws["temporary_workspace_bytes"]
        total = ws["total_kv_footprint_bytes"]

        compressed_bytes = max(stored + metadata, 1)
        compression_ratio = compressed_bytes / max(full_bytes, 1)
        memory_reduction_factor = full_bytes / max(compressed_bytes, 1)

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            memory_reduction_factor=memory_reduction_factor,
            seq_len=compressed.logical_seq_len,
            num_layers=bd.get("__num_layers__", 0),
            stored_kv_bytes=stored,
            materialized_working_kv_bytes=materialized,
            metadata_bytes=metadata,
            temporary_workspace_bytes=temporary,
            total_kv_footprint_bytes=total,
        )

    # ── Verification lifecycle guard (hook-based backends override) ─────────

    @contextmanager
    def verification_mode(self) -> Generator[None, None, None]:
        """Context manager guarding the verification path.

        Default implementation is a no-op — safe for pass-through and future
        offline backends.  Hook-based subclasses (e.g. kvpress) may override to
        assert no forward hooks are active while ``VerificationEngine`` runs.

        ``ExactKVGenerator`` wraps ``verify_sequential`` inside this context
        when the compressor provides ``verification_mode``.
        """
        yield

    def _backend_compress_from_full_state(self, state: FullKVState) -> dict:
        """Replay compression from an authoritative FullKVState.

        Used by hook-based backends (kvpress).  Must NOT mutate ``state`` or
        ``state.past_key_values``.  Return ``backend_data`` with
        ``cache_format`` and backend-specific payload.

        Default: not implemented — only called when ``_compresses_via_full_state``
        is True.
        """
        raise NotImplementedError(
            f"{self.name} declares _compresses_via_full_state but does not "
            "implement _backend_compress_from_full_state"
        )

    # ── Optional override (lossless adapters use default) ───────────────────

    def _get_next_token_id(self, state: FullKVState, backend_data: dict) -> int:
        """Return the compressed model's next-token prediction.

        Default: state.next_token_id — correct for lossless (pass-through)
        adapters.  Lossy backends that differ from full-KV predictions must
        override this to run materialize_for_draft() then a forward pass.
        """
        return state.next_token_id

    # ── Abstract backend hooks (subclasses must implement) ───────────────────

    @abc.abstractmethod
    def _backend_compress(
        self,
        k_tensors: list,
        v_tensors: list,
        cache_format: str,
    ) -> dict:
        """Compress the (already-cloned) KV tensors using the backend.

        Args:
            k_tensors:    Cloned key tensors, one per layer.  May be modified.
            v_tensors:    Cloned value tensors, one per layer.  May be modified.
            cache_format: "tuple" | "dynamic_v4" | "dynamic_v5".

        Returns:
            backend_data dict stored in CompressedKVState.data.
            Must include "cache_format" (string).
            The base class will inject "__full_bytes__" and "__num_layers__"
            via setdefault if not already present.
        """
        ...

    @abc.abstractmethod
    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        """Reconstruct a HF-compatible past_key_values from backend_data.

        Must be deterministic for fixed backend_data.
        Must NOT mutate backend_data.
        Must return an object whose format is detectable by cache/utils._detect_format.
        """
        ...

    @abc.abstractmethod
    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,
        backend_data: dict,
    ) -> dict:
        """Return workspace-aware memory accounting.

        Required keys (all int):
            stored_kv_bytes               — bytes of compressed/encoded tensors.
            materialized_working_kv_bytes — bytes when dequantised for attention.
            metadata_bytes                — scales, zero-points, or similar.
            temporary_workspace_bytes     — transient scratch (conservative).
            total_kv_footprint_bytes      — accounting sum of above four.

        total_kv_footprint_bytes is a conservative accounting sum,
        NOT a measured peak GPU memory value.
        """
        ...


# ---------------------------------------------------------------------------
# PassThroughBackendAdapter — zero-dependency PoC
# ---------------------------------------------------------------------------

class PassThroughBackendAdapter(BackendAdapter):
    """Zero-dependency proof-of-concept adapter for V6 Phase B.

    Stores full-precision KV tensors unchanged and materialises them back via
    cache/utils.rebuild_cache.  Behaves identically to NoOpCompressor:
    every drafted token matches the full-model prediction, so acceptance_rate
    must be 1.0 in any ExactKV run.

    Purpose: exercise the BackendAdapter boundary without integrating any real
    backend.  Confirms that the sealed public API, invariant checks, capability
    metadata, and workspace-memory accounting all work end-to-end.

    NOT a compression backend.  Does NOT reduce memory.  Does NOT provide
    any compression benefit.  backend_name='passthrough' signals this is an
    internal PoC, not an external library.
    """

    name: str = "backend_passthrough"

    capabilities: CompressorCapabilities = CompressorCapabilities(
        name="backend_passthrough",
        compressor_type="identity",
        is_simulated=False,
        supports_real_bytes_claim=False,
        supports_token_dropping=False,
        supports_quantization=False,
        key_bit_width=None,
        value_bit_width=None,
        asymmetric=False,
        backend_name="passthrough",
        backend_version="0",
        adapter_name="PassThroughBackendAdapter",
        adapter_version="0.1.0",
        notes=(
            "V6 Phase B proof-of-concept adapter. "
            "Stores full-precision KV tensors unchanged — no quantisation, "
            "no token dropping, no memory reduction. "
            "Exercises the BackendAdapter boundary without wrapping any real "
            "backend library. "
            "Behaves identically to NoOpCompressor: acceptance_rate == 1.0. "
            "backend_name='passthrough' is an internal identifier, not an "
            "external library name. "
            "supports_real_bytes_claim=False: stored bytes equal full-precision "
            "bytes — no compression has occurred."
        ),
    )

    def _backend_compress(
        self,
        k_tensors: list,
        v_tensors: list,
        cache_format: str,
    ) -> dict:
        """Store cloned key/value tensors unchanged."""
        return {
            "k": k_tensors,
            "v": v_tensors,
            "cache_format": cache_format,
        }

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        """Rebuild a HF-compatible cache from stored tensors."""
        return rebuild_cache(
            backend_data["k"],
            backend_data["v"],
            cache_format,
            0,  # seq_len is unused by rebuild_cache but required by the signature
        )

    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,
        backend_data: dict,  # noqa: ARG002
    ) -> dict:
        """Workspace accounting for a pass-through (no-compression) adapter.

        stored_kv_bytes               = full_kv_bytes (full-precision clone)
        materialized_working_kv_bytes = full_kv_bytes (same layout as stored)
        metadata_bytes                = 0 (no scales or zero-points)
        temporary_workspace_bytes     = 0 (no quantise/dequantise scratch)
        total_kv_footprint_bytes      = stored + materialized (conservative sum)

        total_kv_footprint_bytes is a conservative accounting sum,
        NOT a measured peak GPU memory value.
        """
        stored = full_kv_bytes
        materialized = full_kv_bytes
        metadata = 0
        temporary = 0
        total = stored + materialized + metadata + temporary
        return {
            "stored_kv_bytes": stored,
            "materialized_working_kv_bytes": materialized,
            "metadata_bytes": metadata,
            "temporary_workspace_bytes": temporary,
            "total_kv_footprint_bytes": total,
        }
