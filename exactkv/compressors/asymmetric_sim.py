"""AsymmetricQuantSimCompressor — per-side K/V quantisation with independent widths.

Named no-arg subclasses for registry use
-----------------------------------------
The seven named compressors at the bottom of this module are thin no-arg
subclasses of ``AsymmetricQuantSimCompressor``.  They exist solely to satisfy
the registry contract (``get_compressor(name)`` → ``_REGISTRY[name]()``) and
bind a canonical name and ``(k_bits, v_bits)`` pair:

    K8/V4-sim   k8_v4_sim     is_simulated=True
    K8/V2-sim   k8_v2_sim     is_simulated=True
    K4/V8-sim   k4_v8_sim     is_simulated=True
    Kfull/V4-sim k_full_v4_sim is_simulated=True
    K4/Vfull-sim k4_v_full_sim is_simulated=True
    K8/Vfull    k8_v_full     is_simulated=False  ← no _sim (real storage only)
    Kfull/V8    k_full_v8     is_simulated=False  ← no _sim (real storage only)

The ``_sim`` suffix is present only when a compressor uses a simulated sub-INT8
width (4-bit or 2-bit), where the values are stored in ``int8`` containers
rather than being real packed sub-INT8.  ``k8_v_full`` and ``k_full_v8`` use
only real INT8 and full-precision passthrough — no simulated sub-INT8 — so they
carry no ``_sim`` suffix and report ``is_simulated=False``.


What "asymmetric" means here
-----------------------------
Keys and values are quantised (or left at full precision) **independently**,
each with its own per-tensor symmetric scale.  The K-side and V-side bit-widths
can differ, allowing policies like K8/V4, K-full/V4, K4/V-full, etc.

What "simulated" means here
----------------------------
Sub-INT8 widths (4-bit and 2-bit) are simulated: values are clipped to the
signed N-bit numeric range but stored in ``torch.int8`` containers.  No real
bit-packing is performed.  Compressors that use any simulated width have
``is_simulated=True`` and ``supports_real_bytes_claim=False``.

Supported widths
----------------
* ``None`` / ``"full"``  — passthrough at full precision (no quantisation)
* ``8``                  — per-tensor symmetric INT8; real storage (1 B/element)
* ``4``                  — INT4 range [-8, 7] stored in int8; simulated
* ``2``                  — INT2 range [-2, 1] stored in int8; simulated

Quantisation formula (per-tensor, per-side)::

    denom = max absolute value of the signed range (127 for INT8, 7 for INT4, 1 for INT2)
    scale = max(abs(tensor)) / denom       (1.0 for all-zero tensors)
    q     = round(tensor / scale).clamp(qmin, qmax).to(torch.int8)

Dequantisation::

    out = q.to(model_dtype) * scale

Memory accounting
-----------------
``CompressedBytes`` reflects **actual storage**:
* quantised side — 1 byte/element (int8 container) + 8 bytes (float64 scale)
* full-precision side — actual element_size * nelement bytes (stored as clone)
``full_bytes`` is the fp32 reference (4 bytes/element for all tensors).

Do NOT cite ``compressed_bytes`` for simulated sub-INT8 sides as evidence of
real INT4/INT2 memory savings.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, rebuild_cache
from exactkv.compressors.base import CompressorCapabilities, CompressionStats

# ---------------------------------------------------------------------------
# Width tables
# ---------------------------------------------------------------------------

# Maps bit-width → (qmin, qmax, denom) for symmetric signed quantisation.
_WIDTH_PARAMS: dict[int, tuple[int, int, float]] = {
    8: (-128, 127, 127.0),
    4: (-8,    7,   7.0),
    2: (-2,    1,   1.0),
}

_VALID_WIDTHS = frozenset({None, 8, 4, 2})

# Bytes used to store one float64 scale value.
_SCALE_BYTES = 8
# fp32 reference bytes per element.
_FP32_BYTES = 4


# ---------------------------------------------------------------------------
# Helpers — normalise input width specification
# ---------------------------------------------------------------------------

def _parse_bits(bits: int | str | None, side: str) -> int | None:
    """Convert a user-supplied width spec to a canonical int or None.

    Args:
        bits: One of ``None``, ``"full"``, ``8``, ``4``, ``2``.
        side: ``"k"`` or ``"v"`` — used in error messages only.

    Returns:
        ``None`` for full-precision passthrough, or an integer width.

    Raises:
        ValueError: if ``bits`` is not a recognised value.
    """
    if bits is None or bits == "full":
        return None
    if isinstance(bits, int) and bits in _WIDTH_PARAMS:
        return bits
    raise ValueError(
        f"Invalid bit-width for {side}_bits: {bits!r}. "
        f"Supported values: None, 'full', 8, 4, 2."
    )


# ---------------------------------------------------------------------------
# Per-tensor quantisation / dequantisation
# ---------------------------------------------------------------------------

def _quantize(t: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric quantisation to the signed N-bit range.

    Args:
        t:    Input floating-point tensor.
        bits: Target bit-width (8, 4, or 2).

    Returns:
        (q, scale): q is torch.int8 on the same device as t; scale is a
        Python float.
    """
    qmin, qmax, denom = _WIDTH_PARAMS[bits]
    abs_max = float(t.abs().max().item())
    scale = abs_max / denom if abs_max != 0.0 else 1.0
    q = (t / scale).round().clamp(qmin, qmax).to(torch.int8)
    return q, scale


def _dequantize(
    q: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct an approximation of the original tensor."""
    return q.to(dtype=dtype, device=device) * scale


# ---------------------------------------------------------------------------
# Per-layer compression / materialisation
# ---------------------------------------------------------------------------

def _compress_side(
    t: torch.Tensor,
    bits: int | None,
) -> dict[str, Any]:
    """Compress one K or V tensor for one layer.

    Returns a dict with either:
    * ``{"q": int8_tensor, "scale": float}`` for quantised sides, or
    * ``{"full": cloned_fp_tensor}`` for full-precision passthrough.
    """
    if bits is None:
        return {"full": t.clone()}
    q, scale = _quantize(t, bits)
    return {"q": q, "scale": scale}


def _materialize_side(
    side_data: dict[str, Any],
    bits: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct one K or V tensor for use in a draft forward pass."""
    if bits is None:
        return side_data["full"].to(dtype=dtype, device=device)
    return _dequantize(side_data["q"], side_data["scale"], dtype, device)


def _side_storage_bytes(side_data: dict[str, Any], bits: int | None) -> int:
    """Actual storage bytes for one compressed side."""
    if bits is None:
        t = side_data["full"]
        return int(t.nelement()) * int(t.element_size())
    return int(side_data["q"].nelement()) * 1 + _SCALE_BYTES


# ---------------------------------------------------------------------------
# Capabilities derivation
# ---------------------------------------------------------------------------

def _build_capabilities(
    name: str,
    k_bits: int | None,
    v_bits: int | None,
) -> CompressorCapabilities:
    """Derive CompressorCapabilities from the resolved K/V bit-widths."""
    is_simulated = (k_bits is not None and k_bits < 8) or (
        v_bits is not None and v_bits < 8
    )
    supports_real = not is_simulated
    asymmetric = k_bits != v_bits

    k_desc = "full precision" if k_bits is None else f"INT{k_bits}"
    v_desc = "full precision" if v_bits is None else f"INT{v_bits}"

    sim_parts = []
    if k_bits is not None and k_bits < 8:
        sim_parts.append(f"K (INT{k_bits})")
    if v_bits is not None and v_bits < 8:
        sim_parts.append(f"V (INT{v_bits})")
    sim_clause = (
        f"  Simulated sides ({', '.join(sim_parts)}) use the signed "
        f"INT{{n}} numeric range but are stored in torch.int8 containers; "
        f"no real bit-packing is performed.  Do not cite compressed_bytes "
        f"for these sides as evidence of real packed memory savings."
        if sim_parts
        else "  Both sides use real storage (INT8 or full precision)."
    )

    notes = (
        f"Asymmetric K/V simulated compressor: K={k_desc}, V={v_desc}."
        + sim_clause
    )

    return CompressorCapabilities(
        name=name,
        compressor_type="quantization",
        is_simulated=is_simulated,
        supports_real_bytes_claim=supports_real,
        supports_token_dropping=False,
        supports_quantization=True,
        key_bit_width=k_bits,
        value_bit_width=v_bits,
        asymmetric=asymmetric,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public compressor class
# ---------------------------------------------------------------------------

class AsymmetricQuantSimCompressor:
    """Compressor that quantises K and V at independently specified bit-widths.

    Args:
        k_bits: Bit-width for key tensors. One of ``None``/``"full"``, ``8``,
                ``4``, or ``2``.  ``None``/``"full"`` means passthrough at
                full precision.
        v_bits: Bit-width for value tensors.  Same set of options.
        name:   Optional registry name string.  If ``None``, a canonical name
                is generated from the widths (e.g. ``"asym_k8_v4"``).

    Raises:
        ValueError: if ``k_bits`` or ``v_bits`` is not a recognised value.

    Note: Sub-INT8 widths (4, 2) are simulated — values are clipped to the
    signed N-bit range but stored in ``torch.int8`` containers.  Do NOT use
    ``compressed_bytes`` from ``stats()`` as evidence of real packed-INT4/INT2
    memory savings when ``is_simulated=True``.
    """

    def __init__(
        self,
        k_bits: int | str | None,
        v_bits: int | str | None,
        name: str | None = None,
    ) -> None:
        self._k_bits: int | None = _parse_bits(k_bits, "k")
        self._v_bits: int | None = _parse_bits(v_bits, "v")

        k_label = "full" if self._k_bits is None else str(self._k_bits)
        v_label = "full" if self._v_bits is None else str(self._v_bits)
        self.name: str = name if name is not None else f"asym_k{k_label}_v{v_label}"

        self.capabilities: CompressorCapabilities = _build_capabilities(
            self.name, self._k_bits, self._v_bits
        )

    # ------------------------------------------------------------------
    # KVCompressor protocol
    # ------------------------------------------------------------------

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Compress K and V independently.  Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        layers: list[dict[str, Any]] = []
        for k, v in zip(k_tensors, v_tensors):
            layers.append({
                "k_data": _compress_side(k, self._k_bits),
                "v_data": _compress_side(v, self._v_bits),
            })

        data: dict[str, Any] = {
            "layers": layers,
            "cache_format": cache_format,
            "seq_len": seq_len,
            "dtype": state.dtype,
            "device": state.device,
            "k_bits": self._k_bits,
            "v_bits": self._v_bits,
        }

        return CompressedKVState(
            data=data,
            metadata={"next_token_id": state.next_token_id},
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Dequantise / reconstruct and return a cache for a forward pass.

        Creates fresh tensors — does NOT modify ``compressed.data``.
        """
        d = compressed.data
        dtype: torch.dtype = d["dtype"]
        device: torch.device = d["device"]
        seq_len: int = d["seq_len"]
        cache_format: str = d["cache_format"]
        k_bits: int | None = d["k_bits"]
        v_bits: int | None = d["v_bits"]

        k_out = [
            _materialize_side(layer["k_data"], k_bits, dtype, device)
            for layer in d["layers"]
        ]
        v_out = [
            _materialize_side(layer["v_data"], v_bits, dtype, device)
            for layer in d["layers"]
        ]

        return rebuild_cache(k_out, v_out, cache_format, seq_len)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress from the updated authoritative full state (V1 strategy)."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        """Return byte-level statistics.

        For simulated sub-INT8 sides, ``compressed_bytes`` reflects actual
        int8 container storage, NOT theoretical packed size.
        For full-precision sides, ``compressed_bytes`` equals the actual
        tensor storage (fp32/fp16/bf16).

        ``full_bytes`` is the fp32 reference (4 bytes/element per tensor).
        """
        d = compressed.data
        k_bits: int | None = d["k_bits"]
        v_bits: int | None = d["v_bits"]
        layers = d["layers"]
        seq_len = d["seq_len"]

        full_bytes = 0
        compressed_bytes = 0
        num_layers = len(layers)

        for layer in layers:
            k_data = layer["k_data"]
            v_data = layer["v_data"]

            # Reference element count (same for k and v)
            if k_bits is None:
                k_nelems = int(k_data["full"].nelement())
            else:
                k_nelems = int(k_data["q"].nelement())

            if v_bits is None:
                v_nelems = int(v_data["full"].nelement())
            else:
                v_nelems = int(v_data["q"].nelement())

            full_bytes += (k_nelems + v_nelems) * _FP32_BYTES
            compressed_bytes += (
                _side_storage_bytes(k_data, k_bits)
                + _side_storage_bytes(v_data, v_bits)
            )

        full_bytes = max(full_bytes, 1)
        compressed_bytes = max(compressed_bytes, 1)

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compressed_bytes / full_bytes,
            memory_reduction_factor=full_bytes / compressed_bytes,
            seq_len=seq_len,
            num_layers=num_layers,
        )


# ---------------------------------------------------------------------------
# Named no-arg subclasses — registry-compatible (Phase C)
# ---------------------------------------------------------------------------
# Each class delegates to AsymmetricQuantSimCompressor with fixed (k_bits, v_bits)
# and the canonical registry name.  All seven are no-arg-constructible so the
# registry contract get_compressor(name) → _REGISTRY[name]() works.
#
# Naming rule: _sim suffix iff either side uses a simulated sub-INT8 width.

class K8V4SimCompressor(AsymmetricQuantSimCompressor):
    """K=INT8, V=INT4-sim.  is_simulated=True (V side sub-INT8)."""
    def __init__(self) -> None:
        super().__init__(k_bits=8, v_bits=4, name="k8_v4_sim")


class K8V2SimCompressor(AsymmetricQuantSimCompressor):
    """K=INT8, V=INT2-sim.  is_simulated=True (V side sub-INT8)."""
    def __init__(self) -> None:
        super().__init__(k_bits=8, v_bits=2, name="k8_v2_sim")


class K4V8SimCompressor(AsymmetricQuantSimCompressor):
    """K=INT4-sim, V=INT8.  is_simulated=True (K side sub-INT8)."""
    def __init__(self) -> None:
        super().__init__(k_bits=4, v_bits=8, name="k4_v8_sim")


class KFullV4SimCompressor(AsymmetricQuantSimCompressor):
    """K=full precision, V=INT4-sim.  is_simulated=True (V side sub-INT8)."""
    def __init__(self) -> None:
        super().__init__(k_bits=None, v_bits=4, name="k_full_v4_sim")


class K4VFullSimCompressor(AsymmetricQuantSimCompressor):
    """K=INT4-sim, V=full precision.  is_simulated=True (K side sub-INT8)."""
    def __init__(self) -> None:
        super().__init__(k_bits=4, v_bits=None, name="k4_v_full_sim")


class K8VFullCompressor(AsymmetricQuantSimCompressor):
    """K=INT8, V=full precision.  is_simulated=False (no sub-INT8 side).

    No ``_sim`` suffix: both sides use real storage only (INT8 or full
    precision).  ``supports_real_bytes_claim=True``.
    """
    def __init__(self) -> None:
        super().__init__(k_bits=8, v_bits=None, name="k8_v_full")


class KFullV8Compressor(AsymmetricQuantSimCompressor):
    """K=full precision, V=INT8.  is_simulated=False (no sub-INT8 side).

    No ``_sim`` suffix: both sides use real storage only (full precision or
    INT8).  ``supports_real_bytes_claim=True``.
    """
    def __init__(self) -> None:
        super().__init__(k_bits=None, v_bits=8, name="k_full_v8")
