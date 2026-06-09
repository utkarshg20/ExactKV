"""Serving-style cache lifecycle harness for ExactKV compatibility evaluation.

Models separate authoritative full KV vs compressed draft KV ownership,
logical vs physical sequence length, and block/page mapping — without
integrating vLLM, LMCache, or PagedAttention.

This is a local compatibility harness, not production serving infrastructure.
It does not measure throughput, latency, speedup, runtime, or tokens/sec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len, kv_total_bytes

# ---------------------------------------------------------------------------
# Ownership constants
# ---------------------------------------------------------------------------

CacheOwner = Literal["authoritative_full", "compressed_draft", "serving_harness"]

AUTHORITATIVE_FULL: CacheOwner = "authoritative_full"
COMPRESSED_DRAFT: CacheOwner = "compressed_draft"
SERVING_HARNESS: CacheOwner = "serving_harness"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheBlock:
    """One block/page slot range with logical and physical coordinates."""

    block_id: int
    logical_start: int
    logical_end: int
    physical_start: int
    physical_end: int


@dataclass
class ServingCacheEntry:
    """Registered KV cache entry tracked by the lifecycle harness."""

    cache_id: str
    owner: CacheOwner
    logical_seq_len: int
    physical_seq_len: int
    block_size: int
    blocks: list[CacheBlock] = field(default_factory=list)
    supports_real_bytes_claim: bool = False
    is_simulated: bool = False
    note: str = ""
    stored_kv_bytes: int | None = None
    materialized_working_kv_bytes: int | None = None
    total_kv_footprint_bytes: int | None = None
    compressor_name: str | None = None
    retained_logical_positions: tuple[int, ...] | None = None


# ---------------------------------------------------------------------------
# Block mapping helpers
# ---------------------------------------------------------------------------


def validate_retained_logical_positions(
    retained_logical_positions: Sequence[int],
    *,
    logical_seq_len: int,
    physical_seq_len: int,
) -> tuple[int, ...]:
    """Validate and normalise retained logical position mapping.

    Raises ``ValueError`` when mapping is invalid.
    """
    if physical_seq_len >= logical_seq_len:
        raise ValueError(
            "retained_logical_positions are only required when "
            f"physical_seq_len ({physical_seq_len}) < logical_seq_len "
            f"({logical_seq_len})"
        )

    positions = tuple(int(p) for p in retained_logical_positions)
    if len(positions) != physical_seq_len:
        raise ValueError(
            f"retained_logical_positions length ({len(positions)}) must equal "
            f"physical_seq_len ({physical_seq_len})"
        )

    for idx, pos in enumerate(positions):
        if pos < 0 or pos >= logical_seq_len:
            raise ValueError(
                f"retained_logical_positions[{idx}]={pos} out of range "
                f"[0, {logical_seq_len})"
            )
        if idx > 0 and positions[idx] <= positions[idx - 1]:
            raise ValueError(
                "retained_logical_positions must be strictly increasing; "
                f"found {positions[idx - 1]} then {positions[idx]} at index {idx}"
            )

    return positions


def build_blocks(
    *,
    logical_seq_len: int,
    physical_seq_len: int,
    block_size: int,
    retained_logical_positions: Sequence[int] | None = None,
) -> list[CacheBlock]:
    """Build block/page table covering the physical sequence.

    When ``physical_seq_len == logical_seq_len``, identity mapping is used.
    When ``physical_seq_len < logical_seq_len``, ``retained_logical_positions``
    must be provided explicitly.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if physical_seq_len < 0 or logical_seq_len < 0:
        raise ValueError("sequence lengths must be non-negative")
    if physical_seq_len > logical_seq_len:
        raise ValueError(
            f"physical_seq_len ({physical_seq_len}) cannot exceed "
            f"logical_seq_len ({logical_seq_len})"
        )

    if physical_seq_len == 0:
        return []

    if physical_seq_len < logical_seq_len:
        if retained_logical_positions is None:
            raise ValueError(
                "retained_logical_positions are required when "
                f"physical_seq_len ({physical_seq_len}) < "
                f"logical_seq_len ({logical_seq_len}); "
                "refusing to fabricate a mapping"
            )
        retained = validate_retained_logical_positions(
            retained_logical_positions,
            logical_seq_len=logical_seq_len,
            physical_seq_len=physical_seq_len,
        )
    else:
        retained = None

    blocks: list[CacheBlock] = []
    block_id = 0
    physical = 0
    while physical < physical_seq_len:
        p_end = min(physical + block_size, physical_seq_len)
        if retained is None:
            logical_start = physical
            logical_end = p_end
        else:
            logical_start = retained[physical]
            logical_end = retained[p_end - 1] + 1
        blocks.append(
            CacheBlock(
                block_id=block_id,
                logical_start=logical_start,
                logical_end=logical_end,
                physical_start=physical,
                physical_end=p_end,
            )
        )
        block_id += 1
        physical = p_end

    return blocks


def _seq_len_from_kv_tensors(k_tensors: list) -> int | None:
    if not k_tensors:
        return None
    tensor = k_tensors[0]
    if not hasattr(tensor, "shape") or len(tensor.shape) < 3:
        return None
    return int(tensor.shape[-2])


def infer_physical_seq_len(
    compressed_state: CompressedKVState,
    *,
    explicit: int | None = None,
) -> int:
    """Infer physical KV length from a ``CompressedKVState`` when safe.

    Raises ``ValueError`` when physical length cannot be inferred safely.
    """
    if explicit is not None:
        return int(explicit)

    data = compressed_state.data
    if isinstance(data, dict):
        if "__physical_seq_len__" in data:
            return int(data["__physical_seq_len__"])
        if "seq_len" in data:
            return int(data["seq_len"])
        if "k" in data and isinstance(data["k"], list):
            inferred = _seq_len_from_kv_tensors(data["k"])
            if inferred is not None:
                return inferred
        if "layers" in data and isinstance(data["layers"], list) and data["layers"]:
            layer0 = data["layers"][0]
            if isinstance(layer0, dict) and "k_q" in layer0:
                inferred = _seq_len_from_kv_tensors([layer0["k_q"]])
                if inferred is not None:
                    return inferred
        if "dynamic_cache" in data:
            return kv_seq_len(data["dynamic_cache"])
    else:
        try:
            return kv_seq_len(data)
        except TypeError:
            pass

    raise ValueError(
        "physical_seq_len cannot be inferred safely from CompressedKVState; "
        "pass physical_seq_len explicitly"
    )


def _validate_block_ranges(blocks: list[CacheBlock]) -> None:
    for block in blocks:
        if block.logical_start < 0 or block.physical_start < 0:
            raise ValueError(
                f"block {block.block_id}: negative start positions are invalid"
            )
        if block.logical_end <= block.logical_start:
            raise ValueError(
                f"block {block.block_id}: logical range reversed or empty "
                f"({block.logical_start}, {block.logical_end})"
            )
        if block.physical_end <= block.physical_start:
            raise ValueError(
                f"block {block.block_id}: physical range reversed or empty "
                f"({block.physical_start}, {block.physical_end})"
            )


def _entry_to_dict(entry: ServingCacheEntry) -> dict[str, Any]:
    return {
        "cache_id": entry.cache_id,
        "owner": entry.owner,
        "logical_seq_len": entry.logical_seq_len,
        "physical_seq_len": entry.physical_seq_len,
        "block_size": entry.block_size,
        "blocks": [
            {
                "block_id": b.block_id,
                "logical_start": b.logical_start,
                "logical_end": b.logical_end,
                "physical_start": b.physical_start,
                "physical_end": b.physical_end,
            }
            for b in entry.blocks
        ],
        "supports_real_bytes_claim": entry.supports_real_bytes_claim,
        "is_simulated": entry.is_simulated,
        "note": entry.note,
        "stored_kv_bytes": entry.stored_kv_bytes,
        "materialized_working_kv_bytes": entry.materialized_working_kv_bytes,
        "total_kv_footprint_bytes": entry.total_kv_footprint_bytes,
        "compressor_name": entry.compressor_name,
        "retained_logical_positions": (
            list(entry.retained_logical_positions)
            if entry.retained_logical_positions is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ServingCacheLifecycleHarness:
    """Track serving-style KV cache ownership and lifecycle invariants.

    Wraps existing ``FullKVState`` / ``CompressedKVState`` without mutating
    them.  Verification must conceptually use the ``authoritative_full`` entry;
    the ``compressed_draft`` entry must never overwrite authoritative storage.
    """

    def __init__(self, *, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self._entries: dict[str, ServingCacheEntry] = {}
        self._authoritative_id: str | None = None
        self._compressed_id: str | None = None
        self._next_id = 0

    def _new_cache_id(self, owner: CacheOwner) -> str:
        self._next_id += 1
        return f"{owner}_{self._next_id}"

    def register_authoritative_full(
        self,
        full_state: FullKVState,
        *,
        cache_id: str | None = None,
    ) -> str:
        """Register authoritative full-precision KV without mutating ``full_state``."""
        logical = full_state.seq_len
        physical = kv_seq_len(full_state.past_key_values)
        if physical != logical:
            raise ValueError(
                f"authoritative full KV physical length ({physical}) must match "
                f"logical seq_len ({logical})"
            )

        entry_id = cache_id or self._new_cache_id(AUTHORITATIVE_FULL)
        if entry_id in self._entries:
            existing = self._entries[entry_id]
            if existing.owner == COMPRESSED_DRAFT:
                raise ValueError(
                    f"cache_id {entry_id!r} is owned by compressed_draft; "
                    "cannot replace with authoritative_full"
                )

        full_bytes = kv_total_bytes(full_state.past_key_values)
        blocks = build_blocks(
            logical_seq_len=logical,
            physical_seq_len=physical,
            block_size=self.block_size,
        )

        entry = ServingCacheEntry(
            cache_id=entry_id,
            owner=AUTHORITATIVE_FULL,
            logical_seq_len=logical,
            physical_seq_len=physical,
            block_size=self.block_size,
            blocks=blocks,
            supports_real_bytes_claim=True,
            is_simulated=False,
            note=(
                "Authoritative full-precision KV for verification and commit. "
                "Conservative byte counts from tensor shapes; not measured peak GPU memory."
            ),
            stored_kv_bytes=full_bytes,
            materialized_working_kv_bytes=full_bytes,
            total_kv_footprint_bytes=full_bytes,
        )
        self._entries[entry_id] = entry
        self._authoritative_id = entry_id
        return entry_id

    def register_compressed_cache(
        self,
        compressed_state: CompressedKVState,
        *,
        physical_seq_len: int | None = None,
        retained_logical_positions: Sequence[int] | None = None,
        compressor: Any | None = None,
        cache_id: str | None = None,
    ) -> str:
        """Register compressed/draft KV without mutating ``compressed_state``."""
        entry_id = cache_id or self._new_cache_id(COMPRESSED_DRAFT)
        if entry_id in self._entries:
            existing = self._entries[entry_id]
            if existing.owner == AUTHORITATIVE_FULL:
                raise ValueError(
                    f"cache_id {entry_id!r} is owned by authoritative_full; "
                    "compressed_draft cannot replace authoritative_full"
                )

        logical = compressed_state.logical_seq_len
        physical = infer_physical_seq_len(
            compressed_state, explicit=physical_seq_len
        )

        if physical < logical and retained_logical_positions is None:
            raise ValueError(
                f"physical_seq_len ({physical}) < logical_seq_len ({logical}) "
                "but retained_logical_positions were not provided; "
                "refusing to fabricate a mapping"
            )

        retained: tuple[int, ...] | None = None
        if physical < logical:
            retained = validate_retained_logical_positions(
                retained_logical_positions,  # type: ignore[arg-type]
                logical_seq_len=logical,
                physical_seq_len=physical,
            )

        blocks = build_blocks(
            logical_seq_len=logical,
            physical_seq_len=physical,
            block_size=self.block_size,
            retained_logical_positions=retained,
        )

        caps = None
        if compressor is not None and hasattr(compressor, "capabilities"):
            caps = compressor.capabilities

        stored: int | None = None
        materialized: int | None = None
        total: int | None = None
        if compressor is not None and hasattr(compressor, "stats"):
            stats = compressor.stats(compressed_state)
            stored = stats.stored_kv_bytes
            materialized = stats.materialized_working_kv_bytes
            total = stats.total_kv_footprint_bytes

        note = (
            "Compressed/draft KV for drafting only; verification uses "
            "authoritative_full. total_kv_footprint_bytes is a conservative "
            "accounting sum, not measured peak GPU memory."
        )
        if caps is not None and caps.is_simulated:
            note += " Simulated compressor uses int8 containers, not packed-bit storage."

        entry = ServingCacheEntry(
            cache_id=entry_id,
            owner=COMPRESSED_DRAFT,
            logical_seq_len=logical,
            physical_seq_len=physical,
            block_size=self.block_size,
            blocks=blocks,
            supports_real_bytes_claim=(
                bool(caps.supports_real_bytes_claim) if caps is not None else False
            ),
            is_simulated=bool(caps.is_simulated) if caps is not None else False,
            note=note,
            stored_kv_bytes=stored,
            materialized_working_kv_bytes=materialized,
            total_kv_footprint_bytes=total,
            compressor_name=compressed_state.compressor_name,
            retained_logical_positions=retained,
        )
        self._entries[entry_id] = entry
        self._compressed_id = entry_id
        return entry_id

    def append_committed_tokens(self, count: int) -> None:
        """Advance logical sequence length after a commit round."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return

        for entry in self._entries.values():
            was_identity = (
                entry.retained_logical_positions is None
                and entry.physical_seq_len == entry.logical_seq_len
            )
            entry.logical_seq_len += count
            if was_identity:
                entry.physical_seq_len += count
            entry.blocks = build_blocks(
                logical_seq_len=entry.logical_seq_len,
                physical_seq_len=entry.physical_seq_len,
                block_size=entry.block_size,
                retained_logical_positions=entry.retained_logical_positions,
            )

    def validate_invariants(self) -> None:
        """Validate ownership, alignment, and block-table invariants."""
        if self._authoritative_id is None:
            raise ValueError("no authoritative_full entry registered")

        auth = self._entries[self._authoritative_id]
        if auth.owner != AUTHORITATIVE_FULL:
            raise ValueError(
                f"authoritative entry {auth.cache_id!r} has owner {auth.owner!r}"
            )

        for entry in self._entries.values():
            _validate_block_ranges(entry.blocks)
            if entry.physical_seq_len > entry.logical_seq_len:
                raise ValueError(
                    f"entry {entry.cache_id}: physical_seq_len "
                    f"({entry.physical_seq_len}) exceeds logical_seq_len "
                    f"({entry.logical_seq_len})"
                )
            if (
                entry.physical_seq_len < entry.logical_seq_len
                and entry.retained_logical_positions is None
            ):
                raise ValueError(
                    f"entry {entry.cache_id}: pruned physical cache requires "
                    "retained_logical_positions"
                )

        if self._compressed_id is not None:
            comp = self._entries[self._compressed_id]
            if comp.owner != COMPRESSED_DRAFT:
                raise ValueError(
                    f"compressed entry {comp.cache_id!r} has owner {comp.owner!r}"
                )
            if comp.cache_id == auth.cache_id:
                raise ValueError(
                    "compressed_draft and authoritative_full must be separate entries"
                )
            if comp.logical_seq_len != auth.logical_seq_len:
                raise ValueError(
                    f"logical_seq_len mismatch: authoritative={auth.logical_seq_len}, "
                    f"compressed_draft={comp.logical_seq_len}"
                )

    def summarize(self) -> dict[str, Any]:
        """Return a JSON-serialisable lifecycle summary with memory-honesty fields."""
        entries = [_entry_to_dict(e) for e in self._entries.values()]
        auth_logical: int | None = None
        comp_logical: int | None = None
        if self._authoritative_id is not None:
            auth_logical = self._entries[self._authoritative_id].logical_seq_len
        if self._compressed_id is not None:
            comp_logical = self._entries[self._compressed_id].logical_seq_len

        invariants_ok = False
        invariant_error: str | None = None
        try:
            self.validate_invariants()
            invariants_ok = True
        except ValueError as exc:
            invariant_error = str(exc)

        return {
            "harness_owner": SERVING_HARNESS,
            "block_size": self.block_size,
            "authoritative_cache_id": self._authoritative_id,
            "compressed_cache_id": self._compressed_id,
            "verification_uses": AUTHORITATIVE_FULL,
            "authoritative_logical_seq_len": auth_logical,
            "compressed_logical_seq_len": comp_logical,
            "entries": entries,
            "invariants_valid": invariants_ok,
            "invariant_error": invariant_error,
            "note": (
                "Local serving-context compatibility harness only; not vLLM or "
                "LMCache integration. No throughput, latency, speedup, runtime, "
                "or tokens/sec fields. No active GPU memory measurement."
            ),
        }
