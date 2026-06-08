# 12_COMPRESSOR_INTERFACE.md

# ExactKV Compressor Interface

## Purpose of this document

This document defines the compressor abstraction for ExactKV.

ExactKV must be compressor-agnostic. The verification engine should not know whether the compressed KV cache came from INT8 quantization, INT4 quantization, KIVI, TurboQuant, SnapKV, KVzip, or any future method.

The compressor interface is the boundary between:

```text
compression methods
```

and

```text
ExactKV draft/verify runtime
```

## Why the compressor interface matters

Without this interface, ExactKV becomes a one-off implementation.

With this interface, ExactKV becomes:

- A benchmark suite for KV compressors
- A runtime wrapper for different lossy compression methods
- A research platform for designing acceptance-optimized compressors

## Design goals

The interface should be:

1. **Simple enough for V1**
   - Support basic INT8 compression.

2. **Flexible enough for later versions**
   - Support quantization.
   - Support token dropping.
   - Support external adapters.
   - Support online and offline compressors.

3. **Explicit about limitations**
   - Not all compressors can share exactly the same internal representation.

4. **Friendly to benchmarking**
   - Must expose memory and compression stats.

5. **Friendly to debugging**
   - Must expose metadata and shape information.

## Compression categories

ExactKV needs to support at least two broad classes.

### 1. Quantization compressors

These preserve cache shape but reduce precision.

Examples:

- INT8
- INT4
- KIVI
- KVQuant
- RotateKV
- TurboQuant-style quantization

Shape behavior:

```text
same layers
same sequence length
same heads
same head dimension
lower precision values
```

### 2. Token-dropping compressors

These remove some cached token positions, often per layer or attention head.

Examples:

- SnapKV
- KVzip
- KVzap
- H2O
- StreamingLLM
- DuoAttention

Shape behavior:

```text
same model
different retained token positions
possibly different cache length per layer/head
```

Token-dropping support is harder. Phase 1 should not implement it.

## Versioned interface philosophy

The compressor interface should evolve over versions.

### V1 interface

Support simple quantization only.

### V2 interface

Formalize compressor base class and metrics.

### V3 interface

Support benchmark reporting.

### V4 interface

Support external and shape-changing compressors.

### V5+

Support runtime-specific cache layouts, paged KV, and offload.

## Core data structures

## FullKVState

Represents the authoritative cache.

```python
@dataclass
class FullKVState:
    past_key_values: Any
    input_ids: torch.Tensor
    committed_tokens: list[int]
    device: torch.device
    dtype: torch.dtype
    metadata: dict = field(default_factory=dict)
```

Fields:

- `past_key_values`: model-specific full KV cache
- `input_ids`: full prompt plus committed output IDs if needed
- `committed_tokens`: tokens generated after prompt
- `device`: where full KV currently lives
- `dtype`: full precision dtype
- `metadata`: model-specific information

## CompressedKVState

Represents a compressed cache.

```python
@dataclass
class CompressedKVState:
    data: Any
    metadata: dict
    compressor_name: str
    logical_seq_len: int
    committed_tokens: list[int]
    device: torch.device | None = None
```

Fields:

- `data`: compressor-specific representation
- `metadata`: scales, zero points, dropped indices, bit width, etc.
- `compressor_name`: name of compressor
- `logical_seq_len`: sequence length represented by the cache
- `committed_tokens`: generated tokens committed so far
- `device`: device for compressed data if applicable

## CompressionStats

Every compressor should be able to report stats.

```python
@dataclass
class CompressionStats:
    compressor_name: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bit_width: int | None = None
    retained_token_ratio: float | None = None
    extra_metadata_bytes: int = 0
    notes: dict = field(default_factory=dict)
```

Important:

```python
compression_ratio = compressed_bytes / original_bytes
```

Smaller is better.

## Base interface

```python
class KVCompressor(Protocol):
    name: str

    def compress(self, full_state: FullKVState) -> CompressedKVState:
        ...

    def materialize_for_draft(
        self,
        compressed_state: CompressedKVState,
    ) -> Any:
        ...

    def update_after_commit(
        self,
        compressed_state: CompressedKVState,
        full_state: FullKVState,
        committed_token_ids: list[int],
    ) -> CompressedKVState:
        ...

    def stats(
        self,
        full_state: FullKVState,
        compressed_state: CompressedKVState,
    ) -> CompressionStats:
        ...
```

## Method: compress

### Purpose

Create a compressed representation from full KV.

### Input

```python
full_state: FullKVState
```

### Output

```python
CompressedKVState
```

### Requirements

- Must preserve enough information to draft tokens approximately.
- Must set `logical_seq_len`.
- Must set `compressor_name`.
- Must provide metadata needed for memory estimation and updates.

### V1 behavior

For INT8:

- Iterate over each tensor in `past_key_values`.
- Quantize values to int8.
- Store scale values.
- Return quantized tensors plus scale metadata.

## Method: materialize_for_draft

### Purpose

Prepare compressed KV for use during draft generation.

This may:

- Dequantize INT8 to float tensors.
- Return compressed tensors for a custom attention kernel.
- Construct sparse cache views.
- Return backend-specific structures.

### V1 behavior

For simplicity, V1 may dequantize compressed KV back to the model dtype before drafting.

This does not produce real memory speedup, but it proves correctness and acceptance logic.

### Later behavior

Later versions should avoid full dequantization where possible.

## Method: update_after_commit

### Purpose

Update compressed cache after ExactKV commits tokens.

### Why this is necessary

After verification, the system commits either:

- accepted drafted tokens
- a full-KV correction token
- a bonus verifier token

Both full KV and compressed KV must now represent the same logical sequence.

### V1 behavior

Simplest safe implementation:

- Recompress the updated full KV after every commit round.

This is inefficient but correct.

### V2 behavior

Incrementally quantize and append new KV entries.

### V4+ behavior

Use compressor-specific update logic.

## Method: stats

### Purpose

Report memory and compression information.

### Required outputs

- original bytes
- compressed bytes
- compression ratio
- bit width if applicable
- retained token ratio if applicable
- metadata bytes if known

### Why important

ExactKV must benchmark not just output behavior but memory tradeoffs.

## Optional interface for online compressors

Some compressors update during inference based on activations or hidden states.

Future interface:

```python
class OnlineKVCompressor(KVCompressor):
    def update_layer(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden: torch.Tensor,
        req_offsets: torch.Tensor | None = None,
    ) -> Any:
        ...
```

This is not needed in V1.

## Optional interface for token-dropping compressors

Token dropping requires retained indices.

Future structure:

```python
@dataclass
class TokenDropMetadata:
    retained_indices: dict
    dropped_indices: dict
    retained_ratio_by_layer: dict
    retained_ratio_by_head: dict
```

Potential methods:

```python
def retained_positions(self, compressed_state, layer_idx, head_idx=None):
    ...

def gather_for_attention(self, compressed_state, layer_idx, positions):
    ...
```

Not V1.

## Optional interface for external adapters

External libraries may not use ExactKV's internal data structures.

Adapter pattern:

```python
class ExternalCompressorAdapter(KVCompressor):
    def __init__(self, backend):
        self.backend = backend
```

Examples:

- `KVPressAdapter`
- `KIVIAdapter`
- `TurboQuantAdapter`
- `SnapKVAdapter`

## V1 INT8 compressor specification

### Name

```text
int8
```

### Goal

Provide the simplest compressor that changes KV values enough to test draft/verify behavior.

### Quantization formula

For a tensor `x`:

```python
scale = max(abs(x)) / 127
q = round(x / scale).clamp(-128, 127).to(torch.int8)
```

Dequantization:

```python
x_hat = q.float() * scale
```

### Granularity

V1 can use per-tensor scale.

Later versions can use:

- per-channel scale
- per-head scale
- per-token scale
- asymmetric quantization

### Metadata

Store:

- scale
- original dtype
- original shape
- tensor location
- quantization granularity

### Known weakness

Per-tensor INT8 may be too accurate and produce very high acceptance. That is fine.

If it never diverges, introduce INT4 or artificial noise in V2 for testing rejection paths.

## V2 INT4 compressor specification

### Name

```text
int4
```

### Goal

Create stronger approximation and more meaningful compression.

### Implementation options

#### Simple V2 implementation

Store values in `torch.int8` but clamp to 4-bit range:

```python
q = round(x / scale).clamp(-8, 7).to(torch.int8)
```

This simulates INT4 numerically without bit-packing.

#### Later implementation

Pack two 4-bit values into one byte.

### Why simple first

The goal is acceptance behavior, not compression kernel optimization.

## Debug compressor

A useful test compressor intentionally corrupts KV slightly.

### Name

```text
debug_noise
```

### Purpose

Force mismatches so the verification engine can be tested.

### Behavior

Add small noise to KV values:

```python
x_hat = x + noise_scale * torch.randn_like(x)
```

This should never be used for serious benchmarks.

## Compressor correctness tests

Every compressor must pass:

### Test 1: compress returns valid state

- Non-null data
- Correct compressor name
- Logical sequence length set
- Metadata present

### Test 2: materialize returns model-compatible cache

The returned cache can be used in a forward pass.

### Test 3: stats are non-negative

- original bytes > 0
- compressed bytes > 0
- compression ratio > 0

### Test 4: update preserves logical length

After committing tokens, compressed and full states must have same logical sequence length.

## Compressor benchmark metrics

Each compressor should be evaluated by:

- compression ratio
- memory bytes
- acceptance rate
- average accepted tokens per verification
- first mismatch position
- lossy output exact match rate
- ExactKV output exact match rate
- throughput
- update cost
- materialization cost

## Important warning

A compressor can be good for direct lossy inference but bad for ExactKV.

ExactKV cares about:

```text
acceptance length
```

not just:

```text
semantic quality
```

A future compressor may be optimized specifically to maximize accepted draft length under full-KV verification.

## Final interface principle

ExactKV should make the compressor replaceable.

The verification engine should not care how compression works.

The benchmark suite should treat every compressor uniformly.
