# 05_ARCHITECTURE.md

# ExactKV Architecture

## Purpose of this document

This document defines the architecture for ExactKV.

It should be read after:

- `00_VISION.md`
- `01_PROBLEM.md`
- `02_EXISTING_WORK.md`
- `03_VERICACHE_ANALYSIS.md`
- `04_EXACTKV_THESIS.md`

This file translates the project thesis into a concrete system design that Cursor and future contributors can use when implementing the codebase.

## Architectural principle

ExactKV should be built as a **correctness-first, compressor-agnostic inference runtime**.

The architecture must support two identities:

1. **Runtime**
   - Generate tokens with compressed-KV drafting and full-KV verification.
   - Commit only tokens that match full-KV decoding.

2. **Benchmark suite**
   - Evaluate compressors by acceptance rate, exactness, memory, latency, and throughput.
   - Compare `full`, `lossy`, and `exactkv` modes.

The architecture should avoid premature optimization. Phase 1 should be simple, readable, and correct. Performance-oriented architecture comes later.

## Core architectural idea

ExactKV maintains two conceptual cache paths:

```text
Full KV path
    authoritative
    exact
    used for verification

Compressed KV path
    approximate
    cheaper
    used for drafting
```

The compressed path proposes candidate tokens. The full path decides which tokens are valid.

The fundamental loop is:

```text
1. Build or maintain full KV cache.
2. Compress full KV into compressed KV.
3. Draft N tokens using compressed KV.
4. Verify drafted tokens using full KV.
5. Accept longest matching prefix.
6. Correct first mismatch using full-KV token.
7. Update both caches with committed tokens.
8. Repeat.
```

## High-level data flow

```text
Prompt
  ↓
Tokenizer
  ↓
Prefill full model
  ↓
FullKVState
  ↓
Compressor
  ↓
CompressedKVState
  ↓
CompressedKVDrafter drafts x tokens
  ↓
VerificationEngine verifies draft against FullKVState
  ↓
AcceptanceResult
  ↓
Commit accepted tokens and correction
  ↓
Update FullKVState and CompressedKVState
  ↓
Output tokens
```

## System components

ExactKV should be decomposed into the following components.

```text
exactkv/
├── generation/
│   ├── full_kv.py
│   ├── lossy_kv.py
│   └── exact_kv.py
├── cache/
│   ├── full_state.py
│   ├── compressed_state.py
│   └── cache_utils.py
├── compressors/
│   ├── base.py
│   ├── int8.py
│   ├── int4.py
│   └── token_drop.py
├── verification/
│   ├── engine.py
│   ├── acceptance.py
│   └── rollback.py
├── metrics/
│   ├── exactness.py
│   ├── memory.py
│   ├── throughput.py
│   └── acceptance.py
├── benchmarks/
│   ├── runner.py
│   ├── prompts.py
│   └── report.py
└── utils/
    ├── model_loading.py
    ├── tokenization.py
    └── device.py
```

This layout is not mandatory, but it gives Cursor a clean starting point.

## Component 1: ModelRuntime

### Purpose

Wraps a Hugging Face causal language model and tokenizer.

### Responsibilities

- Load model and tokenizer.
- Move model to selected device.
- Set deterministic generation settings.
- Expose low-level forward calls.
- Normalize differences between model families where possible.

### Phase 1 constraints

- Hugging Face Transformers only.
- Causal LM only.
- Greedy decoding only.
- Single model instance.
- Single device if possible.

### Suggested interface

```python
class ModelRuntime:
    def __init__(self, model_name: str, device: str = "cuda"):
        ...

    def encode(self, prompt: str) -> torch.Tensor:
        ...

    def decode(self, token_ids: torch.Tensor) -> str:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        ...
```

## Component 2: FullKVState

### Purpose

Represents the authoritative full-KV state.

### Responsibilities

- Store full precision `past_key_values`.
- Support prefill.
- Support incremental updates when new tokens are committed.
- Provide the cache needed for verification.

### Important invariant

FullKVState is the source of truth.

If FullKVState and CompressedKVState disagree, FullKVState wins.

### Phase 1 simplification

Full KV can remain on GPU or current device. Phase 1 does not need CPU offload.

Later versions can move full KV to CPU or another memory tier.

### Suggested interface

```python
@dataclass
class FullKVState:
    past_key_values: Any
    input_ids: torch.Tensor
    committed_tokens: list[int]
    device: torch.device
```

## Component 3: CompressedKVState

### Purpose

Represents the compressed approximation of the KV cache.

### Responsibilities

- Store compressor-specific compressed cache.
- Store metadata needed for decompression or approximate attention.
- Track compression ratio and memory footprint.
- Update after committed tokens.

### Important invariant

CompressedKVState must correspond to the same committed output prefix as FullKVState.

The values may be approximate, but the logical sequence length must stay aligned.

### Suggested interface

```python
@dataclass
class CompressedKVState:
    compressed_past_key_values: Any
    metadata: dict
    committed_tokens: list[int]
    compressor_name: str
```

## Component 4: KVCompressor

### Purpose

Defines the plugin interface for all compressors.

### Responsibilities

- Compress full KV.
- Optionally decompress or materialize approximate KV.
- Update compressed cache after new committed tokens.
- Report memory statistics.
- Expose configuration.

### Why this abstraction matters

ExactKV should not be hard-coded to TurboQuant, INT4, KIVI, or any single compressor.

The project is valuable because it can evaluate many compressors under the same verification harness.

### Minimal Phase 1 interface

```python
class KVCompressor:
    name: str

    def compress(self, full_kv_state: FullKVState) -> CompressedKVState:
        ...

    def prepare_for_draft(self, compressed_state: CompressedKVState):
        ...

    def update(
        self,
        compressed_state: CompressedKVState,
        new_token_ids: torch.Tensor,
        new_full_kv=None,
    ) -> CompressedKVState:
        ...

    def memory_bytes(self, compressed_state: CompressedKVState) -> int:
        ...
```

## Component 5: CompressedKVDrafter

### Purpose

Generates candidate tokens using compressed KV.

### Responsibilities

- Given a compressed cache and current committed prefix, generate `draft_len` tokens.
- Run greedy decoding using the approximate cache.
- Return both tokens and optional per-token debug metadata.

### Phase 1 simplification

The drafter can materialize an approximate `past_key_values` compatible with Hugging Face and call model forward repeatedly.

This may not be fast, but it is easier to verify.

### Suggested interface

```python
class CompressedKVDrafter:
    def draft(
        self,
        model_runtime: ModelRuntime,
        compressed_state: CompressedKVState,
        last_token_id: torch.Tensor,
        draft_len: int,
    ) -> DraftResult:
        ...
```

### DraftResult

```python
@dataclass
class DraftResult:
    draft_tokens: list[int]
    draft_logits: list[torch.Tensor] | None
    compressed_state_after_draft: CompressedKVState | None
```

## Component 6: VerificationEngine

### Purpose

Verifies drafted tokens against full-KV decoding.

### Responsibilities

- Run full-KV verification pass.
- Compare full-KV predicted tokens with draft tokens.
- Compute longest matching prefix.
- Produce correction token on mismatch.
- Produce bonus token when all draft tokens match.
- Return an acceptance result.

### Key correctness rule

No token is committed unless it matches the full-KV prediction, except the correction token, which is itself generated by full KV.

### Suggested interface

```python
class VerificationEngine:
    def verify(
        self,
        model_runtime: ModelRuntime,
        full_state: FullKVState,
        draft_tokens: list[int],
    ) -> AcceptanceResult:
        ...
```

### AcceptanceResult

```python
@dataclass
class AcceptanceResult:
    accepted_tokens: list[int]
    rejected_tokens: list[int]
    correction_token: int | None
    bonus_token: int | None
    first_mismatch_index: int | None
    all_matched: bool
    verifier_tokens: list[int]
```

## Component 7: ExactKVGenerator

### Purpose

The top-level generation API.

### Responsibilities

- Run prefill.
- Initialize full and compressed cache states.
- Loop over draft and verify.
- Commit accepted tokens and corrections.
- Update both caches.
- Track metrics.
- Return final output.

### Suggested interface

```python
class ExactKVGenerator:
    def __init__(
        self,
        model_runtime: ModelRuntime,
        compressor: KVCompressor,
        draft_len: int = 8,
        max_new_tokens: int = 128,
    ):
        ...

    def generate(self, prompt: str) -> ExactKVResult:
        ...
```

### ExactKVResult

```python
@dataclass
class ExactKVResult:
    output_ids: list[int]
    output_text: str
    metrics: ExactKVMetrics
    trace: list[AcceptanceResult]
```

## Component 8: BenchmarkRunner

### Purpose

Runs the same prompt set under multiple modes.

### Required modes

```text
full
lossy
exactkv
```

### Mode definitions

#### full

Normal full-KV generation.

This is the ground truth.

#### lossy

Direct compressed-KV generation without verification.

This shows how the compressor behaves if trusted directly.

#### exactkv

Compressed-KV drafting with full-KV verification.

This should match the full output.

### Suggested interface

```python
class BenchmarkRunner:
    def run_prompt(self, prompt: str, config: BenchmarkConfig) -> BenchmarkResult:
        ...

    def run_suite(self, prompts: list[str], config: BenchmarkConfig) -> BenchmarkReport:
        ...
```

## Core invariants

These must hold throughout implementation.

### Invariant 1: Exact output equivalence

Under greedy decoding, ExactKV output must equal full-KV output.

```python
exactkv_output_ids == full_output_ids
```

If this fails, the implementation is wrong.

### Invariant 2: Full KV is authoritative

Compressed KV never decides final output alone.

### Invariant 3: Cache prefix alignment

FullKVState and CompressedKVState must always correspond to the same committed token sequence.

### Invariant 4: Rejected tokens are not committed

Drafted tokens after the first mismatch must be discarded.

### Invariant 5: Metrics must not affect generation

Logging, tracing, and benchmarking should not change model outputs.

## Version 1 architecture

Version 1 should implement the minimal end-to-end system.

### Included

- Hugging Face model loading
- Greedy decoding
- Full-KV baseline
- Simple INT8 compressor
- Compressed draft generation
- Verification engine
- Acceptance trace
- Exactness test
- Small benchmark runner

### Excluded

- vLLM
- LMCache
- CPU offload
- async transfers
- cross-resource staggering
- CUDA streams
- Triton kernels
- sampling
- batching
- multi-GPU

## Version 1 data path

```text
prompt
  ↓
full prefill
  ↓
full_kv_state
  ↓
int8 compressor
  ↓
compressed_kv_state
  ↓
draft 4 to 8 tokens
  ↓
verify with full_kv_state
  ↓
accept/correct
  ↓
update states
  ↓
repeat
```

## Important implementation detail: verification mode

The VeriCache paper verifies multiple drafted positions in one forward pass.

For Phase 1, Cursor may implement verification more simply first:

- Step through drafted tokens using full KV.
- Compare each full-KV next token to draft token.
- Stop at first mismatch.
- Commit accepted tokens and correction.

This is less efficient but easier to get correct.

Later, implement parallel verification over the draft span.

Correctness comes first.

## Device architecture

### Phase 1

Use one device.

```text
model: GPU if available, else CPU
full KV: same device
compressed KV: same device
```

### Later

```text
compressed KV: GPU
full KV: CPU
verification KV: temporarily loaded to GPU
```

### Production target

```text
compressed KV resident on GPU
full KV offloaded to CPU/storage
async transfer before verification
staggered request scheduling
```

## Configuration system

Use explicit config objects.

```python
@dataclass
class ExactKVConfig:
    model_name: str
    compressor: str
    draft_len: int
    max_new_tokens: int
    device: str
    dtype: str
    greedy: bool = True
    seed: int = 0
```

Avoid hidden globals.

## Logging and traces

ExactKV should make the generation process inspectable.

Trace each round:

- round index
- draft length requested
- drafted tokens
- verified tokens
- accepted count
- mismatch position
- correction token
- cumulative accepted tokens
- elapsed time
- memory estimate

Example trace row:

```json
{
  "round": 3,
  "draft_len": 8,
  "drafted": [421, 912, 18, 77],
  "verified": [421, 912, 19],
  "accepted_count": 2,
  "mismatch_index": 2,
  "correction": 19
}
```

## Error handling

ExactKV should fail loudly on correctness problems.

Examples:

- If exact output check fails, raise an error.
- If cache sequence lengths diverge, raise an error.
- If compressor returns invalid shape, raise an error.
- If unsupported model cache format appears, raise a clear error.

Do not silently continue after state corruption.

## Architecture roadmap

### V1

Correctness proof with Hugging Face and simple compression.

### V2

Framework with compressor interface, metrics, and tests.

### V3

Benchmark suite and reports.

### V4

Advanced compressors.

### V5

Performance and offload.

### V6

vLLM/LMCache integration.

## Final architectural stance

ExactKV should start as a simple, transparent research runtime.

It should become a serious systems project only after the core correctness and benchmark logic is reliable.
