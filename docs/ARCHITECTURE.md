# ExactKV Architecture

## High-Level Flow

Prompt
↓
Full KV Cache Creation
↓
Compressed KV Cache Creation
↓
Draft Generation
↓
Verification
↓
Acceptance / Rejection
↓
Output

## Components

### Full Cache Manager

Responsibilities:

* Maintain authoritative KV state
* Generate verification tokens
* Handle rollback logic

### Compressor Interface

All compressors implement:

```python
class KVCompressor:
    def compress(...)
    def update(...)
    def stats(...)
```

Supported compressors:

* Int8
* Int4
* TokenDrop
* TurboQuant backend
* Future compressors

### Draft Engine

Responsibilities:

* Generate candidate tokens
* Operate exclusively on compressed KV

### Verification Engine

Responsibilities:

* Compare draft tokens against full-KV outputs
* Compute longest matching prefix
* Determine acceptance rate

### Metrics Engine

Tracks:

* Throughput
* Acceptance rate
* Memory reduction
* Exactness

## MVP Constraints

* Single GPU
* Greedy decoding only
* Hugging Face models
* PyTorch only
* No Triton
* No CUDA kernels

## Future Extensions

* Triton kernels
* vLLM integration
* CPU KV offloading
* Multi-GPU support
* Async verification
