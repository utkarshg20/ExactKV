# ExactKV V1 Scope Statement

## What V1 is

V1 is the correctness prototype for ExactKV.

Its sole goal is to prove the core ExactKV generation loop is algorithmically correct
using Hugging Face Transformers, greedy decoding, a simple INT8 compressor, and
single-request single-device execution.

## What V1 proves

1. **Full-KV baseline correctness.**
   The custom `generate_full_greedy` function produces exactly the same token IDs as
   `model.generate(do_sample=False, num_beams=1)`. This is the ground truth gate.

2. **Cache alignment invariant.**
   After every draft-verify-commit round, the authoritative `FullKVState` and the
   `CompressedKVState` both represent the same logical committed token prefix.
   Lengths are asserted to match at every round boundary.

3. **Accept/reject/correction bookkeeping correctness.**
   - Accepted tokens are those where `draft_token_id == verifier_token_id` at each
     position, checked left to right.
   - On the first mismatch, the verifier's token is committed as a correction.
   - Rejected tokens (positions after the first mismatch) are never committed.
   - This is tested with mocked token sequences (no model) and with real compressors.

4. **Benchmark plumbing.**
   Three modes — `full`, `lossy`, `exactkv` — run on the same prompts and produce a
   structured JSON report.

5. **Primary correctness criterion.**
   For all prompts in the smoke suite:
   ```python
   exactkv_output_ids == full_output_ids
   ```
   This is a hard requirement. If it fails, no other result from that run is meaningful.

## What V1 explicitly does NOT prove

1. **VeriCache-level throughput or speedup.**
   V1 uses sequential (step-by-step) verification, no CPU offload, no async transfer,
   no cross-resource staggering, and no vLLM scheduler. It will be slower than full-KV
   inference for most settings. This is expected and acceptable.

2. **Parallel verification correctness.**
   V1 verifies one draft token at a time by feeding it to the full model and comparing
   the argmax output. It does not implement the single-forward-pass multi-position
   verify from the VeriCache paper. That belongs to V5.

3. **Production performance claims.**
   V1 is a research and correctness prototype. Token/sec and memory numbers are
   measurement artifacts for later analysis, not performance claims.

4. **Broad model compatibility.**
   V1 targets `Qwen/Qwen2.5-0.5B` (and optionally TinyLlama). Cache format differences
   across model families are known risks and are not addressed in V1.

5. **Sampling-compatible exactness.**
   V1 is greedy-only. Rejection-sampling-compatible verification is future work.

## V1 scope boundary

| In scope | Out of scope |
|---|---|
| Hugging Face Transformers | vLLM, SGLang, TensorRT-LLM |
| Greedy decoding | Sampling, beam search, temperature |
| Single request | Batching |
| Single device (GPU or CPU) | CPU offload, async transfer, CUDA streams |
| NoOp / INT8 / debug_noise compressors | KIVI, KVQuant, TurboQuant, SnapKV, KVzip |
| Sequential verification | Parallel verification (single-pass verify) |
| INT8 (per-tensor symmetric) | INT4, bit-packing, custom kernels, Triton |
| JSON benchmark report | Dashboards, leaderboards |
| Token-ID exact match | Semantic similarity metrics as primary |

## V1 exit criteria

All of the following must pass before V1 is considered complete:

- [ ] `tests/test_full_generation.py` (gate): custom greedy == `model.generate`
- [ ] `tests/test_acceptance_logic.py`: mocked accept/reject cases
- [ ] `tests/test_verification_engine.py`: NoOp verification on real model
- [ ] `tests/test_noop_exactkv.py`: NoOp ExactKV == full, acceptance 100%
- [ ] `tests/test_int8_exactkv.py`: INT8 ExactKV == full
- [ ] `tests/test_debug_noise_exactkv.py`: rejection forced, ExactKV still == full
- [ ] `tests/test_metrics.py`: metric reconciliation
- [ ] `tests/test_benchmark_runner.py`: JSON report, `exactkv_failures == 0`

## Known V1 limitations and brittleness notes

### DynamicCache internal reconstruction (transformers 5.8.1)

ExactKV currently supports transformers >= 5.8 `DynamicCache` through direct
attribute injection into `DynamicLayer` objects (`layer.keys`, `layer.values`,
`layer.is_initialized`, `layer.dtype`, `layer.device`).  This relies on
`DynamicLayer.__dict__` remaining stable.

**Risk:** This reconstruction may break silently across transformers versions if
`DynamicLayer` adds validation, changes internal field names, or introduces a
more restricted interface.

**Future mitigation:** Replace with a version-pinned compatibility layer or
contribute a stable `from_tensors()` / `from_dict()` class method to upstream
`DynamicCache`.  Until then, the minimum viable fix is to pin `transformers` to
a tested version range in `pyproject.toml` and gate the cache utilities with a
version check.

**Scope note:** This limitation is acceptable for V1 because the goal is
*correctness verification*, not production deployment.  All tests must run with
`TRANSFORMERS_OFFLINE=1` (model cached locally) and are validated against
`Qwen/Qwen2.5-0.5B` on transformers 5.8.1.

## Citation and novelty note

The draft-then-verify compressed-KV algorithm is from:

> VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is a
compressor-agnostic, Hugging Face-first implementation, a structured benchmark harness,
and a framework for evaluating compressors by acceptance behavior under full-KV
verification.
