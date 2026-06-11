# GPU Memory Methodology (V11 Phase 4 / Experiment 018)

**Status:** Pilot methodology — device-level observations only; **not** a standard ExactKV metric.
**Distinct from:** V5 `total_kv_footprint_bytes` workspace accounting
([`EXPERIMENT_004_WORKSPACE_MEMORY.md`](EXPERIMENT_004_WORKSPACE_MEMORY.md)).

> This is a **GPU memory methodology pilot**, not a performance benchmark.
> Measurements are **PyTorch CUDA allocation observations**, not universal
> hardware-independent memory claims.
> `total_kv_footprint_bytes` remains a **conservative accounting sum**, not measured peak GPU memory.
> Measured GPU allocation includes **model weights, framework allocator behaviour,
> temporary tensors, and other non-KV allocations** unless carefully isolated.
> ExactKV does **not** claim throughput, latency, speedup, runtime, tokens/sec,
> or production readiness.
> ExactKV does **not** claim active GPU memory as a stable standard metric unless
> methodology is approved in a future scope.

---

## 1. Purpose

Define how ExactKV may **optionally** record PyTorch CUDA device allocation at defined
lifecycle points — separately from V5 byte accounting — and document why the two
numbers must not be conflated.

Experiment **018** pilots this methodology on RunPod GPU with `exactkv_failures == 0`
as a correctness gate. Outcomes: **pilot-success** (caveated observations published)
or **honest deferral** (methodology published, measurements deemed too noisy).

---

## 2. V5 accounting sum vs measured GPU allocation

| Aspect | V5 `total_kv_footprint_bytes` | Pilot GPU observations |
|---|---|---|
| **What it is** | Conservative sum of stored + materialized + metadata + temporary (tensor-shape accounting) | `torch.cuda.memory_allocated` / `max_memory_allocated` at lifecycle points |
| **Device** | CPU or GPU — derived from tensor sizes | **CUDA device only** |
| **Includes model weights?** | **No** — KV/workspace components only | **Yes** — entire process GPU arena |
| **Peak vs point-in-time** | Static estimate after prefill | Peak during generation + point snapshots |
| **Hardware-independent?** | Yes (shape-based) | **No** — GPU, driver, torch, allocator dependent |
| **Standard report schema?** | **Yes** (stable since V5) | **No** — pilot artifact only |

**Rule:** Never substitute pilot GPU bytes for `total_kv_footprint_bytes` in leaderboards,
acceptance tables, or launch claims.

---

## 3. Why `total_kv_footprint_bytes` is not measured peak GPU memory

V5 fields count tensor element sizes × dtype widths for KV-related storage and
conservative workspace estimates. They do **not**:

- Query the CUDA driver or allocator
- Include model parameter weights on GPU
- Include autograd graphs, cudnn workspaces, or flash-attention temporaries
- Reflect fragmentation, caching, or unused reserved pools
- Capture peak memory across an entire multi-round ExactKV loop

Therefore `total_kv_footprint_bytes` answers: *"What is a conservative accounting
footprint for KV-related tensors?"* — not *"How much GPU memory did this run use?"*

---

## 4. What PyTorch CUDA APIs can and cannot measure

### Can observe (with caveats)

| API | Meaning |
|---|---|
| `torch.cuda.memory_allocated()` | Bytes of tensors currently allocated on device |
| `torch.cuda.max_memory_allocated()` | Peak allocated since last `reset_peak_memory_stats()` |
| `torch.cuda.reset_peak_memory_stats()` | Reset peak counter (call after `synchronize`) |
| `torch.cuda.synchronize()` | Wait for GPU kernels — **required** before reading stats |

### Cannot reliably isolate

| Component | Issue |
|---|---|
| **KV cache only** | No public API separates KV tensors from weights in HF forward |
| **Exact peak system-wide** | `memory_allocated` excludes cached blocks in allocator pool |
| **Cross-run comparability** | Allocator retains caches; order of runs affects peaks |
| **Multi-GPU** | Pilot uses device 0 only |
| **Serving stacks** | vLLM/LMCache paging not in scope |

### Field naming (pilot artifact only)

| Field | Lifecycle point |
|---|---|
| `gpu_baseline_model_loaded_bytes` | After sync/reset, model already on GPU |
| `gpu_allocated_after_prefill_bytes` | After `prefill_to_full_state` |
| `gpu_peak_allocated_during_run_bytes` | Peak during `ExactKVGenerator.generate` |
| `gpu_allocated_after_run_bytes` | After generation completes |
| `gpu_allocated_after_cleanup_bytes` | After `gc.collect()` + `empty_cache()` (best-effort) |

**Do not use** `active_gpu_kv_bytes` — reserved for a future approved schema if ever adopted.

---

## 5. Warmup requirements

1. **Model load warmup:** Load model once; run one discarded forward pass optional.
2. **Per-cell protocol:**
   - `torch.cuda.synchronize()`
   - `torch.cuda.reset_peak_memory_stats()`
   - Record baseline `memory_allocated()`
   - Prefill → record post-prefill allocated
   - ExactKV generate → record `max_memory_allocated()` peak
   - Record post-run allocated
   - Best-effort cleanup → record post-cleanup allocated
3. **Do not** compare cells across different GPUs or torch builds without noting hardware.

---

## 6. Model weights vs KV cache vs temporary allocations

On GPU, `memory_allocated()` after model load includes **all parameters and buffers**.
Prefill adds KV tensors. ExactKV generation adds:

- Authoritative full KV growth on commit
- Compressed materialization copies on draft
- Verification deep-copies (temporary spikes)
- Framework temporaries during attention

Pilot observations therefore measure **whole-device tensor allocation**, not KV-only.
Subtracting baseline from peak is a **heuristic delta**, not a certified KV footprint.

---

## 7. Allocator and cache effects

PyTorch's caching allocator reuses freed blocks. `empty_cache()` returns unused
cached memory to the driver but does not guarantee identical baseline across cells.
Peaks can vary ± few percent between identical prompts on the same GPU.

**Implication:** Pilot data supports **relative** comparisons (noop vs int8 on same
hardware/session), not absolute universal claims.

---

## 8. Hardware and runtime specificity

Observations depend on:

- GPU model and VRAM size
- NVIDIA driver version
- PyTorch build (CUDA version)
- `float16` vs `float32`
- transformers / attention implementation
- Prompt length and `max_new_tokens`

**Not portable** across RunPod pods, local laptops, or production clusters.

---

## 9. Why this is not throughput/latency/performance benchmarking

Experiment 018 records **memory allocation snapshots only**. It does **not** measure:

- tokens/sec, throughput, latency, speedup, `runtime_seconds`
- batching efficiency or serving scheduler behaviour
- production-serving readiness

Comparing GPU bytes between compressors is **not** a speedup or efficiency claim.

---

## 10. Pilot artifact policy

| Item | Location |
|---|---|
| Methodology | This document |
| Pilot JSON/CSV | `reports/experiment_018_gpu_memory_pilot.json` (gitignored) |
| Experiment report | [`EXPERIMENT_018_GPU_MEMORY_PILOT.md`](EXPERIMENT_018_GPU_MEMORY_PILOT.md) |
| Standard `validate_report` schema | **Unchanged** — pilot does not use it |
| `active_gpu_kv_bytes` in standard reports | **Not added** |

---

## 11. Decision criteria (pilot-success vs deferral)

**Pilot-success** if:

- CUDA measurements complete for ≥80% of planned cells
- `exactkv_failures == 0`
- Observations are internally consistent (peak ≥ post-prefill ≥ baseline within tolerance)
- Caveats documented; no universal GPU memory claim

**Deferral** if:

- Measurements are too noisy (peak < baseline, wild cell-to-cell variance)
- OOM prevents full panel
- Allocator effects make cross-compressor ordering meaningless

Either outcome closes D14 methodology gate for V11 Phase 4.

---

## Related

- [`V5_SCOPE_STATEMENT.md`](V5_SCOPE_STATEMENT.md) — workspace accounting design
- [`EXPERIMENT_004_WORKSPACE_MEMORY.md`](EXPERIMENT_004_WORKSPACE_MEMORY.md) — V5 validation
- [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) — Phase 4 plan
- `exactkv/metrics/gpu_memory_pilot.py` — pilot helpers
- `scripts/run_experiment_018_gpu_memory_pilot.py` — pilot runner
