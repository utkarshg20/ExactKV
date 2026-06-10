# TurboQuant / TurboQuant+ Integration Research (V9 Phase A)

**Status:** Phase A complete — feasibility research only. **ExactKV does not implement
TurboQuant or TurboQuant+.** No adapter code, no compressor registration, no
Experiment 008 artifacts.

**Date:** 2026-06-09  
**Scope:** Determine whether TurboQuant / TurboQuant+ can be integrated behind
ExactKV's existing `BackendAdapter` in V9 Phase B.  
**Recommendation:** **Proceed to adapter prototype with restrictions** (§23).

> Guardrails inherited from [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md): no
> throughput, latency, tokens/sec, speedup, `runtime_seconds`, `active_gpu_kv_bytes`,
> or production-serving claims as ExactKV results. External TurboQuant+ numbers
> cited below are **upstream / community claims**, not ExactKV measurements.

---

## 1. Purpose

V9 Phase A answers:

> Can a real TurboQuant-family backend be wrapped by `BackendAdapter` so that
> ExactKV can (a) store a compressed KV representation, (b) materialize HF-compatible
> `past_key_values` for draft generation, and (c) keep authoritative full-precision
> KV separate for `VerificationEngine` — without changing generation, verification,
> or report schemas?

This document records installation attempts, API discovery, cache-format analysis,
Qwen2.5 compatibility, adapter-shape requirements, RunPod plans, failure modes,
and a go/no-go recommendation for Phase B.

---

## 2. What TurboQuant / TurboQuant+ is

### TurboQuant (paper)

**TurboQuant** (Zandieh et al., ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
is a data-oblivious vector quantizer for KV cache elements:

1. Extract per-vector L2 norm.
2. Apply a random rotation (WHT + sign flips in production; dense Haar in the Python prototype).
3. Scalar Lloyd-Max quantization per rotated coordinate.
4. Store indices + norms; decompress by inverse rotation and norm rescaling.

The paper's full Algorithm 2 adds a 1-bit QJL residual stage on keys. **Production
TurboQuant+ in llama.cpp drops QJL** on both K and V (documented in upstream README).

### TurboQuant+ (community workspace)

[**turboquant_plus**](https://github.com/TheTom/turboquant_plus) is a research
workspace bundling:

| Artifact | Role |
|---|---|
| **`turboquant/` Python package** | NumPy reference implementation (`PolarQuant`, `TurboQuant`, `TurboQuantMSE`, `KVCacheCompressor`) |
| **`refract-llm` wheel** | Cross-engine KV fidelity scorer (llama.cpp, MLX, vLLM, SGLang) — **not** a HF PyTorch quantizer |
| **`llama-cpp-turboquant` fork** | Production inference: C/Metal/CUDA packed formats `turbo2`/`turbo3`/`turbo4` |
| **MLX fork** (`TheTom/mlx@feature/turboquant-plus`) | `TurboKVCache` for Apple Silicon MLX models |
| **Papers / docs** | Asymmetric K/V, boundary V, sparse V dequant, block-size study |

TurboQuant+ extends TurboQuant with PolarQuant + Walsh–Hadard rotation, asymmetric
K/V cache types (`q8_0`-K + `turbo`-V), layer-aware boundary V, and optional sparse-V
decode optimizations in llama.cpp.

**ExactKV relevance:** ExactKV needs a **compress → store → materialize for draft**
path on **Hugging Face PyTorch** `past_key_values`. The feasible bridge is the
**Python `turboquant` package** plus torch/numpy tensor conversion — not the
llama.cpp runtime loop or GGUF serving stack.

---

## 3. Installation result

### Isolated venv attempt (2026-06-09, macOS, Python 3.13)

| Step | Result |
|---|---|
| `git clone https://github.com/TheTom/turboquant_plus.git` | ✅ Success |
| `pip install numpy scipy` + `PYTHONPATH=<repo>` | ✅ `import turboquant` succeeds |
| `pip install -e ".[dev]"` (refract-llm 0.3.2.3) | ✅ `refract --help` works |
| `python -m pytest tests/test_kv_cache.py tests/test_turboquant.py tests/test_polar_quant.py` | ✅ 43 passed (upstream turboquant tests) |
| `pip install turboquant` from PyPI | ❌ No standalone PyPI package for `turboquant` |

**Important:** `pyproject.toml` ships **only** `refract*` in the wheel. The `turboquant`
Python package is **dev-only** in the same repo — Phase B must vendor, submodule, or
`pip install` from a pinned git URL with `PYTHONPATH` / editable layout documented.

Scratch reproduction:

```bash
git clone https://github.com/TheTom/turboquant_plus.git /tmp/turboquant_plus_research
cd /tmp/turboquant_plus_research
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy
PYTHONPATH=/tmp/turboquant_plus_research python3 -c "from turboquant import KVCacheCompressor; print('ok')"
```

ExactKV repo script (no ExactKV deps changed):

```bash
PYTHONPATH=/tmp/turboquant_plus_research python3 scripts/research/turboquant_phase_a_inspect.py
```

---

## 4. Dependency constraints

| Component | Required for Phase B adapter | Notes |
|---|---|---|
| **numpy** ≥ 1.24 | Yes | Core quantize/dequantize |
| **scipy** ≥ 1.10 | Yes | Lloyd-Max / codebook helpers |
| **torch** | Yes (ExactKV already has) | Tensor bridge only |
| **transformers** | Yes (ExactKV already has) | Model + cache formats |
| **CUDA / Metal** | **No** for Python prototype path | llama.cpp kernels optional for fidelity studies |
| **cmake / C++** | No for Phase B | Only for llama.cpp builds |
| **mlx / mlx-lm** | No | Separate MLX stack |
| **vllm** | No | Out of V9 scope |

**ExactKV default `pyproject.toml` must not gain hard TurboQuant deps.** Follow
Experiment 005 precedent: optional `[turboquant]` extra in a dedicated `.venv-turboquant`.

---

## 5. Python / CUDA / PyTorch / transformers requirements

| Layer | Python | PyTorch | CUDA | transformers |
|---|---|---|---|---|
| **Python `turboquant` (Phase B target)** | ≥ 3.10 tested 3.13 | Bridge only; quant in NumPy | Not required | Uses existing ExactKV pins for cache extract/rebuild |
| **llama.cpp TurboQuant+ (production formats)** | N/A (C++) | N/A | Optional (community CUDA fork) | N/A (GGUF) |
| **MLX TurboKVCache** | mlx-lm ≥ 0.31 | N/A (MLX arrays) | N/A | N/A |
| **REFRACT scorer** | ≥ 3.9 | Optional per backend | Per backend | Optional (`refract-sglang`) |

Phase B smoke on `Qwen/Qwen2.5-0.5B` can run **CPU float32** (same as Experiments 001–007).
GPU is **not** a Phase B blocker for the NumPy adapter path.

---

## 6. Supported models and cache formats

### Upstream model coverage (external claims)

Upstream validates TurboQuant+ on many GGUF / MLX models (Qwen2.5 1.5B–7B, Qwen3.x,
Llama, Mistral, Command-R+ 104B, etc.) via **llama.cpp and MLX**, not HF PyTorch serving.

### HF PyTorch path (ExactKV-relevant)

| Model | Evidence | head_dim | Phase B fit |
|---|---|---:|---|
| **`Qwen/Qwen2.5-0.5B`** | `benchmarks/benchmark_ppl_tq_vs_rq.py` monkey-patches K quant on this model | **64** | ✅ Primary ExactKV model |
| **`Qwen/Qwen3-1.7B`** | `benchmarks/validate_real_model.py` extracts real KV tensors | **128** | ✅ Larger HF validation |
| Other Qwen2.5 variants | Same architecture family; head_dim 64 or 128 typical | — | Likely supported via `head_dim` ctor arg |

### Cache formats

| Format | Where | ExactKV compatibility |
|---|---|---|
| HF `tuple` / `dynamic_v4` / `dynamic_v5` | ExactKV `cache/utils.py` | ✅ Adapter materializes via `rebuild_cache` |
| `CompressedKVCache` (Python) | `turboquant/kv_cache.py` | ✅ Maps to `backend_data` dict |
| llama.cpp `turbo2`/`turbo3`/`turbo4` packed blocks | C structs in `ggml-turbo-quant.c` | ❌ Not directly HF; different runtime |
| MLX `TurboKVCache` | MLX lazy arrays | ❌ Wrong tensor stack |

---

## 7. API surface discovered

### Python `turboquant` exports (`turboquant/__init__.py`)

```
PolarQuant, QJL, TurboQuant, TurboQuantMSE, CompressedVector, KVCacheCompressor
```

### Primary integration class: `KVCacheCompressor`

```python
compressor = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
compressed: CompressedKVCache = compressor.compress(k_cache, v_cache)  # np float arrays
k_hat, v_hat = compressor.decompress(compressed)  # np float arrays
```

**Input shape:** `(num_layers, num_kv_heads, seq_len, head_dim)` — matches HF tensors
after `extract_kv_tensors` + numpy conversion (per-layer list can be stacked).

**K quantizer:** `TurboQuant` (PolarQuant + QJL) — paper-faithful, **not** identical to
production llama.cpp K path (QJL dropped in production).

**V quantizer:** `TurboQuantMSE` (PolarQuant only) — aligned with production V path.

### Lower-level classes

| Class | Use |
|---|---|
| `PolarQuant` | MSE scalar quant with dense rotation matrix |
| `TurboQuantMSE` | V-cache / production-style polar quant |
| `TurboQuant` | K-cache with QJL residual (research) |
| `CompressedVector` | Per-head K batch container |

### REFRACT (`refract` CLI)

Separate evaluation framework scoring llama.cpp / MLX / vLLM servers. **Not** a
compressor API for ExactKV. Useful as external quality reference only.

### llama.cpp CLI (not Phase B integration)

```
--cache-type-k turbo3 --cache-type-v turbo3
```

Operates inside GGUF inference — no export of HF `past_key_values`.

---

## 8. Compression formats discovered

### Python prototype storage

`CompressedKVCache` holds:

- **K:** per-layer, per-head `CompressedVector` (mse_indices, vector_norms, qjl_signs, residual_norms, bit_width)
- **V:** per-layer, per-head `indices` + `norms` arrays (MSE PolarQuant)

Quantization is **per (seq_pos, head_dim) vector** along the head_dim axis (one vector
per token position per head).

### llama.cpp production formats (external)

| Type | Bits/value (approx) | Notes |
|---|---:|---|
| `turbo2` | 2.5 | Extreme compression; asymmetric K recommended |
| `turbo3` | 3.5 (block_size=32) | Default; block_size=128 changes storage (upstream paper) |
| `turbo4` | 4.25 | Best quality in upstream matrix |
| `q8_0` | 8 | llama.cpp baseline quant cache |

Packed layout, WHT fast rotation, Metal/CUDA kernels — **byte layout differs** from
Python `CompressedKVCache`. Phase B honest bytes must count **actual stored numpy
payload**, not theoretical llama.cpp compression ratios.

### Production vs Python divergence (critical)

| Aspect | Python `KVCacheCompressor` | llama.cpp TurboQuant+ |
|---|---|---|
| K algorithm | TurboQuant + QJL | PolarQuant / turbo3/4 **without QJL** |
| Rotation | Dense Haar `random_rotation_dense` | Walsh–Hadamard + sign flips |
| Block packing | Per-vector indices | Block structs (`block_size` 32/128) |
| Sparse V | Not in Python KV layer | Decode-time kernel optimization |

Phase B adapter should document which path it implements. **Recommended:** start with
`TurboQuantMSE` for **both** K and V at configurable bit widths (production-aligned
polar path), or fork `KVCacheCompressor` K path to drop QJL — not raw `TurboQuant` with QJL.

---

## 9. Whether TurboQuant exposes compressed KV directly

**Yes — in the Python prototype.**

`KVCacheCompressor.compress()` returns a `CompressedKVCache` dataclass with structured
numpy fields. This is a **direct compressed representation**, not an opaque GPU handle.

**In llama.cpp:** compressed KV lives inside the runtime's internal cache buffers — no
Python API to extract packed blocks for HF `BackendAdapter`.

**Conclusion for ExactKV:** Phase B must use the **Python `turboquant` package**, not
llama.cpp internals, to populate `CompressedKVState.data`.

---

## 10. Whether TurboQuant can materialize KV for draft generation

**Yes — via dequantize + `rebuild_cache`.**

Flow:

1. `extract_kv_tensors(full_state)` → per-layer torch K/V lists.
2. Convert to numpy → `KVCacheCompressor.compress()`.
3. Store `CompressedKVCache` (+ rotation seeds / quantizer params) in `backend_data`.
4. On draft: `decompress()` → numpy → torch tensors on `compressed.device`.
5. `rebuild_cache(k_list, v_list, cache_format, seq_len)` → HF `past_key_values`.

Upstream `benchmarks/validate_real_model.py` and `benchmark_ppl_tq_vs_rq.py` demonstrate
steps 1–2 and quant-dequant on real Qwen KV tensors.

**Lossy drafting:** compressed predictions may differ from full KV. Adapter must override
`_get_next_token_id` to run `materialize_for_draft` + forward pass (same pattern as other
lossy compressors).

---

## 11. Whether full authoritative KV can remain separate for ExactKV verification

**Yes — no blocker.**

TurboQuant Python path is **offline**: compress cloned tensors, no model hooks, no global
patching. Matches `BackendAdapter` contract:

- `compress()` clones tensors before `_backend_compress` (base class enforced).
- `VerificationEngine` uses `FullKVState` only — never touches compressor.
- No `verification_mode()` hook disable required (contrast kvpress).

**Risk:** If Phase B mistakenly integrated llama.cpp in-process or MLX hooks, ownership
would blur. **Rejected for Phase B** — stay on offline numpy path.

---

## 12. Whether TurboQuant requires CUDA kernels or custom ops

| Path | Custom ops? |
|---|---|
| Python `turboquant` | **No** — pure NumPy/SciPy |
| llama.cpp | **Yes** — Metal/CUDA/HIP kernels in `ggml-turbo-quant.c` |
| MLX | **Yes** — MLX ops in fork |

Phase B adapter: **no custom ops in ExactKV**, no CUDA requirement.

---

## 13. Whether CPU fallback exists

**Yes** for the Python prototype (verified: roundtrip on CPU, pytest pass).

llama.cpp also builds CPU backends, but that path does not help HF ExactKV integration.

---

## 14. Whether Qwen/Qwen2.5-0.5B is supported

**Yes — for the Python HF bridge path.**

| Check | Result |
|---|---|
| Public config `head_dim` | 64 (= 896 / 14 attention heads) |
| `PolarQuant` / `KVCacheCompressor` at d=64 | ✅ Smoke roundtrip passed |
| Upstream benchmark reference | `benchmark_ppl_tq_vs_rq.py` targets `Qwen2.5-0.5B` |
| ExactKV default model | Same checkpoint; `cache/utils.py` tested on it |

**Caveat:** Upstream llama.cpp asymmetric tables focus on Q4_K_M **GGUF** weights;
ExactKV uses **HF float32** weights. Acceptance behaviour may differ — that is what
Experiment 008 measures; not a Phase A blocker.

---

## 15. Whether larger Qwen models are likely supported

**Likely yes** for HF + Python path, with constraints:

| Model class | head_dim | Notes |
|---|---:|---|
| Qwen2.5-0.5B / 1.5B (typical) | 64 | Dense rotation O(d²) — cheap |
| Qwen2.5-7B / Qwen3-1.7B | 128 | Validated upstream on real KV tensors |
| Very large (70B+) | varies | NumPy compress/decompress per round may be slow — **acceptance evaluation still valid on CPU/GPU**; wall-clock is out of scope |

Phase E RunPod validation (≥1.5B) is recommended for credibility, not because 0.5B
is unsupported.

---

## 16. Cache ownership and mutation risks

| Risk | Severity | Mitigation |
|---|---|---|
| Mutating authoritative `FullKVState` tensors | Low if adapter follows base class | `BackendAdapter.compress` clones before `_backend_compress` |
| Mutating `backend_data` during materialize | Low | `_backend_materialize` must not mutate; decompress into new arrays |
| Global monkey-patches (kvpress lesson) | **None** for numpy path | Do not use hook-based upstream benchmarks in adapter |
| llama.cpp shared cache ownership | N/A | Do not embed llama.cpp in Phase B |
| Non-deterministic materialize | Medium | Persist `seed`, bit widths, and quantizer rotation state in `backend_data` |

---

## 17. Workspace-memory implications

Use V5 fields honestly per [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md):

| Field | TurboQuant Python accounting |
|---|---|
| `stored_kv_bytes` | Sum of numpy array nbytes in `CompressedKVCache` (+ rotation matrix storage if persisted) |
| `materialized_working_kv_bytes` | Full dequantized K/V torch tensors (fp32/fp16 as materialized) |
| `metadata_bytes` | Lloyd-Max codebooks, rotation matrices, seeds, config (k_bits, v_bits, norm_correction) |
| `temporary_workspace_bytes` | Conservative estimate for numpy scratch during compress/decompress |
| `total_kv_footprint_bytes` | Sum of above — **accounting sum, not measured peak GPU** |
| `supports_real_bytes_claim` | `True` only if `stored_kv_bytes` < `__full_bytes__` with honest packing count |

**Do not** cite upstream "3.8–6.4× compression" as ExactKV `memory_reduction_factor` —
those are **external llama.cpp format claims** with different packing and algorithms.

`KVCacheCompressor.memory_stats()` provides theoretical bit budgets; adapter should
prefer **actual stored array nbytes** for honesty.

---

## 18. Required BackendAdapter shape

Thin wrapper class (Phase B — not implemented in Phase A):

```text
TurboQuantBackendAdapter(BackendAdapter)
  name = "turboquant_polar"  # illustrative
  capabilities = CompressorCapabilities(
      is_simulated=False,
      supports_real_bytes_claim=True,  # if stored < full
      supports_quantization=True,
      asymmetric=True,               # if k_bits != v_bits
      backend_name="turboquant_plus",
      backend_version="<pinned git sha>",
      adapter_name="TurboQuantPolarAdapter",
      ...
  )

  _backend_compress(k_tensors, v_tensors, cache_format):
      # torch -> numpy stack (layers, kv_heads, seq, head_dim)
      # KVCacheCompressor or production-aligned variant
      # return {compressed_payload, cache_format, head_dim, k_bits, v_bits, seed, ...}

  _backend_materialize(backend_data, cache_format):
      # decompress -> per-layer torch tensors -> rebuild_cache(...)

  _backend_workspace_bytes(full_kv_bytes, backend_data):
      # count numpy + materialized torch bytes

  _get_next_token_id(state, backend_data):
      # override: materialize + forward (lossy)
```

No `_compresses_via_full_state()` — offline path only.

---

## 19. Required changes, if any, to BackendAdapter

**None required for Phase B.**

Existing sealed API in `exactkv/compressors/backend_adapter.py` is sufficient:

- Tensor-list `_backend_compress` / `_backend_materialize` hooks match numpy bridge.
- `verification_mode()` default no-op is correct.
- `_get_next_token_id` optional override already documented for lossy backends.

**Optional Phase B additions (not blockers):**

- Documented optional extra `turboquant` in ExactKV `pyproject.toml` (numpy/scipy pins only).
- Experiment script factory `create_turboquant_adapter()` (kvpress pattern) — not default registry.

---

## 20. Testing plan for Phase B

| Test | Gate |
|---|---|
| Unit: clone safety | Full state tensors unchanged after `compress()` |
| Unit: roundtrip shape | `materialize_for_draft` → `extract_kv_tensors` shapes match `logical_seq_len` |
| Unit: determinism | Same `backend_data` → identical materialized tensors |
| Unit: `stats()` reconcile | `total_kv_footprint_bytes` = sum of components |
| Unit: capabilities honest | `is_simulated=False`, `supports_real_bytes_claim` matches actual nbytes |
| Smoke exactness | 2 prompts × 2 `draft_len` on `Qwen2.5-0.5B`; **`exactkv_failures == 0`** |
| Isolation | Default `pytest` without `[turboquant]` extra passes |
| Forbidden fields | `_assert_no_forbidden_fields` on any experiment JSON |

Experiment 008 (Phase C) extends to full `core` suite (34 prompts) per
[`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) §13.

---

## 21. RunPod GPU plan if needed

### When GPU is needed

| Activity | GPU required? |
|---|---|
| Phase B adapter smoke (Python numpy) | **No** — CPU float32 acceptable |
| Phase C Experiment 008 (0.5B) | **No** — matches prior experiments |
| llama.cpp format fidelity / REFRACT cross-check | Optional — separate from ExactKV gate |
| Phase E ≥1.5B Qwen2.5 validation | **Recommended** — memory + backend-faithful device |
| KVQuant / KIVI (Phase D) | Likely yes — separate track |

### Phase E RunPod commands (illustrative)

```bash
# 1. Pod: 1× A100 40GB or L40S 48GB, CUDA 12.x, Ubuntu 22.04
# 2. Clone ExactKV + create isolated venv
git clone <exactkv-repo> && cd ExactKV
python3 -m venv .venv-turboquant && source .venv-turboquant/bin/activate
pip install -e ".[dev,turboquant]"   # Phase B optional extra (to be added)
git clone https://github.com/TheTom/turboquant_plus.git vendor/turboquant_plus
export PYTHONPATH=$PWD/vendor/turboquant_plus

# 3. Pin versions in experiment manifest (torch, transformers, turboquant git sha)
python scripts/run_experiment_008_turboquant.py --model Qwen/Qwen2.5-1.5B --device cuda

# 4. Gate: exactkv_failures == 0; record GPU model + CUDA in manifest only
```

GPU is for **larger-model exactness validation**, not throughput measurement.

---

## 22. Failure modes that would reject TurboQuant for V9

| Failure mode | Phase A verdict |
|---|---|
| No API to compress HF-style K/V tensors | **Not triggered** — Python path exists |
| Cannot materialize HF `past_key_values` | **Not triggered** — decompress + `rebuild_cache` |
| Requires verify-path hooks | **Not triggered** — offline numpy |
| Only GGUF/llama.cpp runtime (no tensor API) | **Triggered for llama.cpp-only path** — reject embedding llama.cpp in ExactKV |
| Non-deterministic materialize | **Manageable** — persist seeds/state |
| Cannot support Qwen2.5-0.5B head_dim | **Not triggered** — d=64 works |
| Cannot count bytes honestly | **Not triggered** — numpy nbytes countable |
| Python dep incompatible with ExactKV pins | **Not triggered** — numpy/scipy only |

**Would be no-go if:** Phase B could only integrate via llama.cpp/vLLM/MLX with no tensor
bridge — **not the case** given Python `KVCacheCompressor`.

---

## 23. Recommendation

### **Proceed to adapter prototype — with restrictions**

| Restriction | Rationale |
|---|---|
| **Python `turboquant` only in Phase B** | Only path exposing compress/decompress on numpy KV tensors compatible with HF extract/rebuild |
| **Do not integrate llama.cpp / MLX / REFRACT runtime** | Wrong inference stack; no HF `past_key_values` export |
| **Production-aligned K quant** | Prefer `TurboQuantMSE` for K (no QJL) or document deviation from llama.cpp turbo3/4 |
| **Optional `[turboquant]` extra + `.venv-turboquant`** | Keep default ExactKV install unchanged (Experiment 005 pattern) |
| **Not in default registry** | Construct via experiment factory until stable |
| **Honest bytes** | Count actual numpy storage; do not claim llama.cpp packed-bit ratios |
| **Boundary V / sparse V deferred** | Not in `KVCacheCompressor`; optional Phase B+ policy layer |
| **RunPod deferred to Phase E** | Not required for Phase B smoke or Experiment 008 on 0.5B |

### Not recommended now

| Option | Why |
|---|---|
| Defer and try KIVI/KVQuant first | TurboQuant Python path is **more HF-compatible** than KVQuant CUDA hooks |
| Full no-go | Tensor bridge exists; exactness gate unaffected |
| Requires RunPod GPU first | **False** for Phase B/C on 0.5B CPU |

### Phase B entry criteria (met)

- [x] Install/import documented
- [x] API maps to `_backend_compress` / `_backend_materialize`
- [x] Qwen2.5-0.5B head_dim compatible
- [x] Verify isolation confirmed
- [x] `BackendAdapter` sufficient without schema changes

**Commit Phase A before Phase B adapter implementation** — this document is the approval
artifact for Phase B coding.

---

## Related documents

| Document | Relevance |
|---|---|
| [`V9_SCOPE_STATEMENT.md`](V9_SCOPE_STATEMENT.md) | V9 phases; Experiment 008 plan |
| [`BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) | Adapter contract |
| [`EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) | Isolated env precedent |
| [`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md) | §10 TurboQuant+ survey |
| [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) | D1/D2 status update |

## External references (not ExactKV results)

- TurboQuant paper: <https://arxiv.org/abs/2504.19874>
- turboquant_plus repo: <https://github.com/TheTom/turboquant_plus>
- llama-cpp-turboquant: <https://github.com/TheTom/llama-cpp-turboquant>
