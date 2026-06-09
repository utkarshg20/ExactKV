# ExactKV v0.6.0 Release Notes

**Status:** V6 implementation complete (Phases 0–E).
**Base:** Builds on `v0.5.0` (workspace-aware memory accounting, Experiment 004).

> **V6 is an integration and evaluation release, not a performance release.**
> No throughput, latency, speedup, runtime, or production-readiness claims.
> ExactKV does not claim kvpress's external benchmark numbers as ExactKV results.

---

## 1. V6 summary

V6 introduces a **real-backend adapter boundary** (`BackendAdapter`) so an external
KV-cache compression library can plug into ExactKV's existing `KVCompressor` protocol
without changing the draft-verify-commit loop or verification engine.

V6 delivers:

- A documented `BackendAdapter` interface and a minimal **pass-through PoC**
  (`backend_passthrough`) exercising the boundary.
- A **restricted experimental** `KVPressKnormAdapter` wrapping **KnormPress only**
  from the external [kvpress](https://github.com/NVIDIA/kvpress) library.
- Phase C core-suite validation and **Experiment 005** comparing the restricted
  kvpress adapter against ExactKV baselines by exactness, acceptance behaviour,
  and V5 workspace-memory accounting.

V6 evaluates backends by **acceptance behaviour and memory honesty only** — never
by speed or serving performance.

---

## 2. What V6 adds

| Deliverable | Location |
|---|---|
| `BackendAdapter` interface design | [`docs/BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) |
| Pass-through PoC adapter (`backend_passthrough`) | `exactkv/compressors/backend_adapter.py` |
| Restricted `KVPressKnormAdapter` (KnormPress only) | `exactkv/compressors/kvpress_knorm.py` |
| `verification_mode()` hook guard | `BackendAdapter` + `ExactKVGenerator` |
| Optional `[kvpress]` extra (not default install) | `pyproject.toml` |
| Phase C validation report | [`docs/KVPRESS_KNORM_VALIDATION.md`](KVPRESS_KNORM_VALIDATION.md) |
| Experiment 005 report | [`docs/EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md) |
| Experiment 005 runner (isolated env only) | `scripts/run_experiment_005_kvpress_knorm.py` |
| Additive capability fields | `backend_name`, `backend_version` on `CompressorCapabilities` |

**Unchanged from V5:** generation logic, draft-verify-commit loop, report schema
(except additive `kvpress_gates` in Experiment 005 JSON), and the no-performance-claim
policy.

---

## 3. BackendAdapter and backend_passthrough

`BackendAdapter` is a thin wrapper that translates a real backend's compress/decompress
API into ExactKV's `KVCompressor` protocol. The verification engine continues to use
authoritative full-precision KV only; the adapter affects the **draft path** only.

**`backend_passthrough`** (V6 Phase B PoC) is registered in the default compressor
registry. It exercises the adapter boundary with a trivial identity backend — lossless,
`exactkv_failures == 0` — proving the interface without an external dependency.

See [`docs/BACKEND_ADAPTER_INTERFACE.md`](BACKEND_ADAPTER_INTERFACE.md) for the full
design: full-state compression (Option A), capability metadata, workspace-field
population rules, and hook-safety requirements.

---

## 4. Restricted KVPressKnormAdapter

`KVPressKnormAdapter` wraps **KnormPress only** from kvpress 0.5.3. It is:

- **Not** in the default compressor registry.
- Constructed only via `create_kvpress_knorm_adapter()` (e.g. in the Experiment 005
  script or kvpress validation tests).
- Run with **`isolate_compression_model=True`** (default): compression replays prefill
  on a `deepcopy` of the verification model so kvpress hooks do not mutate the model
  used for verify/commit.
- Guarded by **`verification_mode()`**: asserts zero attention forward hooks on the
  verification model before, during, and after verification.

**Byte semantics:** `stored_kv_bytes` and `materialized_working_kv_bytes` reflect
**real pruned `DynamicCache` tensor bytes** (token-dropping), not packed low-bit
quantization. For KnormPress, `materialized == stored`. `supports_real_bytes_claim=True`
applies to this pruned-cache storage only.

---

## 5. kvpress environment restrictions

kvpress is **optional** and **isolated**:

| Item | Default env | `[kvpress]` env (`.venv-kvpress`) |
|---|---|---|
| `pip install exactkv` | No kvpress | `pip install -e ".[kvpress]"` |
| transformers | 5.8.x | 5.2.0 (kvpress pins `<5.3`) |
| `import exactkv.compressors` | Does not load kvpress | May load kvpress only when adapter is constructed |
| Registry | `backend_passthrough` only | `kvpress_knorm_restricted` **not** registered |

**Additional constraints:**

- `import kvpress` may **globally patch** `ALL_ATTENTION_FUNCTIONS`; keep kvpress
  imports out of default ExactKV module loading.
- Python 3.13 requires `fire>=0.7.1` workaround in the kvpress venv only
  (kvpress pins `fire<0.7` which imports removed `pipes`).
- No DecodingPress, AdaKVPress, ComposedPress, or `KVPressTextGenerationPipeline`.

---

## 6. Experiment 005 summary

Full report: [`docs/EXPERIMENT_005_KVPRESS_KNORM.md`](EXPERIMENT_005_KVPRESS_KNORM.md).

| Metric | Value |
|---|---|
| Total cells | **272** |
| Core prompts | **34** |
| Compressors | **8** (`noop`, `int8`, `int4_sim`, `k8_v4_sim`, `k_full_v8`, `k8_v_full`, `backend_passthrough`, `kvpress_knorm_restricted`) |
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| KVPress `compression_ratio` | 0.5 |
| **ExactKV failures** | **0** (all compressors) |

**Gates (kvpress_knorm_restricted, 34 cells):**

| Gate | Result |
|---|---|
| Verification model hooks always 0 | **PASS** |
| Compression model hooks 0 → 24 → 0 | **PASS** |
| Pruning on all prompts (physical `<` logical seq) | **PASS** |
| Workspace-memory reconciliation | **PASS** |
| `supports_real_bytes_claim` for pruned DynamicCache bytes | **PASS** |

**Acceptance highlights (aggregate draft-token accept rate):**

| Compressor | Accept rate |
|---|---|
| `noop` / `backend_passthrough` | 1.000 |
| `k_full_v8` | 0.990 |
| `k8_v_full` / `int8` | ~0.96 |
| `k8_v4_sim` | 0.891 |
| `int4_sim` | 0.628 |
| **`kvpress_knorm_restricted`** | **0.413** |

Lossy draft divergences (102 / 272 cells) are **expected**; final ExactKV output
remains exact because verification uses authoritative full KV.

Artifacts (gitignored): `reports/experiment_005_kvpress_knorm.{json,csv}`.

---

## 7. What V6 does not claim

ExactKV V6 does **not** measure, report, or claim:

- Speedup, throughput, tokens per second, or latency
- Wall-clock runtime (`runtime_seconds`)
- Production-readiness or production serving performance
- vLLM, LMCache, or PagedAttention integration
- kvpress's external speed or serving benchmark numbers as ExactKV results

V6 documents **correctness** (`exactkv_output_ids == full_output_ids`),
**acceptance behaviour**, and **conservative workspace-memory accounting** only.

---

## 8. What remains deferred

| Item | Target |
|---|---|
| Broader kvpress presses (DecodingPress, AdaKVPress, ComposedPress, pipelines) | Future scoped work |
| Default registry integration for `kvpress_knorm_restricted` | Not planned for V6 |
| KIVI adapter | V6 alternate candidate; deferred |
| KVQuant-style adapter (pre-RoPE key quant) | **V7** |
| TurboQuant-style adapter (rotation + quant) | **V7** |
| vLLM / LMCache serving context | **V8** (evaluation context only) |
| Active GPU memory profiling | **V8** at earliest |
| CUDA/Triton kernels, CPU offload, batching, sampling | Out of scope |
| Parallel verification, bonus-token acceptance | Out of scope |

---

## 9. Known limitations

1. **kvpress requires an isolated optional environment** — not part of the default
   install; transformers version fork (`5.8.x` default vs `5.2.x` kvpress).
2. **`import kvpress` globally patches attention functions** — run kvpress adapter
   code only in `.venv-kvpress` or processes that accept this side effect.
3. **`isolate_compression_model=True` is required** — kvpress does not fully restore
   `rotary_emb` assignments on context exit; compression must use a model copy.
4. **KnormPress only** — no decoding-time presses, no composed pipelines.
5. **Not in default registry** — `kvpress_knorm_restricted` is experimental; use
   `create_kvpress_knorm_adapter()` explicitly.
6. **CPU-only evaluation** — Experiment 005 ran on CPU; GPU behaviour not characterised.
7. **Single model** — `Qwen/Qwen2.5-0.5B` only in Experiment 005.
8. **`total_kv_footprint_bytes` is an accounting sum**, not measured peak GPU memory.
   Active GPU memory is not reported.

---

## 10. Upgrade notes / usage notes

### Default install (unchanged)

```bash
pip install -e .
pytest  # kvpress tests skip when kvpress is not installed
```

Default imports remain kvpress-free. `backend_passthrough` is available via
`get_compressor("backend_passthrough")`.

### kvpress optional environment

```bash
python -m venv .venv-kvpress
.venv-kvpress/bin/pip install -e ".[kvpress]"
# Python 3.13 only:
.venv-kvpress/bin/pip install "fire>=0.7.1"

# Validation
.venv-kvpress/bin/pytest tests/test_kvpress_knorm_validation.py -q

# Experiment 005 (generates gitignored reports)
.venv-kvpress/bin/python scripts/run_experiment_005_kvpress_knorm.py
```

### Constructing the restricted kvpress adapter (not in registry)

```python
from exactkv.compressors.kvpress_knorm import create_kvpress_knorm_adapter

compressor = create_kvpress_knorm_adapter(compression_ratio=0.5)
```

Do **not** expect `get_compressor("kvpress_knorm_restricted")` in the default registry.

### Report compatibility

V1–V5 reports remain valid. V6 adds optional `backend_name` / `backend_version` on
`CompressorCapabilities`. Experiment 005 JSON includes additive `kvpress_gates` per
kvpress cell only.

---

## 11. Next version: V7

**V7 — attention-aware and V-specific experiments.** With V6 proving a real backend
can pass the exactness gate behind `BackendAdapter`, V7 evaluates:

- KVQuant-style (pre-RoPE key) and TurboQuant-style (rotation + quant) adapters
- Sparse V dequantization and layer-aware V precision policies
- Real vs simulated asymmetric compressor comparisons
- Attention-aware divergence analysis

Still **no performance claims**. See [`docs/FUTURE_ROADMAP_V6_V8.md`](FUTURE_ROADMAP_V6_V8.md).

**V8** remains serving-stack evaluation context and active GPU profiling (earliest).

---

## Changelog summary

| Phase | What was done |
|---|---|
| V6 Phase 0 | Scope statement (`docs/V6_SCOPE_STATEMENT.md`) |
| V6 Phase A | `BackendAdapter` interface design (`docs/BACKEND_ADAPTER_INTERFACE.md`) |
| V6 Phase B | `PassThroughBackendAdapter` (`backend_passthrough`) + tests |
| V6 Phase C | `[kvpress]` extra; `KVPressKnormAdapter` (KnormPress only); validation report |
| V6 Phase D | Experiment 005 (272 cells); `docs/EXPERIMENT_005_KVPRESS_KNORM.md` |
| V6 Phase E | This release; README/ROADMAP updates; audit |

---

## Attribution

**VeriCache** (draft-then-verify algorithm):

> Yao et al., *VeriCache: Turning Lossy KV Cache into Lossless LLM Inference*, arXiv:2605.17613, 2026.

**kvpress** (external library; KnormPress wrapper only):

> NVIDIA, [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress). ExactKV does not implement kvpress.

---

## Test count

Default env: targeted scaffold + backend adapter tests pass (kvpress tests skipped when optional dep absent).

Kvpress env: scaffold + backend adapter + kvpress validation tests pass (65 tests in combined gate run at Phase D/E).
