# KVPress KnormPress Adapter — Phase C Validation Report

**Date:** 2026-06-09  
**Scope:** V6 Phase C validation only (restricted `KVPressKnormAdapter`; KnormPress only).  
**Status:** Core-suite exactness gate **passed**. Ready for Experiment 005 with documented limitations.

> ExactKV evaluates this backend by acceptance behaviour and memory honesty.
> No throughput, latency, speedup, tokens/sec, runtime_seconds, or production-readiness
> claims are made in this document.

---

## 1. Environment

| Item | Value |
|---|---|
| Virtualenv | `.venv-kvpress` (`pip install -e ".[kvpress]"`) |
| Python | 3.13.3 |
| transformers | 5.2.0 |
| kvpress | 0.5.3 |
| torch | 2.12.0 |
| fire workaround | `fire>=0.7.1` installed manually (Python 3.13 `pipes` removal) |
| Default ExactKV env | transformers 5.8.x, **no kvpress** (unchanged) |

Validation commands:

```bash
# kvpress env
.venv-kvpress/bin/pytest tests/test_kvpress_scaffold.py tests/test_backend_adapter_poc.py -q
.venv-kvpress/bin/pytest tests/test_kvpress_knorm_validation.py -v

# default env (kvpress tests skipped)
pytest tests/test_backend_adapter_poc.py tests/test_kvpress_scaffold.py -q
```

---

## 2. Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| Compressor | `KVPressKnormAdapter` via `create_kvpress_knorm_adapter()` |
| Press | `KnormPress` only (`compression_ratio=0.5`) |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| Prompt suite | **`core`** (34 prompts, `benchmarks/prompts/core.jsonl`) |
| Model isolation | `isolate_compression_model=True` (default) |

---

## 3. Run summary

| Metric | Result |
|---|---|
| Prompt runs | **34** (full core suite) |
| `exactkv_failures` | **0** |
| `exactkv_output_ids == full_output_ids` | **34 / 34** |
| Aggregate acceptance rate (draft tokens) | **0.394** (305 accepted / 774 drafted) |
| Total rejected draft tokens | 469 |
| Total corrections | 184 |
| Prompts with ≥1 rejection | 34 / 34 |
| Prompts with ≥1 correction | 34 / 34 |
| Hook-safety gate | **PASS** |
| Workspace-memory gate | **PASS** |
| Forbidden performance fields | **None found** |

Validation test wall time: ~141 s for `tests/test_kvpress_knorm_validation.py` (3 tests).

---

## 4. Exactness gate

For every core prompt, `ExactKVGenerator` with `KVPressKnormAdapter` produced
**token-identical** output to `generate_full_greedy` on the same model:

```
exactkv_output_ids == full_output_ids   (34 / 34)
exactkv_failures == 0
```

This holds even though the **draft path** is lossy (compressed KV). Verification
and commit always use the authoritative full-precision KV cache; corrections
reconcile draft mismatches so final output matches full greedy.

Per-round alignment invariant held on every trace:

```
full_seq_len_after == compressed_seq_len_after   (logical_seq_len)
```

---

## 5. Lossy draft divergences (expected)

KnormPress pruning causes draft predictions to diverge from full-KV predictions.
This is **expected** and does not violate ExactKV exactness:

| Observation | Detail |
|---|---|
| Draft acceptance rate | ~39.4% aggregate across core suite |
| All prompts had rejections | 34 / 34 (lossy draft on every prompt) |
| Corrections applied | 184 total (verifier overrides bad draft prefix) |
| Final output vs full greedy | **Identical** (exactness gate) |

Example per-prompt acceptance rates: 0.27–0.60 for longer generations; shorter
prompts (e.g. `core_stress_003`) had fewer total drafted tokens.

---

## 6. Hook-safety findings

### Verification model (`runtime.model`)

| Phase | Attention `_forward_hooks` count |
|---|---|
| Before generation | 0 |
| During `verification_mode()` / `verify_sequential` | 0 |
| After generation | 0 |

Confirmed by `verification_mode()` guard and per-prompt hook checks in validation.

### Compression model (`deepcopy(runtime.model)`)

| Phase | Attention `_forward_hooks` count |
|---|---|
| Before `with press(model):` | 0 |
| During `with press(model):` | **24** (24 Qwen2.5 layers) |
| After context exit | 0 |

Hooks are scoped to compression replay only; the verification model is never
hooked because `isolate_compression_model=True` (default).

### Model mutation

Without isolation, kvpress permanently mutates `rotary_emb` references (documented
in `tests/test_kvpress_scaffold.py`). Default isolation keeps the verification
model structurally unchanged.

---

## 7. Full-state immutability during verification

Dedicated test `test_full_state_unchanged_during_verification` confirms after
`verify_sequential` inside `verification_mode()`:

- `kv_total_bytes(full_state.past_key_values)` unchanged
- `full_state.seq_len` unchanged
- Per-layer K/V tensor values unchanged (`torch.equal`)

---

## 8. Physical vs logical sequence length

For all 34 core prompts after prefill compress:

| Field | Behaviour |
|---|---|
| `logical_seq_len` | Equals full prefill `state.seq_len` (alignment invariant) |
| Physical `kv_seq_len(pruned cache)` | **Strictly less** than logical on all 34 prompts |
| `pruning_observed_count` | 34 / 34 |
| `pruning_equal_count` | 0 |

Examples (prefill):

| prompt_id | logical | physical |
|---|---|---|
| `core_nat_001` | 5 | 2 |
| `core_nat_002` | 8 | 4 |
| `core_nat_004` | 13 | 6 |

Ratio consistent with `compression_ratio=0.5` (approximately half the tokens retained).

---

## 9. Workspace-memory findings

`supports_real_bytes_claim=True` reflects **actual pruned `DynamicCache` tensor
bytes**, not packed-bit quantization:

| Field | Value / rule |
|---|---|
| `stored_kv_bytes` | `kv_total_bytes(pruned DynamicCache)` |
| `materialized_working_kv_bytes` | **Equals** `stored_kv_bytes` (token-dropping; no dequantize step) |
| `metadata_bytes` | 0 |
| `temporary_workspace_bytes` | 0 |
| `total_kv_footprint_bytes` | `stored + materialized + metadata + temporary` (exact reconcile) |

Example (`core_nat_001`): `full_bytes=122880`, `stored_kv_bytes=49152` (~40% of full).

`is_simulated=False`; this is a real token-dropping backend, not int8-container simulation.

---

## 10. No-performance-claim audit

Scanned `CompressionStats` dataclass fields for all 34 prompts. **No** occurrences of:

- `tokens_per_second`
- `throughput`
- `latency`
- `speedup`
- `runtime_seconds`

---

## 11. Limitations

1. **KnormPress only** — no DecodingPress, AdaKVPress, ComposedPress, or pipeline API.
2. **Isolated venv required** — `transformers<5.3` pin conflicts with default 5.8.x.
3. **Global attention patch** — `import kvpress` permanently wraps `ALL_ATTENTION_FUNCTIONS`.
4. **Compression model copy** — default `deepcopy` adds memory/time cost per adapter instance.
5. **Not in default registry** — manual `create_kvpress_knorm_adapter(runtime)` only.
6. **Lossy draft** — low draft acceptance rate; corrections frequent; final output still exact.
7. **CPU validation only** — GPU behaviour not evaluated in this gate.
8. **Single model** — only `Qwen/Qwen2.5-0.5B` validated here.

---

## 12. Experiment 005 readiness

| Criterion | Status |
|---|---|
| `exactkv_failures == 0` on core suite | ✅ |
| Hook isolation during verify/commit | ✅ |
| Full-state immutability during verify | ✅ |
| Honest workspace bytes (`supports_real_bytes_claim`) | ✅ |
| Not registered in default compressor registry | ✅ |
| Default env unchanged | ✅ |

**Recommendation:** `KVPressKnormAdapter` is **ready for Experiment 005** (acceptance
+ workspace memory comparison vs noop/int8/sim baselines) under the restrictions above.

Experiment 005 remains **pending** — this validation gate does not constitute
Experiment 005 itself.

---

## 13. Test artifacts

| File | Purpose |
|---|---|
| `tests/test_kvpress_knorm_validation.py` | Core-suite validation gate (kvpress env) |
| `tests/test_kvpress_scaffold.py` | Unit/scaffold safety tests |
| `tests/test_backend_adapter_poc.py` | BackendAdapter boundary regression |

Reproduce summary JSON:

```bash
.venv-kvpress/bin/python tests/test_kvpress_knorm_validation.py
```
