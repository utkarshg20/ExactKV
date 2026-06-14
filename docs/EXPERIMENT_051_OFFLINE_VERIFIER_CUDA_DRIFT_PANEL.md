# Experiment 051: Offline Verifier CUDA Drift Panel (Phase 12F)

**Status:** CUDA float16/bfloat16 exactness panel for reloaded full-KV verifier drift stress — **not** default runtime.

> This is a **CUDA exactness panel for the offline restored-verifier path**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> Lossy draft source uses **existing compressor logic**.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md`](EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md) · `exactkv/cache/offline_verifier.py`

---

## 1. Purpose

Phase 12E validated the offline restored-verifier drift panel on CPU float32. Phase 12F reruns the **same reloaded full-KV verifier path** on CUDA float16 and bfloat16 (when supported) to test whether exact full-greedy output is preserved under GPU execution with real lossy draft divergence and correction.

---

## 2. Setup

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | `cuda` (skip cleanly if unavailable) |
| Prompts | 6-prompt reduced drift panel (optional `--full-panel` for 12) |
| `max_new_tokens` | 32 |
| `draft_len` values | 4, 8 |
| Compressors | `int4_sim`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`, `int8` (baseline) |
| Backends | `InMemoryKVStorageBackend`, `FileKVStorageBackend` |
| Decoding | Deterministic greedy; model eval mode; no sampling |

Reproduce:

```bash
python3 scripts/research/run_exp051_offline_verifier_cuda_drift_panel.py
```

CUDA smoke (one cell):

```bash
EXACTKV_RUN_CUDA_SMOKE=1 pytest tests/test_exp051_offline_verifier_cuda_drift_panel.py -q
```

Report (gitignored): `reports/experiment_051_offline_verifier_cuda_drift_panel.json`

---

## 3. CUDA/dtype configs

| dtype | When tested |
|---|---|
| `float16` | CUDA available |
| `bfloat16` | CUDA available **and** `torch.cuda.is_bf16_supported()` |

`configure_cuda_determinism()` sets `cudnn.deterministic=True` and `cudnn.benchmark=False` before runs. If CUDA attention kernels produce nondeterministic exactness failures, they are recorded in `exactness_blockers` — exactness is **not** silently relaxed.

---

## 4. Drift stress settings

Same drift pressure as Phase 12E:

- **`draft_len` 4 and 8**
- **`max_new_tokens` 32** (16–32 range)
- Drift-targeted prompts (pharmacy semantic, LongBench-style, retrieval copy, tool-call JSON, etc.)
- Four built-in lossy compressors including `k8_v4_boundary4_v8_sim`

Default panel uses six prompts: `drift_001`, `drift_002`, `drift_003`, `drift_005`, `drift_006`, `drift_011`.

---

## 5. Storage backend

Same Phase 12A–12E capture → store → reload path:

1. Capture full prefill KV on CUDA
2. Store via `InMemoryKVStorageBackend` or `FileKVStorageBackend`
3. Reload into `FullKVState` for offline verification

File backend namespace: `exp051/{dtype}/`.

---

## 6. Lossy draft sources

Built-in compressors only — **no registry changes**:

- `int4_sim`
- `k8_v4_sim`
- `k8_v4_boundary4_v8_sim`
- `int8` (baseline)

Draft path: compress → materialize_for_draft → update_after_commit (existing logic).

---

## 7. Verifier source

**Type:** `reloaded_full_kv` — unchanged from Phase 12C–12E.

For each cell: live full greedy reference on CUDA; offline loop verifies draft tokens against **reloaded** full KV; accepts prefix / corrects mismatch.

---

## 8. Results

Fill after running the script on CUDA hardware. When CUDA is unavailable, report status is `blocked` with `cuda_available: false` and empty `cells`.

Key aggregate fields:

| Field | Meaning |
|---|---|
| `cuda_available` | Whether CUDA was present at run time |
| `dtype_supported` | Per-dtype hardware support map |
| `skipped_configs` | Dtype configs not tested (no CUDA, bf16 unsupported, load failure) |
| `draft_divergence_count` | Total verify rounds with lossy draft ≠ verifier prefix |
| `semantic_divergence_count` | Correction rounds on semantic-tagged categories |
| `mean_acceptance` | Aggregate acceptance across cells |
| `exactness_blockers` | CUDA exactness failures (output ≠ live greedy) |

---

## 9. Exactness result

Pass criterion: `exactkv_failures == 0` and final offline output matches live full greedy for every tested cell. Any CUDA nondeterminism or dtype mismatch is reported in `exactness_blockers` — **not hidden**.

---

## 10. Acceptance / correction behavior

When lossy drafts diverge on CUDA, the reloaded full-KV verifier accepts matching prefix tokens and commits a correction token at the first mismatch, then recompresses the draft state — same behavior as Phase 12E.

---

## 11. Real drift examples if any

Per-cell `round_traces[]` record rounds where `all_matched: false`, with `draft_tokens`, `verifier_tokens`, and `correction_token`. Semantic categories (pharmacy, LongBench-style, retrieval copy) increase the chance of meaningful corrections.

---

## 12. CUDA skips/blockers

| Condition | Behavior |
|---|---|
| No CUDA | `status: blocked`, `cuda_available: false`, clean skip |
| bf16 unsupported | Listed in `skipped_configs` with reason |
| Model load failure per dtype | Dtype skipped; reason in `restore_blockers` |
| Exactness failure | `exactness_blockers` with token index when known |

---

## 13. Restore/draft/verification blockers

- **Restore:** capture/storage/reload failures
- **Draft:** compressor compress/materialize/update failures
- **Verification:** sequential verification or alignment failures
- **Exactness:** offline output ≠ live full greedy without upstream blocker

Failures are **not hidden**.

---

## 14. What this proves

- Restored full-KV verifier can preserve exact full-greedy output on CUDA float16/bfloat16 when tested
- Real lossy draft divergence and correction behavior can be observed on GPU under the same isolated path as Phase 12E
- CUDA skips and exactness blockers are reported honestly

---

## 15. What this does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| Compressor ranking / leaderboard | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Throughput or speed benefit | **Not shown** |
| Active memory savings | **Not shown** |
| Production serving | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Universal CUDA determinism | **Not guaranteed** — failures are reported |

---

## 16. Relation to VeriCache parity

Phase 12F extends the dual-cache verify loop CUDA exactness check — still isolated from `ExactKVGenerator`. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

---

## 17. Next step

- See [`RESTORED_VERIFIER_RUNNER.md`](RESTORED_VERIFIER_RUNNER.md) (Phase 12G) for consolidated isolated runner API
- Optional full 12-prompt CUDA panel (`--full-panel`) if runtime acceptable
- Independent human review + locked panel runs remain required before any `full_parity_claim_allowed` upgrade
