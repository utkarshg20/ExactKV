# Restored-Verifier Runner

**Status:** Isolated experimental API (Phase 12G) — **not** default ExactKV runtime.

> This is an **isolated restored-verifier runner**, not default runtime integration.  
> Restored full KV is used only in an isolated experiment path.  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.  
> **Current ExactKV generation and verification behavior is unchanged.**

Companion: [`EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md`](EXPERIMENT_052_RESTORED_VERIFIER_RUNNER_SMOKE.md) · [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md) · `exactkv/cache/restored_verifier_runner.py`

---

## What it is

The restored-verifier runner is a **reusable isolated API** that runs the Phase 12C–12E experiment loop:

1. Capture full prefill KV
2. Store and reload full KV from a `KVStorageBackend`
3. Generate a live full-greedy reference
4. Generate lossy draft tokens via built-in compressors
5. Verify drafts using **reloaded full KV** as verifier source
6. Accept prefix / correct mismatch until output matches live greedy

It wraps existing helpers in `exactkv/cache/offline_verifier.py` and `exactkv/cache/hf_kv_restore.py` — **no second verifier implementation**.

---

## Relation to VeriCache parity

VeriCache separates full-KV verifier residency from compressed draft paths. The runner exercises that dual-cache verify pattern in an **isolated experiment path** only. Phase 11K keeps throughput/memory/serving/full-parity claims **forbidden**.

Prior evidence:

- **Phase 12E** (CPU float32 drift stress): 192/192 token exact match, 264 draft divergence rounds
- **Phase 12F** (CUDA panel): additional CUDA evidence when hardware permits; skipped cleanly when CUDA unavailable
- **Phase 12G** (runner smoke): 12/12 exact via `run_restored_verifier()`
- **Phase 12H** (runner panel): Exp 050-style drift panel via runner only — see [`EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md`](EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md)

---

## How it differs from default ExactKV runtime

| Aspect | Default runtime | Restored-verifier runner |
|---|---|---|
| Entry point | `ExactKVGenerator` | `run_restored_verifier()` |
| Verifier KV source | Live full cache | **Reloaded** stored full KV |
| Wired to generation defaults | Yes | **No** |
| Intended use | Production path (future) | Research / exactness panels |

`ExactKVGenerator` and `VerificationEngine` behavior are **unchanged**.

---

## Stored/reloaded full KV as verifier source

Verifier source is always `reloaded_full_kv`:

- Prefill KV captured via `capture_prefill_kv`
- Serialized through `KVStorageBackend` (`InMemoryKVStorageBackend` or `FileKVStorageBackend`)
- Restored into `FullKVState` for `VerificationEngine.verify_sequential`

This mirrors VeriCache’s stored verifier concept without integrating serving infrastructure.

---

## Why it is isolated

The runner is deliberately **not** wired into:

- Default generation runtime
- vLLM or LMCache integrations
- Remote prefix cache runtime
- Batching or custom CUDA/Triton kernels

It exists so researchers can run controlled “lossy draft + restored verifier” panels without changing production code paths.

---

## API overview

```python
from exactkv.cache.restored_verifier_runner import (
    RestoredVerifierRunConfig,
    default_panel_config,
    default_smoke_config,
    run_restored_verifier,
    report_to_exp053_json,
)

# Smoke (Exp 052)
config = default_smoke_config(
    prompt_ids=["offline_001", "offline_002"],
    compressor_names=["int8", "int4_sim"],
    draft_len=4,
    max_new_tokens=12,
)
report = run_restored_verifier(config)

# Drift panel (Exp 053)
from exactkv.cache.restored_verifier_runner import default_panel_prompt_ids

panel = default_panel_config(prompt_ids=default_panel_prompt_ids(full_panel=True))
report = run_restored_verifier(panel, experiment_id="exp053_restored_verifier_runner_panel")
payload = report_to_exp053_json(report)
```

Key types:

- `RestoredVerifierRunConfig` — model, device, dtype, prompts, backends, compressors, `draft_len_values`
- `RestoredVerifierCellResult` — per-cell exactness, acceptance, drift counts, blockers
- `RestoredVerifierRunReport` — aggregate metrics, `no_real_drift_observed`, exactness gate

---

## What it proves

- ExactKV can expose a **clean experimental runner** for lossy draft + stored/reloaded full-KV verifier
- Final output can match live full greedy when prior Phase 12E/12F exactness gates pass
- Blockers and exactness failures are recorded without hiding failed cells

---

## What it does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| Speedup / latency / throughput benefit | **Not shown** |
| Active GPU memory savings | **Not shown** |
| Production serving | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
| Compressor ranking | **Not shown** |

---

## No speed/memory/serving claims

Runner reports include `claim_note` and `forbidden_claims`. Diagnostic acceptance and draft-divergence counts describe verifier behavior only — they are **not** throughput or memory benchmarks.
