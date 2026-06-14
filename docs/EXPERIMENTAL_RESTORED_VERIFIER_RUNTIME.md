# Experimental Restored-Verifier Runtime

**Status:** Non-default explicit opt-in API (Phase 13A) — **not** default ExactKV runtime.

> This is a **non-default experimental restored-verifier runtime path**.  
> Restored full KV is used **only when explicitly enabled**.  
> **Default ExactKV generation behavior is unchanged.**  
> **vLLM and LMCache are not integrated.**  
> **Remote prefix caching is not implemented.**  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`EXPERIMENT_054_EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md`](EXPERIMENT_054_EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md) · `exactkv/runtime/experimental.py`

---

## What it is

`run_experimental_restored_verifier()` is an **explicit opt-in** runtime wrapper that invokes the Phase 12G restored-verifier runner when `ExperimentalRestoredVerifierConfig.enabled=True` and `mode=RESTORED_VERIFIER_OFFLINE`.

It does **not** modify `ExactKVGenerator`, `VerificationEngine`, default CLI output, or compressor registry globals.

---

## How it differs from default ExactKV runtime

| Aspect | Default runtime | Experimental runtime |
|---|---|---|
| Entry point | `ExactKVGenerator.generate()` | `run_experimental_restored_verifier(config)` |
| Activation | Default path | **`enabled=True` required** |
| Verifier KV | Live full cache during generation | Reloaded stored full KV (offline runner) |
| Env-var activation | N/A | **Forbidden** — no implicit enable |
| CLI impact | Unchanged | Unchanged |

---

## How it calls the restored-verifier runner

When explicitly enabled and validated:

1. `validate_experimental_config(config)`
2. Map to `RestoredVerifierRunConfig`
3. Call `run_restored_verifier()` from `exactkv/cache/restored_verifier_runner.py`
4. Return `ExperimentalRuntimeResult` with exactness failures and blockers preserved

When `enabled=False`: returns `status="disabled"` without calling the runner.

---

## Explicit opt-in only

- `ExperimentalRuntimeMode.DEFAULT` — no restored-verifier path
- `ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE` — requires `enabled=True` plus full explicit config (model, prompts, compressors, storage backends, draft lengths, verifier source, claim note)
- **No** `os.environ` or hidden flags
- **No** global registry mutation

---

## Why default generation is unchanged

`ExactKVGenerator` and `VerificationEngine` are not imported or patched by the experimental module. The experimental path is a **parallel entry point** for research scripts only.

---

## Why it is not vLLM / LMCache / remote prefix / production serving

The experimental wrapper reuses the **offline isolated runner** only. It is **not production serving** and does not integrate serving infrastructure, remote prefix caches, or external inference engines.

---

## What evidence would be needed next

Before any default-runtime integration claim:

1. Locked human-reviewed panel runs with `exactkv_failures == 0`
2. Phase 11K parity gate review
3. Explicit product decision to wire stored verifier into `ExactKVGenerator` (not approved in Phase 13A)
4. Independent serving/memory/throughput evidence (currently **forbidden** claims)

---

## API overview

```python
from exactkv.runtime.experimental import (
    ExperimentalRuntimeMode,
    ExperimentalRestoredVerifierConfig,
    default_experimental_smoke_config,
    run_experimental_restored_verifier,
)

# Disabled — runner not called
result = run_experimental_restored_verifier(
    ExperimentalRestoredVerifierConfig.disabled()
)

# Enabled — explicit opt-in
config = default_experimental_smoke_config()
result = run_experimental_restored_verifier(config)
```

---

## What it proves

- Users can **explicitly opt in** to the restored full-KV verifier path via a structured runtime API
- Disabled configs do not invoke the runner
- Default ExactKV generation remains unchanged

---

## What it does not prove

| Claim | Status |
|---|---|
| Default runtime integration | **Not shown** |
| Production serving | **Not shown** |
| vLLM / LMCache integration | **Not shown** |
| Speedup / throughput / memory benefit | **Not shown** |
| Full VeriCache reproduction | **Not shown** |
