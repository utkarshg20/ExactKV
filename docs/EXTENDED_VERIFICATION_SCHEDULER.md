# Extended Verification Scheduler (Phase 11E)

**Status:** Scheduler contract layer only — **generation and verification behavior unchanged.**

> This is a **scheduler contract layer**, not a verification runtime rewrite.  
> **Current generation and verification behavior is unchanged.**  
> **Bonus-token acceptance remains disabled.**  
> **vLLM and LMCache modes are placeholders only.**  
> No speedup, latency improvement, throughput improvement, memory savings, or production-serving claim is made.  
> ExactKV still does **not** reproduce VeriCache throughput results.

Companion: [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`DUAL_CACHE_ABSTRACTION.md`](DUAL_CACHE_ABSTRACTION.md) · `exactkv/verify/scheduler.py`

---

## 1. Why scheduler contracts (VeriCache Stage 4)

VeriCache discusses extended verification strategies (sequential checks, batched/span verify, optional bonus-token acceptance, serving-aware scheduling). ExactKV already implements **sequential** (default) and **span** (opt-in) in `VerificationEngine` — but there was no portable **policy metadata** describing how verification could be scheduled alongside dual-cache contracts.

Phase 11E adds that policy layer **without** changing runtime behavior.

---

## 2. Current sequential / span status

| Mode | Runtime today | Scheduler contract |
|---|---|---|
| **Sequential** | Default in `ExactKVGenerator` | `VerificationPolicyKind.SEQUENTIAL` |
| **Span** | Opt-in `verify_span` in `VerificationEngine` | `VerificationPolicyKind.SPAN` |
| **Bonus token** | **Disabled** (`bonus_token=None` in V1 traces) | `BONUS_TOKEN_DISABLED` policy documents this |
| **Serving-aware** | **Not implemented** | `SERVING_AWARE_PLACEHOLDER` only |

Exp 029 validated span ≡ sequential exactness on a 600-cell grid. This phase does **not** rerun or extend that grid.

---

## 3. What the scheduler contract adds

| Type | Purpose |
|---|---|
| `VerificationPolicy` | Kind, draft limits, span size, commit semantics, execution mode |
| `VerificationSchedulePlan` | Planned verify steps + optional `DualCacheState` summary |
| Validators | Enforce bonus disabled, exact-prefix-only commits, no active vLLM/LMCache |
| Factories | `sequential_policy`, `span_policy`, `disabled_bonus_token_policy`, `serving_aware_placeholder_policy` |

**Not wired** into `ExactKVGenerator` or `VerificationEngine`.

---

## 4. Bonus-token acceptance remains disabled

All factory policies set `bonus_token_acceptance_enabled=False`. Validation **rejects** any policy that enables bonus tokens. This matches V1/V13 commit semantics: exact prefix + correction only.

---

## 5. Serving / vLLM / LMCache placeholders only

| `VerificationExecutionMode` | Phase 11E status |
|---|---|
| `LOCAL_HF` | Only mode that may be marked conceptually active |
| `FUTURE_VLLM` | Placeholder — `runtime_integration_active` must be **False** |
| `FUTURE_LMCACHE` | Placeholder — `runtime_integration_active` must be **False** |
| `SERVING_AWARE_PLACEHOLDER` | Metadata only; requires explicit placeholder claim note |

Aligns with Exp 017 **no-go** for direct vLLM/LMCache integration.

---

## 6. What this does not prove

| Claim | Status |
|---|---|
| Speedup / latency / throughput | **Not claimed** — no timing fields in policy |
| Memory savings | **Not claimed** |
| Production serving readiness | **Not claimed** |
| VeriCache throughput reproduction | **Not claimed** |
| Parallel verification at runtime | **Not added** |

---

## 7. How Stage 5/6 build on this

| Stage | Connection |
|---|---|
| **Stage 5** — vLLM prototype | Would reference `FUTURE_VLLM` policy with integration gates |
| **Stage 6** — LMCache | Would reference `FUTURE_LMCACHE` policy with restore gates |
| **Runtime wiring** (future) | `build_schedule_plan(policy, dual_cache=...)` composes with 11B–11D contracts |

---

## 8. JSON schema (policy)

```json
{
  "policy_name": "span_default",
  "kind": "SPAN",
  "max_draft_tokens": 8,
  "span_size": 4,
  "bonus_token_acceptance_enabled": false,
  "commit_semantics": "EXACT_PREFIX_ONLY",
  "execution_mode": "LOCAL_HF",
  "requires_dual_cache": true,
  "runtime_integration_active": false,
  "claim_note": "..."
}
```

---

## 9. Claims boundary

| Allowed | Forbidden |
|---|---|
| Scheduler policy metadata exists | Bonus-token acceptance enabled |
| Documents sequential vs span scheduling intent | vLLM/LMCache marked active |
| Placeholder serving-aware policy with caveats | Speed/latency/throughput/memory/serving claims |
| Composes with `DualCacheState` summary in plan | Runtime behavior change without new phase |
