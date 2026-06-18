# Phase 17D: Integration Design Review

**Status:** run `scripts/research/run_exp089_integration_design_review.py`.

> This is an integration design review, not implementation.  
> ExactKV default generation remains unchanged.  
> Streaming attention is not integrated into token commit.  
> Guarded shadow remains diagnostic-only in the implemented path.  
> Before any token-commit path changes, full verification must remain the source of truth.  
> Top-k agreement is supplementary only and is not an exactness guarantee.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_089_INTEGRATION_DESIGN_REVIEW.md`](EXPERIMENT_089_INTEGRATION_DESIGN_REVIEW.md)

---

## 1. Purpose

Produce a claim-safe integration design review for what should happen after Phase 17. Answer: what is the safest next implementation path from guarded diagnostic shadow infrastructure toward real integration, and what gates must pass before any token-commit path changes are allowed?

No runtime changes, no new model experiments, no deeper integration in this phase.

---

## 2. Evidence basis

Local inventory (docs and optional reports):

| Source | Role |
|--------|------|
| `docs/PHASE_16_CLOSEOUT.md` | Phase 16 claim freeze |
| `docs/PHASE_17_CLAIM_SAFE_DEMO.md` | L0 demo packaging |
| `docs/PHASE_17B_BROADER_MODEL_VALIDATION.md` | Broader model panel |
| `docs/PHASE_17C_LONG_CONTEXT_VALIDATION.md` | Longer-context panel |
| `docs/EXPERIMENT_084_GUARDED_DECODE_TIME_SHADOW_PANEL.md` | Expanded guarded shadow |
| `docs/EXPERIMENT_088_LONG_CONTEXT_VALIDATION_PANEL.md` | Long-context validation |
| `docs/CLAIMS_AUDIT.md` | Public claims audit |
| `docs/DEFERRED_WORK_REGISTER.md` | Deferred / no-go items |
| `docs/VERICACHE_SYSTEMS_ROADMAP.md` | VeriCache gap documentation |

Missing files are marked missing; results are not invented.

---

## 3. Integration levels L0–L5

| Level | Title | Status |
|-------|-------|--------|
| **L0** | Claim-safe demo packaging | Implemented (17A) |
| **L1** | External post-hoc shadow observer | Implemented (16K–16O) |
| **L2** | Live + guarded decode-time diagnostic shadow | Implemented (16P–17C) |
| **L3** | Guarded draft shadow (no commit) | Not implemented |
| **L4** | Verifier-mediated compressed draft | Not implemented |
| **L5** | CUDA/Triton/vLLM/LMCache/serving | Deferred |

Each level records evidence, implementation risk, claim risk, required gates, allowed claims, and forbidden claims in the Exp 089 report.

---

## 4. Current implemented level

**L2 — live diagnostic observer** (`L2_live_diagnostic_observer`)

Includes `GuardedDecodeTimeShadowObserver`, live round observer, Phase 17B/17C validation panels, and Exp 084 expanded panel. Shadow is diagnostic-only; safety gates enforce no token-commit influence.

---

## 5. Future levels

- **L3:** compressed-attention draft diagnostics during generation; never commit from shadow
- **L4:** compressed draft proposes tokens; full verifier remains source of truth
- **L5:** real backend integration (CUDA/Triton/vLLM/LMCache/serving) — deferred / no-go where documented

---

## 6. Gate policy before token-commit changes

Before any L4 token-commit path research:

1. Baseline-vs-integrated token parity on fixed greedy settings
2. `exactkv_failures = 0`
3. Shadow output cannot bypass full verification
4. Full verifier remains source of truth
5. Fallback path restores existing generation behavior
6. Deterministic test harness
7. Claim audit pass
8. No performance/memory claims without measurement
9. No production/serving claims without backend validation
10. No broad model claim from small panels

---

## 7. Risk register

Ten risks tracked (severity, mitigation, current status):

- Shadow accidentally influences token commit
- Verifier bypass
- Hidden default runtime change
- Callbacks mutating generation state
- Top-k agreement misrepresented as exactness
- Small-panel results overstated
- Performance claims without measurement
- Memory claims without active measurement
- VeriCache reproduction overclaim
- Production-serving overclaim

Full detail in `reports/experiment_089_integration_design_review.json`.

---

## 8. Allowed claims

Matches Phase 16 claim freeze (`ALLOWED_CLAIMS` in `phase16_closeout.py`):

- Offline streaming-attention diagnostics
- Qwen2.5 offline attention shadow probes
- Diagnostic tolerance policy
- External generation-shadow observer
- Opt-in live round observer
- Guarded decode-time shadow dry-run
- In tested Phase 16 panels: guarded shadow did not change generated tokens
- In tested Phase 16 panels: ExactKV failures remained zero

---

## 9. Forbidden claims

Matches Phase 16 claim freeze (`FORBIDDEN_CLAIMS`):

- Speed, throughput, latency improvement
- Active GPU or production memory savings
- VeriCache serving or throughput reproduction
- Streaming attention integrated into token commit
- Shadow logits or top-k as exactness guarantees
- Production-ready system

---

## 10. Deferred work

From deferred work register and integration review:

- L3 guarded draft shadow (no commit)
- L4 verifier-mediated compressed draft
- L5 CUDA/Triton/vLLM/LMCache/serving
- Direct vLLM / LMCache integration (no-go)
- Measured active GPU memory savings
- Production serving
- Real compressed-attention token commit path

---

## 11. Recommended next phase

**Phase 18A (complete):** integration safety spec — see [`PHASE_18A_INTEGRATION_SAFETY_SPEC.md`](PHASE_18A_INTEGRATION_SAFETY_SPEC.md).

**Phase 18B (proposed):** guarded draft shadow no-commit spec or scaffold — explicit approval required; claim boundaries unchanged.

---

## 12. What not to do next

- Do not implement L4 token-commit paths without passing all gates
- Do not add observer hooks or change `ExactKVGenerator` default behavior
- Do not integrate streaming attention into token commit
- Do not start CUDA/Triton/vLLM/LMCache work ahead of the safety spec
- Do not claim speed, memory, serving, or VeriCache reproduction from diagnostic panels

---

## Run

```bash
python3 scripts/research/run_exp089_integration_design_review.py
```

Report: `reports/experiment_089_integration_design_review.json` (gitignored).
