# VeriCache Parity RC Claim Gate (Phase 11K)

**Status:** Claim classification gate only — **not a parity certification.**

> ExactKV currently reproduces **VeriCache-style algorithmic semantics**, not the full VeriCache serving system.  
> ExactKV does **not** currently reproduce VeriCache throughput results.  
> ExactKV does **not** currently reproduce VeriCache memory benefits.  
> ExactKV does **not** currently implement vLLM or LMCache integration.  
> ExactKV does **not** currently implement production serving.  
> No speedup, latency improvement, throughput improvement, active memory savings, or production-serving claim is made.  
> **Full VeriCache reproduction remains forbidden** until all gates pass and are reviewed.

Companion: [`VERICACHE_PARITY_AUDIT.md`](VERICACHE_PARITY_AUDIT.md) · [`VERICACHE_SYSTEMS_ROADMAP.md`](VERICACHE_SYSTEMS_ROADMAP.md) · [`PAPER_LIKE_REPRODUCTION_PANEL.md`](PAPER_LIKE_REPRODUCTION_PANEL.md) · `exactkv/claims/vericache_parity_gate.py`

---

## 1. Why the claim gate exists

Stages 11B–11J built **contract layers** for VeriCache-equivalent systems work. Stage 10 (this phase) adds a **conservative claim gate** that classifies what may be said today vs what requires future evidence and human review — protecting the repo from overclaiming.

**Contract completion is not evidence of runtime parity.**

---

## 2. What ExactKV can claim today

| Category | Status | Example allowed wording |
|---|---|---|
| **Algorithmic semantics** | `ALLOWED_WITH_SCOPE` | VeriCache-style draft-then-verify on HF harness |
| **Correctness on tested panels** | `ALLOWED_WITH_SCOPE` | `exactkv_failures == 0` on **named** panel (Exp 012, 029, …) |
| **Crash-test harness** | (framing) | Correctness-first KV compression crash-test lab |
| **Benchmark gap** | (framing) | Outcome benchmarks vs ExactKV path equivalence — complementary |

---

## 3. What ExactKV cannot claim today

| Category | Status |
|---|---|
| **Full VeriCache reproduction** | `FORBIDDEN` |
| **Systems parity** | `FORBIDDEN` |
| **Throughput benefit** | `FORBIDDEN` |
| **Memory benefit** | `FORBIDDEN` |
| **Production serving** | `FORBIDDEN` |
| **Remote prefix cache runtime** | `FORBIDDEN` |
| **Paper-like reproduction** | `BLOCKED_PENDING_EVIDENCE` |
| **vLLM / LMCache integration** | `CONTRACT_ONLY` (metadata exists — not integrated) |

---

## 4. Evidence required to unlock each category

| Category | Required evidence |
|---|---|
| **Throughput benefit** | Phase 11I `throughput_claim_allowed`; reproducible baseline panel; exactness gate |
| **Memory benefit** | Active memory measurement; `memory_claim_allowed`; not diagnostic-only |
| **Production serving** | Multi-request serving tests; `serving_claim_allowed` |
| **Paper-like reproduction** | Phase 11J `paper_panel_claim_eligible`; locked panel run |
| **vLLM integration** | Prototype runtime beyond Phase 11F contract; Exp 017 blockers cleared |
| **LMCache integration** | Prototype runtime beyond Phase 11G contract |
| **Remote prefix cache** | Network/runtime tier beyond Phase 11H loopback |
| **Full VeriCache reproduction** | **All** gates above + independent human review |

---

## 5. Why contract completion ≠ runtime parity

Phase 11F–11J documents **what must be true before prototype work**. Validators on those contracts default to `CONTRACT_ONLY`, `DIAGNOSTIC_ONLY`, or `claim_eligible=False`. The claim gate reads those defaults and keeps integration and benefit claims **forbidden**.

---

## 6. Why paper numbers are not ExactKV results

Yao et al. (*VeriCache*, arXiv:2605.17613) report their system on their panel. ExactKV must not cite those figures as ExactKV results. The paper panel contract (11J) keeps `paper_numbers_as_exactkv_results=False`.

---

## 7. Human review before parity RC

Any upgrade toward `REQUIRES_HUMAN_REVIEW` or `full_parity_claim_allowed=True` requires:

1. Locked panel evidence
2. Gate flags true on contracts
3. Claims audit pass
4. Independent reviewer sign-off

Default gate: `full_parity_claim_allowed=False`.

---

## 8. Gate flags (default)

| Flag | Default |
|---|---|
| `audit_passed` | `True` (Phase 11A audit doc exists) |
| `paper_panel_claim_eligible` | `False` |
| `throughput_claim_allowed` | `False` |
| `memory_claim_allowed` | `False` |
| `serving_claim_allowed` | `False` |
| `full_parity_claim_allowed` | `False` |

---

## 9. JSON schema (gate header)

```json
{
  "audit_passed": true,
  "paper_panel_claim_eligible": false,
  "throughput_claim_allowed": false,
  "memory_claim_allowed": false,
  "serving_claim_allowed": false,
  "full_parity_claim_allowed": false,
  "claim_note": "..."
}
```

---

## 10. Claims boundary

| Allowed | Forbidden |
|---|---|
| Algorithmic semantics with scope | Full VeriCache reproduction |
| Panel-bound exactness with citation | Systems parity |
| Contract metadata exists (vLLM/LMCache/panel) | vLLM/LMCache integrated |
| Diagnostic timing/memory with caveats | Throughput or memory **benefit** |
| Human-review process documented | Production-scale deployment readiness |
| | External paper numbers as ExactKV results |
