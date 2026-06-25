# Phase 21I: L4 Trace Schema Adversarial Injection Panel

**Status:** run `scripts/research/run_exp110_l4_trace_schema_adversarial_injection_panel.py`.

> This is adversarial trace schema stress testing, not runtime instrumentation.  
> ExactKV default generation remains unchanged.  
> ExactKVGenerator remains unchanged.  
> Production CLI remains unchanged.  
> Verifier evidence must be explicit.  
> Proposal evidence and verifier evidence must be separate fields.  
> Missing verifier evidence blocks dry-run decisions rather than fabricating one.  
> Runtime instrumentation is not authorized in this phase.  
> L4 runtime commit remains blocked.  
> Passing this panel authorizes only runtime instrumentation design, not implementation or commit integration.  
> No speed, throughput, latency, serving, active GPU memory, or production-memory claim is made.  
> ExactKV still does not reproduce VeriCache throughput or serving results.

Companion: [`EXPERIMENT_110_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md`](EXPERIMENT_110_L4_TRACE_SCHEMA_ADVERSARIAL_INJECTION_PANEL.md)

---

## 1. Purpose

Stress-test L4 verifier evidence trace schema enforcement under adversarial injection across five attack categories.

---

## 2. Relation to Phase 21H

Phase 21H validated canonical schema examples. Phase 21I injects adversarial, poisoned, and silent-failure traces to test robustness.

---

## 3. Adversarial panel boundary

- Synthetic adversarial traces only
- No ExactKVGenerator changes
- No runtime instrumentation
- Trace-only diagnostics

---

## 4. Adversarial categories

1. Missing field attacks  
2. Field forgery attacks  
3. Structural poisoning  
4. Divergence injection (valid control cases)  
5. Silent failure attempts  

---

## 5. Panel classifications

- `pass` — valid divergence handled correctly  
- `blocked_missing_verifier_evidence` — missing verifier blocks  
- `invalid_trace` — schema validation rejects  
- `detected_poisoning` — forgery/alias/poison detected  

---

## 6. Metrics

- adversarial_detection_rate  
- invalid_trace_rejection_rate  
- false_acceptance_rate (must be 0)  
- schema_robustness_score  

---

## 7. What this authorizes

**Phase 21J:** L4 verifier runtime instrumentation design — complete. See [`PHASE_21J_L4_VERIFIER_RUNTIME_INSTRUMENTATION_DESIGN.md`](PHASE_21J_L4_VERIFIER_RUNTIME_INSTRUMENTATION_DESIGN.md).

**Next:** Phase 21K Stage 3 verifier-mediated dry-run design (`phase21k_l4_stage3_verifier_mediated_dry_run_design`).

---

## 8. What this does not authorize

- Runtime verifier instrumentation implementation  
- L4 runtime commit integration  

---

## Run

```bash
python3 scripts/research/run_exp110_l4_trace_schema_adversarial_injection_panel.py
```

```bash
pytest tests/test_exp110_l4_trace_schema_adversarial_injection_panel.py -q
```

Report: `reports/experiment_110_l4_trace_schema_adversarial_injection_panel.json` (gitignored).
