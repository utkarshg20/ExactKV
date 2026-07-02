# ExactKV v0.11.0 Release Notes

**Status:** V-release complete (Phases 0–6; legacy docs: V11). **Tag:** `v0.11.0` — **research release**.
**Base:** Builds on `v0.10.0` (evaluation-suite hardening, Experiments 012–014).

> **V-release** is launch hardening, not a performance or production release (legacy docs: V11).
> V10/V11 suites are **not universal benchmarks**. `_sim` compressors are **not** real packed-bit backends.
> Restricted real backends remain **factory-only**. `total_kv_footprint_bytes` is accounting, not measured peak GPU memory.
> Active GPU memory is **not** a standard schema metric. Repair policies are **pilot-only**, not core defaults.
> ExactKV does **not** claim speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, production readiness, or model accuracy improvement.

---

## 1. V11 summary

V11 closes scale, serving-context, profiling, forensics, and launch-documentation gaps
before a defensible public v1.0.0 — without changing the exactness gate, generation logic,
verification logic, or standard report schemas.

Deliverables:

- **Phases 1–2:** Experiments **015–016** — 1.5B and 3B on full V10 suites.
- **Phase 3:** Experiment **017** — serving sidecar probe; vLLM/LMCache no-go refresh.
- **Phase 4:** Experiment **018** — GPU memory methodology pilot.
- **Phase 5:** Experiment **019** — divergence autopsy.
- **Phase 5b:** Experiment **020** — repair-policy pilot.
- **Phase 6:** Launch readiness package (this release + status + artifact policy + narrative draft).

**Hard gate:** `exactkv_failures == 0` on Experiments 015–020.

---

## 2. Experiments 015–020

| Exp | Focus | Cells | Headline |
|---|---|---:|---|
| 015 | 1.5B V10 suites | 896 | `int8` **0.978**; V10 findings transfer |
| 016 | 3B V10 suites | 896 | `int8` **0.991**; exactness at 3B |
| 017 | Serving sidecar | 32 | Probe pass; vLLM/LMCache **no-go** |
| 018 | GPU memory pilot | 100 | `pilot_success`; schema unchanged |
| 019 | Divergence autopsy | 400 | Logit/KV-layer forensics; attention deferred |
| 020 | Repair-policy pilot | 300 | `fallback_int8` **0.979**; policies not in core |

---

## 3. What changed since v0.10.0

| Area | Change |
|---|---|
| Multi-model validation | 1.5B + 3B on full 128-prompt V10 suites |
| Serving | Sidecar probe + documented no-go refresh |
| Profiling | GPU memory methodology (isolated pilot) |
| Forensics | Mechanistic divergence autopsy (Exp 019) |
| Policy pilot | Category-adaptive / fallback-int8 selectors (Exp 020, experiment-layer) |
| Documentation | Launch readiness, artifact policy, narrative draft |
| Core runtime | **Unchanged** (generation, verification, default registry, standard schema) |

New analysis modules (pilot-only, not core behavior): `gpu_memory_pilot`, `divergence_autopsy`, `repair_policy`, `sidecar_probe`.

---

## 4. What V11 proves

- Exactness gate holds through **3B** on expanded V10 suites.
- Serving sidecar observability is feasible; direct stack integration is not.
- GPU memory can be piloted with caveats; V5 accounting remains primary.
- Divergence is localizable; autopsy-guided policies can improve acceptance in a pilot.

---

## 5. What V11 does not prove

- Speed, throughput, latency, or runtime improvement.
- Active GPU memory savings.
- Production serving readiness.
- Model accuracy improvement.
- Universal benchmark coverage.

---

## 6. Known limitations

- Attention logging deferred (sdpa).
- Repair policies on 25-prompt pilot only.
- No sampling, batching, or bonus tokens.
- Public launch narrative is **draft** only.

---

## 7. Launch-readiness decision

| Item | Status |
|---|---|
| V11 substance | ✅ Complete |
| Tag `v0.11.0` | ✅ Ready |
| Public `v1.0.0` launch | ❌ Not yet — see [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) |

---

## 8. Tag readiness

```bash
git tag -a v0.11.0 -m "V11 launch hardening complete (Experiments 015–020)"
```

Do **not** imply public launch when pushing this tag. v1.0.0 requires separate gate review.

---

## Related

- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md)
- [`PROJECT_STATUS_V0.11.0.md`](PROJECT_STATUS_V0.11.0.md)
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
- [`RELEASE_NOTES_V0.10.0.md`](RELEASE_NOTES_V0.10.0.md)
