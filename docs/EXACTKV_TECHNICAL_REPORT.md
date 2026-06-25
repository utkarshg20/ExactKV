# When Does Compressed KV Start Lying? Token-Level Drift in KV Cache Compression

**ExactKV Technical Report (Phase K)**

Generated from on-disk release artifacts. Values cited below are read from `reports/scale_7b/raw.json`, `reports/public_release/leaderboard_final.json`, and supporting evidence files — not invented.

---

## 1. Abstract

ExactKV is a compressor-agnostic crash-test and leaderboard framework for LLM KV-cache compression. It measures when compressed KV paths first diverge token-by-token, how much of each draft prefix remains accepted by a full-KV verifier, and whether verifier-backed execution preserves exactness on tested panels.

The public release headline is a **1500-cell** real-GPU benchmark (`reports/scale_7b/raw.json`) on **meta-llama/Llama-3.1-8B** and **mistralai/Mistral-7B-Instruct-v0.3** with **`exactkv_failures = 0`**. ExactKV is **not a production serving system** and **does not reproduce VeriCache** serving throughput.

**ExactKV did not start at Phase A.** A long pre-release research arc (**V1–V21**, Experiments 001–113+, Phases 11–21) established verifier-first semantics, trace methodology, demo failure cases, safety boundaries, and no-go claim boundaries before Phases A–J formalized, scaled, packaged, validated, and released the system.

Full historical catalog: [`HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md) (1,176 artifacts).

---

## 2. Motivation

Lossy KV-cache compression can change greedy token generation. Serving stacks often treat compressed KV as an optimization, but token-level drift is easy to under-measure. ExactKV asks a concrete question: **when does compressed KV start lying relative to full-precision greedy decoding**, and can verifier-mediated draft/verify/commit semantics preserve exactness?

---

## 3. Problem statement

Given a model, prompt, compressor, and greedy decoding configuration:

1. A **lossy compressed KV** path proposes draft tokens.
2. A **full-KV verifier** checks each draft token against the reference greedy policy.
3. ExactKV **accepts** matching prefixes and **rejects/corrects** on mismatch.
4. Metrics record **first divergence index**, **acceptance rate**, **verifier agreement**, and **exactkv failures** (final output ≠ full-KV greedy).

The framework is evaluation-first: it measures exactness compatibility, not deployment throughput.

---

## 4. What ExactKV measures

| Metric | Definition | Public status |
|--------|------------|---------------|
| `acceptance_rate` | Fraction of drafted tokens accepted by verifier | Allowed |
| `first_divergence_index` | First token index where lossy path diverges | Allowed |
| `avg_accepted_span` | Mean accepted tokens per verification round | Allowed (panel-scoped) |
| `verifier_agreement` | Agreement between verifier and full-KV reference | Allowed |
| `exactkv_failure_rate` | Cells where final ExactKV ≠ full-KV greedy | Allowed |
| `compression_ratio` | Compressed stored bytes / full bytes | **Stored tensor byte ratio only** |

See [`METRIC_DEFINITIONS.md`](METRIC_DEFINITIONS.md).

---

## 5. Project lineage: from verifier prototypes to release benchmark

**ExactKV did not start at Phase A.** Release Gate R2 archaeology catalogued **1,176** meaningful artifacts, including **1,145** pre–Phase A–J bucket entries, from repository evidence (file tree, git history, experiment docs, tests, and report metadata).

### V1–V21: the pre-release verifier-first arc

The **version arc spans V1–V21** — pre-formal-release prototype milestones **distinct** from phases A–K and from the **1500-cell** public benchmark. V1–V13 have dedicated `V{N}_SCOPE_STATEMENT.md` files; V14–V21 are documented primarily via Phase 14–21 experiment and phase artifacts (evidence status: **partial** — no dedicated V14–V21 scope statements in-repo). Full per-version table: [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md).

| Range | Role |
|-------|------|
| V1–V3 | Early foundation, verifier-first core |
| V4–V11 | Compression simulation, adapters, suite hardening (v0.4–v0.11 tags) |
| V12–V13 | Deferred-work gauntlet, practicality proof, demos |
| V14–V15 | CUDA verifier gate, GPU memory diagnostics, vLLM no-go probes |
| V16 | Shadow observers, streaming quant feasibility |
| V17 | Claim-safe demos, broader model validation |
| V18–V21 | Integration safety, L3 no-commit, L4 dry-run scaffolds |

Representative pre-A arcs (experiments/phases):

| Era | Representative work | Role |
|-----|---------------------|------|
| Early foundation | VISION, V1–V3 scope; Exp 001–003 smoke/core sweeps | Problem framing |
| Verifier core | draft/verify/commit loop; Exp 028–029 span verification | Exactness gate origin |
| Trace correctness | V10 suites (Exp 012–016); Phase G authority | First-divergence methodology |
| Compression studies | int8/int4 sim, KIVI/KVQuant/TurboQuant adapters | Compressor comparisons |
| Demos | Exp 034/034b structured output; terminal crash-test | Illustrative failure cases |
| Safety ladder | L3 draft-shadow no-commit; L4 dry-run specs (Exp 090–091) | Runtime boundary |
| Shadow/runtime probes | Exp 076–085 shadow observers | Diagnostic instrumentation |
| No-go probes | vLLM (Exp 059–065), LMCache path, Exp 027 truth boundary | Claim boundaries |
| Instability analysis | Exp 116–117 regime/phase diagrams | Supporting analysis |

Phases A–J **formalized** cross-compressor benchmarking, kernel microbenchmarks (F), truth engine (G), platform/scale (H+), novelty audit (I), and public release freeze (J). Gates R0/R1/R2 added evidence integrity, Mistral leaderboard repair, and full lineage reconstruction.

Full narrative: [`PROJECT_LINEAGE.md`](PROJECT_LINEAGE.md).

---

## 6. Verifier-first design

ExactKV's core loop:

1. **Draft** from compressed (lossy) KV.
2. **Verify** each token against full-KV greedy predictions.
3. **Commit** accepted prefix; correct on mismatch.
4. **Repeat** from updated authoritative full state.

The hard gate on published panels: **`exactkv_failures == 0`**. ExactKV does **not** claim to have invented compressed-KV verification; VeriCache is the closest conceptual prior art (see §19–20).

---

## 7. Trace-level correctness and first divergence

Trace records capture per-token events: match, first divergence, post-divergence drift. Phase G `FirstDivergenceAuthority` provides canonical divergence classification (`token_mismatch`, `length_drift`, `kernel_inconsistency`, `verifier_disagreement`, `none`).

Pre-A V10 evaluation suites and sensitivity forensics developed these metrics before the formal release panel. Phase G unified truth engine consolidates divergence authority for release artifacts (`reports/phaseG_unified_truth.json`).

---

## 8. Demo and failure-case development

Historical demos (illustrative, not throughput claims):

- **Structured-output drift** — Exp 034/034b pharmacy/JSON correction cases (`reports/demo_pack.json`)
- **Terminal crash-test** — [`EXACTKV_TERMINAL_CRASH_TEST.md`](EXACTKV_TERMINAL_CRASH_TEST.md)
- **LongBench-style drift** — Exp 037

Release demo cards: [`reports/public_release/demo_cards.md`](../reports/public_release/demo_cards.md).

These support storytelling and methodology; the **1500-cell scale_7b panel** is the public headline benchmark.

---

## 9. Safety ladder and no-commit runtime boundary

Pre-A safety work includes L0–L5 ladder artifacts, guarded draft-shadow no-commit scaffolds (Phase 18B / Exp 091), L4 verifier-mediated **dry-run** design specs, and integration safety specs (Phase 18A). **No-commit / dry-run** constraints remain in force — L4 runtime commit paths are not authorized for public claims.

---

## 10. Runtime coupling, shadow observers, and instability analysis

Shadow observer panels (Exp 076–085) and live round observers explored runtime instrumentation without authorizing production commit paths. Exp 116 instability regime analysis and Exp 117 phase diagrams inform visualization and leaderboard insights but are **not** standalone public performance claims.

---

## 11. Serving, vLLM, LMCache, memory, and timing no-go investigations

Investigations established **forbidden claim boundaries**:

- **vLLM** feasibility probes (Exp 059–065) — no production integration claim
- **LMCache** prototype path — storage/offload adjacent, not exactness headline
- **GPU memory / timing** pilots (Exp 018, 027, 030, 031, 057–058) — compression ratios are **stored tensor byte ratios** unless active GPU memory is explicitly measured

ExactKV is **not a production serving system**. These probes inform what must **not** be claimed publicly.

---

## 12. Transition from pre-A research to formal release phases

| Phase | Role |
|-------|------|
| A–C | Formal cross-model benchmark + leaderboard + visuals |
| D | Runtime probe layer |
| E–F | KV compression kernels (F = microbenchmark evidence only) |
| G | Canonical divergence truth engine |
| H/H+ | Public leaderboard platform + 7B/8B scale |
| I | Novelty audit + claim lock |
| J | Public release freeze + reproducibility |
| R0/R1/R2 | Evidence integrity, Mistral repair, lineage archaeology |
| K | Final launch pack + technical report |

Historical Phase A **336-cell** panel remains internal supporting evidence — **not** the public headline.

---

## 13. Benchmark design

- **Authoritative panel:** Phase H+ `scale_7b` — **1500 cells**
- **Device:** real GPU (`cuda`), `float16`, `deterministic_mode=false`
- **Execution:** sequential model execution (RunPod volume constraint)
- **Scoring:** weighted composite of acceptance, verifier agreement, normalized first divergence, exactness success, stability (`reports/public_release/leaderboard_final.json`)
- **Divergence authority:** Phase G `FirstDivergenceAuthority`

Reproduce reports: `python3 scripts/exactkv_repro.py --reports-only`

---

## 14. Models and compressors evaluated

### Models (public release)

| Model | Cells |
|-------|------:|
| `meta-llama/Llama-3.1-8B` | 750 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 750 |

### Compressors (scale panel)

`noop`, `int8`, `int4_sim`, `spectralquant`, `shard` — 300 cells each across both models.

### Adapter honesty

| Adapter | Status |
|---------|--------|
| SpectralQuant | **fallback/proxy** (`spectralquant_available=False`) |
| Shard | **probe-first** heuristic (`probe_only=True`) |

Do **not** claim real SpectralQuant or real Shard integration in the current environment.

---

## 15. Main release results

Source: `reports/scale_7b/scale_summary.json`, `reports/scale_7b/raw.json`

| Field | Value |
|-------|-------|
| Total cells | **1500** |
| ExactKV failures | **0** |
| Deterministic mode | `false` (real GPU) |
| Dtype | float16 |

---

## 16. Public leaderboard

Source: `reports/public_release/leaderboard_final.json`

### Llama-3.1-8B (top rows)

| Rank | Compressor | Score | Acceptance | Failure rate |
|------|------------|------:|-----------:|-------------:|
| 1 | `noop` | 1.000 | 1.000 | 0.000 |
| 2 | `int8` | 1.000 | 1.000 | 0.000 |
| 5 | `int4_sim` | 0.859 | 0.852 | 0.000 |
| 6 | `spectralquant` | 0.859 | 0.852 | 0.000 |
| — | `shard` | 0.544 | 0.632 | 0.000 |

### Mistral-7B-Instruct-v0.3 (top rows)

| Rank | Compressor | Score | Acceptance | Availability |
|------|------------|------:|-----------:|--------------|
| 3 | `noop` | 1.000 | 1.000 | available |
| 4 | `int8` | 0.983 | 1.000 | available |
| 7 | `int4_sim` | 0.851 | 0.837 | available |
| — | `spectralquant` | 0.851 | 0.837 | mock_fallback |
| — | `shard` | 0.727 | 0.623 | probe_only |

Mistral numeric rows restored by Release Gate R1 aggregate repair.

---

## 17. Unified truth engine

Phase G (`reports/phaseG_unified_truth.json`) provides canonical first-divergence authority and divergence typing across cells. It consolidates pre-A trace correctness work into release-grade divergence records without changing Phase G definitions in this launch pack.

---

## 18. Kernel microbenchmark results

Source: `reports/phaseF_kernel_benchmark.json` — **kernel microbenchmark only; not end-to-end inference speedups.**

| Mode | Torch → Triton ratio | Notes |
|------|------------------------|-------|
| int8 | **1.63×** | Tested `kv_shape=[1,8,512,64]`, CUDA |
| int4 | **1.54×** | Same scope |
| block_sparse | **0.98×** | `execution_backend=torch` — **not Triton-accelerated** |

Compression ratios in Phase F are **stored tensor byte ratios** on the tested shape.

---

## 19. Prior art and novelty positioning

See [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) and `reports/novelty_audit.json`.

ExactKV's defensible positioning (without overclaiming):

- Public compressor-agnostic **token-level drift** and **first-divergence** leaderboard
- Phase G canonical divergence authority
- Reproducible artifact pipeline (benchmark → leaderboard → public_release)
- Real 7B/8B scale panel with zero ExactKV failures (current evidence)

**Forbidden:** unqualified uniqueness claims, unqualified production-readiness claims, unqualified SOTA/fastest claims.

---

## 20. Relationship to VeriCache

**VeriCache** is the closest conceptual prior art for compressed-KV draft plus full-KV verification for lossless inference with serving optimizations. ExactKV is inspired by draft/verify semantics but is a **public exactness benchmark and leaderboard platform**, not a VeriCache reproduction.

**ExactKV does not reproduce VeriCache** serving throughput, memory behavior, or deployment stack.

---

## 21. Relationship to ShardCache / shard-kv

**ShardCache (shard-kv)** is primarily a cache database / LMCache storage benchmark system — adjacent but not equivalent to transformer KV-cache token-drift exactness benchmarking. ExactKV's `shard` compressor slot is **probe-first heuristic analysis**, not full Shard or ShardCache product integration.

---

## 22. Limitations

- **Not a production serving system**
- **Does not reproduce VeriCache**
- Phase F results are **kernel microbenchmark only** (not end-to-end speedups)
- Compression ratios = **stored tensor byte ratios** (not active GPU memory unless measured)
- SpectralQuant = **fallback/proxy** in current environment
- Shard = **probe-first** heuristic
- Sequential model execution on scale run (volume constraint)
- Panel-scoped greedy decoding — not universal worst-case bounds
- 1,176 historical artifacts require editorial curation for external narrative

---

## 23. Reproducibility

```bash
python3 scripts/exactkv_repro.py --reports-only   # regenerate public reports
python3 scripts/exactkv_repro.py --release-check  # validators + tests
python3 scripts/build_launch_pack.py              # demo cards + launch manifest
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 24. Release artifacts

| Artifact | Path |
|----------|------|
| Source of truth | `reports/scale_7b/raw.json` |
| Public leaderboard | `reports/public_release/leaderboard_final.json` |
| Technical report | `docs/EXACTKV_TECHNICAL_REPORT.md` |
| Project lineage | `docs/PROJECT_LINEAGE.md` |
| Historical inventory | `reports/historical_artifact_inventory.json` |
| Demo cards | `reports/public_release/demo_cards.json` |
| Launch manifest | `reports/public_release/launch_manifest.json` |
| Claim boundaries | `docs/CLAIM_BOUNDARIES.md` |
| Evidence status | `reports/release_evidence_status.json` |

Index: [`ARTIFACT_INDEX.md`](ARTIFACT_INDEX.md) · Checklist: [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)

---

## 25. Conclusion

ExactKV evolved through a long verifier-first research arc before formal release phases packaged a **1500-cell** real-GPU benchmark with **`exactkv_failures = 0`** on Llama-3.1-8B and Mistral-7B-Instruct-v0.3. The framework measures **when compressed KV starts lying** at token granularity and ranks compressors on exactness-compatible metrics — with explicit claim boundaries around serving, VeriCache, kernel microbenchmarks, adapter honesty, and stored-byte compression ratios.

ExactKV is a research-grade evaluation platform. It is **not** a production serving system and **does not reproduce VeriCache**.
