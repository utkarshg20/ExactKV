# ExactKV Project Lineage

> **ExactKV did not start at Phase A.** The formal A–J release pipeline formalized, scaled, packaged, and validated a much larger pre-release research arc spanning experiments, demos, safety ladders, runtime probes, and claim-boundary work.

Generated: 2026-06-26T14:58:00.590242+00:00

## 1. Executive summary

ExactKV evolved over a long research arc before the formal Phase A–J release pipeline. The repository contains **1214** catalogued meaningful artifacts, including approximately **1177** pre–Phase A–J bucket entries, **121** experiment documents, and **31** phase documents discovered from the file tree — not from memory.

Phase A–J **formalized** cross-compressor benchmarking, kernel microbenchmarks, truth-engine divergence authority, public leaderboard packaging, novelty audit, and release gates. It did **not** originate verifier-mediated exactness, trace-level drift measurement, or demo-driven failure-case development.

The **version arc spans V1–V21** — pre-formal-release prototype milestones distinct from phases A–K. V1–V13 have dedicated scope statements; V14–V21 are documented primarily via Phase 14–21 experiment/phase artifacts (see [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md)).

## 2. Methodology

This lineage was reconstructed by:

1. **File tree scan** — `git ls-files` across `exactkv/`, `scripts/`, `tests/`, `docs/`, `reports/`
2. **Git history** — tags (`v0.1.0`–`v0.11.0`), commit messages referencing exp/phase/L3/L4/shadow/runtime
3. **Report metadata** — `phase_id`, `total_cells`, `exactkv_failures` from JSON reports where present
4. **Document titles** — experiment index, V-series scope statements, phase closeouts, claim audits

Tooling: `exactkv/platform/project_archaeology.py` · `scripts/build_project_lineage.py`

## 3. Full chronological timeline (discovered buckets)

| Era (bucket) | Artifacts found |
|--------------|----------------:|
| `early_foundation` | 11 |
| `verifier_core` | 163 |
| `trace_correctness` | 19 |
| `compression_simulation` | 12 |
| `structured_output_demos` | 331 |
| `v_series_demos` | 15 |
| `benchmark_prototypes` | 23 |
| `safety_ladder` | 42 |
| `shadow_observer_runtime` | 64 |
| `l3_draft_shadow` | 13 |
| `l4_verifier_mediated_dry_run` | 29 |
| `instability_analysis` | 13 |
| `visualization_layer` | 11 |
| `no_go_serving_probe` | 47 |
| `memory_timing_claim_boundary` | 28 |
| `external_compressor_investigation` | 103 |
| `formal_phase_A_to_C` | 9 |
| `runtime_probe_phase_D` | 5 |
| `kernel_phase_E_F` | 4 |
| `truth_engine_phase_G` | 5 |
| `evidence_gate_R0_R1` | 4 |
| `novelty_release_phase_I_J` | 5 |
| `launch_phase_K_preparation` | 5 |
| `unknown` | 253 |

## 4. Early problem framing

[`docs/VISION.md`](VISION.md) and V1–V3 scope statements frame the original problem: **lossy KV-cache compression can change greedy token generation**. ExactKV asked whether compressed caches remain compatible with full-precision greedy decoding through draft/verify/commit semantics.

## 4b. Version arc V1–V21

| Range | Role |
|-------|------|
| V1–V3 | Early foundation: correctness prototype, framework, sweeps |
| V4–V9 | Compression simulation, adapters, serving harness, backend gauntlet |
| V10–V-release | Evaluation suite hardening, launch hardening (public tag `v-release`) |
| V12–V13 | Deferred-work gauntlet, practicality proof, demos, external methods |
| V14–V15 | CUDA restored verifier, GPU memory diagnostics, vLLM no-go probes |
| V16 | Shadow observers, streaming quant feasibility, Phase 16 closeout |
| V17 | Claim-safe demo packaging, broader model / long-context validation |
| V18–V19 | Integration safety, L3 guarded draft-shadow no-commit, round-log sources |
| V20–V21 | L4 pre-gate, verifier-mediated dry-run scaffolds (no runtime commit) |

Full per-version evidence: [`VERSION_LINEAGE.md`](VERSION_LINEAGE.md) · `reports/version_lineage.json`

## 5. Verifier-first core

Early foundation and verifier-core artifacts include `exactkv/` generator/verifier modules, Experiments 001–002 smoke/core sweeps, and span verification work (Exp 028–029). The invariant **`exactkv_failures == 0`** on tested panels became the hard gate.

## 6. Trace-level correctness work

Trace correctness spans acceptance rate, first divergence index, verifier agreement, and ExactKV failure tracking — developed across V10 suites (Exp 012–016), sensitivity forensics (Exp 013), and Phase G `FirstDivergenceAuthority`.

## 7. Compression simulation and compressor studies

Discovered work includes int8/int4 simulators, asymmetric K/V (Exp 003), layer-aware V (Exp 006), TurboQuant/KIVI/KVQuant adapters (Exp 008–010, 023–024), SpectralQuant probes (Exp 042–045), and Shard bounded probes (Exp 038–041). Current release uses **fallback/proxy** for SpectralQuant and **probe-first** for Shard where real dependencies are absent.

## 8. Demo and failure-case development

Structured-output and adversarial demos include Exp 034/034b pharmacy correction, terminal crash-test demos (`EXACTKV_TERMINAL_CRASH_TEST.md`), LongBench-style drift demo (Exp 037), and V-series visual packages (Exp 035–036). These are **illustrative exactness evidence**, not throughput claims.

## 9. Safety ladder and runtime boundary work

Discovered L0–L5 / Phase 16–21 artifacts include guarded draft-shadow no-commit scaffolds (Phase 18B/Exp 091), L4 verifier-mediated **dry-run** design specs, integration safety specs (Phase 18A), and pre-L4 safety gate reviews. **No-commit / dry-run** constraints remain in force.

## 10. Runtime coupling and live-probe work

Shadow observer panels (Exp 076–085), live round observers (Exp 081–082), and decode-time shadow smoke tests (Exp 083–084) explored runtime instrumentation without authorizing production commit paths.

## 11. Instability analysis and visualization

Exp 116 instability regime analysis and Exp 117 phase diagrams connect to visualization layers (Exp 035, public visual package). These inform leaderboard insights but are not standalone public performance claims.

## 12. Serving, vLLM, LMCache, memory, and speed investigations

Experiments 017, 059–065 (vLLM feasibility), LMCache prototype path docs (Phase 11G), GPU memory pilots (Exp 018, 031, 057–058), and Exp 027/030 performance-memory truth boundaries established **no-go** and **forbidden claim** lists for serving throughput, unqualified end-to-end speedups, and unqualified GPU memory savings claims.

## 13. Transition into formal Phase A–J release pipeline

| Phase | Role |
|-------|------|
| A–C | Formal cross-model benchmark + leaderboard + visuals |
| D | Runtime probe layer |
| E–F | KV compression kernels (Phase F = kernel microbenchmark only) |
| G | Canonical divergence truth engine |
| H/H+ | Public leaderboard platform + 7B/8B scale |
| I | Novelty audit + claim lock |
| J | Public release freeze + reproducibility |
| R0/R1 | Evidence integrity + Mistral leaderboard repair |

## 14. Current release evidence (authoritative)

- **1500-cell** Phase H+ `reports/scale_7b/raw.json` (real GPU, float16, `deterministic_mode=false`)
- Models: `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-Instruct-v0.3`
- **`exactkv_failures = 0`**
- Phase F kernel microbenchmark (int8 ~1.63×, int4 ~1.54×) — **not end-to-end**
- Phase G truth engine; public bundle under `reports/public_release/`

## 15. What earlier work still supports

- Verifier/draft semantics and trace metrics methodology
- Demo narratives (terminal crash-test, structured-output drift)
- Claim boundaries (`CLAIMS_AUDIT.md`, Exp 027, VeriCache parity gate)
- Compressor adapter honesty disclosures
- Test infrastructure validating exactness gates

## 16. What earlier work was superseded

- Phase A 336-cell panel → **scale_7b 1500-cell** public headline
- Legacy live correction demo → terminal crash-test demo
- Stale per_model_tables without Mistral → R1 aggregate repair

## 17. What earlier work is intentionally not claimed

- vLLM/LMCache integration (probes only)
- Production serving / throughput / latency wins
- Active GPU memory savings (forbidden as public claim unless explicitly measured)
- Real SpectralQuant / full Shard product integration in current environment
- L4 runtime commit paths (dry-run / scaffold only)

## 18. How to read the release

| Tier | Artifacts |
|------|-----------|
| **Authoritative** | `reports/scale_7b/raw.json`, `reports/public_release/*`, `reports/release_evidence_status.json` |
| **Supporting historical** | Experiment docs, V-series suites, demos, safety ladder specs |
| **Exploratory / no-go** | vLLM probes, serving sidecars, timing harnesses — claim-boundary evidence only |

See also: [`ARTIFACT_INDEX.md`](ARTIFACT_INDEX.md) · [`HISTORICAL_ARTIFACT_INVENTORY.md`](HISTORICAL_ARTIFACT_INVENTORY.md)
