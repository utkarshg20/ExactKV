# ExactKV Version Lineage V1–V21 (Release Synthesis — Part 2)

The **version arc** is a pre-formal-release research timeline, **distinct** from
the lettered Phase A–K release arc and from the authoritative 1500-cell benchmark
(`reports/scale_7b/raw.json`). V1–V13 are **verified** by dedicated
`V{N}_SCOPE_STATEMENT.md` files and/or git release tags; V14–V21 are **partial**
(documented via numbered Phase/experiment docs, no dedicated `V{N}` scope file).

Source: distilled from [`docs/VERSION_LINEAGE.md`](../docs/VERSION_LINEAGE.md) and
git tags. No version's evidence status was upgraded beyond what the repo supports.

| Version | Status | Theme / contribution | Evidence | Relation to current release | Caveats | Confidence |
|---------|--------|----------------------|----------|------------------------------|---------|------------|
| V1 | verified | Correctness prototype; verifier-first core | `docs/V1_SCOPE_STATEMENT.md`; tag `v0.1.0-phase1` | Origin of draft/verify/commit semantics | Historical prototype; not the 1500-cell headline | high |
| V2 | verified | Framework generalization; registry + CLI | `docs/RELEASE_NOTES_V0.2.0.md`; tag `v0.2.0` | Framework that scale panel builds on | Historical | high |
| V3 | verified | Serious benchmarking; sweeps + markdown reports | `docs/RELEASE_NOTES_V0.3.0.md`; tag `v0.3.0` | Benchmark methodology origin | Historical | high |
| V4 | verified | Asymmetric K/V compression (Exp 003) | `docs/RELEASE_NOTES_V0.4.0.md`; tag `v0.4.0` | `k8_v4_sim` compressor lineage | Historical | high |
| V5 | verified | Workspace-aware memory accounting (Exp 004) | `docs/RELEASE_NOTES_V0.5.0.md`; tag `v0.5.0` | Stored-byte accounting origin (not VRAM) | Historical; byte-ratio only | high |
| V6 | verified | Backend adapter interface (kvpress KnormPress) | `docs/RELEASE_NOTES_V0.6.0.md`; tag `v0.6.0` | Adapter pattern for external compressors | Historical | high |
| V7 | verified | Layer-aware V policies (Exp 006/006C) | `docs/RELEASE_NOTES_V0.7.0.md`; tag `v0.7.0` | Layer-aware compressor studies | Historical | high |
| V8 | verified | Serving harness (Exp 007) | `docs/RELEASE_NOTES_V0.8.0.md`; tag `v0.8.0` | Serving-context feasibility (no-go for prod claims) | Historical; not production serving | high |
| V9 | verified | Real-backend gauntlet (Exp 008–011) | `docs/RELEASE_NOTES_V0.9.0.md`; tag `v0.9.0` | TurboQuant/KIVI/KVQuant adapter probes | Historical; restricted adapters | high |
| V10 | verified | Eval-suite hardening (Exp 012–014) | `docs/RELEASE_NOTES_V0.10.0.md`; tag `v0.10.0` | Trace metrics + prompt suites | Historical | high |
| V11 | verified | Launch hardening (Exp 015–020) | `docs/RELEASE_NOTES_V0.11.0.md`; tag `v0.11.0` | Divergence autopsy, repair-policy pilot | Historical | high |
| V12 | verified | Deferred-work completion gauntlet (Exp 021–027) | `docs/V12_SCOPE_STATEMENT.md` | Performance/memory truth boundary (no-go list) | Historical | high |
| V13 | verified | Practicality proof; span verification, demos, external methods | `docs/V13_SCOPE_STATEMENT.md`; Exp 028–050 | 600-cell span grid (0 failures); SpectralQuant/Shard probes | Historical | high |
| V14 | partial | CUDA restored-verifier gate & GPU-memory diagnostics | Exp 055–058 / Phase 14 docs | Memory accounting boundary | No `V14_SCOPE_STATEMENT.md`; inferred | medium |
| V15 | partial | vLLM feasibility probes (**no-go**) | Exp 059–065 / Phase 15 docs | Establishes vLLM-integration forbidden claim | No scope file; inferred | medium |
| V16 | partial | Shadow observers; streaming-quant attention feasibility | Exp 066–085 / Phase 16 docs | Diagnostic instrumentation only | No scope file; inferred | medium |
| V17 | partial | Claim-safe demo; broader-model & long-context validation | Exp 086–089 / Phase 17 docs | Demo packaging discipline | No scope file; inferred | medium |
| V18 | partial | Integration safety; L3 guarded draft-shadow **no-commit** | Exp 090–094 / Phase 18 docs | Safety ladder L3 | No scope file; inferred; no-commit | medium |
| V19 | partial | L3 round-log proposal source | Exp 095–097 / Phase 19 docs | Proposal-source provenance | No scope file; inferred | medium |
| V20 | partial | L4 pre-gate; verifier-mediated design spec | Exp 098–101 / Phase 20 docs | Safety ladder L4 design | No scope file; inferred; dry-run | medium |
| V21 | partial | L4 scaffolds; trace-only **dry-run** | Exp 102–113 / Phase 21 docs | L4 trace schema, adversarial-injection panel | No scope file; inferred; no runtime commit | medium |

## Versions requiring manual source attachment

V14–V21 lack dedicated `V{N}_SCOPE_STATEMENT.md` files. Their version identity is
**inferred** from numbered Phase 14–21 documentation and experiment docs. They are
historical lineage only and **must not** be cited as release-benchmark evidence.
