# ExactKV evaluator guide

Short path for external reviewers who cannot run the full 7B/8B GPU panels.

## 1. What is ExactKV?

A **compressor-agnostic crash-test harness** for KV-cache compression under greedy decoding.
It measures **first-divergence index**, **acceptance rate**, **verifier agreement**, and
**exactness failures** per cell. It is **not** a production serving system and does **not**
claim end-to-end speedup or active GPU memory savings.

## 2. What claim does it prove?

That lossy KV compressors **drift at different rates by task, context, and compressor class**,
and that a verifier-mediated draft/commit loop can preserve full-KV greedy outputs when
verify/commit semantics are implemented correctly.

Primary scientific metrics: **divergence rate**, **acceptance rate**, **first-divergence index**.
`exactkv_failures = 0` is a **harness safety gate** on cited panels, not proof that compression
is practically useful on its own.

## 3. What claim does it not prove?

- Production throughput or latency
- Active VRAM savings at serving time
- Official LongBench / MBPP / BFCL leaderboard scores
- That fallback/proxy compressors (SpectralQuant, Shard probe) are real integrations

See [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md).

## 4. Smallest demo (CPU, ~1 minute)

```bash
git clone https://github.com/utkarshg20/ExactKV.git
cd ExactKV
pip install -e ".[dev]"
python3 scripts/exactkv_terminal_crash_test.py --speed fast
```

Expected: banner `EXACTKV CRASH TEST`, a drift case, verifier correction, failures `0`.

## 5. Cheap repro (CPU, no model download)

```bash
bash scripts/smoke_test.sh
python3 scripts/exactkv_repro.py --reports-only
python3 scripts/check_site_claims.py
python3 scripts/audit_public_claims.py
python3 -m pytest tests/test_acceptance_logic.py tests/test_capture_divergence_topk.py -q
```

Expected: `SMOKE TEST PASSED`, claim audits pass, acceptance/top-k unit tests green.

## 6. Main table reproduction (GPU + HF access)

Headline **1,500-cell** panel:

```bash
python3 scripts/run_phase_a_scale_benchmark.py --device cuda --dtype float16
```

Artifact: `reports/scale_7b/raw.json`

Public leaderboard JSON: `reports/public_release/leaderboard_final.json`

External **8,132-cell** supplement: `reports/external_panels/` (see
[`EXTERNAL_BENCHMARK_PANELS.md`](EXTERNAL_BENCHMARK_PANELS.md)).

## 7. Compressor tiers (read before interpreting tables)

| tier | examples | meaning |
|------|----------|---------|
| **Built-in real** | `noop`, `int8` | Implemented in-repo, used in headline panels |
| **Built-in simulated** | `int4_sim`, `int6_sim`, `h2o_sim` | Diagnostic simulators, not upstream product ports |
| **Fallback / proxy** | `spectralquant`, `shard` | Mock or probe-only when real deps unavailable |
| **Faithful adapter** | `snapkv_experimental`, `turboquant_experimental`, `kivi_offline_r32` | Upstream code paths on a separate grid (appendix §6.17) |

## 8. Version source of truth

| field | value |
|-------|-------|
| Package version | `exactkv.__version__` / `pyproject.toml` |
| Latest git tag | `v0.11.0` |
| Research release narrative | [`RELEASE.md`](../RELEASE.md) |

Historical phase docs live under `docs/` and `docs/archive/` (when present). Start here, not there.

## 9. Key files for code review

| concern | file |
|---------|------|
| Draft / verify / commit | `exactkv/runtime/exactkv_generator.py` |
| Acceptance logic | `exactkv/verification/acceptance.py` |
| Benchmark cell runner | `exactkv/benchmarks/runner.py` |
| External panels | `exactkv/benchmarks/external_panel.py` |
| Claim boundaries | `docs/CLAIM_BOUNDARIES.md` |
