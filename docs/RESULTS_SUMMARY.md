# ExactKV Results Summary (Phase J)

Values read from on-disk artifacts — not invented.

---

## Authoritative public release: Phase H+ scale_7b

| Field | Value | Source |
|-------|-------|--------|
| Total cells | **1500** | `reports/scale_7b/scale_summary.json` |
| ExactKV failures | **0** | `reports/scale_7b/scale_summary.json` |
| Deterministic mode | `false` (real GPU) | `reports/scale_7b/scale_summary.json` |
| Device | `cuda` | `reports/scale_7b/scale_summary.json` |
| Dtype | float16 (per run manifest) | `reports/scale_7b/raw.json` |

### Models

- `meta-llama/Llama-3.1-8B` — 750 cells
- `mistralai/Mistral-7B-Instruct-v0.3` — 750 cells

### Compressors (scale panel)

`noop`, `int8`, `int4_sim`, `spectralquant`, `shard` — 300 cells each across both models.

---

## Top leaderboard rows (Llama-3.1-8B)

From `reports/scale_7b/leaderboard.json` / `reports/public_release/leaderboard_final.json`:

| Rank | Compressor | Score | Acceptance |
|------|------------|------:|-----------:|
| 1 | `noop` | 1.0 | 1.0 |
| 2 | `int8` | 1.0 | 1.0 |
| 3 | `int4_sim` | 0.684 | 0.852 |
| 4 | `spectralquant` | 0.684 | 0.852 |
| 5 | `shard` (probe-first) | 0.544 | 0.632 |

### Mistral-7B-Instruct-v0.3 (Phase H+ scale_7b)

| Rank | Compressor | Score | Acceptance | Availability |
|------|------------|------:|-----------:|----------------|
| 1 | `noop` | 1.0 | 1.0 | available |
| 2 | `int8` | 0.983 | 1.0 | available |
| 3 | `int4_sim` | 0.851 | 0.837 | available |
| 4 | `spectralquant` | 0.851 | 0.837 | mock_fallback |
| 5 | `shard` | 0.727 | 0.623 | probe_only |

Values from `reports/scale_7b/leaderboard.json` after Release Gate R1 aggregate repair (750 raw cells per model).

---

## Historical Phase A (internal supporting evidence)

| Field | Value |
|-------|-------|
| Cells | 336 |
| Models | Qwen 0.5B family, Mistral-7B, Llama-3.1-8B (4-model panel) |
| Role | Cross-model drift illustrations; **not** the final public release headline |

---

## Phase F kernel microbenchmark

Source: `reports/phaseF_kernel_benchmark.json`

| Mode | Speedup (torch→triton) | Caveat |
|------|------------------------|--------|
| int8 | **1.63×** | Kernel microbenchmark only |
| int4 | **1.54×** | Kernel microbenchmark only |
| block_sparse | **0.98×** | `execution_backend=torch` (not Triton win) |

**Not** end-to-end inference speedups.

---

## Adapter status (current environment)

| Adapter | Status | Source |
|---------|--------|--------|
| SpectralQuant | **fallback/proxy** (`spectralquant_available=False`) | release evidence checker |
| Shard | **probe-first** heuristic (`probe_only=True`) | `shard_real` adapter |

Do not claim real SpectralQuant or real Shard integration.

---

## Limitations

- ExactKV is **not a production serving system**.
- Does **not reproduce VeriCache** serving throughput.
- Compression ratios are **stored tensor byte ratios** unless active GPU memory is measured.
- Scale run used **sequential** model execution (disk quota constraint).
- Public leaderboard requires repaired `per_model_tables` when sequential merges omit derived aggregates (Release Gate R1).

---

## Reproduce summary artifacts

```bash
python3 scripts/exactkv_repro.py --reports-only
```
