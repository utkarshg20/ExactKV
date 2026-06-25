# ExactKV Reproducibility Guide

Commands to reproduce benchmarks, audits, and public release artifacts **without inventing new results**.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Hugging Face cache directory — use a **large volume** (e.g. `/workspace/.cache/huggingface`) |
| `HF_TOKEN` | Hugging Face API token for gated models |
| `HUGGING_FACE_HUB_TOKEN` | Alias for `HF_TOKEN` |
| `TRANSFORMERS_OFFLINE` | Set `1` for pytest without network when weights are cached |

**Security:** Never paste real HF tokens into committed scripts, docs, or logs. Rotate any exposed token immediately.

---

## Disk guidance

- Model caches for Llama-3.1-8B (~30GB) and Mistral-7B-Instruct-v0.3 (~14GB) exceed typical root disks.
- Point `HF_HOME` at a large mount (RunPod `/workspace`, external volume, etc.).
- Sequential model execution (Llama then Mistral) may be required when volume quota is limited.

---

## Reports-only (no inference)

```bash
python3 scripts/exactkv_repro.py --reports-only
```

Regenerates:
- `reports/public_release/` from `reports/scale_7b/`
- `docs/NOVELTY_AUDIT.md`, `reports/novelty_audit.json`
- `reports/release_evidence_status.json`

---

## Phase F kernel microbenchmark

```bash
python3 scripts/run_phase_f_kernel_benchmark.py   # if present
# artifact: reports/phaseF_kernel_benchmark.json
```

Requires CUDA. Results are **kernel microbenchmark only** — not end-to-end inference speedups.

Locked evidence (current artifact):
- INT8: ~1.63× (torch vs triton, tested shape)
- INT4: ~1.54×
- block_sparse: torch execution path (~0.98×)

---

## Phase H+ scale_7b benchmark (expensive — GPU)

```bash
export HF_HOME=/path/to/large/volume/.cache/huggingface
export HF_TOKEN=...   # do not commit
python3 scripts/exactkv.py run full-scale-7b --config exactkv/configs/scale_7b_8b.yaml
```

Outputs:
- `reports/scale_7b/raw.json`
- `reports/scale_7b/leaderboard.json`
- `reports/scale_7b/scale_summary.json`

Current locked run: **1500 cells**, `exactkv_failures=0`, `deterministic_mode=false`.

---

## Publish bundle

```bash
python3 scripts/exactkv.py run publish
```

---

## Novelty audit (Phase I)

```bash
python3 scripts/run_novelty_audit.py
```

---

## Claim / evidence / secret checks

```bash
python3 scripts/audit_public_claims.py
python3 scripts/check_release_evidence.py
python3 scripts/check_no_secrets.py
python3 scripts/check_public_release.py
```

---

## Full validation suite

```bash
python3 scripts/exactkv_repro.py --release-check
python3 -m pytest -q
```

---

## Full repro (requires explicit confirmation)

```bash
python3 scripts/exactkv_repro.py --full --confirm-expensive
```

Without `--confirm-expensive`, this **fails** by design.

Use `--skip-gpu` with `--full` to run non-inference steps only:

```bash
python3 scripts/exactkv_repro.py --full --confirm-expensive --skip-gpu
```

---

## Repro manifest

Every `exactkv_repro.py` run writes `reports/repro_manifest.json` with:
- timestamp, git commit, Python/torch versions
- CUDA/triton availability
- commands run, skipped steps, failures
