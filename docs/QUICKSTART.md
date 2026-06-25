# ExactKV Quickstart

CPU-safe paths to explore ExactKV without expensive GPU inference.

> **Status:** Prelaunch research prototype — not a production serving system. See [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md).

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Details: [`INSTALL.md`](INSTALL.md)

---

## CPU-safe quick command

```bash
bash scripts/smoke_test.sh
python3 scripts/exactkv_terminal_crash_test.py --speed fast
```

No GPU or model download required for the terminal demo.

---

## Reports-only command (no inference)

Regenerate public release bundle, novelty audit, and evidence status from on-disk artifacts:

```bash
python3 scripts/exactkv_repro.py --reports-only
```

**Expected outputs:**
- `reports/public_release/` — README, leaderboard, methodology, manifest
- `docs/NOVELTY_AUDIT.md`, `reports/novelty_audit.json`
- `reports/release_evidence_status.json`, `docs/RELEASE_EVIDENCE_STATUS.md`
- `reports/repro_manifest.json`

---

## Release validation (recommended before launch)

```bash
python3 scripts/exactkv_repro.py --release-check
# or
bash scripts/repro_all.sh
```

---

## GPU benchmark command (expensive)

Real 7B/8B inference requires:
- CUDA GPU with sufficient VRAM
- Hugging Face authentication (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`)
- Large disk volume for model caches (prefer `HF_HOME` on `/workspace` or equivalent — not root disk)
- Hours of runtime

```bash
export HF_HOME=/workspace/.cache/huggingface   # example: large volume
export HF_TOKEN=your_token_here                # never commit tokens
python3 scripts/exactkv.py run full-scale-7b --config exactkv/configs/scale_7b_8b.yaml
```

**Requires explicit confirmation for full repro wrapper:**

```bash
python3 scripts/exactkv_repro.py --full --confirm-expensive
```

Without `--confirm-expensive`, `--full` refuses to run expensive inference.

---

## Public release regeneration

```bash
python3 scripts/exactkv.py run publish
```

Authoritative public evidence: **1500-cell Phase H+ scale_7b** (`reports/scale_7b/raw.json`).

---

## What you should see

| Step | Success signal |
|------|----------------|
| Smoke | `SMOKE TEST PASSED` |
| `--reports-only` | `wrote reports/repro_manifest.json`, public_release updated |
| `--release-check` | evidence PASS, claim audit PASS, public release validator PASS |

---

## What ExactKV is / is not

**Is:** Compressor-agnostic crash-test and leaderboard for LLM KV-cache compression exactness.

**Is not:** Production serving, VeriCache reproduction, end-to-end speedup proof, or v1.0 product.

- [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md)
- [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md)
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)
