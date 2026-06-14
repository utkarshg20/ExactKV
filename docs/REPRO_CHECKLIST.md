# Reproduction Checklist (V13 Phase 9A)

**Purpose:** Verify that demos, leaderboard, and core tests work from a clean checkout — without running full GPU experiment sweeps.

> This checklist supports **prelaunch hardening**, not public launch.
> GPU is optional for most items below.

Companion: [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) · [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) · [`LAUNCH_VALIDATION_REPORT.md`](LAUNCH_VALIDATION_REPORT.md) (Phase 9C)

---

## 1. Quick repro (no GPU required)

Run from repository root after install:

```bash
bash scripts/smoke_test.sh
make audit
python3 scripts/audit_public_claims.py
python3 scripts/check_docs_links.py
python3 scripts/check_report_hygiene.py --require-public
```

`pip install -e ".[dev]"` includes `matplotlib` and `pillow` for leaderboard and public visual scripts.

### Expected outcomes

| Command | Exit code | Expected output |
|---|---|---|
| `exactkv_terminal_crash_test.py --no-delay --plain` | 0 | `EXACTKV CRASH TEST`, `DRIFT DETECTED`, `REJECTED`, `COMMITTED`, failures `0` |
| `exactkv_leaderboard.py --plain` | 0 | `EXACTKV CRASH-TEST LEADERBOARD`, `FULL PANEL RESULTS`, tier sections |
| `visualize_experiment_035.py` | 0 | 8 PNGs in `docs/assets/exp035_*.png`; updates `leaderboard.md` + `leaderboard.html` |
| `render_public_visuals_036.py` | 0 | `public_*.png` cards in `docs/assets/` |
| `pytest` (above tests) | 0 | All tests pass |

---

## 2. Clean clone check

```bash
git clone <repo-url> exactkv-repro-test
cd exactkv-repro-test
pip install -e ".[dev]"
```

Then run §1 quick repro.

See [`LAUNCH_VALIDATION_REPORT.md`](LAUNCH_VALIDATION_REPORT.md) for Phase 9C clean-clone results (2026-06-14).

| Step | Pass? |
|---|---|
| Clone succeeds | |
| `pip install -e ".[dev]"` succeeds | |
| Quick repro commands exit 0 | |
| No missing committed assets (PNG, HTML) | |

**Note:** Full `pytest tests/` may require cached `Qwen/Qwen2.5-0.5B` weights and takes longer. Use §1 subset for smoke.

---

## 3. Dependency check

| Dependency | Required for | Install |
|---|---|---|
| Python 3.10+ | All | system |
| `pip install -e ".[dev]"` | Package + pytest | `pyproject.toml` |
| `torch`, `transformers` | Model tests / sweeps | dev extras |
| `matplotlib` | Exp 035 / 036 visuals | dev extras |
| `Pillow` | Video renderer (optional) | dev extras |
| `ffmpeg` | MP4 encode (optional) | system |
| `kvpress` | SnapKV adapter only | optional venv |
| CUDA GPU | Full experiment reproduction | optional; RunPod scripts |

---

## 4. GPU optional vs required

| Task | GPU required? |
|---|---|
| Terminal crash-test demo (replay) | **No** |
| Leaderboard from committed CSVs | **No** (if `reports/*.csv` present locally) |
| Leaderboard on clone without reports | **Partial** — restricted/static rows only; full panel may be empty |
| Exp 035 / 036 visuals | **No** |
| Subset pytest (§1) | **No** |
| Full `pytest tests/` with model tests | **No** (CPU ok; slow) |
| Exp 030–034 GPU sweeps | **Yes** (CUDA recommended) |
| Exp 033 Llama-3.1-8B | **Yes** + HF license |

---

## 5. Files not to commit

Per `.gitignore` and [`RAW_ARTIFACT_POLICY.md`](RAW_ARTIFACT_POLICY.md):

| Pattern | Reason |
|---|---|
| `reports/*.json` | Large generated experiment output |
| `reports/*.csv` | Large generated experiment output |
| `*.jsonl` (except `benchmarks/prompts/`) | Benchmark output |
| `.env`, tokens, credentials | Secrets |
| Local model caches | User machine specific |

**Prelaunch check:** `git status` should not show accidental `reports/*.json` or `reports/*.csv` staged.

---

## 6. Committed assets that must exist

| Path | Used by |
|---|---|
| `docs/leaderboard.md` | GitHub-readable leaderboard |
| `docs/leaderboard.html` | Public HTML view |
| `docs/assets/public_*.png` | README / social |
| `docs/assets/exp035_*.png` | Research figures |
| `benchmarks/prompts/*.jsonl` | V10 suites |

If `reports/*.csv` are missing locally, regenerate from experiment scripts (GPU) or accept reduced full-panel rows in leaderboard terminal output.

---

## 7. One-command entry points (Phase 9B)

| Goal | Command |
|---|---|
| Smoke test | `bash scripts/smoke_test.sh` or `make smoke` |
| Terminal demo | `python3 scripts/exactkv_terminal_crash_test.py --speed fast` or `make demo` |
| Record terminal (transcript) | `bash scripts/record_terminal_demo.sh` |
| Leaderboard | `python3 scripts/exactkv_leaderboard.py` or `make leaderboard` |
| Prelaunch audits | `make audit` |
| Full test suite (needs model weights) | `TRANSFORMERS_OFFLINE=1 pytest tests/ -q` |

---

## 8. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Leaderboard full panel empty | `reports/experiment_012_*.csv` missing | Run Exp 012 or copy reports locally |
| `ModuleNotFoundError: exactkv` | Not installed editable | `pip install -e ".[dev]"` |
| Model tests skip/fail | Weights not cached | Download Qwen2.5-0.5B per README |
| `visualize_experiment_035` import error | Run from repo root | `cd` to repo root |
| HTML leaderboard blank tab | JS disabled | Enable JavaScript; or use `leaderboard.md` |
| SnapKV tests fail | kvpress not installed | Expected; optional adapter |
| Llama tests fail | HF license / token | Accept Meta license; set `HF_TOKEN` |

---

## 9. Full experiment reproduction (out of scope for smoke)

See [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) for GPU RunPod commands. Do **not** require full sweeps for prelaunch smoke.

---

## 10. Sign-off

| Check | Owner | Date | Pass? |
|---|---|---|---|
| Quick repro §1 | | | |
| Clean clone §2 | | | |
| No raw reports committed §5 | | | |
| Claims match [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md) | | | |

**Launch repro gate:** All §1 commands exit 0 on a clean clone with `pip install -e ".[dev]"` only.
