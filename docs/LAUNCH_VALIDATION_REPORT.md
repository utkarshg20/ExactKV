# Launch Validation Report (Phase 9C / 9D)

**Purpose:** Validate whether the repo can support a `v0.13.0-rc` **research preview** tag.

> **Not a public launch approval.** No git tag was created in this phase.

Companion: [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md) · [`assets/exactkv_terminal_demo_recording_plan.md`](assets/exactkv_terminal_demo_recording_plan.md) · [`BENCHMARK_GAP_ANALYSIS.md`](BENCHMARK_GAP_ANALYSIS.md)

---

## Summary (latest — Phase 9D)

| Field | Value |
|---|---|
| **Validation date** | 2026-06-14 |
| **Phase** | **9D** — RC blocker fixes |
| **Git commit tested** | `83f585d` + 9D working-tree fixes (commit before tag) |
| **Clean clone path** | `/tmp/exactkv_rc_validation/ExactKV-clean` (working-tree copy) |
| **Python version** | 3.13.3 |
| **Install command** | `pip install -e ".[dev]"` only |
| **Verdict** | **`research-preview-rc-ready`** (pending MP4 polish + commit) |

---

## Phase 9D — RC blocker fixes (2026-06-14)

### Fixes applied

| Blocker (9C) | Fix (9D) | Status |
|---|---|---|
| `matplotlib` / `pillow` not in `[dev]` | Added to `[project.optional-dependencies].dev` in `pyproject.toml` | ✅ Fixed |
| `RESTRICTED BACKEND` heading mismatch | Canonical md/html tier heading `## RESTRICTED BACKEND`; test aligned | ✅ Fixed |
| Clean clone missing `reports/*.csv` | Published full-panel + repair-policy static anchors in `build_tiered_leaderboard()` when CSVs absent | ✅ Fixed |
| Full pytest failure | Heading fix + static anchors | ✅ Fixed |

### Re-validation — dev workspace

| Check | Result |
|---|---|
| `pytest -q` | **PASS** — 1754 passed, 83 skipped (~11 min) |
| `audit_public_claims.py` | **PASS** |
| `check_docs_links.py` | **PASS** |
| `check_report_hygiene.py --require-public` | **PASS** |
| `git diff --check` | **PASS** |

### Re-validation — clean install (`pip install -e ".[dev]"` only)

| Command | Exit | Result |
|---|---|---|
| `bash scripts/smoke_test.sh` | 0 | **PASS** (61 pytest subset passed) |
| `exactkv_terminal_crash_test.py --no-delay --plain` | 0 | **PASS** |
| `exactkv_terminal_longbench_drift.py --no-delay --plain` | 0 | **PASS** |
| `exactkv_leaderboard.py --summary` | 0 | **PASS** — 11 full-panel rows from static anchors |
| `exactkv_leaderboard.py --plain` | 0 | **PASS** — includes `FULL · REAL-BYTE` |
| Audits (claims, links, hygiene) | 0 | **PASS** |

**Note:** Clean-clone test used a working-tree copy of 9D fixes. After commit, `git clone` reproduces the same gate.

### Remaining blockers (9D)

| # | Blocker | Severity |
|---|---|---|
| 1 | Fresh terminal MP4 not recorded in 9C/9D | **Launch polish** (optional for RC tag) |
| 2 | 9D fixes must be **committed** before `git clone` validation | **Process** |
| 3 | Explicit public launch / v1.0 not granted | **Policy** |

### Verdict (9D)

**`research-preview-rc-ready`** — clean install, smoke, terminal demos, leaderboard, audits, and full pytest pass. Recommend `v0.13.0-rc1` **after commit**; MP4 recording remains optional polish.

---

## Phase 9C — initial validation (historical)

| Field | Value |
|---|---|
| **Verdict (9C)** | `research-demo-ready` |
| **Git commit** | `83f585dec0b0e3ec3e7f07df1d94c73be49c0ed0` |

### 1. Clean clone validation (Phase A)

| Step | Command | Result |
|---|---|---|
| A1 — documented dev install | `pip install -e ".[dev]"` | **PASS** |
| A2 — optional viz deps (per [`INSTALL.md`](INSTALL.md)) | `pip install matplotlib pillow` | **PASS** (required for leaderboard) |

Clone source: local path `/Users/utkarshgupta/Documents/ExactKV` (same commit as HEAD).

### Command results — strict install (`[dev]` only)

| Command | Exit | Result |
|---|---|---|
| `bash scripts/smoke_test.sh` | 1 | **FAIL** — `exactkv_leaderboard.py` needs `matplotlib` (not in `[dev]`) |
| `python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain` | — | Not re-run in strict pass (smoke failed first) |
| `python3 scripts/exactkv_leaderboard.py --plain` | 1 | **FAIL** — `ModuleNotFoundError: matplotlib` |

### Command results — with optional `matplotlib pillow`

| Command | Exit | Result |
|---|---|---|
| `bash scripts/smoke_test.sh` | 1 | **PARTIAL** — sections 1–5 pass; pytest subset 3 failures (missing CSV reports / leaderboard string expectations) |
| `python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain` | 0 | **PASS** |
| `python3 scripts/exactkv_terminal_longbench_drift.py --no-delay --plain` | 0 | **PASS** |
| `python3 scripts/exactkv_leaderboard.py --summary` | 0 | **PASS** (note: 15 expected CSVs missing; 0 full-panel rows) |
| `python3 scripts/exactkv_leaderboard.py --plain` | 0 | **PASS** (restricted/smoke tiers populated from static rows) |
| `python3 scripts/audit_public_claims.py` | 0 | **PASS** |
| `python3 scripts/check_docs_links.py` | 0 | **PASS** |
| `python3 scripts/check_report_hygiene.py --require-public` | 0 | **PASS** |

**Clean-clone gate:** **FAIL** on `pip install -e ".[dev]"` alone. **PARTIAL PASS** after documented optional viz deps; full-panel leaderboard empty without local `reports/*.csv`.

---

## 2. Developer workspace validation (Phase C)

Run from repository root at commit `83f585d`:

| Check | Exit | Result |
|---|---|---|
| `python3 scripts/audit_public_claims.py` | 0 | **PASS** |
| `python3 scripts/check_docs_links.py` | 0 | **PASS** |
| `python3 scripts/check_report_hygiene.py --require-public` | 0 | **PASS** |
| `git diff --check` | 0 | **PASS** |
| `git status --short` | — | No staged raw `reports/*.json` or `reports/*.csv` |

**README launch wording:** Correctly states prelaunch / not public-launch-ready / not v1.0.

**Leaderboard artifacts:** `docs/leaderboard.md` and `docs/leaderboard.html` committed; regenerate with `python3 scripts/exactkv_leaderboard.py --md --html` when local CSV reports change.

---

## 3. Regression tests (Phase D)

### Selected pytest (developer workspace)

```bash
pytest tests/test_exactkv_terminal_crash_test.py \
       tests/test_exactkv_terminal_longbench_drift.py \
       tests/test_benchmark_gap_analysis.py \
       tests/test_exactkv_leaderboard.py \
       tests/test_shard_combined_stress_report.py \
       tests/test_spectralquant_restricted_panel.py \
       tests/test_spectralquant_real_kv_smoke.py \
       tests/test_spectralquant_adapter_smoke.py \
       tests/test_spectralquant_probe.py -q
```

| Result | Detail |
|---|---|
| **PASS** | **99 passed** in ~16s |

### Full pytest (developer workspace)

```bash
pytest -q
```

| Result | Detail |
|---|---|
| **FAIL** | **1753 passed**, 83 skipped, **1 failed** in ~21 min |

Failure:

- `tests/test_render_public_visuals_036.py::test_public_visual_package_doc` — expects `"RESTRICTED BACKEND"` in `docs/leaderboard.md`; generated heading is `"Restricted backends"`.

---

## 4. Terminal demo recording readiness (Phase B)

| Item | Status |
|---|---|
| Recording plan | ✅ [`assets/exactkv_terminal_demo_recording_plan.md`](assets/exactkv_terminal_demo_recording_plan.md) |
| Helper script | ✅ [`scripts/record_terminal_demo.sh`](../scripts/record_terminal_demo.sh) |
| Fresh MP4 recorded in Phase 9C | **No** — existing `docs/assets/exactkv_crash_test_demo.mp4` not re-recorded (by design) |
| Terminal replay demo | **PASS** — crash-test and LongBench-style demos exit 0 |

---

## 5. Remaining blockers

| # | Blocker | Severity |
|---|---|---|
| 1 | `matplotlib` / `pillow` not in `[dev]` but required by `smoke_test.sh` leaderboard step | **RC gate** |
| 2 | Clean clone has no `reports/*.csv` → empty full-panel leaderboard; smoke pytest subset fails | **RC gate** |
| 3 | Full pytest: 1 leaderboard heading string mismatch | **Minor** |
| 4 | Terminal MP4 not freshly recorded in this validation | **Launch polish** |
| 5 | Explicit launch / v1.0 decision not granted | **Policy** |

---

## 6. Verdict

### `research-demo-ready`

The repo supports terminal demos, tiered leaderboard replay, benchmark-gap docs, claims audits, and selected regression tests **in a developer workspace with local reports and optional viz deps**.

### Not `research-preview-rc-ready`

Clean-clone repro with documented install alone does not pass smoke (matplotlib) or full leaderboard pytest subset (missing CSV reports). Full pytest has one failure.

### Not `blocked`

Core demos, audits, and selected tests pass on the developer checkout.

---

## 7. Tag recommendation (manual — not executed)

After committing 9D fixes:

```bash
git tag -a v0.13.0-rc1 -m "ExactKV V13 research preview RC1 — see docs/LAUNCH_VALIDATION_REPORT.md"
git push origin v0.13.0-rc1
```

Optional before tag: record terminal demo per [`assets/exactkv_terminal_demo_recording_plan.md`](assets/exactkv_terminal_demo_recording_plan.md).

---

## 8. Claims boundary (unchanged)

No speedup, active memory savings, production serving, or model accuracy improvement claim. Research preview only — not v1.0.
