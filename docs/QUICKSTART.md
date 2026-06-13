# ExactKV Quickstart

**Five commands** to see what ExactKV is — no GPU, no model download.

> Status: **prelaunch hardening** — not public launch, not v1.0.

---

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Details: [`INSTALL.md`](INSTALL.md)

---

## 2. Smoke test (validates install)

```bash
bash scripts/smoke_test.sh
```

Or via Makefile:

```bash
make smoke
```

---

## 3. Terminal crash-test demo

```bash
python3 scripts/exactkv_terminal_crash_test.py --speed fast
```

Replay of verified Exp 034b trace (`drop` → `pickup`). No model inference.

```bash
make demo
```

---

## 4. Crash-test leaderboard

```bash
python3 scripts/exactkv_leaderboard.py
```

Terminal table + updates `docs/leaderboard.md` and `docs/leaderboard.html`.

```bash
make leaderboard
```

Open the HTML view:

```bash
open docs/leaderboard.html   # macOS
```

---

## 5. Prelaunch audits (optional)

```bash
make audit
```

Runs claims, link, and report-hygiene checks.

---

## What you should see

| Step | Success signal |
|---|---|
| Smoke | `SMOKE TEST PASSED` |
| Demo | `DRIFT DETECTED`, `REJECTED`, `COMMITTED`, failures `0` |
| Leaderboard | `FULL PANEL RESULTS` table with tier sections |

---

## What ExactKV is / is not

**Is:** A KV-cache compression **crash-test lab** — lossy KV drafts, full-KV verifies, exact greedy output on tested panels.

**Is not:** A speedup library, VRAM saver, production serving stack, or v1.0 product.

Full claims boundary: [`CLAIMS_AUDIT.md`](CLAIMS_AUDIT.md)

---

## Next steps (optional)

| Goal | Command / doc |
|---|---|
| Full test suite (needs model weights) | `TRANSFORMERS_OFFLINE=1 pytest tests/ -q` |
| Research figures | `pip install matplotlib pillow && python3 scripts/visualize_experiment_035.py` |
| Launch readiness audit | [`LAUNCH_READINESS_GAP_AUDIT.md`](LAUNCH_READINESS_GAP_AUDIT.md) |
