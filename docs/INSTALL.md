# ExactKV Install Guide

**Status:** Prelaunch research prototype — not v1.0, not public-launch-ready.

> See also: [`QUICKSTART.md`](QUICKSTART.md) · [`REPRO_CHECKLIST.md`](REPRO_CHECKLIST.md)

---

## Requirements

| Item | Version / notes |
|---|---|
| **Python** | **3.10+** (`requires-python` in `pyproject.toml`) |
| **pip** | Recent pip with editable install support |
| **OS** | macOS, Linux, or WSL recommended |
| **GPU** | **Optional** for smoke test, terminal demo, and leaderboard replay |
| **Disk** | ~2 GB if downloading `Qwen/Qwen2.5-0.5B` weights (not required for smoke) |

---

## Standard install (editable + dev tools)

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install -e ".[dev]"
```

This installs:

- **Runtime:** `torch`, `transformers`, `accelerate`, `safetensors`, `numpy`, `tqdm`
- **Dev:** `pytest`, `pytest-timeout`

There is **no** `requirements.txt` — use `pyproject.toml` as the source of truth.

---

## Verify install

```bash
python3 -c "import exactkv; print('exactkv OK')"
bash scripts/smoke_test.sh
```

Smoke test is CPU-safe and does **not** download model weights.

---

## Optional: model weights (full test suite)

Full `pytest tests/` may require cached Hugging Face weights:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
"
```

Then run with offline cache:

```bash
TRANSFORMERS_OFFLINE=1 pytest tests/ -q
```

**Llama-3.1-8B** (Exp 033) requires Meta license acceptance and `HF_TOKEN` — not needed for smoke or demos.

---

## Optional: visualization dependencies

Scripts `visualize_experiment_035.py` and `render_public_visuals_036.py` require **matplotlib** and **Pillow**. These are not in `[dev]` extras today:

```bash
pip install matplotlib pillow
```

Not required for smoke test, terminal demo, or leaderboard.

---

## Optional isolated backends (not default)

| Extra | Install | Purpose |
|---|---|---|
| `kvpress` | `pip install -e ".[kvpress]"` | SnapKV experimental adapter only |
| `turboquant` | `pip install -e ".[turboquant]"` + clone `vendor/turboquant_plus` | TurboQuant Python adapter |
| KIVI / KVQuant | Separate venvs per experiment docs | Restricted backend gauntlet |

Do not install these for smoke test or quickstart.

---

## What does **not** require GPU

| Task | GPU? |
|---|---|
| `bash scripts/smoke_test.sh` | No |
| `python3 scripts/exactkv_terminal_crash_test.py` | No (replay mode) |
| `python3 scripts/exactkv_leaderboard.py` | No |
| Prelaunch audit scripts | No |
| `pytest tests/test_exactkv_terminal_crash_test.py` etc. | No |
| Full experiment sweeps (Exp 030–034) | Yes (recommended) |

---

## Common install issues

| Problem | Fix |
|---|---|
| `import exactkv` fails | Run from repo root; `pip install -e ".[dev]"` |
| Torch install slow/fails | Use platform-appropriate PyTorch install per pytorch.org, then `pip install -e ".[dev]"` |
| Smoke test pytest fails | Ensure venv activated and dev extras installed |
| Leaderboard full panel empty | Local `reports/*.csv` missing — copy or regenerate; not an install failure |

---

## Claims boundary

ExactKV install gets you a **correctness-first crash-test lab**. It does **not** imply speedup, active GPU memory savings, production serving, or v1.0 readiness.
