# Raw Artifact Policy

**Status:** V11 Phase 6 — curated bundle policy for release artifacts.
**Applies to:** Experiments **001–020** and future published runs.

> Raw JSON/CSV under `reports/` remain **gitignored**. Committed Markdown experiment
> reports are the canonical published narrative. This policy describes how to package
> curated artifacts for a release without bloating the git repository.

---

## 1. Which raw JSON/CSV reports remain gitignored

All of the following stay out of git (see root `.gitignore`):

| Pattern | Examples |
|---|---|
| `reports/*.json` | `experiment_012_eval_suite_expansion.json`, `experiment_020_repair_policy_pilot.json` |
| `reports/*.csv` | Matching CSV exports for each experiment |

**Committed instead:** `docs/EXPERIMENT_*_*.md` reports, experiment index, release notes,
readiness assessments, and reproduction commands in each experiment doc.

---

## 2. Why raw reports are not committed directly

- **Size:** Full experiment JSON can be hundreds of KB to MB per run; cumulative
  history would dominate the repository.
- **Reproducibility:** Markdown reports + pinned scripts + documented commands are
  the stable reference; raw files are reproducible outputs.
- **Secrets and paths:** RunPod hostnames, local pickle paths, and environment-specific
  metadata may appear in manifests — better kept in optional bundles than in git history.
- **Schema stability:** Standard report schema is documented; pilot artifacts (Exp 018,
  019, 020) are explicitly isolated and may use additive fields outside the standard schema.

---

## 3. How to package curated artifacts

1. Reproduce or collect gitignored JSON/CSV from `reports/` after a successful run
   (`exactkv_failures == 0` on included experiments).
2. Create a staging directory, e.g. `exactkv-reports-bundle-staging/`.
3. Copy **only** approved experiment artifacts (see §5 for exclusions).
4. Add `manifest.json` (§5) at the bundle root.
5. Generate SHA-256 checksums for every file in the bundle.
6. Write `checksums.sha256` (one line per file: `hash  filename`).
7. Create the archive from the staging directory:

```bash
tar czf exactkv-reports-v0.11.0-bundle.tar.gz -C exactkv-reports-bundle-staging .
```

8. Attach the tarball to a GitHub **Release** asset (not a git commit).

---

## 4. Suggested artifact bundle filename

```
exactkv-reports-v0.11.0-bundle.tar.gz
```

For v1.0.0 public launch, use `exactkv-reports-v1.0.0-bundle.tar.gz` with an updated
manifest listing all experiments through the launch tag.

---

## 5. Checksum manifest

**`manifest.json`** (bundle root) — minimum fields per experiment file:

```json
{
  "bundle_version": "v0.11.0",
  "created_utc": "2026-06-11T00:00:00Z",
  "experiments": [
    {
      "id": "012",
      "filename": "experiment_012_eval_suite_expansion.json",
      "sha256": "<hex>",
      "cells": 896,
      "exactkv_failures": 0,
      "model": "Qwen/Qwen2.5-0.5B"
    }
  ],
  "note": "Curated research artifacts only. Not a performance benchmark."
}
```

**`checksums.sha256`** — standard format:

```
a1b2c3d4...  experiment_012_eval_suite_expansion.json
e5f6g7h8...  experiment_015_qwen15b_v10_suites.json
```

Generate checksums:

```bash
shasum -a 256 experiment_*.json > checksums.sha256
```

---

## 6. What not to include

| Exclude | Reason |
|---|---|
| Model weights (`~/.cache/huggingface`, RunPod caches) | Huge; users download via Transformers |
| Quantizer pickles (`*.pickle`, `EXACTKV_KVQUANT_QUANTIZERS`) | Large; environment-specific unless explicitly intended for a restricted repro kit |
| Virtual environments (`.venv*`, `.venv-turboquant`, etc.) | Not portable; use documented env setup |
| Huge logs (`exp019.log`, full RunPod session logs) | Noise; not needed for artifact integrity |
| Private RunPod paths, SSH keys, API tokens | Security |
| Unredacted secrets in JSON manifests | Audit before bundling |

---

## 7. Reproduction commands live in experiment docs

Each experiment's Markdown report under `docs/EXPERIMENT_*.md` contains the authoritative
reproduce command. See also [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) §Reproduction commands.

Pilot artifacts (018–020) use isolated schemas documented in their respective reports;
they are optional bundle inclusions labeled `artifact_type: isolated_pilot` in manifests.

---

## Related

- [`V11_LAUNCH_READINESS.md`](V11_LAUNCH_READINESS.md) — launch gates and v1.0.0 decision
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) — experiment catalog
- [`V11_SCOPE_STATEMENT.md`](V11_SCOPE_STATEMENT.md) §15 — scope-era policy summary
