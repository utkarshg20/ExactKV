# ExactKV versioning (read this first)

ExactKV uses **separate naming layers**. They are **not** the same counter.

| System | Example | What it means | Use publicly? |
|--------|---------|---------------|---------------|
| **Git tag / GitHub Release** | `v-release` | Cited public research artifact | **Yes — only tag on GitHub** |
| **Evidence bundle label** | `research release` | Headline 8,132-cell + 1,500-cell panel set in `RELEASE.md` | **Yes** |
| **Package / pip** | `0.11.0` | `pyproject.toml`, `exactkv.__version__` (implementation detail) | Optional |
| **V-milestones (V1–V21)** | `V-release`, `V13` | Internal research arc docs in `docs/` | **No — archive only** |

## What you should cite

```text
ExactKV research release
Git tag: v-release
Commit: see GitHub Releases
```

Install from the release tag:

```bash
pip install git+https://github.com/utkarshg20/ExactKV.git@v-release
```

## Git tags on GitHub

**Only one tag exists:** **`v-release`**.

Earlier tags (`v0.1.0-phase1`–`v0.10.0`, `v0.11.0`, `v0.13.0-rc1`) were **removed** in July 2026. They caused confusion with V-milestone numbers and semver ordering. Historical work remains in `docs/` under V1–V13, but **no longer has git tags**.

## V-milestones vs public release (not the same counter)

| V-milestone | Theme | Public git tag? |
|-------------|-------|-----------------|
| V1–V10 | Prototype through suite hardening | *(removed — docs only)* |
| **V-release** | Launch hardening + public research release | **`v-release`** |
| V12 | Deferred-work gauntlet | *(docs only)* |
| V13 | Practicality proof | *(docs only; old rc tag removed)* |
| V14–V21 | L3/L4 integration ladder | *(docs only)* |

**V-release is the shipped public artifact.** Older docs may still say **V11** or **`v0.11.0`** — same milestone, retired names. V12/V13 describe later internal work; they do **not** mean a newer product version exists on GitHub.

## Evidence bundle “research release”

`RELEASE.md` describes the **research release** artifact (technical report + 8,132 external cells + faithful appendix):

```text
ExactKV research release
Git tag: v-release
```

**Note:** `v3.0` in the technical report and `reports/external_panels/v30/` refers to a **specific GPU panel batch** inside this release, not a separate public version number.

## For reviewers

- **One git tag:** `v-release`. **One GitHub Release.** **One public name:** research release.
- **V1–V21** in `docs/` are internal milestone names, not semver.
- Start here: [GitHub Release v-release](https://github.com/utkarshg20/ExactKV/releases/tag/v-release), [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md).

Historical depth: [VERSION_LINEAGE.md](VERSION_LINEAGE.md), [PROJECT_LINEAGE.md](PROJECT_LINEAGE.md).
