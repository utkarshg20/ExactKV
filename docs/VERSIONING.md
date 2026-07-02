# ExactKV versioning (read this first)

ExactKV uses **three separate naming systems**. They are **not** the same counter.

| System | Example | What it means | Use publicly? |
|--------|---------|---------------|---------------|
| **Package / pip** | `0.11.0` | `pyproject.toml`, `exactkv.__version__` | **Yes** |
| **Git tag / GitHub Release** | `v0.11.0` | Cited public research artifact | **Yes — only tag on GitHub** |
| **Evidence bundle label** | `research release` | Headline 8,132-cell + 1,500-cell panel set in `RELEASE.md` | Yes (paper/README) |
| **V-milestones (V1–V21)** | `V11`, `V13` | Internal research arc docs in `docs/` | **No — archive only** |

## What you should cite

```text
ExactKV 0.11.0 (research release)
Git tag: v0.11.0
Commit: 6a67201 (see GitHub Releases)
```

Install from the release tag:

```bash
pip install git+https://github.com/utkarshg20/ExactKV.git@v0.11.0
```

## Git tags on GitHub

**Only one tag exists:** **`v0.11.0`**.

Earlier internal milestone tags (`v0.2.0`–`v0.10.0`, `v0.13.0-rc1`) were **removed** in July 2026. They caused confusion with V-milestone numbers and semver ordering. The work they marked is still documented under V1–V13 in `docs/`, but **no longer has git tags**.

## V-milestones vs public release (not the same counter)

| V-milestone | Theme | Public git tag? |
|-------------|-------|-----------------|
| V1–V10 | Prototype through suite hardening | *(removed — docs only)* |
| **V11** | Launch hardening + public research release | **`v0.11.0`** |
| V12 | Deferred-work gauntlet | *(docs only)* |
| V13 | Practicality proof | *(docs only; old rc tag removed)* |
| V14–V21 | L3/L4 integration ladder | *(docs only)* |

**V11 is not “missing”.** It **is** `v0.11.0`. V12/V13 docs describe later internal work; they do **not** mean a newer product version exists on GitHub.

## Evidence bundle “research release”

`RELEASE.md` describes the **research release** artifact (technical report + 8,132 external cells + faithful appendix). That label is independent of pip version `0.11.0`:

```text
ExactKV 0.11.0 — research release
```

**Note:** `v3.0` in the technical report and `reports/external_panels/v30/` refers to a **specific GPU panel batch** inside this release, not a separate public version number.

## For reviewers

- **One git tag:** `v0.11.0`. **One GitHub Release.** **One package version:** `0.11.0`.
- **V1–V21** in `docs/` are internal milestone names, not semver.
- Start here: [GitHub Release v0.11.0](https://github.com/utkarshg20/ExactKV/releases/tag/v0.11.0), [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md).

Historical depth: [VERSION_LINEAGE.md](VERSION_LINEAGE.md), [PROJECT_LINEAGE.md](PROJECT_LINEAGE.md).
