# ExactKV versioning (read this first)

ExactKV uses **three separate naming systems**. They are **not** the same counter.

| System | Example | What it means | Use publicly? |
|--------|---------|---------------|---------------|
| **Package / pip** | `0.11.0` | `pyproject.toml`, `exactkv.__version__` | **Yes** |
| **Git tag / GitHub Release** | `v0.11.0` | Cited public research artifact | **Yes — only this tag** |
| **Evidence bundle label** | `v3.0` | Headline 8,132-cell + 1,500-cell panel set in `RELEASE.md` | Yes (paper/README) |
| **V-milestones (V1–V21)** | `V11`, `V13` | Internal research arc docs in `docs/` | **No — archive only** |
| **Old git tags** | `v0.4.0` … `v0.10.0` | Historical milestone snapshots from June 2026 | **No — ignore for release order** |

## What you should cite

```text
ExactKV v0.11.0 (research artifact v3.0)
Git tag: v0.11.0
Commit: 6a67201 (release artifact; see GitHub Releases)
```

Install: `pip install exactkv==0.11.0` (from git tag `v0.11.0`).

## Why the tags look “out of order”

Development was **not** linear in tag order:

1. **V1–V10** were tagged `v0.2.0`–`v0.10.0` in early June 2026 as internal milestones.
2. **V12** (Experiments 021–027) completed **without** a `v0.12.0` git tag.
3. **`v0.13.0-rc1`** was an internal research-preview tag (V13 work) created **before** the public **`v0.11.0`** release was tagged. The number `13` refers to **V13 milestone**, not semver “newer than 0.11”.
4. **`v0.11.0`** is the **only canonical public release**. It maps to **V11 launch-hardening** work **plus** the v3.0 evidence bundle (Phases A–K external panels).

The confusing tag **`v0.13.0-rc1` has been removed** from GitHub. It was never a public release.

## V-series vs v0.x tags (not 1:1 after V10)

| V-milestone | Theme | Git tag (if any) | Public? |
|-------------|-------|------------------|---------|
| V1–V3 | Prototype → benchmark | v0.2.0–v0.3.0 | Historical |
| V4–V10 | Compression + suites | v0.4.0–v0.10.0 | Historical |
| **V11** | Launch hardening | **v0.11.0** | **Yes — cite this** |
| V12 | Deferred-work gauntlet | *(no tag)* | Internal docs only |
| V13 | Practicality proof | ~~v0.13.0-rc1~~ *(withdrawn)* | Internal docs only |
| V14–V21 | L3/L4 / integration ladder | *(no tags)* | Internal docs only |

**V11 is not “missing”.** It is **`v0.11.0`**, the public release. V12 and V13 continued in parallel in docs; they never supersede `v0.11.0` as the cited artifact.

## Evidence bundle “v3.0”

`RELEASE.md` uses **v3.0** for the **research evidence bundle** (technical report + 8,132 external cells + faithful appendix). That is independent of pip version `0.11.0`:

```text
ExactKV v0.11.0 — research release v3.0
```

## For reviewers

- **Do not** sort git tags by date and assume semver order.
- **Do not** treat `V13` docs as “version 13” of the product.
- **Do** use [GitHub Release v0.11.0](https://github.com/utkarshg20/ExactKV/releases/tag/v0.11.0) and [EVALUATOR_GUIDE.md](EVALUATOR_GUIDE.md).

Historical depth: [VERSION_LINEAGE.md](VERSION_LINEAGE.md), [PROJECT_LINEAGE.md](PROJECT_LINEAGE.md).
