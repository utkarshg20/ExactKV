# Changelog v2.4.1 → v2.4.2

**Date:** 2026-06-27

Polish-only pass. No new GPU runs, no claim changes.

## Changes

1. **Reproducibility appendix expanded** (§17 in `.md`, Appendix in `.tex`):
   - Added external smoke / MBPP / analysis-pack / validator / panel-summary commands.
   - Explicit subsections: verification, headline, evidence-plus, external Llama-only, MBPP both-model.
   - Added `mbpp_gpu_raw.json` as a named source of truth.

2. **External tables: columns added** (Tables 4 / 4b in `.md`; Tables `external-smoke` / `external-mbpp` in `.tex`):
   - Added Model, Context (K), MNT (max_new_tokens), compressor set to both tables.
   - Table captions explicitly note compressor set: `noop/int8/int4_sim`.

3. **Leaderboard score table added** (§4 in `.md`, after validated-compressors in `.tex`):
   - 6-row table from `reports/public_release/leaderboard_final.json` (built-in compressors only).
   - `spectralquant` and `shard` explicitly excluded from ranked table with note.
   - Formula retitled "leaderboard-style" throughout.

4. **Abstract/positioning phrasing softened**:
   - "leaderboard framework" → "evaluation crash-test framework with leaderboard-style reporting"
   - Applied in abstract (`.md` §1, `.tex` abstract), Positioning section, §13 Novelty.

## Target rating

Strict research draft: **8.3/10** (polish closes presentation gaps, no evidence change).
