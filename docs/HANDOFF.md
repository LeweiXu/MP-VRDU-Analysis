# Handoff: ops/scripts v4 fixes + annotation rework

Date: 2026-07-09. This picks up the loose ends after the v4 restructure, focused
on `ops/scripts/` and the manual-annotation workflow. Everything below is landed
and verified unless it's under "Open items".

## What changed

### Scripts fixed after the restructure

Three scripts failed to import because the v4 restructure moved `scripts/` into
`ops/scripts/`, dissolved `covariates/` / `metrics/` / `gates/`, and replaced
`doc_type`-derived binning with manual annotation.

- **`ops/scripts/annotate_docs.py`** — reworked. It imported the deleted
  `doc_type_bin` and `scripts.split_docs_by_type`. Now:
  - vocab comes from `data/annotations.py` (`BIN_LABELS` = text-dominant /
    mixed-modality / visual-dominant, `SCAN_LABELS`, `VISUAL_KINDS`), one source
    of truth.
  - the old `score` that compared human bins to a `doc_type`-derived `auto_bin`
    is gone (that assumption is exactly what v4 drops). `score` now reports label
    distributions plus human-vs-auto *scan* agreement.
  - `dominant_visual` is optional: `annotate --no-dominant-visual` skips the
    prompt, and it never gates row completion (only `bin_label` + `scan_label`
    do). Matches the pivot's "exploratory only" note.
  - new Cohen's kappa flow: `kappa-sheet --n 25 --seed 0` draws a subset from the
    labelled sheet and blanks the labels (so a second annotator is blind);
    `annotate --sheet …` fills it; `kappa` reports kappa per field vs a 0.75 gate
    with the specific disagreements.
- **`ops/scripts/split_docs_by_type.py`** — dropped the deleted `DOC_TYPE_TO_BIN`
  import (was only used to print an "assumed bin" per type). PDF-splitting for
  browsing is intact.
- **`ops/scripts/inspect_results.py`** — inlined the retired `gates/viewer.py`
  (the plan said inspect_results absorbs it). Adjusted for v4: `ResultRow` now in
  `schema.py` (rebuilt via `ResultRow.from_dict`), `load_result_rows` moved to
  `reporting.build` and returns dicts, `experiment_paths` in
  `experiments.engine.paths`. Also surfaces `bin_label` / `scan_label`.
- **`ops/scripts/dump_docstrings.py`** — ran fine but its `SUMMARY_OVERRIDES` map
  was 100% stale (pre-v4 paths) and its two still-existing keys injected
  "v4 should…" roadmap text into the auto-generated `REPO_STRUCTURE.md`, which the
  repo rule forbids. Emptied the map so the file map builds from live docstrings.
  Fixed two stale `scripts/…` path strings too.

### Dead code stripped

- **Deleted `ops/scripts/run_probe.py`** — the v3 Stage-1 feasibility CLI. Its
  probes included `boxes` (bounding-box channel) and `doc-type` distribution, both
  abandoned in v4; the surviving GPU feasibility check is `resolution_probe.py`.
  Cleaned every live-tree reference: `ops/scripts/__init__.py`, `ops/kaya/kaya.py`
  epilog, `README.md`, both `ops/kaya/KAYA_*_GUIDE.md`, and the `docs/DECISIONS.md`
  copy-list. While in the agent guide, also corrected a stale `--skip-tool-caches`
  flag to the real `--skip-parsers`.

### prestage verified (no change)

`ops/scripts/prestage.py` is already v4-adapted: current-only docstring, correct
`ops.kaya` / `ops.scripts` imports, and `ops/kaya/config.json` stages exactly the
v4 set — Qwen3-VL 2B/4B/8B + InternVL3-8B, the text/vision retrieval rungs
(bge-m3, Qwen3-Embedding-4B / colmodernvbert, colqwen2.5, colqwen3-4B), and the
three parsers (PaddleOCR-VL, MinerU2.5, Unlimited-OCR). All three `download_hf`
helpers resolve; `--help` works. Left untouched.

### Docs

- **`docs/ANNOTATION_GUIDE.md`** — moved here from `ops/scripts/` and rewritten for
  v4: new bin vocab with the modality-dominance definitions (incl. the scanned
  handwriting = text-dominant case), `ops.scripts.` module paths, the
  `docs/requirements/annotate.txt` install path, dropped the removed doc-type
  scoring, documented `--no-dominant-visual`, and a section on the blind-subset
  Cohen's kappa workflow.
- **`docs/DECISIONS.md`** — copy-list table updated with the outcome of this pass
  (reworked / verified / removed, dated 2026-07-09).

## Annotation loader is now strict (context for whoever continues)

`data/annotations.py::load_annotations` and `data/binning.py::stamp_bins` were
tightened (not by me, but relevant): once `annotations/doc_labels.csv` exists it's
authoritative. A missing required column (`doc_id` / `bin_label` / `scan_label`)
or an out-of-set label raises; blank-`bin_label` rows are skipped as in-progress;
and `stamp_bins(..., require_complete=True)` raises if any corpus doc is
unlabelled. So a partial or malformed sheet fails loudly rather than silently
binning blank. The `annotate_docs` output sheet is compatible with this (verified
end-to-end).

## Verification

- All 12 remaining `ops/scripts/*.py` + the package import cleanly.
- `annotate_docs` all five subcommands `--help` OK; kappa math unit-tested against
  a known value; blind-subset + kappa flow tested end-to-end; output sheet loads
  through `load_annotations`.
- `inspect_results` imports and `--help` runs.
- `dump_docstrings.build_section()` runs with no roadmap leak.
- 151 tests still collect; none reference the changed/removed scripts.

## Open items (flagged, not done)

- **`docs/pivot_v4_implementation.md`** still lists `run_probe` as
  "copied-pending-rework" in two spots (lines ~244, ~355). That's the frozen plan;
  the removal is logged in `DECISIONS.md` instead. Update the plan text only if you
  want it to match reality.
- **KAYA guides + README repo-map are broadly pre-restructure stale.** Both
  `ops/kaya/KAYA_*_GUIDE.md` still use `kaya.kaya` (vs `ops.kaya.kaya`) and
  `scripts/` (vs `ops/scripts/`) prefixes throughout; `README.md`'s repo map still
  lists `gates/` and `cli/`. I only touched the run_probe lines. A dedicated doc
  reconcile pass (the plan's final-cleanup step) is still needed.
- **`docs/REPO_STRUCTURE.md` not regenerated.** After emptying the
  `dump_docstrings` overrides, run `python -m ops.scripts.dump_docstrings` to
  refresh the per-file map when you want it updated.
