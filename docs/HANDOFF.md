# Handoff (2026-07-17, end of day): parser+g3 recovered, build rewritten to base+sweeps

## One-line status

Kaya went down at **17:00 for migration** (hostname no longer resolves). The parser
(unlimited) and g3 arms are fully recovered, judged, and in the tables (error=0). g2
inference is partial (~36%). The big deliverable is a **from-scratch rewrite of the
table build** to the base+sweeps design, plus doc_type-collapsed summary tables.
**Nothing is committed** — the whole session is in the working tree for review.

## Data state (all local, judging needs no Kaya)

| arm | state |
| --- | --- |
| parser (unlimited) | **DONE** — 1585 ok / 109 oom / **0 error**, judged, in tables. |
| parser (paddle, mineru) | done (paddle = the representation run; mineru judged). |
| g3 hallucination | **DONE** — 2698 ok / 230 oom / 0 error, judged. |
| all G1 sweeps (representation, reasoner/size+family, quant, resolution, scanned halves) | done + judged. |
| g2 retrieval | **PARTIAL** — 5563/15246 (~36%) pulled locally; judging still running (`logs/g2_judge_now.log`, slow). |

The remaining `oom` cells (parser 109, g3 230, g2 ~1.25k) are genuine reasoner OOMs on
the 16 GB V100 — not fixable on Kaya; need an A100/H100 `--failed-only` sweep if/when
one is available.

## What happened today (recovery)

1. **Parser ParserCacheMiss (144 cells, 12 landscape docs) fixed.** Root cause was
   Unlimited-OCR **gundam crop-mode CUDA OOM** (not an aspect crash) — the worker
   swallowed it, so it surfaced downstream as `ParserCacheMiss`. Fix in
   `tools/parser_worker.py`: `_unlimited_markdown` now falls back gundam → base-1024 →
   base-640 and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`. Smoke (job 1063873)
   recovered 4/4. Recovered via a **normal resume** (job 1063881, `--skip-oom`, NOT
   `--failed-only`) after dropping the 144 error rows from the remote predictions
   (backup kept). 138 → ok, 6 → genuine reasoner OOM.
2. **Verdict-cache poisoning cleaned (HANDOFF §42 risk, hit for real).** The earlier
   judge chain had written stale `error`/`oom` verdicts for cells that recovery later
   turned `ok`, so a re-judge scored 0. Dropped the stale verdicts (backups kept) and
   re-judged: parser 144 rescored, g3 35 rescored. **Audited every arm** for this — only
   the two recovered arms (parser, g3) were affected; both fixed.

## The build rewrite (main deliverable)

The generation side moved to base+sweeps in the 2026-07-10 yaml-expander change, but
`reporting/build.py` was knowingly left routing by task identity. Rewrote it end to end:

- **`config.BASELINE`** — per-task baseline (one value per axis), the source of truth
  for held-fixed caption values.
- **`reporting/plan.py`** — a declarative registry: each analysis table's source
  run_tag(s), swept axis, builder, caption. Replaces `TASK_TO_TABLES`.
- **`reporting/tables/_load.py`** — cross-run loaders (scan-merge, parser/scale/quant
  merges) + the per-column-n footer helper.
- **`ops.build`** writes one CSV per table + a combined **`results/tables/all_tables.md`,
  flat** (no more `full-<run_tag>/` dirs). Each table carries a **caption** (swept axis +
  full held-fixed baseline, so it is explainable on its own) and accuracy grids carry a
  **per-column `n` footer**. Cross-run merges give a real 3-parser comparison and an
  all-specs scale table (the old per-task fragmentation is gone).
- **`ops/mine.py` folded in and deleted** (its `docs/generated/mined_tables.md` too).
- **Summary tables (markdown-only).** Nine long/doc_type-repeated tables also emit a
  doc_type-collapsed summary (headline, parser, resolution, quantization,
  scan_vs_digital, prefill_cost, vram_headroom, oom_frontier). The 217-row
  retrieval_accuracy detail is **hidden from the .md** (CSV kept) and replaced there by a
  compact best-F1-per-method summary. Summaries are `<key>_summary`, `md`-only.
- Condition-format fixes across 7 builders (the stale `condition=="oracle"` filters that
  silently fell back), rewrote `parser.build` (multi-parser) and `hallucination`'s
  prompt-mode parse, and escaped `|` in markdown cells (joint retriever names).
- **Per-column `n` footer on every table** (`-` where inapplicable, e.g. routing/retrieval
  metric columns; real per-level n on the grids). **scan_vs_digital** no longer shows an
  unlabelled column — `_load` backfills a blank `scan_label` from `annotations/auto_scan.csv`
  (every doc is labelled there; ~28 docs were generated before the auto-scan pass). And a
  **generation report** (cells / ok / oom / error / OOM% per run_tag, + a total) now heads
  `all_tables.md`, above the baseline preamble.
- **Frozen interfaces untouched** (build only reads caches/specs). **243 tests pass.**
- Docs: `docs/AGENT_GUIDE.md` now documents the generation + build paradigm;
  `docs/DECISIONS.md` has the changelog entry.

Run it: `python -m ops.build` → `results/tables/all_tables.md` + 18 CSVs.

## Background processes still running (harmless)

- `logs/g2_judge_now.log` (pid ~6396) — judging the local g2 (5563 cells), slow. Its
  final rebuild step calls the **old** `ops.build --task/--run-tag` CLI which no longer
  exists → it will log a harmless failure. **After it finishes, just re-run
  `python -m ops.build`** to fold g2's judged accuracy into the tables.
- `logs/g2_final_watch.log` (pid ~6318) — polling a now-dead Kaya; it will fail to pull
  and eventually give up. The g2 resume's extra cells (beyond 5563) are stranded on
  Kaya and lost to the migration unless Kaya returns.

## Uncommitted (this session, for review/commit)

Build rewrite: `config.py` (BASELINE), `ops/build.py`, `reporting/build.py`, new
`reporting/plan.py` + `reporting/tables/_load.py`, `reporting/tables/_{common,markdown}.py`,
the builder fixes/summaries (`headline, parser, resolution, scale, composition, routing,
hallucination, mined_*`), `tests/test_mined_and_guards.py` (stale fixture), deleted
`ops/mine.py` + `docs/generated/mined_tables.md`, docs (`AGENT_GUIDE`, `DECISIONS`).
Parser fix: `tools/parser_worker.py`. Regenerated `docs/generated/build_status.md`.
(Pre-existing at session start: `docs/G1_Representation.md` / `docs/G2_Retrieval.md`
deletions.) Scratchpad holds the transient recovery scripts.

## Next steps

1. Review + commit the working tree (build rewrite + parser fix).
2. When `g2_judge_now` finishes, re-run `python -m ops.build` for final g2 numbers.
3. When Kaya returns: an A100/H100 `--failed-only` sweep for the OOM cells (parser 109,
   g3 230, g2 ~1.25k) and, if wanted, finishing g2 inference (was ~36%).
4. Still-open known issue: interrupt-unsafe `--failed-only`
   (`experiments/engine/driver.py::_prepare_failed_only`) — we avoided it this session by
   using normal resumes; the recommended fix (in-memory skip set + merge) is unimplemented.
