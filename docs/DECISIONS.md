# DECISIONS — the changelog

This is the **only** doc that carries history. Every pivot, superseded design, and
real judgement call is recorded here, newest first (one line each: what, why, what
it affected). `README.md` and `docs/AGENT_GUIDE.md` describe the system as it is
now, present tense; anything about how it *changed* lives here. See `CLAUDE.md` for
the documentation discipline.

Pivots are folded in here once implemented: the standalone `pivot_v4.md` and the
v5 pivot notes (binning + the G4/routing collapse) are superseded by the entries
below and should not be kept as separate live files.

**RQ-aligned table mining: four new builders, RQ sectioning, gap specs (2026-07-19).** Added `integration` (accuracy by evidence hop, `hop=none` dropped as answerable-with-no-gold-pages rather than unanswerable), `fidelity_transition` (the paired within-question TL→TLV and T→TLV verdict transitions by evidence source, paired n 717/716 — the lead RQ1 number), `attribution` (representation/retrieval/reasoning split, stamped PROVISIONAL because its retrieval column reads the ~36% G2 pool, and its reasoning residual is emitted as a raw uncorrected upper bound), and `source_stratification`. That last one is **deliberately degenerate**: `metadata.source_dataset` is the loader's dataset id (hardcoded `"mmlongbench"`, `data/loader.py:170`), not upstream QA provenance, and MMLongBench-Doc publishes none, so the inherited-vs-native memorisation channel is *unmeasurable* rather than measured-and-null; the table says so instead of implying a null result, and it gets no `_summary` because it has no doc_type axis to collapse. `all_tables.md` now renders in four sections (RQ1/RQ2/RQ3/Appendix) via an `rq` field on `AnalysisTable` and `Table`, defaulting to Appendix so an unmapped table is retained, never dropped; CSV paths and every existing builder's logic are unchanged. Surfaced the two known discrepancies as captions (G2's `bge-m3` inference arm vs `BASELINE`'s `bm25`; G3's "similarity" label vs the emitted `retrieved_text_k3` base). Added five specs under `ops/specs/`: three runnable (`g1_prompting_answerable`, `g1_failed_only_a100`, `g2_retrieval_full_rerun`) and two `_target_`-prefixed proposals that name the pipeline change and frozen interface they would touch and implement none of it. 32 tables build clean, 243 tests pass, frozen interfaces untouched.

**`CODEBASE_GUIDE.md` + de-silenced oracle-filter fallback (2026-07-19).** Renamed the paper-facing `docs/methods_appendix.md` to `docs/CODEBASE_GUIDE.md` and added an operational layer (row schema, cache layout, the plan-driven build + table inventory + how-to-extend, current data state, spec-only vs code-change axes) for a non-coding collaborator; the methods content is preserved as Part B. Audited the table builders for the silent-pool class: the `[condition=="oracle"] or list(rows)` idiom (and the `is_unanswerable or list(rows)` twin) now route through `_common.rows_for_condition`/`unanswerable_rows`, which warn and name a condition-format drift before falling back instead of quietly pooling every condition. Behaviour-preserving (oracle rows exist, so `all_tables.md` is byte-identical and 243 tests pass); the earlier scan-label backfill and `restrict_to_primary_spec` fixes are verified still clean. Touched `_common.py` + 10 builders.

**Build rewritten to the base+sweeps design; one explainable table per variable (2026-07-17).**
The generation side moved to base+sweeps (one variable off a fixed baseline) back in the
yaml-expander change (2026-07-10), but `reporting/build.py` was knowingly left routing tables
by *task identity* (`TASK_TO_TABLES`), so every G1 run emitted the same six tables regardless
of what it swept (the reasoner run's `parser.csv` was meaningless), non-swept axes were stamped
with `config` scalar defaults, output fragmented into `results/tables/full-<run_tag>/` dirs, and
several builders (`parser`/`routing`/`hallucination`) silently misfired on the current
`condition` format (`oracle__none`, matched by the `or list(rows)` fallback). Rewrote it
plan-driven: a new `reporting/plan.py` maps each analysis table to its source run_tag(s) + swept
axis + builder; `config.BASELINE` holds the per-task baseline as the source of truth; `ops.build`
writes one CSV per table plus a combined `results/tables/all_tables.md`, flat. Every table now
carries a caption stating its swept axis and the held-fixed baseline (so results are explainable
on their own) and accuracy grids carry a per-column `n` footer (columns differ under OOM). Tables
comparing out-of-key axes merge across run_tags (parser paddle+mineru+unlimited; digital+scanned
scan-merge). Added `reporting/tables/_load.py` (cross-run loaders + footer), condition helpers
(`split_condition`/`base_condition`) in `_common`, and folded `ops/mine.py` (deleted, with its
`docs/generated/mined_tables.md`) into the build. Fixed a stale `prompt-<mode>` condition in
`tests/test_mined_and_guards.py`. Build only reads caches/specs; frozen interfaces
(`schema.py`, `pipeline/` ABCs, orchestrator cache key/`ResultRow`) untouched; 243 tests green.

**Judge re-judge no-op + coverage line, reporting guards, mined tables (2026-07-14).**
For the judge+build+mine pass over the near-complete cache: (1) `ops/judge.py::judge_run`
now checks the result cache **before** calling `judge.score()` on an `ok` cell, so a
re-judge of an already-scored cell makes no Gemini call (a true no-op top-up after the
pending reruns land, not just a zero-write); it also prints a per-run coverage line
(cells/ok/oom/err + answerable/unanswerable). (2) Reporting safeguards: a shared
`restrict_to_primary_spec` (`reporting/tables/_common.py`) keeps the single-reasoner
tables (headline/parser/resolution/composition/routing) from silently pooling a
multi-`model_spec` sweep into one accuracy cell; `scale` (the model-size/quant sweep)
keeps every spec. `scale` and `routing` gained a clean `prefill_ms` column with a
caveat that `latency_ms` is decode-inflated (~20x by the verbose-answer change). (3)
Six `mined_*` deployment tables (`reporting/tables/mined_*.py`) driven by `ops/mine.py`
(kept out of the task->table routing so they never misfire on the wrong run_tag), with
a candidates summary at `docs/generated/mined_tables.md`; the H100-dependent
evidence-survival table is defined-but-blocked. Additive only; frozen contracts and
cache keys untouched.

**g3 `<image>` sentinel collision fixed (2026-07-14).** g3 recorded 18 errors on one doc
(`2306.05425v1.pdf`, a VLM paper): its text literally contains `<image>`, which is also the
`IMAGE_PLACEHOLDER` sentinel, so the backends' placeholder-count-vs-image check rejected the
prompt (`prompt has N image placeholders but M images`), even in text-only rungs. Fix:
`ModelInput.to_local_prompt` (`models/payload.py`) now replaces a literal `<image>` in
document text with `[image]` before inserting real sentinels, so only true image slots are
counted (covers both qwen3vl and internvl local backends). Behaviour fix inside the frozen
`ModelInput`; signature unchanged and the prediction cache key does not include prompt text,
so no cached cells are invalidated. Regression: `tests/test_image_placeholder_collision.py`.

**InternVL3-8B einops fix + failed-cell rerun specs (2026-07-14).** The post-run sweep found
InternVL3-8B produced zero valid cells (3388) because its remote modeling code imports `einops`,
which was absent from the Kaya `core` env. Added `einops==0.8.2` to `docs/requirements/core.txt`
and installed it into `envs/core`; verified via transformers `check_imports` (no submit). New
specs `kaya_failed_rerun.yaml` (the 4 failed run_tags, verbatim configs, run with `--failed-only`)
and `kaya_failed_rerun_smoke.yaml` (same, 1 question each, isolated `*-smoke` run_tags, run fresh).

**Centralised Tier-1/2 experiment knobs into `config.py` (2026-07-13).**
Followed up the qwen3-seq-cap move. Now single-sourced in `config.py`, with the old
homes importing back: the **representation ladder** `("T","TL","TLV","V")` as
`REPRESENTATION_LADDER` (was re-declared in `scoring/frontier.RUNG_ORDER`,
`pipeline/representation.RUNGS`, and the `ExperimentConfig.representations` default), and
the **modality bins** as `DEFAULT_BINS` (`data/annotations.BIN_LABELS` now imports it;
`data/binning.BINS` already did). Science params moved out of modules into `config.py`:
bootstrap `N_BOOTSTRAP`/`BOOTSTRAP_SEED`/`BOOTSTRAP_CI_LOW`/`_HIGH` (`scoring/accuracy`),
`ABSTENTION_FORMS` (`scoring/abstention`), `SCANNED_MIN_CHARS_PER_PAGE` (`data/render`), and
`JUDGE_SYSTEM_PROMPT` + `JUDGE_GPT_MODEL`/`JUDGE_GEMINI_MODEL` (`pipeline/judge`). No cycle:
`config.py` imports only stdlib. Tier-3 (vision embed knobs, model-ID registry, prompt
headers) deferred. Frozen interfaces untouched (only defaults moved).

**Cheap visual-retrieval DPI sweep (2026-07-13).** New retrieval-only study: re-embed and
re-rank pages at several render DPIs, compare page P/R/F1. `RetrievalEvalRow` gained a `dpi`
field (set from `config.dpi`); `reporting/tables/retrieval_accuracy.build_by_dpi` builds a
`retrieval_dpi` table grouped by (retriever, k, dpi), wired into `G2_retrieval`. Old rows
get `dpi` backfilled from `config.dpi` at build (`_enrich_retrieval_rows`, alongside the
doc_type backfill). `complete_retrieval.py --parser-dpi N` overrides the render DPI so one
spec sweeps DPIs (each keys its own memo, so runs never collide). Spec: `kaya_g2_dpi.yaml`
(vision-only; methods come from `--vision-methods`). Cheap because retrieval-only (no
reasoner, no judge), vision-only (text spans are dpi-independent), and lower DPI is faster
than the 200 baseline. Caveat: the colqwen processors resize internally, so the informative
range is low→moderate DPI.

**`check_run.py`: `--check-all` renamed to `--all`, skips template/smoke specs (2026-07-13).**
The all-specs sweep now filters out any `ops/specs/*.yaml` whose name matches
`(template|smoke)` (regex `ALL_SKIP_RE`), since those aren't real runs. Flag renamed
`--check-all` → `--all` (dest still `check_all`).

**Added `retrieval_accuracy_overall` table (2026-07-13).** A second retrieval table
alongside `retrieval_accuracy`, grouped by (retriever, modality, k) only (no doc_type
split), one macro P/R/F1 row per method/k over all 847 questions. `build_overall` in
`reporting/tables/retrieval_accuracy.py`, wired into `G2_retrieval`'s table set. The
per-doc_type `retrieval_accuracy` table is unchanged.

**Retrieval side-artifact carries `doc_type`; old `retrieval.jsonl` backfilled at build (2026-07-13).**
The 2026-07-12 rename grouped `retrieval_accuracy` by the native `doc_type`, but
`RetrievalEvalRow` (`scoring/retrieval.py`) only ever carried the modality `bin_label`,
so every retrieval row read back a blank `doc_type` and the whole table collapsed into
one `(unknown)` bucket per method/k. Fix: added a `doc_type` field to `RetrievalEvalRow`
(set from `question.doc_type`, so fresh `retrieval.jsonl` carries it; `asdict` picks it
up), and `reporting/build.py::_backfill_retrieval_doc_type` fills a blank `doc_type` on
older rows by joining `doc_id → doc_type` from the corpus (best-effort; if the dataset
can't load, rows stay `(unknown)` and the table still builds). The `g2-retrieval-full`
table now breaks out all 7 classes (n sums to 847 per method/k). Watch-out for the
manual memo fold-in: the qwen3-embedding regen (`retrieval_qwen3.jsonl`) also re-ranked
`colqwen3`, which is already in `retrieval.jsonl`, so a blind `cat >>` double-counts
colqwen3 at k=1,3,5; fold in only the `qwen3-embedding*` rows.

**`ops.build --run-tag` + kaya.py split into `ops/kaya/runner/` (2026-07-12).**
- **`ops.build` gained `--run-tag`.** It built from `ExperimentConfig()` (un-tagged
  cache) and so found nothing for a run-tagged generation; `--run-tag g1-representation-full`
  points `config.paths.cache_dir` at `results/cache/<run_tag>/…`. Verified end to end on
  the finished G1 representation `results.jsonl` (headline/parser/resolution/scale/routing/
  composition tables assemble). Table `n` is the per-group cell count (`len(rows)`),
  shared across builders; per-column accuracy still uses the per-rung counts.
- **Tables now group by the native mmlongbench doc_type (7 classes), not the modality
  bin.** `reporting/tables/_common.py`: `bin_of`/`ordered_bins`/`BIN_ORDER` became
  `doc_type_of`/`ordered_doc_types`/`DOC_TYPE_ORDER` (Academic paper, Administration/
  Industry file, Brochure, Financial report, Guidebook, Research report / Introduction,
  Tutorial/Workshop; unknown last). The `bin` column across headline/parser/resolution/
  matched_cross/retrieval_accuracy is renamed `doc_type`. The modality `bin_label` still
  drives `per_bin` sampling and annotation; only the report grouping changed.
- **`ops/kaya/kaya.py` (1200+ lines) split into an `ops/kaya/runner/` subpackage**, one
  module per slice: `config` (dataclasses/constants/quoting), `remote` (ssh exec + the
  prelude/env exports), `sync` (push/pull), `sources` (`# kaya:` headers + repo-local file
  resolution + spec run_tag), `slurm` (sbatch gen/submit), `jobs` (squeue/wait/logs/cancel
  listing), `status`, and `commands` (the handlers). `kaya.py` is now just the parser +
  dispatch and re-exports `load_config` / `spec_arg` / `KayaConfig` / `push` / `pull` for
  existing importers. `ops/scripts/kaya_status.py` folded into `runner/status.py` and is
  reachable as `python -m ops.kaya.kaya status` (test import updated to the new path).

**`--skip-oom`, qwen3-embedding batch=1, and V100 walltimes corrected (2026-07-12).**
Three changes from sizing the full runs for the Kaya migration deadline:
- **`--skip-oom`** (`ops/generate.py` → `driver.generate(skip_oom=...)`, and `g2_rerun.py`
  passes it through). Drops every cell already recorded `oom` from the run, prewarm
  included. A cached oom row is a cache hit at inference anyway, but the driver's prewarm
  still hit `render_pages` + the isolated parser for TL/TLV cells on every resume; this
  cuts that re-parse. It is the resume counterpart to `--failed-only` (which retries oom
  cells); do not pass both. `_oom_cell_ids` (unlike `_prepare_failed_only`) does not
  rewrite predictions.jsonl.
- **qwen3-embedding encodes at `batch_size=1` + `max_seq_length=4096`.** First tried
  `batch_size=1` with the cap removed (aiming for no truncation), but a memo-regen job
  (1033382) OOM'd on a dense page after ~24 questions: batch=1 bounds the batch dimension,
  but attention is O(seq^2), so one long page still spikes past the fp16 weights on its own
  forward. Restored the 4096 cap (fits one V100 at batch=1 with headroom; only the rare
  very long page truncates). Only affects the qwen3-embedding retrieval memo build (stage
  1), not inference (bge-m3 / colqwen2.5). The cap and encode batch now live in `config.py`
  (`QWEN3_EMBEDDING_MAX_SEQ_LEN` / `QWEN3_EMBEDDING_ENCODE_BATCH`), imported by
  `retrievers/text.py`, instead of being hard-coded in the module. Dense-retriever memo rows now also carry
  truncation telemetry (`seq_len_cap`, `page_token_lens`, `truncated_pages`) so you can see
  which pages the cap clipped; the fields are additive (the memo loader ignores unknown
  keys) and populate only for embedders exposing an HF tokenizer (qwen3-embedding, not
  bge-m3's wrapper).
- **Retrieval benchmark isolates failures per question, records them, and can rerun from
  scratch.** `write_retrieval_eval` used a per-*method* try/except, so one question's OOM
  dropped the whole method. Now a load failure still skips the method, but once loaded each
  question is ranked independently: an OOM records a memo status row (`status` +
  `skipped_reason`, empty `ranking`, written once via `MemoizedRetriever.persist_failure`)
  and the loop continues; failed questions are left out of the scored rows. `fresh=True`
  (`complete_retrieval --fresh`, `g2_rerun --fresh-complete`) deletes each method's memo
  first so the rung re-ranks uniformly (no mixing capped/uncapped rows). On the inference
  side, `build_retrievers(reuse_only=…)` (set on a `--skip-retrieval` pass) makes a memo
  miss raise `RetrievalMemoMiss` instead of silently re-ranking, so the inference cell is
  recorded as a failure (with the reason, carrying any earlier retrieval failure) and
  rides on — keeping failures self-contained and rerunnable rather than guarding the run.
- **README walltimes rebased on observed rate (~33 s/8B cell, wall-clock incl. prewarm),
  replacing the old ~18 s/cell guess that underestimated.** Added the V100 no-FlashAttention
  note and the recommendation to run G2 inference on the supervisor H100 (V100 image cells
  are ~45-57 s and ~15k reduced-k cells is ~130 h).

**Generate/judge split: predictions.jsonl unjudged, results.jsonl is the judge's
(2026-07-11).** Generate no longer scores inline. This supersedes the behaviour where
`Orchestrator.run_cell` ran a (stub) judge and wrote a fully-scored `results.jsonl`
while `predictions.jsonl` held only ok cells.
- **Generate writes one unjudged file.** `run_cell` builds a new
  `schema.PredictionRow` (answer + all covariates + telemetry + `status`, no judge) and
  the driver writes it to `predictions.jsonl`, one row per cell **including failures**.
  The inline judge, `ResultCache` in generate, and the `judge_spec` spec key are gone
  (`judge_spec` dropped from `yaml_spec.py` and every `ops/specs/*.yaml`); `ops.generate`
  lost its `--judge-spec` flag.
- **`ops/judge.py` is now real** (was a no-op that logged counts). `--spec <yaml>`
  resolves the run_tag + dataset, loads the corpus for gold answers, scores each ok
  prediction with `--judge-spec` (stub default; gemini / gpt-4o-mini also), and writes
  the full `ResultRow` to `results.jsonl`. Failed cells pass through unscored (score 0),
  so `results.jsonl` is a **strict superset** of `predictions.jsonl`, row for row.
  Deduped on `result_key`, so a second judge writes disjoint rows and re-judging is
  resumable. Loop mirrors the retired `old/experiments/artifacts.py`.
- **Frozen interfaces intact.** `PredictionRow` = `ResultRow` minus
  `{result_key, judge_spec, score, correct, abstained}`;
  `PredictionRow.to_result_row(score, result_key)` builds the judged row. `ResultRow`'s
  shape and the `prediction_key` / `result_key` formulas are unchanged — only the
  caching contract's *content* grew (predictions.jsonl now stores the richer
  `PredictionRow`, replacing `CachedPrediction`; results.jsonl is written by judge, not
  generate). Removed the dead `generate_results.jsonl` path.
- **Fallout.** `check_run` reads `status` from `predictions.jsonl`; `inspect_results`
  reads `PredictionRow`; `final_probe` runs a stub judge step before the table build.

**Flat spec format + one unified spec-driven task (2026-07-11).** The generation
layer was collapsed to a single mechanism, and `pivot_v4.md` is folded in here and
deleted. This supersedes the "Per-sweep YAML expander (2026-07-10)" entry below.
- **One pipeline, `task_name` is a label.** `G1OracleLadder` / `G2Retrieval` /
  `G3Hallucination` are gone; `experiments/tasks/task.py::Task` is the only
  generation task. It reads its behaviour from the config: `pool` (answerable /
  unanswerable), `retrieval_representation` (`oracle` = gold pages via
  `OracleConditioner`; else the text/vision inference arms), the ladder, `k`, and
  `prompt_modes`. `task_name` names only the cache dir + parallel job, not a type.
  The registry resolves any label to `Task(name)`.
- **Flat, fully-explicit specs — no `base`, no `sweeps`, no `task`.** The nested
  base/sweeps + `retrieval`/`inference` form is replaced: every run lists the full
  variable set explicitly under a `task_name`, and a list-valued axis is the set of
  values to run over (cross-product). `dataset`/`parser` expand to one run_tag each;
  `reasoner_spec` x `quantization` fold into `reasoner_specs`; `visual_resolution`
  becomes the driver-looped list; representations / k / prompt_modes are cell
  dimensions. New keys: `corpus.pool`, `parser_dpi` (was `dpi`),
  `retrieval_representation` (in {T, V, oracle}), merged benchmark method lists
  (`text_retrievers` / `vision_retrievers` / `joints` at top level, no separate
  `retrieval:` block). `ops/specs/template.yaml` is the reference menu + the three
  worked tasks; all specs rewritten; the old nested specs deleted.
- **Pool is spec-driven, not task-bound.** `config.pool` (from `corpus.pool`)
  replaces `pool_for_task(name)` / `UNANSWERABLE_TASKS`.
- **Enforcement.** A run whose benchmark lists are non-empty must include `bge-m3`
  and `colqwen2.5` (the fixed inference arms), and any inference pick must be a
  benchmarked method (`SpecError`).
- **Prompt mode rides the conditioner name.** The prediction key has no prompt
  field, so the mode is appended (`retrieved_text_k3__none`, `oracle__none`);
  answerable G1/G2 runs use `none` (not the old `targeted` default) per the specs.
- **Oracle is a retriever too.** `retrievers/oracle.py::OracleRetriever` returns the
  gold pages, for uniformity / a perfect-retrieval reference; reasoner oracle cells
  still select via `OracleConditioner` (all gold pages, no top-k).

**G2 retrieval stage-drift fixed + BGE-M3 inference arm (2026-07-10/11).**
- **Retrieval benchmark is stage 1, before inference.** It was written *after* all
  reasoner cells and rebuilt every method from raw retrievers; now it runs first
  (gated on `config.text_retrievers`), persists each method's ranking to the shared
  memo (`MemoizedRetriever`, `<cache>/retrieval/`), and the inference arms reuse
  those rankings instead of ranking twice. `retrieval.jsonl` is written incrementally
  (per method, flushed) instead of buffered to one `"w"` at the end.
- **Inference text arm bm25 -> bge-m3** (G2 specs); the vision arm builds with
  `allow_text_fallback=False` so a load failure is an honest miss, not a silent
  order ranking. `build_retrievers` tolerates `none` arms (oracle / vision-less runs).

**How this drifted from `pivot_v4.md` (folded, then deleted).** The pivot still
framed three tasks (G1/G2/G3) with a `base` + `sweeps` YAML and G2-specific
`retrieval:`/`inference:` blocks (pivot §7). The implementation went further toward
the pivot's own "few tasks, one pipeline" principle: there is now exactly one task,
and specs are flat and fully explicit (no fallbacks), which the pivot's staged
plan did not describe. The pivot's science (cost-ordered ladder, bins, RQs,
retrieval cost rungs, telemetry, answerable/unanswerable split, machine split = the
retry) is unchanged and now lives in `README.md`; the code structure lives in
`docs/AGENT_GUIDE.md`.

**Per-sweep YAML expander wired (2026-07-10).** The nested `base` + `sweeps` (and G2
`retrieval` / `inference`) form in `ops/specs/target_architecture.yaml` is now
expanded end to end by `experiments/corpus/yaml_spec.py`: one flat `Spec` per sweep at
parse time, so the driver is unchanged. One name per axis (scalar in `base`, list in a
sweep); precedence sweep > task `base` > file `base`.
- **run_tag strategy is dictated by the frozen cache key.** Axes in the key (reasoner
  spec incl. quant suffix, `visual_resolution`) sweep under ONE run_tag as a
  driver-looped list; axes NOT in the key (`parser`, `dataset`) get one run_tag per
  value (`<base>-<sweep>-<value>`). This is a deliberate deviation from the HANDOFF's
  `parsers:`/`datasets:` list idea, forced by the frozen key (parser/dataset are not
  in it, so a shared run_tag would collide / mix corpora). Quant folds into
  `reasoner_specs` suffixes (`bf16` = no suffix); a G3 `prompt` sweep folds into the
  single run's cells (no extra run_tag).
- **Config-driven, no longer hardcoded:** G2 benchmark method sets
  (`text_retrievers` / `vision_retrievers` / `joints`, `joints: matched` = zip of the
  two lists), the inference retriever picks (`driver.build_retrievers` via the
  registries) validated as a subset of the benchmark lists, `inference_representations`
  / joint on-off / `joint_k_values`, G3 `prompt_modes`, and the `dataset` -> loader map
  in `ops/generate.py` (mmlongbench / longdocurl). Flat specs (`kaya*.yaml`) are
  unchanged: `matched` reproduces the old joint tier-pairs and the inference defaults
  match the old module constants. Cache keys untouched (additive config only).
- **`per_doc_type: N` now means EXACTLY N questions per doc_type label** (seven labels,
  so `per_doc_type: 1` = seven questions). Previously the doc-coherent draw kept whole
  documents and overshot N. The draw still selects whole documents, then caps to
  exactly N, which can slice the last drawn document — a partial-document break the
  plain draw never did. Kept for the exact-count requirement; **caveat for the
  doc-level bootstrap on small N** (flagged for confirmation).

**Vision fixes validated; persist-cache poisoning found (2026-07-10).** Re-smoke (job
1026235, 5 min) after staging the colpali bases + the fp16 fix: the retrieval
side-artifact now writes real rows for `colqwen2.5`, `qwen3-embedding` (fp16, no OOM),
`bm25`, `bge-m3`, and the mid joint `bge-m3|colqwen2.5` (4/6 methods + 1/3 joints).
Two follow-ups: (a) `colmodernvbert` still fails with its base staged: its adapter repo
does carry processor/tokenizer files, but the base is a custom `modernvbert` model_type
whose text_config points at `ettin-encoder-150m`, so `from_pretrained` still reaches for
an uncached repo offline (tracked in the env/prestage handoff). (b) **MemoizedRetriever
persist-cache poisoning:** the pre-fix run 1026182 persisted colqwen2.5 *fallback*
rankings to `results/cache/g2-doctype50-smoke/retrieval/colqwen2.5__dpi200.jsonl`; the
re-smoke read those back (cache hit) instead of recomputing, so the smoke's G2 inference
vision cells still used stale fallback pages (only k=1 matched the fresh benchmark
ranking). The real run's `g2-doctype50` retrieval cache is separate and empty, so it will
compute real colqwen2.5 rankings. Cleared the smoke's poisoned retrieval cache to avoid
confusion. Underlying design risk (persisting a silent fallback ranking indistinguishably
from a real one) is noted for the fallback fix in the handoff.

**Kaya smoke diagnosis + vision-retriever fixes (2026-07-10).** First full smoke
(job 1026182, limit:2, 2B) passed the health gate (G1 18 / G2 36 / G3 6 cells ok, 0
failures), but the six-method retrieval side-artifact only produced bm25 + bge-m3.
Root causes found in the job log: (1) the ColPali vision rungs are PEFT **adapters**
whose **base models were never prestaged** — `vidore/colqwen2.5-base` and
`ModernVBERT/colmodernvbert-base` — so `from_pretrained` failed offline for both
colqwen2.5 and colmodernvbert (and colqwen2.5 silently fell back to text/order in the
inference pre-pass, so the G2 vision arm was bogus too); (2) `qwen3-embedding-4B` OOM'd
a 16 GB V100 because SentenceTransformer loads fp32; (3) `colqwen3` (Ops-Colqwen3-4B)
has **no class in the installed colpali_engine** (only ColPali/ColQwen2/ColQwen2_5/
ColModernVBert/ColIdefics3 exist). Fixes: added the two base models to
`config_minimal.json` and re-staged; load Qwen3-Embedding in fp16 on CUDA
(`retrievers/text.py`). colqwen3 is left to fail-fast-and-skip (needs a colpali_engine
upgrade or a compatible repo) — the ladder keeps 5/6 methods (both cheap+mid vision
rungs, all three text rungs) and the cheap+mid joints; the expensive-vision rung and
expensive joint are dropped for now. My colmodernvbert/colqwen2.5 class-name guesses
were correct; only colqwen3's was not (no such class).

**Docs consolidated to three (2026-07-10).** Reduced the authored docs to
`README.md` (user + experiment), `docs/AGENT_GUIDE.md` (agent: structure, frozen
interfaces, implementation reference), and this changelog. The former
`PROJECT_SPEC.md`, `USER_GUIDE.md`, `ANNOTATION_GUIDE.md`, and the standalone
`pivot_*.md` files are folded into these three; the Kaya runbook lives at
`ops/kaya/KAYA.md`. Cause of the prior contradictions: `AGENT_GUIDE` described a
"v3 now, v4 pending" migration state that rotted once v4 shipped, while `README`
described shipped v4 — so they disagreed. Fix: present-tense-only in the two
authored docs, single-source-of-truth per fact, and history confined to this file
(rules added to `CLAUDE.md`).

**Open (⚠ PENDING v5):** (a) **Binning source** — manual annotation is optional and
not the working default (see the 2026-07-10 entry below); the direction is to bin
by representative document domains or by `evidence_source`, and to define which
experiments need the full corpus vs a frozen random subset. (b) **G4 / routing** —
LANDED 2026-07-10 (entry below): G4 is removed as a task, routing is fully
build-time over G1, and the classifier is folded into G3 as an optional one-shot
side artifact. Still open: predicted-domain *routing accuracy* (routing uses the
classifier only for its latency price so far). (c) **Dependencies** — evaluating the
remaining strip (the vLLM-drop verdict is already recorded below). These are
tracked here; the two authored docs describe only shipped behaviour and mark these
spots `⚠ PENDING v5`.

---

## G4 folded into G3 (2026-07-10)

**G4 removed as a generation task; the classifier becomes G3's optional side
artifact; routing stays fully build-time.** Routing accuracy needs no inference of
its own — every policy (uniform-cheapest/strongest, oracle, type-aware) is a
selection over G1's already-cached ladder rows — so the only GPU work routing ever
needed was the modality-bin classifier. That one-shot pass now rides on G3 (a small
reasoner task with a spare `run_side` slot) instead of a standalone
`G4_classifier_pricing` task.

- **Task set is now G1/G2/G3** (`experiments/registry.py`);
  `experiments/tasks/G4_classifier_pricing.py` deleted. `reporting/build.py` reads
  `classifier.jsonl` from G3's side dir, and G3 feeds the `routing` table.
- **Classifier scope.** It prices G1's answerable doc set (answerable pool +
  `per_doc_type` sample), not G3's unanswerable cells, because routing only ever
  routes G1's documents. New `config.classifier_spec` (spec key `classifier`)
  enables it; `none`/unset skips it and routing reports the gold-bin ceiling only.
- **Driver change.** `run_side` now receives the full corpus + the smoke `limit`
  (was: the task's pool), so a side writer whose scope differs from its task's cells
  can resolve its own set. `G2_retrieval.run_side` re-filters to its answerable pool
  so the retrieval benchmark stays answerable-only (pivot_v4 §3.3).
- **Deviation from the G4 pivot note.** The pivot intent floated putting the
  classifier under `ops/scripts/`; the merge decision puts it on G3 instead (smaller
  surface, one fewer entry point). That pivot intent is folded into this entry.
- **Still open (⚠ PENDING):** predicted-domain *routing accuracy* (routing uses the
  classifier only for its latency price, not to re-route by predicted bin); and the
  per-sweep YAML expander (`ops/specs/target_architecture.yaml`), deferred to a
  separate change.

---


## Phase 0 — capture (2026-07-08)

- **Reference commit (v3 snapshot):** `e73ee892b9e627f313ab780ec2c199470175bb5c`
  ("pivot v4", branch `main`). This is the v3 state the do-over forks from.
- **v3 structural fixtures preserved** at `tests/fixtures/v3_results/`, copied from
  `results/cache/{bf16-lowres,yaml-g1-g2-g5-rerun}/full/`. jsonl only (no
  render/marker/ocr blobs). Covers task shapes G1/G2/G3/G5/G6 incl. judged
  `results.jsonl`, retrieval and classifier side-artifacts. Labelled **v3-shaped,
  values NOT comparable to v4** (see the fixtures README).
- **Gitignore:** added `!tests/fixtures/v3_results/**/*.jsonl` to override the
  blanket `*.jsonl` ignore so the fixtures are tracked.

---

## Phase 1 — probes & decisions

### 1a. Resolution probe

Script written at `ops/scripts/resolution_probe.py`. It sweeps the five presets
(`min`/`low`/`med`/`high`/`full`) on the V rung (image-only, parser-independent),
worst case ~10 pages, with the 8B primary reasoner loaded exactly like production
(HF, `device_map="auto"`, 5GiB/GPU reserve, memory-efficient SDPA). Per preset it
records per-GPU peak VRAM and OOM; the highest preset that stays under 16GiB (else
the highest that does not OOM) is the deployment resolution.

**Submitted to Kaya (2026-07-08): job `1017226`**, partition `gpu`, `gpu:v100:2`,
30 min. The `gpu` partition was saturated (the long `g1g2g5-full` run holds a
node), so the scheduler estimated a start up to ~2026-07-10; backfill may run it
sooner. Pull results with `python3 -m kaya.kaya watch 1017226` (or `pull`); the
verdict lands in `results/probes/resolution_probe.json`.

_Verdict (chosen deployment resolution preset): PENDING job 1017226._

### 1b. Environment / dependency decision

**vLLM verdict: DROP.** Evidence (2026-07-08):

- The live inference path already uses plain HF `transformers`
  (`Qwen3VLForConditionalGeneration.from_pretrained(device_map="auto")` +
  `model.generate(do_sample=False)` under the memory-efficient SDPA kernel, in v3
  `models/local_vlm.py`). It never imports vLLM.
- The only real `import vllm` in the whole tree is the retired feasibility probe
  `scripts/run_probe.py:578`, which v4 replaces. The other two hits are docstring
  mentions.
- v4 reasoning is batch-1 and latency-measured, which is exactly the regime where
  vLLM's serving/throughput machinery buys nothing.

**What dropping vLLM frees** (these pins existed only for it):

- `openai` — `requirements.txt` documents `openai<=1.90.0` as a vLLM 0.9.2 cap.
  Freed; judge SDKs can use a current `openai`.
- `torch==2.7.0` exact — relax to a cu126-compatible range (Kaya is `cuda/12.6.3`).
- `transformers==4.57.6` exact — keep the **floor** `>=4.57` (Qwen3-VL's
  `Qwen3VLForConditionalGeneration` landed there); relax the ceiling.
- `pillow==10.4.0` / `pillow<11` — the `<11` pressure came from marker/docling,
  which v4 drops; can move to pillow 11.

**Env partition (target):**

- **Core reasoning env** — the reasoners (Qwen3-VL + InternVL via timm), quant
  (bitsandbytes), the T-rung cheap text (PyMuPDF), all retrievers (rank-bm25,
  FlagEmbedding/BGE-M3, colpali-engine for ColQwen/ColModernVBERT, Qwen3-Embedding),
  the judge SDKs (openai, google-genai), and data/util libs. No vLLM.
- **One isolated env per parser that will not co-exist** — PaddleOCR-VL, MinerU 2.5,
  Unlimited OCR. They are heavy, separately-pinned VLM stacks; they cross to the
  reasoner only via the disk cache (the pre-pass warms the parser cache), so they
  never share the reasoner's env or its VRAM. Parsers that happen to `pip check`
  clean together may share one `parse` env.
- **Dropped stacks** (and any pins that existed only for them): vLLM, marker-pdf,
  docling, and the old paddleocr/paddlex/paddlepaddle parser stack (superseded by the
  isolated PaddleOCR-VL env). Marker caches may be kept as appendix continuity.
- Keep the local-Blackwell (RTX 5070) env as-is.

**Still to verify empirically (Phase 4, per the plan):** the exact package
names/pins for PaddleOCR-VL / MinerU 2.5 / Unlimited OCR, and a clean `pip check`
per env. That is decided by attempting installs, not statically, so it is
deliberately deferred to Phase 4 finalization. This Phase-1 deliverable is the
verdict + partition, which are settled above.

---

## Phase 2 — park & scaffold (2026-07-08)

- **v3 snapshot in `old/`** (untouched, 100 modules): `config.py`, `schema.py`,
  root `__init__.py`, and the packages `cli covariates data experiments gates
  metrics models pipeline reporting tools scripts specs kaya`, plus the v3
  `test_*.py` under `old/tests/`. Deleted in the final commit once v4 is green.
- **Phase-0 fixtures kept at root** `tests/fixtures/v3_results/` (not moved into
  `old/`), so Phase-3 I/O tests can read them.

**Direct-copy set (to v4 homes):**

| File(s) | v4 home | Mark |
|---|---|---|
| `kaya/{kaya.py,__init__.py,config.json,KAYA_*_GUIDE.md}` | `ops/kaya/` | copied-pending-rework (see below) |
| `download_hf,gpu_test,kaya_status,setup_env,dataset_stats,profile_datasets,split_docs_by_type` | `ops/scripts/` | clean copy |
| `dump_docstrings.py` | `ops/scripts/` | reworked (2026-07-09): emptied stale SUMMARY_OVERRIDES so the map builds from live docstrings |
| `annotate_docs.py` | `ops/scripts/` | reworked (2026-07-09): manual-bin vocab, optional dominant_visual, Cohen's kappa subset mode |
| `inspect_results.py` | `ops/scripts/` | reworked (2026-07-09): inlined the retired `gates/viewer.py`, v4 ResultRow/paths |
| `split_docs_by_type.py` | `ops/scripts/` | reworked (2026-07-09): dropped the retired `DOC_TYPE_TO_BIN` summary |
| `prestage.py` | `ops/scripts/` | verified v4 (2026-07-09): stages the three parsers + text/vision retrieval rungs; no change needed |
| `run_probe.py` | (removed) | deleted (2026-07-09): v3 Stage-1 probes (`boxes`, `doc-type`) abandoned in v4; superseded by `resolution_probe.py` |
| `ANNOTATION_GUIDE.md` | `docs/` | reworked (2026-07-09): moved from `ops/scripts/` to `docs/`, updated to v4 bin vocab |
| `specs/*.yaml` | `ops/specs/` | clean copy |

**Deviations recorded:**

- `ops/kaya/kaya.py`: `LOCAL_ROOT` was `Path(__file__).resolve().parents[1]`
  (repo root under `kaya/`); at `ops/kaya/` that resolved to `ops/`, breaking the
  rsync source and program-path anchoring. Changed to `parents[2]`. Verified:
  `python3 -m ops.kaya.kaya show-config` resolves the repo root and config. The
  Kaya driver is now invoked as **`python3 -m ops.kaya.kaya`** (was `kaya.kaya`);
  the live resolution-probe job `1017226` is pulled with
  `python3 -m ops.kaya.kaya watch 1017226`.
- `ops/scripts/dump_docstrings.py`: its `SUMMARY_OVERRIDES` still key off v3 paths
  and contain "v4 should" plan-talk (which the new docstring rule forbids). It is
  copied for reference but must have the overrides cleared and be regenerated in
  Phase 4. Reclassified clean-copy -> copied-pending-rework.

**Scaffold:** empty v4 tree created with 1-3 sentence module docstrings; all 65
spine modules + the `ops` entry points import cleanly. `docs/generated/` created
and the generated outputs (`dataset_stats.md`, `dataset_label_distributions.csv`)
moved there. `docs/REPO_STRUCTURE.md` written (tree + auto-gen marker; per-file map
regenerated in Phase 4). The `CLAUDE.md` module-docstring rule was added.

---

## Phase 3 — tests first (2026-07-08)

- v3 tests are parked in `old/tests/` (deleted from the active tree). `pytest.ini`
  scopes collection to `tests/` and excludes `old/` and the local `.cache/` conda
  package tree.
- v4 suite written as executable specs of the `pivot_v4.md` invariants. Result:
  **150 tests, 0 collection errors, 128 green, 22 red** against the stubs — the red
  ones are the invariants Phase 4 must satisfy.
- Design: `tests/conftest.py::require(module, attr)` fetches an intended v4 symbol
  and fails cleanly if the stub lacks it, so unfinished work reads as a red test
  rather than a collection-time ImportError. Fixtures loaded from
  `tests/fixtures/v3_results/`.
- Green now (real guards, not stubs): every spine module imports; the module-
  docstring rule holds on all v4-authored modules; `config` has no input-token cap
  symbols; `pipeline/representation.py` references no bbox and imports no model
  backend; the four fixture shapes parse. Red now: registry task discovery, schema
  telemetry + truncation canary, cell robustness + `--failed-only`, machine-
  independent keying, corpus sampling modes, YAML `parse_spec`, and the v4
  jsonl reader / build grouping.
- Test files: `test_imports_registry`, `test_schema_telemetry`,
  `test_config_cap_removed`, `test_engine_robustness`, `test_keying`,
  `test_representation`, `test_corpus_scope`, `test_yaml_spec`, `test_io_fixtures`,
  `test_docstrings`.

---

## Environment partition (pre-Phase-4, 2026-07-08)

Built the v4 env partition. **Target Kaya (V100, sm_70, cu126) now**; local
(sm_120) and supervisor (sm_90) are specified but not built yet.

**Four isolated conda envs** (`envs/<name>`), one framework boundary between the
core reasoner and each parser (parsers cross only via the disk cache):

| Env | Framework | Requirements | Model |
|---|---|---|---|
| `core` | torch (no vLLM) | `docs/requirements/core.txt` | Qwen3-VL + InternVL + retrievers + judges + PyMuPDF |
| `parse-paddleocrvl` | **PaddlePaddle** | `parse-paddleocrvl.txt` | `PaddlePaddle/PaddleOCR-VL` |
| `parse-mineru` | torch | `parse-mineru.txt` | `opendatalab/MinerU2.5-2509-1.2B` |
| `parse-unlimited` | torch | `parse-unlimited.txt` | `baidu/Unlimited-OCR` |

Key findings that shaped this:

- **vLLM dropped** (already decided §1b): core env is HF transformers only.
- **PaddleOCR-VL page-level parsing needs PaddlePaddle**, not just transformers
  (the transformers path is element-level only). So its env is Paddle-native
  (`paddleocr[doc-parser]` + `paddlepaddle-gpu` from Paddle's index), zero torch.
- **MinerU** uses `mineru[vlm]==3.4.3` (transformers VLM backend, no vLLM/gradio),
  torch >=2.6.
- **Unlimited-OCR** pins transformers==4.57.1 + torch==2.10.0 (upstream-tested),
  so it can't share the core env's torch — hence its own env.

**Three machine configurations** live as a matrix in `setup_env.py` (CUDA index +
framework versions per machine) plus `docs/requirements/README.md`; the dependency
files are shared across machines. Chose a shared-deps + machine-matrix layout over
three duplicated file trees to avoid drift; only torch/paddle build differs by
machine.

**Script reorg:**
- `setup_env.py` rewritten: `--machine {kaya,local,supervisor} --env {core,parse-*,all}`,
  per-(machine,env) framework install + `pip check`. Import fixed to `ops.kaya.kaya`.
- `prestage.py` rewritten to own **all** downloads incl. the three parser models
  (`config.parsers`); dropped the v3 marker/docling/paddleocr tool-warmup (those
  tools are gone). Per-parser-env aux-model warmup + v4 tool smoke deferred to
  Phase 4 (needs `tools/parser.py`).
- Old root `requirements.txt` + `requirements-local-rtx5070.txt` removed;
  `requirements-annotate.txt` -> `docs/requirements/annotate.txt`.
- `config.json`: `paths.env` -> `envs/core`; added `parsers`.

**Not removed yet:** `envs/mpvrdu` (local + Kaya) — Kaya's is in use by the running
`g1g2g5-full` job; local removal was descoped ("don't worry about local"). Removed
once the new envs are validated and jobs finish.

**Build result (Kaya, verified `pip check` clean on all four):**

| Env | Framework | Notes |
|---|---|---|
| `core` | torch 2.7.0+cu126 | transformers 4.57.6, no vLLM |
| `parse-mineru` | torch 2.7.0+cu126 | mineru 3.4.3 |
| `parse-unlimited` | torch 2.10.0+cu126 | transformers 4.57.1 |
| `parse-paddleocrvl` | paddlepaddle-gpu 3.0.0 | paddleocr 3.7.0 |

`setup_env.py` fix: its post-install framework-version check imported the
framework, but `paddlepaddle-gpu` needs `libcuda.so.1` which is absent on the
GPU-less login node (torch loads CUDA lazily and is fine; paddle hard-fails). The
check now falls back to package metadata on import failure, so a paddle env is no
longer a false build failure. The env itself was already valid (pip check clean);
the paddle binary loads on a GPU node.

Still open: a GPU smoke (load each parser model + a reasoner on a V100) needs the
weights (`prestage.py`, not yet run) and a GPU slot; that is the next validation
beyond "deps resolve." `config.retrieval_models` still lists the v3 retriever ids;
the v4 retriever catalog (BGE-M3 / Qwen3-Embedding-4B / ColModernVBERT / ColQwen3)
is set in Phase 4.

**Doc debt (Phase 4):** the Kaya guides / README / AGENT_GUIDE and the `.bashrc`
`kaya` alias still say `kaya.kaya` and `envs/mpvrdu`; reconciled during the Phase-4
doc pass (the driver is now `python -m ops.kaya.kaya`, core env `envs/core`).

---

## Containment paradigm — confirmed and enforced (2026-07-08)

Confirmed the v3 paradigm survives into v4: **everything (envs, model + parser
weights, datasets, all caches) lives under the project root, gitignored and
rsync-excluded**, so each machine keeps its own heavy dirs and nothing lands in
`$HOME` or a shared system path.

Already held: `envs/`, HF weights (`HF_HOME=.cache`), datasets (`.data`), and
paddle/torch/pip caches are pointed in-project by
`ops/kaya/kaya.py::artifact_exports`, applied by `remote_prelude` for every
`run`/`submit`. `.gitignore` + `rsync_excludes` cover `envs/ .cache/ .data/
results/ logs/`.

Gaps found and closed (they had leaked ~2.2 GB into Kaya `$HOME` during the env
builds):
- **conda package cache** — no `CONDA_PKGS_DIRS`, so `conda create` wrote to
  `~/.conda/pkgs` (1 GB). Added `CONDA_PKGS_DIRS=.cache/conda-pkgs`.
- **pip cache** — 1.2 GB in `~/.cache/pip`. `PIP_CACHE_DIR` was already set but
  some invocations escaped; added `XDG_CACHE_HOME=.cache/xdg` as a catch-all.
- **MinerU/ModelScope** — would download aux models to `~/.cache/modelscope`.
  Added `MODELSCOPE_CACHE=.cache/modelscope` + `MINERU_MODEL_SOURCE=huggingface`.
- Added `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` for GPU-run compile caches.
- Dropped the now-unused `DOCLING_CACHE_DIR`.
- `prestage.py::prepare_hf_cache_env` sets the same vars so `--local` is contained.

The leaked `~/.conda/pkgs` and `~/.cache/pip` on Kaya were cleaned (`conda clean
-ay` + `rm`); the four envs still import (hardlinks intact). `artifact_exports`
runs locally to build the remote prelude, so the fix is live for the next
`run`/`submit` without a code push.

**Requirements moved to `docs/requirements/`** (from `ops/requirements/`) at the
user's request; `setup_env.py::REQUIREMENTS` updated. (Deviation from the impl
plan's "docs/ = authored prose only"; done on explicit instruction.)

---

## Deviation & decision log (Phase 4+)

_One line per real judgement call: what, why, what it affected._

- **Stage 0 — cache namespace bump (2026-07-08).** Added `config.CACHE_VERSION =
  "v4"` and nested `ProjectPaths.cache_dir` under `results/cache/v4/`, so v3 and
  v4 cached cells can never co-mingle. Affects every cache path; wired further in
  Stage 2 (`experiments/engine/paths.py`).
- **Stage 0 — retriever catalog swapped to the v4 set (2026-07-08).** Updated
  `ops/kaya/config.json::retrieval_models` to text = {BGE-M3, Qwen3-Embedding-4B}
  (BM25 needs no weights), vision = {ColModernVBERT, ColQwen2.5-v0.2 kept,
  ColQwen3-4B}. **Not re-staged yet** — deferred until the retriever code lands
  (Stage 4) and the ids are locked. `ColQwen3-4B` id is **tentative**: there is no
  canonical `vidore/colqwen3` repo; using `OpenSearch-AI/Ops-Colqwen3-4B`
  (Qwen3-VL-4B, ColPali-style) as the best-available candidate, to confirm before
  staging. `ColModernVBERT` = `ModernVBERT/colmodernvbert` (confirmed).
- **Stage 1 — cap removed, resolution presets kept (2026-07-08).** `config.py`
  dropped `max_input_tokens` / `MAX_INPUT_TOKENS_BY_SIZE` /
  `max_input_tokens_for_spec` and the size-aware `MAX_PIXELS_BY_SIZE` (resolution
  is now one fixed preset, not size-aware). Kept `VISUAL_RESOLUTION_PRESETS`
  (min/low/med/high/full). Also updated defaults to v4: bins renamed to
  text-dominant/mixed-modality/visual-dominant, conditions dropped `buried` and
  added `similarity`, `k_values` = {1,3,5,7,10}. Greens `test_config_cap_removed`.
- **Stage 1 — DEPLOYMENT_RESOLUTION = "med" is a PLACEHOLDER (2026-07-08).** Set so
  the pipeline has a concrete preset to run at; **not final**. The operational
  resolution probe (job 1017226) decides the real value; its verdict replaces this
  constant. Re-check if the parser path shifts the sequence profile.
- **Stage 1 — ResultRow moved to `schema.py` + telemetry added (2026-07-08).** v3
  kept `ResultRow` in `pipeline/orchestrator.py`; v4 moves it to `schema.py` (the
  telemetry contract imports it from `schema`) and extends it additively with the
  §6 per-cell telemetry (`status`, `skipped_reason`, `bin_label`, `scan_label`,
  `machine`, `total_*`/`text_tokens_fed`/`tokens_dropped` tokens, prefill/decode
  latency split, `peak_vram_bytes`). Truncation fields are a zero-canary
  (`schema.tokens_dropped` / `truncation_occurred`). `Question` gained `bin_label`
  / `scan_label`. Greens `test_schema_telemetry`. This touches a frozen contract
  (ResultRow shape) — recorded here per the frozen-interface rule; the change is
  additive (new fields + a relocation), not a reshape.
- **Stage 0 follow-up — v4 retrievers staged (2026-07-08).** Re-ran `prestage.py`
  after the catalog swap; all five v4 retrieval models staged clean on Kaya,
  including the tentative `OpenSearch-AI/Ops-Colqwen3-4B` (valid snapshot, so the
  id is downloadable; whether it is the right "ColQwen3-4B" for the science is
  still open). The v3 leftovers (bge-small, colpali, colqwen2) remain on disk as
  dead weight, prunable later.
- **Stage 2 — engine keying + robustness (2026-07-08).** `experiments/engine/paths.py`
  gets machine-independent `prediction_key` / `result_key` (SHA-256 over identity
  only, no dpi/hostname/cuda; resolution preset is a manifest field) plus the
  lifted `experiment_paths` / `free_gpu` / `mode` / `configure_logging` /
  `write_phase_status`. `experiments/engine/driver.py` gets the pure primitives
  `run_cells` (one row per cell; failures classified oom vs error with a
  `skipped_reason`), `select_failed`, and `merge_failed_only` (failed-only re-run
  upgrades rows in place). The model-lifecycle half of the driver (parse pre-pass,
  reasoner load/free, systemic-abort threshold) is deferred to Stage 5. Greens
  `test_keying` (3) + `test_engine_robustness` (3). Suite now 12 red / 138 green.
- **Stage 3 — data + corpus (2026-07-08).** `experiments/corpus/resolve.py` gets
  `sample_per_bin` (doc-coherent draw, groups by `bin_label` not doc_type),
  `resolve_corpus` (full / per_bin / limit / ids modes), `pool_for_task`
  (G3 -> unanswerable, else answerable) + `filter_by_pool`. Dropped the v3
  oversized-evidence exclusion (the retry handles overflow now, no exclusion
  list). Data layer: `data/render.py` + `data/loader.py` lifted clean (loader
  gains `split_answerable`); `data/annotations.py` is a new 3-label reader
  (bin/scan/dominant_visual, validates, tolerates a missing sheet);
  `data/binning.py` rewritten to stamp `bin_label`/`scan_label` from the
  annotation table. **Blocker:** `annotations/doc_labels.csv` does not exist yet
  (needs the manual labelling pass), so real runs get blank bins until then;
  tests pass because they fabricate their own labelled corpora. Greens
  `test_corpus_scope` (5). Suite now 7 red / 143 green.
- **Stage 4 — tools + retrievers + representation (2026-07-08).** `tools/text.py`
  trimmed to `embedded_text` (T channel); `tools/visual.py` lifted (dropped
  region_crop, added `tokens_for_pixel_cap`); `tools/parser.py` is a new
  disk-cache interface (`parser_markdown` reads warmed markdown, `warm_parser_cache`
  is GPU-deferred with a lazy backend, so the read path never loads a model, no
  bounding boxes). `pipeline/representation.py` rewritten to the four cost-ordered
  rungs (T=embedded, TL/TLV=parser markdown, V=image), imports no model backend.
  Retrievers: base ABC + helpers + memoization in `retrievers/__init__.py`;
  `retrievers/text.py` (BM25 / BGE-M3 / Qwen3-Embedding, dense loads lazy with
  BM25 fallback); `retrievers/vision.py` (ColModernVBERT / ColQwen2.5 / ColQwen3,
  lazy ColPali-family load with a deterministic fallback; exact per-repo
  model/processor class for ColModernVBERT + ColQwen3 confirmed at GPU bring-up);
  `retrievers/joint.py` (order-preserving dedup union). Greens `test_representation`.
  Suite now 6 red / 144 green.
- **Stage 5 — models + pipeline + orchestrator (2026-07-09).** `models/payload.py`
  + `pipeline/reasoner.py` lifted; `models/__init__.py` `get_reasoner` adapted
  (dropped `max_input_tokens`, dispatches `qwen3vl`/`internvl3`). `models/qwen3vl.py`
  (renamed from local_vlm) and `models/internvl.py`: dropped `_truncate_context`
  (cap gone), populate the new split-token `Prediction` fields (`text_tokens_fed ==
  total_text_tokens`, the zero-canary) and `peak_vram_bytes`; qwen3vl measures a
  prefill/decode latency split (prefill via a timed forward, decode = generate -
  prefill), internvl leaves the split at 0 because `chat()` hides the boundary.
  `models/classifier.py` reworked to predict `bin_label` directly (three v4 bins,
  gold from annotation) instead of the retired doc_type->bin map. `pipeline/judge.py`
  lifted (abstention import repointed to `scoring.abstention`, which was pulled in
  as a leaf). `pipeline/conditioner.py`: kept oracle/retrieved/full, added
  `SimilarityTopK` (similarity provenance, for the hallucination study), dropped
  BuriedOracle. `pipeline/orchestrator.py` rewritten: `ResultRow` from `schema`,
  keys from `experiments.engine.paths` (page_indices, no dpi), two caches
  (`CachedPrediction` carries the new telemetry), captures full per-cell telemetry
  incl. the truncation canary. **Deferral:** the driver's generate/judge task-loop
  (engine lifecycle) needs the GenerationTask ABC, so it lands with Stage 6; the
  orchestrator (single-cell machinery) + prewarm are done here. No direct tests;
  validated at import level, no regression (6 red / 144 green).
- **Stage 6 — tasks + registry + yaml (2026-07-09).** `experiments/tasks/base.py`
  lifted (retriever import + task-bound `resolve_questions` via `pool_for_task`);
  four tasks: `G1OracleLadder` (oracle ladder), `G2Retrieval` (matched/cross
  k-sweep + retrieval side artifact), `G3Hallucination` (unanswerable x similarity
  pages x TLV), `G4ClassifierPricing` (side-only). `experiments/registry.py`
  exposes `TASKS` + `get_task` (exactly the four G-tasks, no legacy names).
  `experiments/engine/side_artifacts.py` ported (v4 retrievers, classifier emits
  bin_label gold; scoring imports stay lazy). `experiments/corpus/yaml_spec.py`
  rewritten: `parse_spec` returns a `Spec` (task/representations/corpus), rejects
  a `machine` field and unknown keys; `config_from_spec` builds an
  ExperimentConfig. Greens `test_imports_registry` (2) + `test_yaml_spec` (2).
  **Deferrals:** the driver generate/judge task-loop (engine lifecycle) still
  pending, needed for the smoke test; G3's 3-prompt-condition sweep needs a
  reasoner prompt-mode interface (flagged, not guessed). Suite now 2 red / 148 green.
- **Stage 7 — scoring + reporting (2026-07-09).** `experiments/engine/driver.py`
  gained `read_rows` (jsonl reader). Scoring ported from `metrics/` + the
  surviving `gates/` math: `scoring/accuracy.py` (doc-level bootstrap CI),
  `scoring/cost.py` (v4 token names + prefill/decode/peak-VRAM aggregation),
  `scoring/frontier.py` (sufficiency rule), `scoring/retrieval.py` (page P/R/F1,
  `bin_label` added), `scoring/agreement.py` (`cohen_kappa` + threshold, dropped
  the F1/F2/F3 gate scaffolding). `reporting/build.py` gets `group_rows`
  (prediction-identity grouping), `load_result_rows`, and an explicit
  `TASK_TO_TABLES` routing map. Greens `test_io_fixtures` (2). **All 150 tests
  green.** **Deferred to the smoke step:** the content-named table builders
  (`reporting/tables/*` still stubs) and the driver generate/judge task-loop +
  ops entry points (Stage 8) have no unit tests, so they are built and validated
  by the local smoke test rather than by pytest.
- **Stage 8 — ops entry points + driver loop (2026-07-09).** `experiments/engine/driver.py`
  gained the generate loop: `build_retrievers`, a spec-only reasoner for the parse
  pre-pass (warm retrieval + render caches, unload retrievers, free GPU, then load
  the reasoner), `generate()` running cells through the orchestrator via `run_cells`
  (ok rows cached by run_cell, failures written as status rows by `_failed_result_row`),
  reasoner freed between specs. Entry points at `ops/` root: `generate.py` (task +
  reasoner-spec + quantization + visual-resolution + limit), `judge.py`, `build.py`
  (loads results, groups by cell). Validated at import level; 150/150 still green.
  The smoke test is its runtime acceptance.
- **Local smoke test PASSED (2026-07-09).** Ran all four gen tasks locally on
  `envs/mpvrdu-local-gpu` (torch 2.8.0+cu128, RTX 5070) with qwen3vl-2b-local-4bit,
  `visual_resolution=min`, stub judge, 2 questions/task. Every task wrote
  well-formed `results.jsonl` cells, one row per cell: G1 T/V `ok` (real answers +
  populated tokens/prefill/decode/peak_vram), G1/G2/G3 TL/TLV `error` with
  `skipped_reason` (parser cache cold, proving the disk-boundary + failure-row
  path); G2 `retrieval.jsonl` scored P/R/F1 (ColQwen2.5 hit the gold page); G3 drew
  from the unanswerable pool; G4 `classifier.jsonl` priced a doc via image-only V.
  Canary `tokens_dropped=0` on every cell. Marker: `results/phase4_smoke_done.txt`.
  **Deliberate choice:** `QwenBinClassifier` representation set to `V` (image-only)
  so modality classification does not depend on the parser cache. **All 8 stages +
  smoke complete; the pytest suite is 150/150 green.**
- **Parser backend implemented (2026-07-09).** `tools/parser.py::warm_parser_cache`
  is now a real subprocess-into-isolated-env runner (no longer a NotImplementedError
  stub): it batches the uncached TL/TLV pages, resolves the parser env python
  (`MPVRDU_PARSER_PYTHON_<TOOL>`, then `MPVRDU_PARSER_PYTHON`, then
  `envs/parse-<tool>/bin/python`), and spawns `tools/parser_worker.py` which writes
  each page's markdown to the same disk cache `parser_markdown` reads. The worker
  imports nothing from the project so a minimal parser env can run it; backends load
  lazily. `paddleocrvl` is verified locally (PaddleOCR det+rec floor; the PaddleOCR-VL
  / PP-StructureV3 markdown paths are gated behind `MPVRDU_PADDLE_RICH`); `mineru` and
  `unlimited` are written to a transformers VLM path and stay Kaya-env-verified (no
  local env). Warming is wired into the driver pre-pass (`_warm_parser_cache`, one
  batched load per run, before the reasoner loads, non-co-resident). Added
  `config.ExperimentConfig.parser_tool` and threaded it through the orchestrator's
  `get_representation` calls. A parser that cannot run raises `ParserUnavailable`,
  logged (not fatal); TL/TLV then record a parser-miss row rather than fabricating text.
- **Reporting table builders implemented (2026-07-09).** The ten content-named
  builders in `reporting/tables/*` are no longer stubs. `_common.py` (Table container,
  ok-row loading with per-cell collapse, bin/rung ordering, accuracy/cost/frontier
  formatting) and `_markdown.py` (report rendering) back: `headline`, `parser`,
  `resolution`, `scale`, `composition` (G1); `matched_cross`, `kdepth`,
  `retrieval_accuracy` (G2 + retrieval side-artifact); `hallucination` (G3); and
  `routing` (build-time assembly from G1 ladder rows + the G4 classifier price).
  `reporting/build.py` gains `assemble_tables` (builds every table its inputs exist
  for, one bad builder logged not fatal) and `write_tables` (CSV per table +
  `all_tables.md`); `ops/build.py` now writes them under `results/tables/<partition>/`.
  Scoring skips non-ok rows and blank bins bucket to `(unlabeled)`. Reruns the local
  smoke with the parser on: all four tasks now `status=ok` (TL/TLV carry real
  paddleocrvl markdown), and all ten tables build from those rows.
- **G3 prompt sweep wired (2026-07-09).** The hallucination task now runs the three
  prompt conditions (`none` / `generic` / `targeted`) instead of one. The instruction
  strings live in `config.PROMPT_MODES` (targeted = the previous frozen preamble, so
  answerable G1/G2 cells are byte-identical); `DEFAULT_PROMPT_MODE="targeted"`,
  `G3_PROMPT_MODES=(none,generic,targeted)`. The mode rides on the conditioner name
  (`similarity_<r>_k3_prompt-<mode>`, mirroring how retrieval k rides the name), so each
  mode is its own cache cell; `Cell.prompt_mode` carries it and the driver passes it to
  `Orchestrator.run_cell`, which sets `reasoner.prompt_instruction` per cell.
  `render_prompt` in both backends (qwen3vl, internvl) took an `instruction` arg;
  `Reasoner` ABC gained `prompt_instruction`. The hallucination table groups by the
  parsed prompt mode. Frozen key composition, `ModelInput`, and `ResultRow` unchanged
  (only the `condition` value varies, as it already did for k). Local G3 smoke: 6 rows
  (3 modes x 2 q), targeted abstains 100%, none/generic 0% — the prompt changes behaviour.
- **Annotation table treated as authoritative when present (2026-07-09).** `data/annotations.py`
  now raises on a sheet that exists but is missing a required column (`doc_id`/`bin_label`/
  `scan_label`), and skips rows whose `bin_label` is still blank (in-progress annotation)
  rather than erroring on them. `data/binning.stamp_bins` gained `require_complete=True`:
  once any doc is labelled, every corpus document must be labelled or the run stops with an
  actionable message (which docs are blank); an all-blank/absent sheet still degrades to
  blank labels so dev/smoke runs proceed. `stamp_bins` leaves `doc_type` untouched, so a
  cell's telemetry keeps the native document type alongside the manual `bin_label` (both
  were already in `ResultRow`; confirmed populated end to end).

## Full G1-G3 doc_type-sampled Kaya submission (2026-07-10)

Autonomous session driving the HANDOFF.md build to submission. Deviations, one line each:

- **G2 scope split confirmed with the user (2026-07-10).** The reasoner *inference*
  k-sweep uses bm25 (text) + colqwen2.5 (vision) + their free union only; the full
  six-method ladder is the accuracy-only side-artifact (no reasoner). This matches pivot 7
  (Scorer A vs Scorer B) and corrects the handoff's "sweep all 6 through the reasoner"
  wording. Affected `matched_cross_sweep_cells`, `G2_retrieval`, `write_retrieval_eval`.
- **JointTopK conditioner added (2026-07-10).** `pipeline/conditioner.py` gains `JointTopK`
  (free deduped union of two retrievers' top-k via `retrievers/joint.union`, name
  `retrieved_joint_k{k}` so the reporting regex buckets it as modality `joint`). Additive,
  frozen cache key unchanged. Fed to the reasoner as bm25 u colqwen2.5 at joint k in {1,3,5}.
- **G2 inference reps = TLV and V (2026-07-10).** `matched_cross_sweep_cells` now sweeps a
  `representations` iterable instead of pinning TLV; `G2_retrieval` reads
  `config.representations ∩ {TLV,V}` (default both). `build_retrievers` stays bm25 +
  colqwen2.5 (all inference needs); `Retrievers` stays the 2-slot dataclass (the 6-method
  ladder builds its own retrievers in the side-artifact), so G3 (`retrievers.text` = bm25)
  is untouched. Adding the V rows is additive to cache.
- **Retrieval cost telemetry, full rigor (2026-07-10, locked decision).** `RetrievalEvalRow`
  gains additive `retrieval_latency_s` + `index_build_amortized_s`; `score_retrieval` takes
  them as kwargs. Each concrete retriever (bm25 got a per-doc index cache, dense, vision)
  now times its per-doc index/embed build (`index_build_s`, cumulative) separately from the
  per-query score (`last_query_s`). `write_retrieval_eval` reads these: per-query latency on
  each row, and the method's total build time stamped as the amortized value on every row
  for that method (pivot 6.3 "once per method x corpus").
- **Six-method side-artifact, robust per-method (2026-07-10).** `write_retrieval_eval`
  rewritten from a `pairs` list to score all 6 methods + 3 matched-tier joints
  (bm25 u colmodernvbert, bge-m3 u colqwen2.5, qwen3-embedding u colqwen3) at single k /
  joint k, building each via `get_text_retriever`/`get_vision_retriever`. Each method is
  built+ranked in its own try/except with `free_gpu()` after, so a big-model V100 OOM
  (qwen3-embedding-4B, colqwen3-4B) loses only that method's rows. Reasoner is already freed
  before `run_side`, so the warm has the GPU to itself.
- **Vision loader dispatches per rung; failed load errors, no silent garbage (2026-07-10,
  best-effort per decision 2).** `ColVisionRetriever._load` now tries a per-subclass
  `model_classes` list of same-family (model_cls, processor_cls) candidate names from
  `colpali_engine.models`, raising a clear error if none load (rather than forcing every
  rung through ColQwen2_5). The side-artifact builds vision rungs with
  `allow_text_fallback=False` so a load failure is an honest miss, not a degraded ranking.
  **Unvalidated offline:** colmodernvbert (`ColModernVBert*`) and colqwen3 (`ColQwen3*`)
  class names are best-guess; if the installed colpali_engine lacks them those two rungs
  error out in the benchmark (run continues). Confirm on the first Kaya run.
- **G3 `none` prompt restored + full pool (2026-07-10).** `config.G3_PROMPT_MODES` back to
  `(none, generic, targeted)`; the stale `test_g3_prompt_modes_drop_none` was rewritten to
  assert all three (pivot 7 needs the unprompted baseline). G3 runs the full unanswerable
  pool via a per-run `corpus: {sampling: full}` spec override (verified it clears the base
  `per_doc_type`). No G3 code change (it already sweeps `G3_PROMPT_MODES` and uses bm25).
- **Specs rewritten (2026-07-10).** `kaya.yaml` run_tags -> g1-doctype50 / g2-doctype50 /
  g3-full, G2 reps [TLV,V], G3 `sampling: full`. `kaya_smoke.yaml` fixed to distinct
  `*-smoke` run_tags (it wrongly reused the real tags), 2B, per_doc_type 1, G3 limit 4.
- **Env: rebuilt core + parse-paddleocrvl, then prestaged (2026-07-10).** Per the user's
  choice. `build_env` skips `conda create` when the prefix exists, so the "rebuild" was an
  idempotent pip re-verify (both pip-check clean; paddle 3.3.1 confirmed). Prestage staged
  all 5 retriever models + 2 reasoners + PaddleOCR-VL + the paddlex pdx cache against
  `config_minimal.json`; mmlongbench already present. Did NOT `--env all` (mineru/unlimited
  are broken and unused).
- **Cron wake mechanism caveat (2026-07-10).** The requested 01:15 wake is a session-only
  CronCreate job (in-memory, fires only while this REPL is alive+idle, auto-expires in 7
  days). It re-enters this plan + HANDOFF and reconstructs state from git + Kaya + this log,
  since a literal same-session resume is not guaranteed once usage resets.
- **pytest 179/179 green (2026-07-10)** after the changes (added 4 per_doc_type tests,
  updated the G3 prompt-mode test).
- **BLOCKED on Kaya connectivity at submit (2026-07-10 ~01:52 UTC).** The smoke submit
  failed at the push step: `kaya.hpc.uwa.edu.au` no longer resolves and no VPN interface is
  up (general internet is fine). The UWA HPC tunnel/VPN that was up when Task A ran (16:16
  UTC) dropped overnight; it can't be restored autonomously (needs the user's VPN
  credentials/2FA). Code + tests + specs are fully ready. **To finish, once the UWA VPN is
  reconnected, run:**
  `python3 -m ops.kaya.kaya submit --gres gpu:1 --mem 48G --time 00:30:00 ops/generate.py -- --spec ops/specs/kaya_smoke.yaml`
  then `python3 -m ops.kaya.kaya pull` + `python3 -m ops.scripts.check_run --spec ops/specs/kaya_smoke.yaml`;
  fix any failures; then the real run:
  `python3 -m ops.kaya.kaya submit --gres gpu:2 --mem 64G --time 08:00:00 ops/generate.py -- --spec ops/specs/kaya.yaml`.
  A recurring connectivity-poll cron was added to auto-run this the moment Kaya resolves.
- **Manual annotations made optional, not required (2026-07-10, user decision).** The first
  smoke on Kaya (job 1026030) failed in 20s at `stamp_bins`: the sheet labels 107/135 docs
  and the `require_complete=True` guard stopped the run on the 28 unlabelled. The user
  decided to abandon the manual annotation pass for now and make `doc_labels.csv` an optional
  enrichment. Flipped `data/binning.stamp_bins` default to `require_complete=False` (partial
  or absent sheet runs; uncovered docs keep blank bins -> `(unlabeled)` in reporting; sampling
  is by native `doc_type` so it is unaffected). `ops/generate.py`: default is now permissive;
  added opt-in `--require-complete-annotations` for strictness, and kept `--allow-unlabelled`
  as a deprecated no-op so `final_probe.py` / `local_test.yaml` still parse. This reverses the
  2026-07-09 "annotation table treated as authoritative" strictness default (the reader still
  validates columns and skips blank rows; only the completeness gate is now opt-in). Tests
  179/179 green.
- **Smoke resized to a fixed limit:2; render cost flagged (2026-07-10).** Smoke 1026049 hit
  the 30-min TIMEOUT (not a code error): `per_doc_type:1` resolved to 39 questions, and the
  G1 pre-pass spent ~27 min rendering ~77 gold pages to PNG (~21s/page) before the parser
  even loaded. Renders cache to disk (`.cache/renders/<pdf>__dpi{dpi}/page_NNNN.png`, guarded
  by exists), so it is a one-time amortized cost (the slow `/group` network FS, not a bug),
  but it means the real run needs generous walltime. Changed `kaya_smoke.yaml` to a fixed
  `limit:2` per task (2 questions, a few page renders) and bumped smoke walltime to 50 min so
  the six-method side-artifact's model loads (incl. the 4B retrievers, where the
  colmodernvbert/colqwen3 class-name guesses get validated) fit. The real run keeps
  per_doc_type:50 and will get an 8h walltime.
- **Target YAML architecture proposed, awaiting user reaction (2026-07-10).** Wrote
  `ops/specs/target_architecture.yaml` (a mock-up, not wired): each main task gets a `base` +
  named per-axis sub-sweeps (G1: size/family/quantization/resolution/parser/dataset; G2 split
  into `retrieval:` 6-method x k and `inference:` chosen-retrievers x TLV/V; G3: prompt),
  each independently collapsible. Chose independent per-axis sweeps over a full cross-product
  (coupled axes + combinatorial blowup). Implementation deferred until the user confirms.