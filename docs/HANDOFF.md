# Session handoff - 2026-07-08

This session was a documentation + code refactor to make future experiments
easier to navigate. No pipeline behavior changed; everything is local working-tree
edits (nothing committed, nothing pushed to Kaya). Full test suite is green:
**99 passed, 1 skipped**.

## Cluster job (from the previous session, not touched here)

The full G1/G2/G5 rerun (`1012121`, run tag `yaml-g1-g2-g5-rerun`, spec
`specs/g1_g5_rerun.yaml`, 2x V100) was still running at the last handoff. This
session did **not** push to Kaya, so the remote job runs from its own synced
snapshot and is unaffected by these local edits. Job state was not re-checked this
session (watcher session `18442`, logs under `logs/g1g2g5-full_1012121.*`). After it
finishes, score and build:

```bash
python -m cli.judge --spec specs/g1_g5_rerun.yaml --judge <judge>
python -m cli.build --run-tag yaml-g1-g2-g5-rerun --bootstrap 1000
```

Note the judge CLI is now `--spec` (or `--full --run-tag`); the deprecated
`--generation` flag was removed (see below).

## Documentation changes

- **`docs/PROJECT_SPEC.md`** rewritten from the README (it was the stale original
  spec). Corrected: judge is Gemini-2.5-Flash default / GPT-4o-mini paid; retrievers
  are `bge-small-en-v1.5` + `colqwen2.5-v0.2`; CIs are document-level bootstrap;
  Table 4 is a held-out MMLongBench subset (not LongDocURL); 70 text-heavy docs (not
  54); plus OCR routing and per-bin sampling.
- **`docs/AGENT_GUIDE.md`** cut ~50% (dropped the build log and the blow-by-blow OOM
  saga; kept decisions, frozen interfaces, caching contract, and the models/data/
  tools/evaluation reference). Recorded the Phase-A checkpoint (shared side-artifact
  writers).
- **`docs/USER_GUIDE.md`** §1-10 refreshed to match PROJECT_SPEC (same fact fixes:
  70 docs, bge-small, Gemini default, document-level bootstrap, held-out Table 4, and
  the V100 hardware-limits risk replacing the resolved transformers-availability one).
- **Dead `implementation_plan.md` refs** cleaned out after the user deleted the file:
  the real breakage was `config.py::project_root` (root marker is now `README.md` +
  `config.py`); also fixed `CLAUDE.md` and `USER_GUIDE.md`. Same for the deleted
  `SINGLE_GPU_8B_FEASIBILITY.md` (two code comments repointed to `AGENT_GUIDE.md`).

## Code changes (refactor)

- **Deleted dead code:** `scripts/attn_probe.py`, `scripts/single_gpu_probe.py`
  (zero-reference probes), `models/api_vlm.py` (empty placeholder).
- **Side-artifact de-duplication:** new `experiments/side_artifacts.py`
  (`write_retrieval_eval` / `write_classifier_eval`). `G5`, `G6`, and the YAML task
  all call it now, so the two task systems can't drift. Emission order preserved, so
  existing `retrieval.jsonl` / `classifier.jsonl` caches stay byte-identical.
- **Removed the deprecated `--generation` CLI path** from `cli/generate.py` and
  `cli/judge.py` (YAML `--spec` is the only entry point now). `generate.py` lost its
  now-dead config flags; the driver's `run_generate`/`run_judge`/`judge` functions
  were kept (still test-covered engine internals).
- **Reorganized `experiments/` into top-level packages** (it now holds generation
  only):
  - `reporting/` = `tables/` (per-table builders) + `build.py` (table -> source-task
    routing).
  - `gates/` = `core.py` (Section-2 gate logic) + `viewer.py` (the shared cached-cell
    viewer, moved out of `experiments/inspect.py`) + `__main__.py`. The gate CLI is
    now **`python -m gates`** (was `python -m scripts.gates`).
  - `scripts/inspect_results.py` stays as the debug CLI, importing `gates.viewer`.
- **Split `reporting/tables.py`** (was 1004 lines) into a `reporting/tables/`
  package: one `T*_*.py` module per table (`T1_headline` .. `T8_scale`, mirroring the
  `G*` task naming), `_common.py` (shared helpers), `_markdown.py` (the two `.md`
  renderers), and `__init__.py` as the entry point that re-exports everything and
  owns the all-tables aggregation (`build_all_tables` / `write_all_tables`). The
  `"table1".."table8"` dict keys are unchanged, so `reporting.build`'s routing and the
  CSV filenames are untouched.

Frozen interfaces were not touched. `GenerationTask.run_side` (refactored in Phase A)
is not a frozen interface; the checkpoint is noted in `AGENT_GUIDE.md`.

## Verification

- `envs/mpvrdu/bin/python -m pytest` -> 99 passed, 1 skipped.
- Entry points import/run: `python -m gates --help`, `python -m cli.{generate,judge,build} --help`,
  `python -m scripts.inspect_results --help`, and `config.ROOT` still resolves.
- Phase-A behavior was verified via the test suite only. The **byte-identical
  side-artifact check needs a GPU** (retrievers/classifier), so it could not run on
  the local sm_120 box; worth running the smoke spec on Kaya when the queue frees up.

## Open items / next steps

- Nothing is committed. We are on `main`; the plan is to commit this session on a
  branch (the plan file is `~/.claude/plans/cheerful-toasting-lerdorf.md`).
- All planned refactor phases (A-E) are done; none remain outstanding.
- After job `1012121` finishes: `kaya.kaya pull`, then judge/build as above, then run
  the F1 frontier gate (`python -m gates frontier --full --run-tag yaml-g1-g2-g5-rerun`).

---

# Addendum - second parallel session (2026-07-08): pre-pass fix + G2/G5 rerun

This ran in parallel with the refactor session above. It was a bug-fix session
(the parse pre-pass delay, plus the G2/G5 rerun failures), not a refactor. The
notes below **add to** the handoff and, where noted, supersede the cluster-job
status in the first section.

> Git note: both sessions share one working tree and `.git`. By the time of this
> addendum the shared tree (both sessions' edits) had been committed to `main` as
> `60791f7 "update docs"`, so the first section's "nothing is committed" is no
> longer true. My fixes below are inside that commit; anything I edited *after* it
> shows as fresh working-tree changes.

## Cluster job update (supersedes "Cluster job" above)

- Job **`1012121` was cancelled** this session. It had been stuck ~2.5h in the G1
  parse pre-pass with no `predictions.jsonl` (the symptom in
  `docs/PREPASS_INFERENCE_DELAY_ISSUE.md`).
- Resubmitted as **`1012581`** with the identical resources (`gpu:v100:2`, 4 CPU,
  64G, 24h) and the same spec/run tag (`specs/g1_g5_rerun.yaml`,
  `yaml-g1-g2-g5-rerun`). Unlike the refactor session, **this session DID push to
  Kaya** (the `kaya.kaya submit` pre-run rsync), so the remote now runs the fixed
  code. `results/` is excluded from the rsync, so the warm render/Marker/OCR
  caches were preserved.
- `1012581` finished **FAILED** (~5.5h). **G1 succeeded** (all 1232 cells cached).
  **G2_family OOM'd** on cell 54/1232 (InternVL3-8B, a `TL` text cell on
  `2024.ug.eprospectus.pdf`), and **G5 never ran** because the driver aborted the
  whole task loop on the first task failure. Both of those are fixed below.

## Root causes + fixes (all in the working tree / committed in 60791f7)

1. **Pre-pass delay** (`docs/PREPASS_INFERENCE_DELAY_ISSUE.md`):
   - RC1: the reasoner was constructed *before* the parse pre-pass. Reordered
     `experiments/driver.py::generate` so the pre-pass runs with a spec-only
     reasoner (`_SpecOnlyReasoner`, no weights) and `reasoner_for` is called after.
     (Caveat: reasoner weight-loading is lazy, so the original doc overstated the
     VRAM harm; it was still a real code/doc-order mismatch.)
   - RC2: the PaddleOCR engine was rebuilt on every `ocr()` call (per scanned page
     on a cold cache). Added a process-local shared engine + `reset_ocr_engine()`
     in `tools/text.py`, and wired the reset into the driver right after the
     pre-pass. Verified on a local 2B run: engine built **exactly once** across a
     cold pre-pass over 4 scanned docs.
2. **G2 OOM:** `models/__init__.py`'s InternVL branch dropped `max_input_tokens`,
   and `models/internvl.py` had **no input truncation at all**, so a long `TL`/text
   context OOMs the V100 O(seq^2) attention. Forwarded `max_input_tokens` and added
   `_truncate_context` mirroring the Qwen backend (keeps image placeholders, trims
   text from the tail). This is *not* the >10-evidence-page filter, which is global
   and already reaches G2.
3. **Cross-task isolation:** `run_generate` / `run_generate_tasks` now always record
   a failed task and continue to the next (a G2 OOM no longer drops G5); cell-level
   skipping still rides `--continue-on-error`. `cli/generate.py` exits non-zero if
   any task failed, so a partial failure still surfaces as SLURM `FAILED`.

## Docs

- Rewrote the README "generation tasks" section (pre-pass ordering, per-rung
  preprocessing, disk caches). Fixed two stale claims: retrieval and the classifier
  render at `config.dpi` (144), not "dpi 96".

## Known validity caveat (OPEN - decide before trusting TL/TLV numbers)

Input truncation is not free. Measured on the real run's cached pages with the
Qwen3-VL-8B tokenizer, joining each oracle question to its evidence pages:

- **T: 1%** of cells truncated, **TL: 29%**, **TLV: 34%** (median ~56% of
  text+layout tokens dropped among truncated cells).
- The verbose bbox **layout** JSON is the bloat and, because it's serialized after
  the text, it's the first thing dropped. So truncation preferentially degrades the
  exact channel the T->TL contrast measures, and could bias the frontier toward
  vision.

Options discussed, none actioned: (1) recompute the sufficiency frontier on the
non-truncated subset as a robustness check; (2) compact the layout serialization so
`L` fits (root-cause fix, needs a fresh run); (3) A100 + FlashAttention to raise or
drop the cap for the main tables.

## Tests

Full suite green: **102 passed, 1 skipped** (`envs/mpvrdu/bin/python -m pytest`).
New tests: pre-pass ordering regression, PaddleOCR engine reuse, InternVL
truncation + registry forwarding, and cross-task isolation.

## New / changed files this session

- New: `specs/smoke_prepass_fix.yaml` (local 2B smoke that reproduced + verified the
  pre-pass fix).
- Edited: `tools/text.py`, `experiments/driver.py`, `models/internvl.py`,
  `models/__init__.py`, `cli/generate.py`, `README.md`, and the two test files
  (`tests/test_experiments.py`, `tests/test_reasoner.py`).

## Next steps (this session)

- To finish the rerun: resubmit `specs/g1_g5_rerun.yaml` with the same run tag. G1
  is all cache hits, G2 reruns with truncation (pre-OOM cells are cache hits), and
  G5 finally runs. Resubmit command:
  `envs/mpvrdu/bin/python -m kaya.kaya submit --job-name g1g2g5-full --gres gpu:v100:2 --cpus-per-task 4 --mem 64G --time 1-00:00:00 --no-wait cli/generate.py -- --spec specs/g1_g5_rerun.yaml`
- Then the judge/build/gate steps from the first section's next-steps still apply.
- Resolve the truncation validity caveat above before reporting TL/TLV frontier
  results.
