# Handoff (2026-07-11): flat-spec refactor done, smokes green, experiments ready

## Update (2026-07-11): G2/G3 specs corrected, README run-times refreshed

Fixed mistakes in the two full specs and updated the README Kaya table to match:

- **G2 (`kaya_g2_full.yaml`):** reasoner is now `qwen3vl-8b-local` (was 2B),
  `k_values` [1,3] → [1,3,5,7,10], `joint_k_values` [1,3] → [1,3,5]. Since inference
  loops over `k_values` (task.py), the reasoner cell count jumped ~9.6k → ~24k, and 8B
  means it now needs **2 V100s** (was 1). README row updated to
  `--gres gpu:v100:2 --time 72:00:00`; that walltime will hit the partition cap, so
  expect to submit at the cap and resume from cache across submissions.
- **G3 (`kaya_g3_full.yaml`):** `text_retrievers` [] → [bm25] so the T-representation
  arm actually retrieves. Reasoner is unchanged (8B), so its GPU/walltime row in the
  README is unchanged.

Everything below is implemented and pytest is green (204). Nothing is git-committed
yet: the whole refactor + the new specs live in the working tree. The three Kaya
smokes passed end to end. The six real experiment specs are written and verified.

## What changed (trust the code + docs/DECISIONS.md, not old prose)

- **One spec-driven task.** `G1OracleLadder`/`G2Retrieval`/`G3Hallucination` are gone;
  `experiments/tasks/task.py::Task` is the only generation task. `task_name` is a
  label (cache dir + parallel job), not a type. Behaviour comes from the config.
- **Flat spec format** (`experiments/corpus/yaml_spec.py`): no `base`, no `sweeps`, no
  `task`. Every run lists the full variable set under a `task_name`; a list axis is
  the set of values to run over (cross-product). `dataset`/`parser` expand to one
  run_tag each; `reasoner_spec` x `quantization` fold into `reasoner_specs`;
  `visual_resolution` -> driver-looped list; reps/k/prompt_modes are cell dimensions.
  `ops/specs/template.yaml` is the reference menu + three worked tasks.
- **Corpus is a 3-stage funnel** (`corpus:` block, applied in `Task.resolve_questions`):
  `scan` (document: any/digital/scanned, PyMuPDF auto-detect cached to
  `annotations/auto_scan.csv`) -> `pool` (question: answerable/unanswerable/all) ->
  `sampling` (full / per_doc_type / per_bin / limit / ids). All five sampling
  strategies are wired (via `resolve.resolve_corpus`).
- **G2 stage-drift fixed (Phase 1):** the retrieval benchmark runs as stage 1 before
  inference, persists rankings to the shared memo (`<cache>/retrieval/`), and the
  inference arms reuse them; `retrieval.jsonl` is written incrementally. Inference
  text arm is **bge-m3** (was bm25); vision arm has fallbacks off.
- **Oracle is a retriever** (`retrievers/oracle.py`); oracle cells still select via
  `OracleConditioner` (all gold pages).
- **Docs:** `pivot_v4.md` folded into `docs/DECISIONS.md` and deleted; `README.md` and
  `docs/AGENT_GUIDE.md` updated to the code. README now has the sampling explanation
  and the Kaya run commands + walltimes.

## Kaya smokes (all COMPLETED, 1 V100, 2B, per_doc_type:1)

- G1 `1028085` COMPLETED (9m) — oracle ladder; a few TL/TLV cells OOM'd and the job
  continued gracefully (this is expected and desired).
- G3 `1028088` COMPLETED (22m).
- G2 `1028087` TIMEOUT at 1h (clean, no critical errors), **resumed** from cache as
  `1028586` COMPLETED (27m). Confirms cache-resume works.

Cache layout: `results/cache/<run_tag>/<smoke|full>/<task_name>/{predictions,results}.jsonl`
+ `results/cache/<run_tag>/retrieval/` (memo). `results/` is rsync-excluded, so the
remote cache survives pushes.

## How to run the real experiments

See README "Running the experiments on Kaya" for the six specs, submit commands, and
walltimes. Short version: `ops.kaya.kaya submit --no-wait --gres gpu:v100:<N> --time
<H> ops/generate.py -- --spec ops/specs/<spec>.yaml` (8B needs 2 V100s, 2B needs 1).
A timeout resumes from cache on resubmit; V100 OOM cells complete on the supervisor
via `--failed-only`.

## Open items / gotchas

- **`kaya_g1_quantization_per_doc_type_80.yaml` does not actually sweep quantization**
  (`quantization: [bf16]` = one value). Set `[bf16, 8bit, 4bit]` for the quant table.
- **`results.jsonl` has one row per cell incl. OOM/error; `predictions.jsonl` only ok
  cells.** The difference is the failed cells (status `oom`/`error`, `skipped_reason`,
  `oom_occurred`). `ops.judge` re-scores predictions only.
- **`docs/REPO_STRUCTURE.md` is absent** (deleted in an earlier commit) and its
  generator `ops/scripts/dump_docstrings.py` has a `REPO` path bug (`parent.parent`
  resolves to `ops/`, needs `parents[2]`) and expects the file to already exist.
  README/AGENT_GUIDE/CLAUDE.md still reference it. Restore/regenerate separately.
- **`ops.judge` / `ops.build` take `--task` but no `--run-tag`** — confirm they target
  the right run_tag before scoring the real multi-run outputs.
- **32B** only runs on the supervisor H100 (V100s can't fit it). `h100_main.yaml` is
  currently a single G1 flat run.
- **Nothing is committed.** Review the working tree (`git status`) before committing;
  it includes the deleted old task files, the new `task.py`/`oracle.py`, the rewritten
  specs, and the doc updates.

## Pipeline invariant worth keeping in mind

Only one model type is on the GPU at a time: retrievers load one at a time and unload
(`write_retrieval_eval`), the parser runs in its own isolated subprocess before the
reasoner loads, and inference reads rankings/parser text from disk caches so it never
reloads them. Don't collapse these stages.
