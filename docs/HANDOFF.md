# Handoff: efficiency pass, YAML run-splitting, per-cell resolution, supervisor handoff

Date: 2026-07-09. This session did a big round of work on the run machinery and the
handoff to the H100 supervisor. Everything below is landed and the test suite is
green (175 passing) unless it's under "In progress" or "Open items".

## Efficiency fixes (hot paths)

- **qwen3vl backend: no more double prefill** (`models/qwen3vl.py`). `answer()` used
  to run a full `model(**inputs)` forward just to time prefill, then `generate()`
  which prefills again. Now a `_FirstTokenTimer` streamer captures time-to-first-
  token inside the single `generate`, so prefill is paid once. Same
  prefill/decode split telemetry, ~half the prefill cost on every cell.
- **Retrieval amortized** (`retrievers/`). `MemoizedRetriever` now caches the full
  ranking per `(question, page_count)` and slices per k, so a k-sweep computes the
  ranking once. Added `Retriever.rank()`. Page embeddings are cached per document
  (vision: CPU-resident LRU; dense text: per-doc), so a doc's pages are embedded
  once, not per question. Added an optional on-disk page-set cache
  (`MemoizedRetriever(persist_dir=...)`, wired in `driver.build_retrievers` and the
  retrieval side-artifact) so rankings survive across processes.
- **Doc page text cached** (`retrievers/__init__.py::document_page_texts`) and
  **render spans cached in-process** (`data/render.py::_SPAN_CACHE`), so BM25/dense
  stop re-parsing the PDF per question and repeat renders skip `get_text`.
- **Cache writers keep one open handle** (`pipeline/orchestrator.py`
  ResultCache/PredictionCache) instead of open/close per row, and the driver
  pre-pass conditions each cell once (feeds the same page set to the parser warm).

## YAML runs: base + runs (multi-run per file)

`experiments/corpus/yaml_spec.py` grew `parse_specs` / `load_yaml_specs`. A spec
file is either flat (one run) or `base:` + `runs:` (several runs merged over a
shared base, each isolated by a unique `run_tag`). `ops/generate.py` iterates the
runs. This is how a task is split into chunks and across machines: give each
machine's file the same `run_tag` but a different slice of a sweep. The flat form
still works. There is still no `machine` field.

## Model-size sweep

`config.ExperimentConfig.reasoner_specs` (+ the `reasoner_specs` spec key) is a
list; when set, `GenerationTask._reasoner_specs` returns it and the driver runs one
pass per spec, freeing the GPU between them. Each spec keys its own rows, so the
size sweep lives in one run. `ops/kaya/config.json` gained the 32B id
(`Qwen/Qwen3-VL-32B-Instruct`) for prestage.

## Resolution is a per-cell key axis (frozen-interface change)

Recorded as a checkpoint in `AGENT_GUIDE.md`. Resolution used to be a per-run
manifest field, out of the cache key. It is now part of both keys and stamped on
`ResultRow`/`CachedPrediction`:

- `prediction_key` / `result_key` (`experiments/engine/paths.py`) take
  `visual_resolution`.
- `Orchestrator(visual_resolution=...)` threads it into the keys and rows.
- Config grew `visual_resolutions` (a list); the driver loops resolutions inside
  the spec loop, reusing the loaded reasoner and just changing `max_pixels`
  (a processor-side downscale), so a resolution sweep is one run.
- `IDENTITY_FIELDS` (both copies in `reporting/`) gained `visual_resolution`, and
  `reporting/tables/resolution.py` now pivots by the per-cell preset.

Verified live at 2B: V visual tokens scale with the preset and each (rung,
resolution) is a distinct key.

**Resolution presets** are now just `low / med / high` (dropped min/full), bumped
~25% to nice values: low 400, med 640, high 960 tok/page (`config.py`).
`RES_ORDER` and the `--visual-resolution` help match.

## `--failed-only` retry (the machine-split mechanism)

`ops.generate --failed-only` (driver `_prepare_failed_only` + `_cell_identity`,
which include resolution) reads what a run wrote, drops the non-`ok` rows, re-runs
only those cells, and upgrades them in place; `ok` cells and side artifacts are
left alone. Verified locally: flipped one cell to error, `--failed-only` retried
exactly that cell back to ok with no duplicates.

## G3 prompt sweep

`config.G3_PROMPT_MODES` dropped the `none` arm (G1 covers unprompted behaviour);
it is now `(generic, targeted)`.

## dpi -> 200

`ExperimentConfig.dpi` default is 200 (was 144). dpi is the OCR/parser render
resolution; the VLM downsamples to the resolution preset, so this only affects
TL/TLV parse quality. It is also a spec field (`dpi:`), so it's per-run settable.

## Run health check: `ops/scripts/check_run.py`

New. Given a spec, it scans each run_tag's `results.jsonl` and reports per task how
many cells are ok / oom / error, missing vs expected, and the top failure reasons.
Exits nonzero if anything looks broken, so it gates a run. This is the "did it work"
tool (there was none before, only a per-cell debug viewer).

## kaya `kill`

`python -m ops.kaya.kaya kill [JOB_ID]` cancels a single SLURM job (defaults to
`.kaya_last_job`). `handle_kill` in `ops/kaya/kaya.py`.

## Supervisor handoff (H100)

- **`setup_env.py --local`** builds the envs into this checkout's `envs/` (matching
  where `parser_env_python` and the reasoner look), so the supervisor can run it
  directly instead of through the Kaya SSH runner. flash-attn install is now
  best-effort (warns and continues, since the reasoner falls back to SDPA).
- **`ops/generate.py` sets the HF cache env** (`config.hf_cache_environ`, setdefault
  so Kaya's own exports win), so a direct run finds prestaged weights with no manual
  exports. `prestage.py` uses the same helper.
- **`prestage.py --config PATH`** lets prestage read a trimmed model list.
  `ops/kaya/h100_main.json` stages only what `h100_main.yaml` needs: the 8B
  reasoner, the paddleocrvl parser, and MMLongBench (no retrievers, no other sizes).
- **`docs/H100_RUNBOOK.md`** walks the supervisor through setup_env -> prestage
  (with `--config ops/kaya/h100_main.json`) -> generate `h100_main.yaml` ->
  `check_run` -> hand back `results/`. Generate-only; judging/building happen
  elsewhere. Retry section uses `--failed-only`.

## Parser bug fixed (would have hit the supervisor)

`tools/parser_worker.py::_paddleocrvl` read `res.get("markdown")`, which returns
`None` in paddleocr 3.7 (markdown is a computed `.markdown` property there, not a
stored key). The result was **empty TL/TLV text**. Fixed to prefer the `.markdown`
property (a dict with `markdown_texts`) with a fallback to the old mapping. Found by
building the parser env locally and actually running it. Verified end to end through
`warm_parser_cache`: real markdown out (4843 chars, "# Is the service safe?...")
where the old code wrote an empty string.

## Shipped spec files (`ops/specs/`)

- `h100_main.yaml` - 8B G1 over T/TL/TLV/V, full set, at med. The supervisor's run.
- `kaya.yaml` - G1 size sweep (2B/4B/8B), G1 resolution sweep at 8B (low/med/high),
  G3. (G4 line is present but commented.)
- `kaya_smoke.yaml` - kaya.yaml capped at 2 questions with `-smoke` run_tags.
- `kaya_replication.yaml` - future: InternVL3-8B replication + 4bit/8bit quant sweep.
- `local_test.yaml` - 2B correctness test (edited by the user to the full T/TL/TLV/V
  ladder, limit 5). Needs the local parse env (below).
- `template.yaml` - documents every field and both spec forms.
- `kaya_probe.yaml` - pre-existing go/no-go probe.

## Local parser env (for local_test.yaml)

Built `envs/parse-paddleocrvl` locally: `paddleocr[doc-parser]==3.7.0` + CPU
`paddlepaddle==3.3.1` (Blackwell has no GPU paddle wheels; paddle 3.0 hit a PIR op
bug, 3.3.1 fixed it). `parser_env_python` finds it at the default path, so no env
var needed. The PaddleOCR-VL model bundle is cached at `~/.paddlex/official_models`.
CPU parsing is ~5-6 min/page, so a local TL/TLV run is slow but functional; this is
purely a local limitation (the H100 uses GPU paddle).

## Testing status

- `pytest`: 175 passing. New tests: `test_check_run.py`, `test_resolution_and_retry.py`
  (resolution-in-key, `--failed-only` helpers, G3 modes, config validation), plus
  multi-run and reasoner_specs coverage in `test_yaml_spec.py`.
  `test_config_cap_removed.py` updated to the low/med/high preset set.
- Ran real 2B generations locally to verify: the resolution sweep (unique keys per
  preset, tokens scale low<med<high), `--failed-only`, and G1+G4 end to end with
  `check_run` green.

## Next step (not done)

- **The full `local_test.yaml` run has NOT been launched.** The parser env and the
  parser fix are both verified, so the pieces are in place. Next step: run it in the
  background with the local GPU env and `check_run` it. TL/TLV will be slow on CPU
  paddle (~5-6 min/page), so the whole thing is a long run.

## Open items

- **32B has no home now.** Reducing the H100 to the 8B headline dropped the
  `g1-size-32b` run, so the size sweep is 2B/4B/8B. If you want the 32B point back it
  needs its own H100 spec (and the full prestage, not the h100_main.json trim).
- **`h100_optional.yaml` (G2) was removed** and the runbook is main-only, so G2
  isn't staged by `h100_main.json` (it has no retrievers). Whoever runs G2 needs the
  retrievers prestaged.
- **Corpus `per_bin` / `ids` sampling isn't wired into generation** (only `full` and
  `limit` take effect in the generate path). Fine for the full run and the `limit`
  smoke; a spec that expects a per_bin subset would silently run full.
- **`REPO_STRUCTURE.md` not regenerated** for the new `check_run.py` (run
  `python -m ops.scripts.dump_docstrings` when you want the file map refreshed).
- **Annotations must be complete for real runs.** The smokes use
  `--allow-unlabelled`; drop that for kaya.yaml/h100_main.yaml or the bins come out
  blank and the bin-axis analysis breaks.
