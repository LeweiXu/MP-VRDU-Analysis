# DECISIONS — pivot-v4 changelog + probe/env verdicts

This is the pivot-v4 decision log called for by `pivot_v4_implementation.md`. It
records the do-over reference point, the Phase-1 probe/env verdicts, and every
real judgement call made while building v4 (one line each: what, why, what it
affected).

> Note on where this lives: the repo's `CLAUDE.md` says the DECISIONS content was
> folded into `AGENT_GUIDE.md`. The v4 implementation plan reintroduces a standalone
> `docs/DECISIONS.md` as the pivot changelog. This file is that changelog. Final
> reconciliation of `AGENT_GUIDE.md` / `PROJECT_SPEC.md` / `README.md` to v4 (and
> folding `pivot_v4.md` in here) happens in the Phase-4 cleanup, not now.

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
| `dump_docstrings.py` | `ops/scripts/` | copied-pending-rework (stale SUMMARY_OVERRIDES) |
| `annotate_docs,prestage,inspect_results,run_probe` | `ops/scripts/` | copied-pending-rework (per plan §Phase 2) |
| `ANNOTATION_GUIDE.md` | `ops/scripts/` | clean copy |
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
