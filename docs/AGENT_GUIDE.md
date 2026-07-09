# Agent guide (decisions, frozen interfaces, reference)

The coding agent's reference for how this repo is built: the fixed decisions, the
tree-to-paper map, the frozen interfaces, the caching contract, and the
implementation reference for models/data/tools/evaluation. It absorbed the former
`MODELS.md`, `DATA.md`, `TOOLS.md`, and `EVALUATION.md`.

Where to look for what:

- `docs/pivot_v4.md` - the v4 decision record: what changes and why (the
  cost-ordered ladder, manual binning, reworked RQs, four tasks, telemetry
  schema, hardware split). The newest intent; the "Fixed decisions" below are
  reconciled to it.
- `README.md` - how the pipeline is actually built **today** (the cell, the
  ladder, the tasks, the repo map). The mechanism ground truth for the current
  code, which still implements the v3 mechanisms; the v4 code migration is
  pending.
- `docs/PROJECT_SPEC.md` - what the paper claims and why (thesis, RQs, gates), v4.
- `docs/USER_GUIDE.md` - the same thesis plus the local runbook.
- `kaya/KAYA_AGENT_GUIDE.md` - Kaya operations, SLURM hazards, offline-cache setup.
- **this file** - fixed decisions and the implementation reference.

Treat the fixed decisions and frozen interfaces as binding; changing one is a
checkpoint recorded here, not a silent edit. You may edit this file directly to
record implementation-relevant decisions and deviations.

## v4 pivot: adopted, code migration pending

`docs/pivot_v4.md` is the current intent and supersedes the v3 spec on five areas.
The decisions below are reconciled to it, but **the code still runs the v3
mechanisms** (README is the ground truth for what executes today). The deltas the
coding agent still has to land:

- **Ladder (cost-ordered, not cumulative).** `T` becomes **PyMuPDF embedded text**
  (digital-born only); `TL` becomes **parser-derived markdown** that *replaces*
  `T`'s text; **bounding-box JSON is dropped entirely**. `TLV` = parser text +
  image; `V` unchanged. `T ⊄ TL`. Names kept; "L" is now vestigial.
- **Parser comparison** at `TL`/`TLV`: PyMuPDF floor, PaddleOCR-VL, MinerU 2.5,
  Unlimited OCR, scored by downstream QA accuracy. Marker dropped from the set.
  Each parser is an isolated env, pre-warmed to the parser cache in the pre-pass.
- **Binning by manual annotation** (`bin_label` = text-dominant / mixed-modality /
  visual-dominant; `scan_label`; exploratory `dominant_visual`), not native
  `doc_type`. Stamped per-cell so tables need no join.
- **Tasks consolidate 6 -> 4:** G1 (oracle ladder + all sweeps as YAML runs), G2
  (retrieval, two scorers), G3 (hallucination/prompting), G4 (routing/classifier).
- **Telemetry schema fixed** (`pivot_v4.md` §6): per-cell (bin/scan labels, machine,
  cap_used, text/visual token split, prefill/decode latency, peak VRAM,
  oom/skip reason), per-run manifest, retrieval side-artifact. Collected every run.
- **Hardware split marked per run** (`machine: kaya | supervisor`) with the
  evidence-page partition (Kaya <=2 pages, supervisor >2), pooled at build time.
- **Answerable/unanswerable split:** ~250 unanswerable pulled out of RQ1/RQ2,
  used only in the RQ3 hallucination study.

Kept from v3: the thesis, the deployable/analytical axes, doc-level bootstrap CIs,
the 3-point margin, the judge-family-≠-reasoner κ gate, the frozen pipeline shape,
the two-layer cache, the YAML/judge/build role split, and the `T/TL/TLV/V` names.

## Fixed decisions (reconciled to v4)

The study is one EACL thesis: **the representation an MP-VRDU system needs is a
function of document type**, sharpened toward a deployment / practical-use framing.

- **Dataset:** MMLongBench-Doc primary (only source with doc type, evidence-modality
  labels, gold pages, and the unanswerable signal). Table 4 is a held-out
  MMLongBench document subset (not LongDocURL, which is kept as a possible future
  replication).
- **Hardware split (v4, marked per run):** two machines, **Kaya** (2xV100 16GB,
  sm_70, no FlashAttention-2) and the supervisor's **A100 / H100**. Every YAML run
  carries `machine: kaya | supervisor`. The size sweep (esp. 32B), quantization
  sweep, resolution sweep, and the RQ2 k-sweep (to k=10) are supervisor-only. The
  evidence-page partition sends <=2 evidence-page questions to Kaya (cap can be
  raised) and the >2 remainder (~50 q) to the supervisor at the same fixed cap,
  pooled at build time via per-cell `machine`/`cap_used` and per-run
  `evidence_page_filter`. The 8B primary runs on 2xV100 (bf16, `device_map="auto"`
  shards it) or on 1xV100 via 4-bit quantization.
- **Reasoner:** Qwen3-VL-8B primary; InternVL3-8B replicates the RQ1 headline only;
  2B is also the smoke model. The size sweep (2B/4B/8B/32B) and quantization sweep
  (4/8/16-bit) are **cost-frontier** studies on supervisor hardware (accuracy-per-
  VRAM, accuracy-per-latency), placement decided post-hoc. All behind one `Reasoner`
  ABC. Closed models are for comparison/judging, not the deployment recommendation.
- **Ladder (v4, cost-ordered):** `T` (PyMuPDF embedded text, digital-born only),
  `TL` (parser-derived markdown that *replaces* `T`'s text; **no bbox JSON**),
  `TLV` (parser text + page image), `V` (page image). `T ⊄ TL`; ordered by compute
  cost. The parser at `TL`/`TLV` is under comparison (PyMuPDF floor, PaddleOCR-VL,
  MinerU 2.5, Unlimited OCR; Marker dropped), each an isolated env pre-warmed to the
  parser cache. Only `TLV`/`V` attach images (modality boundary, enforced in
  `Payload`). *Current code still runs the v3 ladder (Marker text + serialized bbox
  JSON, PaddleOCR for scanned); see README section 4. Migration pending.*
- **Binning (v4, manual annotation)** in `annotations/doc_labels.csv`: `bin_label`
  = **text-dominant / mixed-modality / visual-dominant** (document-level dominant
  modality, not scan status, not page count), `scan_label` = digital / scanned, and
  exploratory `dominant_visual`. Native `doc_type` encodes domain, not modality, so
  it is not the bin axis. *Current `data/binning.py` still exposes the v3 Option-A
  domain bins (text_heavy = Administration/Industry file + Academic paper + Research
  report/Introduction, 578 Q / 70 docs; in_between = Financial report + Guidebook +
  Tutorial/Workshop, 412 Q / 50 docs; visual_heavy = Brochure, 101 Q / 15 docs);
  the manual-annotation bins are the pending swap behind the same signature.*
- **Metrics:** cost = latency@batch1 (primary), with **prefill/decode split**;
  text/vision tokens, peak VRAM, truncation (secondary); sufficiency margin 3 points
  (sensitivity {2,3,5}); **document-level** bootstrap 95% CIs (1000 resamples over
  docs, not questions). The full telemetry schema (`pivot_v4.md` §6) is collected on
  every run.
- **Judge:** a *different family* from the reasoner, gated by Cohen's kappa >= 0.75
  vs 200 hand labels **on the answerable-only set**. `GeminiJudge` (gemini-2.5-flash,
  default, free tier) and `GPT4oMiniJudge` (OpenAI, paid); `StubJudge` is offline
  plumbing.
- **Retrievers (v4, cost rungs):** text = BM25 / BGE-M3 / Qwen3-Embedding-4B; vision
  = ColModernVBERT / ColQwen2.5-v0.2 / ColQwen3-4B; plus free post-hoc joint unions
  (matched-tier pairs, k in {1,3,5} per method). RQ2 inference runs at `TLV`/`V`
  only; the k-sweep (to k=10) runs on the supervisor. *Current code ships BM25 + BGE
  (`BAAI/bge-small-en-v1.5`) text and ColQwen (`vidore/colqwen2.5-v0.2`) vision only;
  the extra rungs and joint unions are pending.*
- **Paths:** root-relative; `.cache/ .data/ envs/ results/ logs/` under the repo
  root on both machines. Two-machine Kaya model (local edit + sync; login for
  staging; compute for offline GPU jobs); all Kaya source/config/docs in `kaya/`.

## Architecture (tree <-> paper)

Data flow: `Question` -> `InputConditioner.condition` -> render pages ->
`Representation.build` -> `Payload` -> `ModelInput.from_payload` ->
`Reasoner.answer` -> `Prediction` -> `Judge.score` -> `Score` -> `ResultRow`
(cached). `Retriever` and `DocTypeClassifier` are covariates. README section 1
narrates the same flow.

| Path | Role |
|---|---|
| `schema.py` | Data contracts: `Question`, `PageSet`, `Page`, `Payload`, `Prediction`, `Score`, `TextPart`/`ImagePart`. |
| `config.py` | `ExperimentConfig` + root-relative `ProjectPaths`; resolution presets, per-size caps. |
| `data/{loader,render,binning}.py` | Loader, PDF->pages substrate, Option-A binning. |
| `pipeline/conditioner.py` | Stage A: `Oracle`, `RetrievedTopK`, `FullDoc`, `BuriedOracle`. |
| `pipeline/representation.py` | Stage B: `T`/`TL`/`TLV`/`V` ladder; modality boundary. |
| `pipeline/reasoner.py` | Stage C: `Reasoner` ABC (swap point). |
| `pipeline/judge.py` | Stage D: `StubJudge`, `GPT4oMiniJudge`, `GeminiJudge`. |
| `pipeline/orchestrator.py` | Composes A->D per cell; owns the two cache layers. |
| `models/` | `ModelInput` + adapters; `get_reasoner(spec)` registry; Qwen/InternVL backends. |
| `covariates/{retriever,classifier}.py` | Retrieval + doc-type classifier covariates. |
| `tools/{text,layout,visual}.py` | Non-VLM channel functions the composers call. |
| `metrics/*` | accuracy (doc-level CI), retrieval, cost, frontier, abstention. |
| `experiments/G*_*.py` + `base.py`, `registry.py` | One `GenerationTask` per file (G1..G6); base is the ABC + cell factories; registry collects them. |
| `experiments/driver.py` | The generate (GPU) + judge (local) engine; phase-2 guards; `config_from_args`. |
| `experiments/side_artifacts.py` | Shared retrieval/classifier side-artifact writers (G5/G6 and YAML both call them). |
| `experiments/{paths,corpus,yaml_spec}.py` | Cache/table layout + status/logging; corpus resolver; YAML spec loader. |
| `reporting/tables/` (one `T*_*.py` per table) + `build.py` | Per-table builders + the table -> source-task routing that writes CSVs + `.md`. |
| `gates/{core,viewer}.py` + `__main__.py` | Section-2 gate logic; shared cached-cell viewer; `python -m gates` CLI. |
| `cli/{generate,judge,build}.py` | The three runnable roles: generate on GPU, judge/build locally. |
| `scripts/*` | Standalone utilities (probes, inspection, annotation, staging). |

**Frozen interfaces (Stage-3 freeze; change only via a checkpoint recorded here):**
`schema.py` contracts; `models/payload.py::ModelInput` +
`from_payload`/`to_chat_messages`/`to_local_prompt`;
`InputConditioner.condition(question, page_count)`; `Representation.build(pages)`
(takes rendered `Page`s, not a `PageSet`, so the composer stays a pure
page-encoder); `Reasoner.answer(question, model_input)`;
`Judge.score(question, prediction)`; `Retriever.retrieve(question, page_count, k)`;
`DocTypeClassifier.classify(question)`; the orchestrator cache key + `ResultRow`
shape. Additive optional kwargs and side caches behind these are not freeze
changes.

**Checkpoint 2026-07-09: `visual_resolution` added to the cell key + `ResultRow`.**
Resolution used to be a per-run manifest field, deliberately *out* of the key, so a
resolution sweep meant one run per preset. It is now a per-cell axis: the preset is
part of both cache keys and is stamped on `ResultRow`/`CachedPrediction`, so a
single run can sweep resolution via `visual_resolutions` and the presets never
collide. A lower-res image is a genuinely different (lossier) input, so this is the
honest identity. Machine-independence holds (resolution is a config value, not a
device property). `IDENTITY_FIELDS` in reporting gained `visual_resolution` to
match, and `resolution.build` now pivots by the per-cell preset.

**Caching (two layers, both under `results/cache/`).** (1) `ResultCache` - one
`ResultRow` per cell keyed by SHA-256 over `{question_id, doc_id, condition,
representation, model_spec, page_indices, visual_resolution, judge_spec}`;
idempotent + resumable from disk. (2) `PredictionCache` (additive) - the reasoner
output keyed the same way **minus judge_spec**, so one prediction is scored by any
judge without re-running the model. `k` is encoded in the conditioner name
(`retrieved_k3`). Model spec and resolution are in both keys, so scaling / family /
resolution sweeps produce distinct, mergeable rows. dpi is *not* in the cell key
(it keys the render/parser disk caches instead).

**Swap point.** The pipeline never imports a backend; it asks `get_reasoner(spec)`
for a `Reasoner` and hands it a `ModelInput`. Adding a Qwen size or
InternVL/GPT/Gemini is a new registry entry, no pipeline change.

## Role split: YAML generate, artifact judge, build

Generation is **YAML-first**. A YAML spec defines one or more data-collection runs
as explicit cell grids over questions, conditions, representations, and model
specs. `cli/generate.py --spec <file.yaml>` is the only GPU entry point; the spec
carries all run config. (`experiments/driver.py` still has the `run_generate` /
`run_judge` registry-selector loops for tests, but no CLI exposes them.) Judge/build
are artifact-driven: they read manifests/predictions/results/side artifacts under a
run tag, so they don't repeat the generate flags.

- **YAML specs.** `specs/full_generation.yaml` (G1/G2/G3/G5/G6),
  `specs/smoke_generation.yaml` (small smoke). Specs support arbitrary ordered
  channel combinations over `T`/`L`/`V`, but the paper experiments use the rung set
  `[T, TL, TLV, V]` only. Cache dirs are
  `results/cache/<run-tag>/<smoke|full>/<run-name>/`.
- **Bridge.** `experiments/yaml_spec.py` loads YAML into dynamic `GenerationTask`
  objects; `experiments/driver.py` owns the generate loop, reasoner/retriever
  construction, parse pre-pass, and cache writes.
- **Shared side-artifact writers (checkpoint 2026-07-07).** The retrieval and
  classifier side artifacts have one implementation in
  `experiments/side_artifacts.py` (`write_retrieval_eval` / `write_classifier_eval`).
  Both the fixed `G5Retrieval`/`G6Classifier.run_side` and the dynamic
  `YamlGenerationTask.run_side` call it, so the two task systems can't drift.
  `write_retrieval_eval` takes the ordered `(modality, k)` pairs to score, so each
  caller keeps its own selection (G5: both modalities across the k-sweep; a YAML run:
  the pairs in its retrieved conditions) and its emission order stays byte-identical
  to before. `run_side` is not a frozen interface.
- **Table routing (replaces `depends_on`).** `reporting/build.py`'s `TABLES`
  registry declares each table's source task(s): table1<-G1, table3<-G1+G2,
  table4<-G3, table6<-G5 (+retrieval side), table7<-G1 (+classifier side),
  table8<-G4. The `tables.py` builders mostly don't filter by `model_spec`, so a
  table is only correct when handed exactly its source tasks' rows.
- **Why roles split across machines.** Reasoner/retrievers/classifier need a GPU;
  the judge needs the internet - on Kaya those never coexist. Generation runs on
  Kaya (GPU, offline); `cli.judge` and `cli.build` run **locally** after a
  `kaya.kaya pull`. Judge loads no models; it scores `predictions.jsonl` directly.
  Judge keys stay in the local `.env` (only `HF_TOKEN` is forwarded to Kaya).
  Run-tagged builds write tables to `results/tables/<mode>-<run-tag>/`.

**Default per-bin subset for full runs.** A full mmlongbench run defaults to ~100
questions per Option-A bin (`ExperimentConfig.per_bin_sample`, default 100;
`sample_seed` default 0) instead of all 1091.
`experiments/corpus.py::sample_questions_per_bin` draws **whole documents** per bin
until the bin reaches the target; bins below target are kept whole (visual-heavy
stays 101 Q / 15 docs). Default subset = 309 Q. CLI: `--per-bin-questions N` (0 =
whole corpus), `--sample-seed N`; an explicit `--questions N` cap still overrides.
mmlongbench full runs only (smoke and LongDocURL ignore it). **Gate provenance:**
the F1 frontier gate as specced wants the whole corpus, so a subset run is a fast
preview; record which one an F1 verdict came from.

**Quantized reasoner (model-spec suffix).** To run the 8B on one 16GB V100,
quantization is a spec suffix, not a cache-key field: `qwen3vl-8b-local-4bit` /
`-8bit`. `ModelSpec.parse` strips the trailing `-4bit`/`-8bit` into
`ModelSpec.quantization` while `name` keeps the full string, so quantized runs get
their own cache rows and `size` still resolves to `8b`. `get_reasoner` passes it to
`LocalVLMBackend` (bitsandbytes 4-bit NF4 double-quant or 8-bit).
`--quantization` on `cli.generate` must match between generate and judge.
`bitsandbytes==0.49.2` is in the remote env only. Mains stay bf16; 4-bit is
single-GPU iteration + a possible appendix quant-sensitivity row.

**Full-run knobs.** `--visual-resolution {full,high,med,low,min}`
(`config.visual_resolution`) fixes the per-page vision-token budget for every
reasoner, overriding the size-aware default. `--run-tag TAG` namespaces the whole
cache tree under `results/cache/<TAG>/` and tables under
`results/tables/<mode>-<TAG>/` (needed so concurrent jobs don't corrupt the render
/ prediction caches); judge/build must pass the same tag. `cli.build` also writes
one combined `all_tables.md` with all eight tables. None of these are in the cache
key, so clear or re-tag the cache when changing resolution for one spec.

## V100 constraints (resolved; final state)

Kaya's V100s are Volta (sm_70): no FlashAttention-2, so attention can fall back to
a math kernel that materializes the full `[heads, seq, seq]` score matrix
(O(seq^2)). Long multi-page cells OOM even after quantization, and the 8B doesn't
fit one V100 in bf16. The mechanisms that keep cells inside 16GB (all additive, no
freeze change; README sections 5 and 7 narrate them):

- **Efficient-attention kernel** forced in the backend (cutlass SDPA, O(seq),
  runs on Volta); a harmless no-op that prefers the efficient kernel on an A100.
- **Per-size vision-pixel cap** (`config.max_pixels_for_spec` / `MAX_PIXELS_BY_SIZE`;
  8B ~768 tok/page, 32B ~520, 2B/4B ~1280), overridable by `--visual-resolution`.
  Applied via the per-image `max_pixels` key `qwen_vl_utils` honors.
- **Per-size input-token cap** (`config.max_input_tokens`; 8B 4096, 32B 3072).
  `LocalVLMBackend._truncate_context` trims the *text* to the budget after
  reserving for images + template, **keeping every image placeholder** (images
  first, then trimmed text). This truncates very long `T`/`TL`/`TLV` cells on the
  main runs; forced by V100 hardware. The `TL` bbox-JSON is the main offender and a
  candidate for a more compact serialization later.
- **2xV100 shard headroom:** `LocalVLMBackend._max_memory_map` reserves ~5GiB/GPU
  when sharding so activation/KV peaks don't tip a GPU over.
- **Many-page cells:** vision tokens are not total-capped, so a question with many
  gold pages can still OOM. Handled by (a) dropping questions with >10 gold
  evidence pages up front (`experiments/corpus.py::load_questions`,
  `MAX_EVIDENCE_PAGES=10`, 7 questions on the full corpus, applied *after* per-bin
  sampling so it doesn't perturb which docs get drawn), and (b) per-cell skip: a
  cell that raises is logged, the GPU freed, and the loop continues
  (`--continue-on-error`). Both phases resolve the same filtered set so judge/build
  agree. A true total-vision cap remains the "correct" fix and is unimplemented.

**Parser/reasoner co-residence.** The Marker/Surya parser, the retriever, and the
reasoner must never share VRAM. Marker/OCR output is disk-cached
(`results/cache/.../marker/`, `.../ocr/`); the **parse pre-pass**
(`driver.py::generate` + `Orchestrator.prewarm_cell`) warms condition->render->build
for every cell with the reasoner *not* loaded, then unloads retrievers and frees
the GPU. `free_gpu()` (gc + `empty_cache` + `synchronize`) runs after the pre-pass,
after each spec's reason pass (`LocalVLMBackend.free()` drops weights), and after
`run_side`. Retrievers gained `unload()`. On a warm cache the reason phase never
loads Surya/PaddleOCR.

## Judge-phase robustness

- **Transient retry.** `pipeline/judge.py::_with_retry` wraps both API judges with
  exponential backoff on 429/5xx; non-transient (400/401) still raise on the first
  try. Free-tier gemini flash returns sporadic 503s that used to kill a whole judge
  run even though scored rows are cached.
- **Partial-cache tolerance.** The judge only re-scores cached predictions; a cell
  generate never produced hits a guard that raises `CacheMiss` (subclass of
  `RuntimeError`). `--continue-on-error` makes the judge skip those and log the
  count, so a partial cache still builds a partial table.

## Section-2 gates (F1-F6)

Gate tooling is `gates/core.py`, exposed via `python -m gates`;
run-tag-aware path resolution via `experiment_paths`. Commands in
`docs/USER_GUIDE.md` (Runbook).

- **F1 frontier divergence.** `gates frontier` (`python -m gates`) reads the full Table-1 CSV,
  returns Go when >=2 Option-A bins have different frontiers. Pending the full
  Qwen3-VL-8B run.
- **F2 judge-human agreement.** `agreement-sample [--render]` writes the 200-row
  labelling sheet (+ a page-image viewing packet reusing `gates/viewer.py`);
  `agreement-score` computes Cohen's kappa over `correct`/`incorrect`/`abstained`,
  gate 0.75. Pending human labels.
- **F3 classifier feasibility.** `classifier-pilot --full` samples 100 docs, runs
  the first-two-page Qwen3-VL-2B classifier, gates top-1 bin accuracy at 0.70.
  Pending.
- **F4-F6.** Implemented, not yet run at full scale. F4: Table-2 analytical slice,
  the InternVL3-8B Table-3 backend, and the (held-out MMLongBench) Table-4
  experiment. F5: evidence-composition mediation + matched-vs-cross retrieval rows
  from real cached predictions/retrieval records. F6: one corpus-level row per
  routing policy, amortizing classifier latency as total classifier time / evaluated
  rows.

## Inspection + annotation tooling

- `gates/viewer.py` (+ `scripts/inspect_results.py`): join a task's
  `predictions.jsonl` + `results.jsonl` back to the `Question` + PDF, render the fed
  pages, dump each cell into `./inspect/<slug>/` (copied PDF, page PNGs, an
  `info.md` listing every `CachedPrediction`/`ResultRow` field). Filters by
  question/doc/representation/condition/incorrect/abstained. Reuses the run's shared
  render cache. Limitation: the judge's free-text rationale is not persisted.
- `scripts/annotate_docs.py`: interactive, resumable per-document annotation of the
  135 docs (text/visual bin, scanned vs digital, dominant visual element,
  multi-column), seeded with `doc_type_bin` + `classify_scanned` priors, writing
  `annotations/doc_labels.csv` (committed, not gitignored) after every doc. `score`
  reports human-bin-vs-`auto_bin` agreement (tests the three-domain assumption) plus
  the scanned fraction.

## Environment and dependencies

Kaya env: `envs/mpvrdu` (Python 3.11, `requirements.txt`). Local RTX 5070
(Blackwell, sm_120) env: `envs/mpvrdu-local-gpu` (`requirements-local-rtx5070.txt`,
torch 2.8+cu128, no vLLM) because the Kaya `torch==2.7.0+cu126` wheel has no sm_120
kernels. Both `pip check` clean.

The repo pulls four heavy, fast-moving stacks into one env - vLLM (serving),
ColPali/ColQwen (retrieval), Marker/Surya (parsing), PaddleOCR/PaddleX (OCR) - plus
Qwen3-VL, which only landed in `transformers` 4.57. Each pins
`transformers`/`torch`/`pillow` independently and they barely overlap:

- `transformers==4.57.6`: Qwen3-VL needs >=4.57; colpali `<4.58`; surya `>=4.56.1`;
  marker `<5`; vLLM `>=4.51.1`. Usable window is essentially just 4.57.x.
- `torch==2.7.0` (+cu126): vLLM 0.9.2 pins it **exactly**; that pin dominates.
- `pillow==10.4.0`: marker/surya require `<11`.
- `huggingface_hub==0.34.4` (transformers 4.57 needs >=0.34); added `hf_xet`.
- `paddleocr==3.1.0` must pair with `paddlex 3.1.x`.
- `openai` is **capped at <=1.90.0 by vLLM 0.9.2** - do not raise it. `google-genai`
  is already transitive via Marker, so the Gemini judge adds no new dependency.

Mitigation: isolate the one exact-pinned troublemaker (vLLM) in a separate local
env without it.

---

# Implementation reference

Condensed from the former `MODELS.md`, `DATA.md`, `TOOLS.md`, and `EVALUATION.md`.
The frozen contracts are above; this is the "how each layer behaves" reference.

## Models (reasoner backends + prompt)

- **Load path.** `transformers==4.57.6` exposes `Qwen3VLForConditionalGeneration` /
  `...MoeForConditionalGeneration` / `Qwen3VLProcessor` without moving Marker,
  Surya, vLLM, or ColPali outside their windows.
- **Registry.** `qwen3vl-{2b,4b,8b,32b}-local` -> the shared HF backend
  (`Qwen/Qwen3-VL-*-Instruct`); `internvl3-8b-local` ->
  `models.internvl.LocalInternVLBackend` (`OpenGVLab/InternVL3-8B`, same
  `Reasoner.answer` contract). A trailing `-4bit`/`-8bit` selects a
  bitsandbytes-quantized load. Other families stay stubbed.
- **Frozen prompt.** Qwen template `m3-qwen3vl-v1` (InternVL `f4-internvl3-v1`), one
  fixed template across the four rungs; `ModelInput.to_local_prompt()` supplies
  `{context}` and each `<image>` placeholder becomes a Qwen image block in page
  order. Decoding is greedy (`do_sample=False`).
- **Accounting** per `Prediction`: `input_text_tokens` (image placeholders
  stripped), `input_visual_tokens` (Qwen `image_grid_thw` estimate), `output_tokens`,
  `latency_s` (batch=1 wall clock), plus metadata (backend, model id, template
  version, `max_new_tokens`, `max_pixels`, `max_input_tokens`, `quantization`, image
  count).
- **Closed models** are comparison/judge only, behind the same ABC via
  `ModelInput.to_chat_messages()`; the pipeline never imports vendor SDKs.

## Data layer

- **Paths** (root-relative both machines): dataset `.data/mmlongbench`, parquet
  `.data/mmlongbench/data/*.parquet`, PDFs `.data/mmlongbench/documents/*.pdf`,
  render cache `results/cache/renders/<pdf-stem>__dpi<N>/page_XXXX.png` (144 DPI
  base).
- **`load_mmlongbench()` -> `Question`:** `id` (`mmlongbench:000000`), `doc_id`,
  `question`, `gold_answer`, `answer_format`, `doc_type`, `evidence_pages`
  (normalised 1-based -> 0-based; original in `raw_fields`), `evidence_sources`,
  `hop` (from evidence-page count), `is_unanswerable` (gold == "Not answerable"),
  `raw_fields` (+`source_dataset="mmlongbench"`).
- **`render_question_pages()`** resolves the PDF and renders the gold pages;
  unanswerable questions with no gold pages render page 0. Each `Page` carries the
  0-based index, PDF path, optional cached PNG, and PyMuPDF line spans.
- **LongDocURL loader** (`load_longdocurl()`) still exists but Table 4 no longer
  uses it; kept for a possible future replication.

## Tools (non-reasoner channels)

- **Primary ladder text/layout.** `tools.text.text_channel(pages)` feeds
  `T`/`TL`/`TLV`: digital-born docs -> `tools.layout.marker_text(pages)`, scanned
  docs -> `tools.text.ocr(pages)`, routed by `annotations/doc_labels.csv`
  (`scan_label` if filled, else `auto_scan`). `marker_bbox_json(pages)` is the
  layout channel for `TL`/`TLV`; `tools.visual.full_page(pages)` /
  `resolution(pages, scale)` feed `TLV`/`V`. Marker (`marker-pdf==1.10.2`, run
  without LLM mode) is primary for digital text/layout. Marker output disk-cached
  under `results/cache/marker/`, OCR under `results/cache/ocr/` (PyMuPDF fallback is
  not cached).
- **Appendix/fallback.** `tools.text.embedded` (PyMuPDF), Docling parser-swap, and
  `tools.visual.region_crop` which degrades to full page (MMLongBench has no in-page
  boxes). The pymupdf fallback in `marker_bbox_json` exists only so local tests run
  before Marker is installed; `prestage --smoke` calls Marker with
  `allow_fallback=False`.
- **Prestage.** `scripts/prestage.py [--smoke]` stages Qwen weights, BGE, ColQwen,
  Marker/Surya, PaddleOCR, Docling (idempotent, offline-probing).

## Evaluation

- **Judge.** `GeminiJudge` (gemini-2.5-flash, default, free tier) and
  `GPT4oMiniJudge` (OpenAI, paid); `StubJudge` offline plumbing. Each returns verdict
  (`correct`/`incorrect`/`abstained`) + extracted answer + rationale; an abstaining
  verdict on a native-unanswerable question counts correct. Keys in the local `.env`
  only. Gate F2 computes Cohen's kappa vs 200 human labels, gate 0.75.
- **Accuracy.** `metrics.accuracy.accuracy_summary()` = mean correctness + 95%
  bootstrap CI resampled at the **document level** (draw `doc_id`s with replacement,
  take all their rows), 1000 draws, seed 0.
- **Cost.** `metrics.cost.cost_summary()` = mean latency@batch1 (primary) + split
  text/vision/output token sums (secondary).
- **Frontier.** `metrics.frontier.sufficiency_frontier()` orders `T->TL->TLV->V`;
  the frontier is the cheapest rung whose upper CI reaches within the margin
  (default 3 points) of the strongest rung's point estimate. `gates frontier` (`python -m gates`)
  gates F1 (Go when >=2 Option-A bins differ).
- **Retrieval.** `metrics.retrieval` scores page precision/recall/F1 vs gold
  `evidence_pages`, sliced by `<retrieval-modality>:<evidence-source>` (e.g.
  `text:table`) so matched/cross separates locating from evidence modality.
- **Composition (Table 5).** Each bin decomposed into normalized evidence-source
  shares (text/table/chart/figure/layout, summing to 1); predicted bin frontier =
  strongest per-modality frontier among modalities with >=10% share.
- **Classifier (Table 7 covariate).** `QwenDocTypeClassifier` renders the first two
  pages, builds `TLV`, asks Qwen3-VL-2B for a native doc_type, maps it through
  Option-A binning. Predicted routing counts classifier cost as total classifier
  latency / evaluated rows, its own `classifier_latency_bs1_s` column. Gate F3 gates
  top-1 bin accuracy at 0.70.
- **Tables 1-8** are emitted by `reporting.tables`; **Table 4 is a held-out
  MMLongBench subset** (disjoint documents for text_heavy/in_between, reused
  visual_heavy), binned by the same three domains as Table 1.
