# Agent guide — structure, frozen interfaces, implementation reference

The coding agent's reference for how this repo is built: the repository structure,
the frozen interfaces you must not break, the caching contract, and the per-layer
implementation reference for models / data / tools / evaluation.

This file describes the system **as it is now**, present tense. It does not narrate
how the code changed over time — that is `docs/DECISIONS.md`, the only file that
carries history. It does not restate the experiment concept — that is `README.md`.
Every fact lives in one place: when this guide needs a concept the README owns (the
cell, the ladder, the bins, the tasks), it points there rather than re-describing
it. See `CLAUDE.md` for the documentation discipline.

Treat the **frozen interfaces** and the **caching contract** as binding. Changing
one is a checkpoint recorded in `docs/DECISIONS.md`, not a silent edit. You may
edit this file directly to keep the implementation reference accurate, in the same
commit as the code change it describes.

---

## Repository structure (tree ↔ paper)

The science spine is flat at the root; operational tooling groups under `ops/`. A
file's home is decided by what it *is*, not who imports it. The full tree with the
generic per-file purpose is in `README.md` ("Repository structure") and, generated
from the docstrings, in `docs/REPO_STRUCTURE.md`. The role map below is the
agent-facing view: which file owns which paper-facing responsibility.

| Path | Role |
|---|---|
| `schema.py` | Frozen data contracts: `Question`, `PageSet`, `Page`, `Payload`, `Prediction`, `Score`, `ResultRow`, `TextPart`/`ImagePart`, plus the per-cell telemetry fields. |
| `config.py` | `ExperimentConfig` + root-relative `ProjectPaths`; cache version, resolution presets, bins, prompt modes, sampling defaults. No input-token cap. |
| `data/loader.py` | MMLongBench (and LongDocURL) rows → `Question`; answerable / unanswerable split. |
| `data/annotations.py` | Read + validate the per-document label sheet. |
| `data/binning.py` | Stamp `bin_label` / `scan_label` onto each `Question`. ⚠ PENDING v5 (bin source under revision — see README §3). |
| `data/render.py` | PDF page → cached PNG + embedded-text spans. |
| `tools/text.py` | Cheap embedded PyMuPDF text (the `T` channel). |
| `tools/parser.py` | Parser markdown for `TL`/`TLV`, read from a disk cache; warms the cache by running a parser in its isolated env. |
| `tools/parser_worker.py` | Subprocess entry the isolated parser env runs (no project imports). |
| `tools/visual.py` | Page-image parts + vision-token estimation from the resolution preset. |
| `retrievers/{__init__,text,vision,joint}.py` | `Retriever` ABC + ranking/memoization; the text and vision cost rungs; the free joint union. |
| `models/__init__.py` | `ModelSpec` parse + `get_reasoner(spec)` registry (the model-family swap point). |
| `models/{qwen3vl,internvl}.py` | Reasoner backends behind one ABC. |
| `models/classifier.py` | First-pages modality-bin classifier (routing side tool). |
| `models/payload.py` | Backend-neutral `ModelInput` + chat / local adapters. |
| `pipeline/conditioner.py` | Stage A: page selection — `oracle` / `retrieved-topk` / `similarity` / `full`. |
| `pipeline/representation.py` | Stage B: the `T`/`TL`/`TLV`/`V` composer; the modality boundary (only `TLV`/`V` attach images). |
| `pipeline/reasoner.py` | Stage C: `Reasoner` ABC (the swap point) + per-cell prompt instruction. |
| `pipeline/judge.py` | Stage D: `StubJudge`, `GeminiJudge`, `GPT4oMiniJudge`. |
| `pipeline/orchestrator.py` | Composes the reasoner path (A→C) per cell into an unjudged `PredictionRow`; owns the cell caches + telemetry capture. |
| `scoring/*` | `accuracy` (doc-level CI), `cost`, `frontier`, `retrieval`, `abstention`, `agreement` (judge-human κ). |
| `experiments/tasks/` | The single spec-driven `Task` + the base ABC; `task_name` is a label, not a type. |
| `experiments/engine/` | The generate driver (robustness, `--failed-only`), side-artifact writers, cache/table paths + cell keys. |
| `experiments/corpus/` | Question-set resolver + sampling; flat YAML spec loader. |
| `experiments/registry.py` | Any `task_name` label → the unified `Task`. |
| `reporting/plan.py` | The build plan: each analysis table's source run_tag(s), swept axis, builder, and baseline caption. |
| `reporting/build.py` | Plan-driven table assembly + cell-grouping readers; writes one CSV per table + `all_tables.md` flat to `results/tables/`. |
| `reporting/tables/` | Per-table builders (headline, parser, resolution, scale, composition, routing, matched_cross, kdepth, retrieval_accuracy, hallucination, and the mined telemetry tables) + shared `_common` / `_markdown` / `_load` helpers. |
| `ops/{generate,judge,build}.py` | The three role entry points. |
| `ops/kaya/` | SLURM sync/submit runner + config + cluster guide. |
| `ops/specs/` | YAML run specs. |
| `ops/scripts/` | Standalone utilities (prestage, annotate_docs, resolution_probe, inspect_results, …). |

Data flow: `Question` → `InputConditioner.condition` → render pages →
`Representation.build` → `Payload` → `ModelInput.from_payload` → `Reasoner.answer`
→ `Prediction` → `Judge.score` → `Score` → `ResultRow` (cached). `Retriever` and
the classifier are side covariates, not stages. README §1 narrates the same flow
for the user.

---

## Frozen interfaces

Change only via a checkpoint recorded in `docs/DECISIONS.md`. Additive optional
kwargs and side caches behind these are not freeze changes.

- `schema.py` contracts.
- `models/payload.py::ModelInput` + `from_payload` / `to_chat_messages` /
  `to_local_prompt`.
- `InputConditioner.condition(question, page_count)`.
- `Representation.build(pages)` — takes rendered `Page`s (not a `PageSet`), so the
  composer stays a pure page-encoder.
- `Reasoner.answer(question, model_input)`.
- `Judge.score(question, prediction)`.
- `Retriever.retrieve(question, page_count, k)`.
- `DocTypeClassifier.classify(question)`.
- The orchestrator cell-key composition + the `ResultRow` shape.

**Swap point.** The pipeline never imports a backend; it asks `get_reasoner(spec)`
for a `Reasoner` and hands it a `ModelInput`. Adding a Qwen size or an
InternVL / GPT / Gemini backend is a new registry entry, no pipeline change.

---

## Caching contract (two files, both under `results/cache/`)

1. **`PredictionCache`** (`predictions.jsonl`, written by generate) — one
   `PredictionRow` per cell **including failures**, keyed by SHA-256 over
   `{question_id, doc_id, condition, representation, model_spec, page_indices,
   visual_resolution}` (no judge). Carries the answer, all covariates, telemetry, and
   `status`/`skipped_reason`; no judging. Idempotent and resumable from disk.
2. **`ResultCache`** (`results.jsonl`, written by the judge phase) — one `ResultRow`
   per cell = the `PredictionRow` plus the judge verdict (`result_key`, `judge_spec`,
   `score`, `correct`, `abstained`), keyed by the prediction key **plus `judge_spec`**.
   `results.jsonl` is a strict superset of `predictions.jsonl` (failed cells pass
   through unscored), and one prediction can be scored by any number of judges into
   disjoint rows without re-running the model.

Rules that make the two-machine model and the sweeps work:

- **`k` and the prompt mode are encoded in the conditioner name**
  (`retrieved_text_k3__none`, `oracle__none`), not separate key fields.
- **`model_spec` and `visual_resolution` are in both keys**, so scaling / family /
  quantization / resolution sweeps produce distinct, mergeable rows in a single run.
- **`dpi` is *not* in the cell key** — it keys the render / parser disk caches
  instead.
- **The cell key is machine-independent** (only cell identity + config values, no
  device property, hostname, or count). A re-run on another box produces the same
  key and completes the *same* file — this is what makes the `--failed-only` retry
  a file copy, not a merge (README §13).
- **`visual_resolution` is a per-cell axis**, part of both keys and stamped on
  `ResultRow` / `PredictionRow`; a lower-res image is a genuinely different
  (lossier) input, so this is the honest identity. `IDENTITY_FIELDS` in reporting
  includes it and `resolution.build` pivots by the per-cell preset.

---

## Role split: YAML generate, artifact judge, build

Generation is **YAML-first**. A spec defines one or more data-collection runs as
explicit cell grids over questions, conditions, representations, and model specs.
`ops/generate.py --spec <file.yaml>` is the only GPU entry point; the spec carries
all run config. Judge and build are **artifact-driven**: they read
manifests / predictions / results / side artifacts under a run tag, so they never
repeat the generate flags.

- **One-variable-at-a-time design (base + sweeps).** The experiment is one
  end-to-end pipeline measured at a single baseline configuration; each run changes
  exactly one variable (a *sweep*) off that baseline and holds the rest fixed, so
  every result isolates one variable's effect. The per-task baseline lives in
  `config.BASELINE`. Axes in the frozen cache key (`reasoner_spec` incl. quant
  suffix, `visual_resolution`) sweep under ONE run_tag as a driver-looped list; axes
  not in the key (`parser`, `dataset`) get one run_tag per value. The flat spec files
  are the per-sweep expansion of this design (one file = one sweep).
- **Specs.** `ops/specs/template.yaml` is the reference menu plus the three worked
  tasks; `kaya.yaml` / `h100_main.yaml` are the real runs and
  `kaya_smoke_g{1,2,3}.yaml` the per-task smokes. Cache dirs are
  `results/cache/<run-tag>/<smoke|full>/<task>/`.
- **Bridge.** `experiments/corpus/yaml_spec.py` loads YAML into flat `Spec`s and
  `experiments/engine/` owns the generate loop, reasoner / retriever construction,
  the parse pre-pass, and cache writes.
- **Flat spec format.** Every flat spec lists the full variable set explicitly under
  a `task_name` (the expanded per-sweep form of the base+sweeps design above;
  `task_name` is a label, not a type). A list-valued axis is the set of values to run
  over (cross-product). `dataset` and `parser`
  expand to one run_tag each (`<run_tag>-<dataset>-<parser>`); `reasoner_spec` x
  `quantization` fold into the driver-looped `reasoner_specs` (`bf16` = no suffix);
  `visual_resolution` becomes `visual_resolutions`; `reasoner_representations` /
  `k_values` / `joint_k_values` / `prompt_modes` / `retrieval_representation` are
  cell/benchmark sets within the one run. `retrieval_representation` is `oracle`
  (gold pages) or the reps the retriever ranks over (`T` = PyMuPDF text, `V` = page
  image). Benchmark method lists (`text_retrievers` / `vision_retrievers` / `joints`,
  `joints: matched` = zip of the two lists) are top-level; a run with a non-empty
  benchmark must include `bge-m3` + `colqwen2.5`, and the inference picks
  (`inference_text_retriever` / `inference_vision_retriever`) must be benchmarked
  methods (`SpecError`). `config_from_spec` maps a `Spec` to `ExperimentConfig`;
  `parser_dpi` is the render/parser DPI (keys the render cache, not the cell).
  `decode_budget` (mapping with a required `default`, other keys prompt modes) and
  `final_answer_delimiter` (`none` = whole answer to the judge) are validated at
  parse time and scoped to the run_tag (see the prompt-modes bullet above).
- **Shared side-artifact writers.** The retrieval and classifier side artifacts
  have one implementation in `experiments/engine/`; each caller passes the method
  sets it wants scored. The retrieval benchmark runs as **stage 1 before** the
  reasoner cells (gated on `config.text_retrievers`), persisting rankings the
  inference arms reuse; the classifier runs after inference. `run_side` /
  `run_retrieval_benchmark` are not frozen interfaces.
- **Build (plan-driven, one table per swept variable).** `reporting/plan.py` declares
  each analysis table: its source run_tag(s), the swept axis, and the builder.
  `ops.build` assembles them all and writes one CSV per table plus a combined
  `results/tables/all_tables.md` (flat, no per-run_tag dirs). Each table carries a
  caption naming the swept axis and the held-fixed baseline (from `config.BASELINE`),
  so it is explainable on its own; accuracy grids also carry a per-column `n` footer
  (columns differ where cells OOM). Tables that compare an out-of-key axis merge
  across run_tags (parser: paddle from the representation run + mineru + unlimited;
  scan: a digital run_tag + its scanned half). The mined telemetry tables (prefill
  cost, VRAM headroom, OOM frontier, quant sensitivity, scan-vs-digital) are folded
  into this build. G3 hallucination keeps its own classification shape (abstention by
  prompt mode over the unanswerable pool).
- **Why roles split across machines.** The reasoner / retrievers / classifier need
  a GPU; the judge needs the internet — on Kaya those never coexist. Generation
  runs on the cluster (GPU, offline); `ops.judge` and `ops.build` run locally after
  a pull. Judge loads no models; it scores `predictions.jsonl` directly. Judge keys
  stay in the local `.env` (only `HF_TOKEN` is forwarded to the cluster).
  `ops.build` writes every table flat to `results/tables/` (one CSV each + `all_tables.md`).

**Sampling for full runs.** A full MMLongBench run resolves its corpus from the
YAML `corpus:` scope: `full` (all 1091), `per_bin: N` (draw **whole documents** per
bin to N — doc-coherent, so the doc-level bootstrap CIs stay valid), `per_doc_type: N`
(draw whole documents per native `doc_type`, then cap to **exactly N questions per
label** — so `per_doc_type: 1` runs one question per label; the exact cap can slice
the last drawn document), or a `limit` / id-list for smoke. The pool is declared by
the spec (`corpus.pool`: answerable / unanswerable), applied by
`experiments/corpus/resolve.py::filter_by_pool` — it is a run variable now, not
keyed off the task name. ⚠ PENDING v5: which experiments require the full corpus
(binning-dependent) vs a frozen random subset is being finalised — see README §3
and `docs/DECISIONS.md`.

**Quantized reasoner (model-spec suffix).** Quantization is a spec suffix, not a
cache-key field: `qwen3vl-8b-local-4bit` / `-8bit`. `ModelSpec.parse` strips the
trailing `-4bit`/`-8bit` into `ModelSpec.quantization` while `name` keeps the full
string, so quantized runs get their own cache rows and `size` still resolves to
`8b`. `get_reasoner` passes it to the backend (bitsandbytes 4-bit NF4 double-quant
or 8-bit). Judge reads the reasoner spec from the same `--spec`, so quantization
matches generate automatically. Mains run bf16; 4-bit is single-GPU iteration plus a
possible appendix quant-sensitivity row.

---

# Implementation reference

The frozen contracts are above; this is the "how each layer behaves" reference.

## Models (reasoner backends + prompt)

- **Registry.** `qwen3vl-{2b,4b,8b,32b}-local` → the shared Qwen3-VL HF backend
  (`Qwen/Qwen3-VL-*-Instruct`); `internvl3-8b-local` → the InternVL backend
  (`OpenGVLab/InternVL3-8B`, same `Reasoner.answer` contract). A trailing
  `-4bit`/`-8bit` selects a bitsandbytes-quantized load. Other families stay stubbed.
- **Frozen prompt.** One fixed template across the four rungs (Qwen `m3-qwen3vl-v1`;
  InternVL `f4-internvl3-v1`). `ModelInput.to_local_prompt()` supplies `{context}`
  and each `<image>` placeholder becomes an image block in page order. Decoding is
  greedy (`do_sample=False`, capped `max_new_tokens`).
- **Prompt modes.** The instruction preamble is swappable (`config.PROMPT_MODES`),
  built by composition from named fragments: `none` / `grounded` / `abstain` /
  `abstain_balanced` / `cot` / `extract_cot`, plus the frozen legacy aliases
  `generic` (= `grounded`) and `targeted` (= `abstain`) that keep old cached cells
  interpretable. The mode rides the conditioner name (the prediction key has no
  prompt field), so each mode is its own cell. Two run_tag-scoped companions,
  never in the cell key: `decode_budget` (per-mode max_new_tokens, rebound on the
  reasoner per cell by the orchestrator) and `final_answer_delimiter` (judge-time
  extraction of the text after the LAST occurrence; the raw answer stays on the
  row). A `run_settings.json` sidecar next to `predictions.jsonl` refuses a
  generate/judge pass whose budget/delimiter differ from the tag's recorded ones.
  Backends record `output_truncated` (hit the decode budget with no EOS) in row
  `metadata` — the output-side canary the input-side truncation fields cannot see
  (InternVL omits it: `chat()` exposes no token count).
- **Accounting** per `Prediction`: `input_text_tokens` (image placeholders
  stripped), `input_visual_tokens` (vision-token estimate), `output_tokens`, the
  `prefill` / `decode` latency split and end-to-end `latency_s` (batch=1), peak
  VRAM, plus metadata (backend, model id, template version, `max_new_tokens`,
  `max_pixels`, `quantization`, image count).
- **Closed models** are comparison / judge only, behind the same ABC via
  `ModelInput.to_chat_messages()`; the pipeline never imports vendor SDKs.

## Data layer

- **Paths** (root-relative both machines): dataset `.data/mmlongbench`, parquet
  `.data/mmlongbench/data/*.parquet`, PDFs `.data/mmlongbench/documents/*.pdf`,
  render cache `results/cache/renders/<pdf-stem>__dpi<N>/page_XXXX.png` (144 DPI
  base).
- **`load_mmlongbench()` → `Question`:** `id` (`mmlongbench:000000`), `doc_id`,
  `question`, `gold_answer`, `answer_format`, `doc_type`, `evidence_pages`
  (normalised to 0-based; original in `raw_fields`), `evidence_sources`, `hop`
  (from evidence-page count), `is_unanswerable` (gold == "Not answerable"),
  `raw_fields` (+ `source_dataset`).
- **`render_question_pages()`** resolves the PDF and renders the gold pages;
  unanswerable questions with no gold pages render page 0. Each `Page` carries the
  0-based index, PDF path, optional cached PNG, and PyMuPDF line spans.
- **LongDocURL loader** exists but the held-out replication uses a held-out
  MMLongBench subset; LongDocURL is kept for a possible future replication.

## Tools (non-reasoner channels)

- **`T` — text.** `tools/text.py` extracts cheap embedded PyMuPDF text; digital-born
  only, empty on scans by design.
- **`TL`/`TLV` — parser markdown.** `tools/parser.py` reads parser markdown from a
  disk cache (`parser_markdown`); the parser under comparison (PyMuPDF floor,
  PaddleOCR-VL default, MinerU 2.5, Unlimited-OCR) runs in its **own isolated env**
  via `tools/parser_worker.py` and pre-warms the cache. The parser VLM never
  co-resides with the reasoner; a cold cache raises `ParserCacheMiss` so the cell
  records a failure row rather than pulling a parser model into the reasoner
  process. **No bounding-box JSON anywhere** — the "L" is a name only.
- **`TLV`/`V` — image.** `tools/visual.py` builds page-image parts and estimates
  vision tokens from the resolution preset.
- **Prestage.** `ops/scripts/prestage.py [--smoke]` stages the reasoner weights,
  retrievers and adapter bases, parser weights, Paddle's separate pipeline cache,
  and MMLongBench (idempotent, with local-cache probing).

## Evaluation

- **Judge.** `GeminiJudge` (gemini-2.5-flash, default, free tier) and
  `GPT4oMiniJudge` (OpenAI, paid); `StubJudge` is offline plumbing. Each returns a
  verdict (`correct` / `incorrect` / `abstained`) + extracted answer + rationale;
  an abstaining verdict on a native-unanswerable question counts correct. Keys live
  in the local `.env`. The judge is a **different family** from the reasoner, gated
  by Cohen's κ ≥ 0.75 vs 200 hand labels on the answerable-only set
  (`scoring/agreement.py`).
- **Accuracy.** `scoring/accuracy.py` = mean correctness + 95% bootstrap CI
  resampled at the **document level** (draw `doc_id`s with replacement, take all
  their rows), 1000 draws, seed 0.
- **Cost.** `scoring/cost.py` = mean latency@batch1 (primary) + the prefill/decode
  split and text / vision / output token sums (secondary), peak VRAM.
- **Frontier.** `scoring/frontier.py` orders `T → TL → TLV → V`; the sufficiency
  frontier is the cheapest rung whose CI upper bound reaches within the margin
  (default 3 points, sensitivity {2,3,5}) of the strongest rung's point estimate.
- **Retrieval.** `scoring/retrieval.py` scores page precision / recall / F1 vs gold
  `evidence_pages`, sliced by `<retrieval-modality>:<evidence-source>` so
  matched / cross separates locating from evidence modality.
- **Composition.** Each bin is decomposed into normalized evidence-source shares
  (text / table / chart / figure / layout, summing to 1).
- **Classifier (routing side tool).** `models/classifier.py` renders the first
  pages, builds `TLV`, and asks the small reasoner for a bin; routing counts the
  classifier's own latency. It is G3's optional one-shot side artifact
  (`classifier.jsonl`, priced over G1's answerable docs when `classifier_spec` is
  set); routing is a pure build-time assembly over G1's rows (README §11).

---

## Environment and dependencies

The core environment is `envs/core` (Python 3.11). Each parser runs in its own
`envs/parse-*` environment behind the parser disk cache, so parser dependency
stacks never share an environment with the reasoner. `setup_env.py --machine
{V100,H100}` installs the matching framework wheels and checks every environment
with `pip check`.

⚠ PENDING v5 — the dependency set is being re-examined (evaluating dropping vLLM,
which exact-pins torch and caps `openai`; stripping stacks the v4 ladder no longer
uses). Until that lands in `docs/DECISIONS.md`, treat the shipped `requirements*.txt`
as the source of truth for what is installed, not any pin list quoted in prose.
