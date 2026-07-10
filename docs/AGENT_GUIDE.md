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
| `pipeline/orchestrator.py` | Composes A→D per cell; owns the two cache layers + telemetry capture. |
| `scoring/*` | `accuracy` (doc-level CI), `cost`, `frontier`, `retrieval`, `abstention`, `agreement` (judge-human κ). |
| `experiments/tasks/` | The four `G[num]_[name]` generation tasks + the base ABC. |
| `experiments/engine/` | The generate/judge driver (robustness, `--failed-only`), side-artifact writers, cache/table paths + cell keys. |
| `experiments/corpus/` | Question-set resolver + sampling; YAML spec loader. |
| `experiments/registry.py` | Task name → task. |
| `reporting/build.py` | Task → table routing, cell grouping, build-time routing assembly; writes CSV + `.md`. |
| `reporting/tables/` | Content-named table builders (headline, parser, resolution, matched_cross, kdepth, retrieval_accuracy, hallucination, routing, scale, composition). |
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

## Caching contract (two layers, both under `results/cache/`)

1. **`ResultCache`** — one `ResultRow` per cell, keyed by SHA-256 over
   `{question_id, doc_id, condition, representation, model_spec, page_indices,
   visual_resolution, judge_spec}`. Idempotent and resumable from disk.
2. **`PredictionCache`** (additive) — the reasoner output keyed the same way **minus
   `judge_spec`**, so one prediction is scored by any judge without re-running the
   model.

Rules that make the two-machine model and the sweeps work:

- **`k` is encoded in the conditioner name** (`retrieved_k3`), not a separate key
  field.
- **`model_spec` and `visual_resolution` are in both keys**, so scaling / family /
  quantization / resolution sweeps produce distinct, mergeable rows in a single run.
- **`dpi` is *not* in the cell key** — it keys the render / parser disk caches
  instead.
- **The cell key is machine-independent** (only cell identity + config values, no
  device property, hostname, or count). A re-run on another box produces the same
  key and completes the *same* file — this is what makes the `--failed-only` retry
  a file copy, not a merge (README §13).
- **`visual_resolution` is a per-cell axis**, part of both keys and stamped on
  `ResultRow` / `CachedPrediction`; a lower-res image is a genuinely different
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

- **Specs.** `ops/specs/full_generation.yaml` and `ops/specs/smoke_generation.yaml`.
  Specs support arbitrary ordered channel combinations over `T`/`L`/`V`, but the
  paper experiments use the rung set `[T, TL, TLV, V]` only. Cache dirs are
  `results/cache/<run-tag>/<smoke|full>/<run-name>/`.
- **Bridge.** `experiments/corpus/yaml_spec.py` loads YAML into dynamic
  `GenerationTask` objects; `experiments/engine/` owns the generate loop,
  reasoner / retriever construction, the parse pre-pass, and cache writes.
- **Shared side-artifact writers.** The retrieval and classifier side artifacts
  have one implementation in `experiments/engine/` so the fixed tasks and the
  dynamic YAML tasks cannot drift; each caller passes the ordered `(modality, k)`
  pairs it wants scored. `run_side` is not a frozen interface.
- **Table routing.** `reporting/build.py`'s routing registry declares each table's
  source task(s); routing is a build-time assembly over G1's rows (plus the
  classifier price). A table is only correct when handed exactly its source tasks'
  rows.
- **Why roles split across machines.** The reasoner / retrievers / classifier need
  a GPU; the judge needs the internet — on Kaya those never coexist. Generation
  runs on the cluster (GPU, offline); `ops.judge` and `ops.build` run locally after
  a pull. Judge loads no models; it scores `predictions.jsonl` directly. Judge keys
  stay in the local `.env` (only `HF_TOKEN` is forwarded to the cluster).
  Run-tagged builds write tables to `results/tables/<mode>-<run-tag>/`.

**Sampling for full runs.** A full MMLongBench run resolves its corpus from the
YAML `corpus:` scope: `full` (all 1091), `per_bin: N` (draw **whole documents** per
bin to N — doc-coherent, so the doc-level bootstrap CIs stay valid), or a
`limit` / id-list for smoke. The answerable pool is bound by the task (G1/G2
answerable, G3 unanswerable), enforced in `experiments/corpus/resolve.py` so a spec
cannot cross-contaminate. ⚠ PENDING v5: which experiments require the full corpus
(binning-dependent) vs a frozen random subset is being finalised — see README §3
and `docs/DECISIONS.md`.

**Quantized reasoner (model-spec suffix).** Quantization is a spec suffix, not a
cache-key field: `qwen3vl-8b-local-4bit` / `-8bit`. `ModelSpec.parse` strips the
trailing `-4bit`/`-8bit` into `ModelSpec.quantization` while `name` keeps the full
string, so quantized runs get their own cache rows and `size` still resolves to
`8b`. `get_reasoner` passes it to the backend (bitsandbytes 4-bit NF4 double-quant
or 8-bit). `--quantization` must match between generate and judge. Mains run bf16;
4-bit is single-GPU iteration plus a possible appendix quant-sensitivity row.

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
- **Prompt modes.** The instruction preamble is swappable (`config.PROMPT_MODES`):
  `none` / `generic` / `targeted`. Answerable cells use `targeted`; the
  hallucination task sweeps all three. The mode is part of cell identity.
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
  the retrievers, and each parser env (idempotent, offline-probing).

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
  classifier's own latency. ⚠ PENDING v5 — routing is moving fully to build-time
  over G1 and the classifier to an optional one-shot prediction pass (README §11,
  `docs/DECISIONS.md`).

---

## Environment and dependencies

Cluster env: `envs/mpvrdu` (Python 3.11, `requirements.txt`). Local RTX 5070
(Blackwell, sm_120) env: `envs/mpvrdu-local-gpu` (`requirements-local-rtx5070.txt`,
torch 2.8+cu128, no vLLM) because the cluster torch wheel has no sm_120 kernels.
Each reasoner-independent parser runs in its **own isolated env** behind the parser
disk cache, so a parser's dependency stack never has to co-exist with the reasoner
stack. Every env is `pip check` clean.

⚠ PENDING v5 — the dependency set is being re-examined (evaluating dropping vLLM,
which exact-pins torch and caps `openai`; stripping stacks the v4 ladder no longer
uses). Until that lands in `docs/DECISIONS.md`, treat the shipped `requirements*.txt`
as the source of truth for what is installed, not any pin list quoted in prose.