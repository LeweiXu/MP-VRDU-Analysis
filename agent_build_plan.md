# Coding Agent Build Plan: MP-VRDU Component Analysis

> Read `context.md` first — it defines the study, the three-stage pipeline, the
> sub-studies, the tool inventory, and the fixed scaffolding. This file is the
> staged engineering plan to implement it.
> Last updated: 2026-06-08.

## Operating principles (read before starting any stage)

1. **Local-first, Kaya-for-scale.** Dev box = RTX 5070 (~12GB VRAM) + 64GB RAM.
   Kaya = A100-class for full runs. Local proves CORRECTNESS on a tiny data
   slice; Kaya proves SCALE. Every stage must be runnable and testable locally
   on a slice that fits 12GB before it goes to Kaya.
2. **The 12GB ceiling is a design constraint, not a blocker.** The 7B generator
   does NOT fit comfortably at full precision on 12GB. Local generation testing
   uses one of: (a) a tiny VLM (e.g. Qwen2.5-VL-3B) or 4-bit quantised 7B as a
   STAND-IN purely to exercise code paths, (b) a mocked generator that returns
   canned output. The real 7B/32B generator runs on Kaya. NEVER report numbers
   from the local stand-in — it exists only to test plumbing.
3. **One variable per experiment** (see context.md sub-studies). The harness must
   make conditions swappable via config, never via code edits.
4. **Every stage ends with a concrete TEST and an intermediary deliverable.** Do
   not advance to stage N+1 until stage N's test passes. State the test result.
5. **Determinism + logging.** Fix seeds. Log every config, every per-question
   result to disk (JSONL), so any run is reproducible and auditable.
6. **Config-driven.** One run = one config file (YAML). A config fully specifies:
   representation, retrieval method, top-k, generation modality, generator,
   judge, dataset slice. Results filenames encode the config hash.

## Repository layout (create in Stage 0)

```
mpvrdu/
  configs/                 # one YAML per experimental condition
  src/
    data/                  # dataset loading, PDF rendering, slicing
    represent/             # Stage 2: parsers/OCR -> text/markdown + chunking
    retrieve/              # Stage 1: BM25, dense, ColPali/ColQwen, hybrid
    generate/              # Stage 3: VLM wrappers; image/text/both input builders
    eval/                  # metrics: accuracy, F1, recall@k; LLM-judge
    pipeline.py            # wires stages per config; emits per-question JSONL
  scripts/                 # CLI entry points, SLURM submission scripts
  tests/                   # unit + slice-level integration tests
  results/                 # JSONL outputs, never committed (gitignore)
  data/                    # local cache of dataset + a tiny "dev slice"
  context.md
  kaya_slurm_cheatsheet.md
  agent_build_plan.md
```

------------------------------------------------------------------------------
## STAGE 0 — Scaffolding & environment (local)

**Goal:** repo skeleton, dependency env, config system, logging — no ML yet.

Tasks:
- Create the repo layout above. Init git; gitignore `results/`, `data/`, weights.
- Define the config schema (a dataclass or pydantic model) covering every field
  in Operating Principle 6. Write a loader that validates a YAML into it.
- Set up structured logging + a results writer that appends one JSON object per
  question to a JSONL file named by config hash + timestamp.
- Pin dependencies in `requirements.txt` / `environment.yml`. Two extras files:
  one CPU/light for local plumbing tests, one with the heavy GPU stack.

**TEST:** load a sample YAML config, validate it, write 3 dummy result rows to
JSONL, read them back. Assert round-trip equality. Run `pytest tests/`.

**DELIVERABLE:** runnable skeleton; `python -m mpvrdu.pipeline --config
configs/smoke.yaml` runs end-to-end with all stages mocked, producing a JSONL.

------------------------------------------------------------------------------
## STAGE 1 — Data layer + the "dev slice" (local)

**Goal:** load MMLongBench-Doc, render pages to images, carve a tiny dev slice.

Tasks:
- Loader for MMLongBench-Doc (HF `yubo2333/MMLongBench-Doc`; PDFs in repo data
  dir). Parse fields incl. `evidence_pages` (note: stored as a string like
  "[3, 5]" — parse to a list of ints). Handle the unanswerable questions
  (no evidence pages) explicitly.
- PDF -> per-page PNG renderer (PyMuPDF/`fitz`). Configurable DPI. Cache renders
  to disk so they're computed once.
- **Dev slice builder:** select ~5 documents spanning short/long and
  single-page/cross-page/unanswerable question types, ~20-30 questions total.
  This slice is what ALL local tests run on. Small enough to be instant.
- A tiny `inspect` script that prints a question, its evidence pages, and shows
  the rendered evidence page, so a human can eyeball correctness.

**TEST:** assert the loader returns the expected question count; assert
`evidence_pages` parsed to ints; assert each referenced page renders to a
non-empty image; assert the dev slice covers all 3 question types.

**DELIVERABLE:** `data/dev_slice/` cached; documented loader API. NOTE the
name-collision warning in context.md — verify you pulled the doc-VQA benchmark.

------------------------------------------------------------------------------
## STAGE 2 — Eval harness + metrics + no-retrieval baseline (local plumbing,
##           first real numbers on Kaya)

**Goal:** the spine of the whole project — a generator answers questions and the
metric scores them. Build this BEFORE any retrieval.

Tasks:
- **Generator interface** (abstract): `answer(question, images=None, text=None)
  -> str`. Implementations:
  - `MockGenerator` (returns canned/gold-ish text) — for local plumbing tests.
  - `LocalSmallVLM` (Qwen2.5-VL-3B or 4-bit 7B) — local code-path testing only.
  - `KayaVLM` (full Qwen2.5-VL-7B, later 32B) — real runs.
  All share one input-builder so image/text/both packing is identical across
  implementations.
- **Metrics:** implement the benchmark's accuracy + F1. Study how MMLongBench-Doc
  scores (extraction + normalisation + compare; answer_format-aware). If using
  an LLM judge, wrap it behind a `Judge` interface; allow a deterministic
  rule-based judge for local tests so plumbing doesn't need an API.
- **No-retrieval baseline condition:** feed the generator the first-N pages (or
  truncated text) — the lower bound.
- Wire `pipeline.py`: config -> data slice -> generator -> metric -> JSONL.

**TEST (local):** run the full pipeline on the dev slice with `MockGenerator` +
rule-based judge. Assert metrics compute, JSONL is well-formed, a deliberately
correct mock scores ~100% and a deliberately wrong mock scores ~0%. This proves
the SCORING is correct independent of model quality.

**TEST (Kaya, first real run):** run no-retrieval baseline with the real 7B
generator on the FULL benchmark. Produce the first honest accuracy/F1 number.

**DELIVERABLE:** working eval harness; first real baseline number on Kaya;
documented metric implementation with a note on the judge used.

------------------------------------------------------------------------------
## STAGE 3 — Oracle upper bound (local plumbing, number on Kaya)

**Goal:** the ceiling. Feed gold `evidence_pages` to the generator.

Tasks:
- Oracle "retriever" that returns exactly the gold evidence pages for each Q.
- Handle unanswerable questions (no evidence) — the correct oracle behaviour is
  to give the generator no evidence and expect an abstention/"not answerable".
- Run through the same pipeline (only the evidence-selection step differs).

**TEST (local):** on the dev slice with the small/stand-in generator, assert the
oracle condition feeds exactly the gold pages and runs end-to-end.

**TEST (Kaya):** full oracle run with real 7B. Sanity: oracle accuracy MUST
exceed no-retrieval accuracy. If not, something is wrong (likely input packing
or judge). The oracle-vs-no-retrieval gap is your headline framing number.

**DELIVERABLE:** floor and ceiling numbers on Kaya. Now every retrieval method
has a meaningful range to fall within.

------------------------------------------------------------------------------
## STAGE 4 — Retrieval, method by method (Sub-study A) (local build, Kaya scale)

**Goal:** implement retrieval methods behind one interface; measure each on
RECALL before touching downstream accuracy.

Build order (one at a time; each gets its own test):
1. **Text representation default** for this sub-study: pick ONE parser
   (PyMuPDF4LLM) to produce the text retrievers index over. Fix it.
2. **Retriever interface:** `index(units)` + `retrieve(query, k) ->
   ranked_unit_ids`. Implementations in order:
   a. **BM25** (rank_bm25 or pyserini).
   b. **TF-IDF** (sklearn) — second sparse point.
   c. **Dense text** (SentenceTransformers; cache embeddings).
   d. **ColPali** then **ColQwen2.5** (visual; from images, NOT the text index —
      enforce in code that visual retrievers ignore the Stage-2 text path).
   e. **Hybrid** (RRF over sparse+dense, and/or text+visual) — LAST.
3. **Retrieval-recall metric:** recall@k / evidence-page hit-rate vs
   `evidence_pages`. This is computed WITHOUT the generator — fast, local-able.

**TEST (local, per retriever):** on the dev slice, assert the retriever returns
k ranked ids and that recall@k is sane (e.g. oracle-pages-present check). Compare
recall numbers across retrievers on the slice — the RANKING should be plausible
before any Kaya run. A retriever scoring ~0 recall where evidence exists is
broken — fix before proceeding.

**TEST (Kaya):** full retrieval-recall for every method (no generator yet =
cheap). Then full pipeline (retrieve -> generate -> score) for each method, run
as two blocks: generation=image and generation=text (per Sub-study A). top-k
swept (e.g. 1,2,4,8) within each.

**DELIVERABLE:** retrieval-recall table + downstream accuracy/F1 table per
method, per generation-modality block, per top-k. This is the core result.

------------------------------------------------------------------------------
## STAGE 5 — Document representation (Sub-study B) (local build, Kaya scale)

**Goal:** hold the best text retriever from Stage 4 fixed; vary parser/OCR.
Text retrievers ONLY (visual retrievers don't consume Stage-2 text).

Tasks:
- **Parser interface:** `parse(pdf_page_or_doc) -> text/markdown + structure`.
  Implementations: PyMuPDF4LLM, MinerU, Tesseract (primary). Each behind the
  same interface so the index step is identical downstream.
- **Chunking module** (cross-cutting): chunk vs page vs section. Section-level
  requires structure-aware parser output — wire so it gracefully degrades when a
  parser gives no structure.
- Re-index the fixed best retriever over each parser's output.

**TEST (local):** on the dev slice, assert each parser produces non-empty output
for a born-digital page; assert Tesseract works on a rasterised page; eyeball one
table's serialisation per parser (Markdown vs raw) via the inspect script.

**TEST (Kaya):** full retrieval-recall + downstream accuracy per parser, with the
retriever fixed. Isolates "how much does extraction quality matter."

**DELIVERABLE:** representation-comparison table; note licensing (PyMuPDF AGPL).

------------------------------------------------------------------------------
## STAGE 6 — Generation modality (Sub-study C) (Kaya)

**Goal:** fix retrieval + representation at best settings; flip image/text/both.

Tasks:
- The input-builder from Stage 2 already supports all three; this stage just runs
  the three conditions on the fixed best pipeline.
- For the "text" and "both" conditions over a VISUAL retriever, note that the
  retrieved pages must be parsed (re-invokes Stage 2) — fix one default parser
  and state it in methodology.

**TEST (Kaya):** all three modalities on the full benchmark with everything else
fixed. Sanity: results differ only by modality.

**DELIVERABLE:** the image-vs-text-vs-both comparison — one clean question.

------------------------------------------------------------------------------
## STAGE 7 — Consolidation & analysis (local)

**Goal:** turn JSONL outputs into the dissertation's tables and figures.

Tasks:
- Aggregation scripts: JSONL -> result tables (accuracy/F1/recall per condition).
- Figures: top-k curves (accuracy vs k, recall vs k, cost vs k); method
  comparison bars with the no-retrieval floor and oracle ceiling drawn in.
- **Error analysis:** break results by question type (single/cross-page/
  unanswerable) and evidence source (text/table/chart/image/layout) — where does
  each method fail? Cross-page and chart/image are where the interesting gaps are.
- Stochasticity: if the judge or generator is sampled, report variance over a
  few seeds on a subset.

**TEST:** regenerate every table/figure from raw JSONL with one command
(reproducibility). Assert numbers match committed result snapshots.

**DELIVERABLE:** the results section's tables + figures, reproducible from JSONL.

------------------------------------------------------------------------------
## SECONDARY STAGES (only after primary is solid — see context.md goals)

- **S8 — LongDocURL:** add as a second dataset behind the Stage-1 loader
  interface (accuracy-only). Re-run key conditions.
- **S9 — Cross-domain:** add SlideVQA / medical / text-only / scene-image
  corpora behind the same loader. Re-run key conditions; ask whether the method
  ranking holds across domains. NOTE: domains differ in which parsers work
  (scans/slides may force OCR) — the Stage-2 comparison changes character.
- **S10 — Proposed method:** assemble the best (representation x retrieval x
  modality x k) combination found and benchmark it against published systems.

------------------------------------------------------------------------------
## Local-vs-Kaya quick reference

| Stage | Local (RTX 5070, 12GB) | Kaya (A100) |
|-------|------------------------|-------------|
| 0 Scaffolding | all | - |
| 1 Data | all (rendering, slicing) | - |
| 2 Harness/metrics | plumbing w/ mock + small VLM | full no-retrieval w/ real 7B |
| 3 Oracle | plumbing on slice | full oracle w/ real 7B |
| 4 Retrieval | build + recall on slice; embeddings | full recall + downstream |
| 5 Representation | parsers on slice | full re-index + downstream |
| 6 Gen modality | input-builder unit tests | full 3-way run |
| 7 Analysis | all | - |

Rule of thumb: anything WITHOUT the big generator (retrieval, recall, parsing,
metrics on canned answers) is fully local. Anything that needs the real 7B/32B
generating over many questions goes to Kaya. Embedding indexes (ColPali/ColQwen)
are ~2B models and may fit locally at small batch for slice testing; build the
full index on Kaya.