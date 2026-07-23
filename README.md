# MP-VRDU: multi-page visually-rich document understanding

This repo measures **which document representation is actually needed** to answer
questions over long, visually-rich PDFs, and **what each representation costs** on
deployable local models. The one-line thesis: *the representation an MP-VRDU
system needs is a function of the document type.* The benchmark is
MMLongBench-Doc; the primary reasoner is Qwen3-VL.

This README is the definitive guide to the **experiment**: what a cell is, the
representation ladder, how documents are binned, the retrieval and prompting
sweeps, the telemetry, and the three generation tasks. It describes the system **as
it is now**; the history of how it got here (pivots, superseded designs) lives in
`docs/DECISIONS.md`. The coding agent's reference for code structure and frozen
contracts is `docs/AGENT_GUIDE.md`. Cluster operations are in
`ops/kaya/KAYA.md`.

## Quick start

```bash
# tests (CPU, fast)
envs/mpvrdu/bin/python -m pytest -q

# generate (GPU, spec-driven) -> judge (needs a Gemini/OpenAI key in .env) -> build
envs/mpvrdu/bin/python -m ops.generate --spec ops/specs/g3_representation.yaml
envs/mpvrdu/bin/python -m ops.judge    --spec ops/specs/g3_representation.yaml --judge-spec gemini-flash
envs/mpvrdu/bin/python -m ops.build

# specs are named g<E-number>_<name>.yaml after the E1-E5 failure modes;
# g0_*.yaml are the interventions (parser, quantization, reasoner, resolution, ...)
```

The three phases split on purpose: **generate** is GPU-only and offline (runs on
the cluster), **judge** and **build** are local and need API keys. Everything is
cached and resumable, so a re-run only fills missing cells.

## Corpus selection: scan → pool → sampling

A run narrows the corpus in three ordered stages (`corpus:` in the spec):

1. **`scan`** (document-level): `any` (no filter), `digital`, or `scanned`. When set,
   each document is labelled by PyMuPDF auto-detection (cached to
   `annotations/auto_scan.csv`), and only documents of that kind — with all their
   questions — pass on.
2. **`pool`** (question-level): `answerable`, `unanswerable`, or `all` (both).
3. **`sampling`** (over what survives 1–2):
   - **`full`** — every remaining question.
   - **`{per_doc_type: N, seed: S}`** — draw whole documents per native `doc_type`
     (shuffled by seed) then cap to **exactly N questions per doc_type** (so
     `per_doc_type: 1` = one question per label; the exact cap can slice the last
     drawn document).
   - **`{per_bin: N, seed: S}`** — draw whole documents per `bin_label` until the bin
     reaches **about N** questions (never slices a document, so the doc-level
     bootstrap stays valid).
   - **`{limit: N}`** — the first N questions (a fast smoke slice).
   - **`{ids: [q1, q2, …]}`** — exactly those question ids.

   `per_doc_type` caps to *exactly* N (can slice a document); `per_bin` keeps *whole*
   documents to *about* N. Both draw whole documents so document-level CIs stay valid.

## Running the experiments

Each experiment is one spec under `ops/specs/`. A run has three stages, each its own
entry point that shares the run's cache by `run_tag`: **generate** (GPU, on Kaya) ->
**judge** (local, needs an API key) -> **build** (local, assembles the tables).

### 1. Generate (Kaya)

Submit generation to SLURM (non-blocking with `--no-wait`). 8B needs two V100s, 2B fits
one; the job writes one `predictions.jsonl` row per cell (including failures), unjudged.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait \
  --gres gpu:v100:2 --time <HH:MM:SS> ops/generate.py -- --spec ops/specs/<spec>.yaml
```

Size `--time` by cell count at roughly 33 s per 8B cell on 2×V100 (wall-clock, prewarm
and model load included; scanned docs and image-heavy cells run slower). A timeout is
safe: resubmitting the same spec resumes from cache, skipping done cells. Two V100
caveats: there is no FlashAttention on these GPUs, so image-heavy G2 inference is far
faster on the supervisor H100 (run the retrieval memo on Kaya, the inference on the
H100); and big-context TLV/V cells OOM by design — they write `oom` rows, the job keeps
going, and the supervisor finishes them with `--failed-only` (add `--skip-oom` on a
resume so it does not re-parse cells already known to OOM). Check queue/node state with
`ops.kaya.kaya status`.

### 2. Judge (local)

Generation does not score. Judging is a separate local phase (Kaya's compute nodes are
offline); it needs a Gemini/OpenAI key in `.env` and the same spec, so it finds the run
by `run_tag`.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya pull                                    # bring predictions back
envs/mpvrdu/bin/python -m ops.judge --spec ops/specs/<spec>.yaml --judge-spec gemini-flash
```

It reads `predictions.jsonl`, looks up each gold answer, and writes one `results.jsonl`
row per prediction (a strict superset: prediction + `score`/`correct`/`abstained`/
`judge_spec` verdict). It loads no models, so it is CPU-only; the only limit is the judge
API quota, so judge one run at a time and set `GEMINI_API_KEY_SECONDARY` for the two-key
fallback. Re-running with a different `--judge-spec` writes its own rows keyed by judge,
so re-scoring never re-generates. Unscored `oom` cells (no answer) carry through, so
judge again after the supervisor completes them if you want them in the tables.

### 3. Build (local)

Assemble the CSV/Markdown tables from `results.jsonl`. Pass the run's `--run-tag` so it
reads that run's cache (omit it only for the un-tagged cache); tables group by the seven
native mmlongbench doc_type classes.

```bash
envs/mpvrdu/bin/python -m ops.build --task <task-or-all> --run-tag <run_tag>
```

---

## Repository structure

The science spine sits flat at the root; operational tooling groups under `ops/`.
A file's home is decided by what it *is*, not who imports it. (`docs/REPO_STRUCTURE.md`
holds the same tree plus an auto-generated per-file docstring map.)

```
mpvrdu/
├── config.py                 run knobs: paths, cache version, resolution presets,
│                             bins, prompt modes, sampling defaults (NO input-token cap)
├── schema.py                 frozen data contracts (Question, PageSet, Payload,
│                             Prediction, Score, ResultRow) + the per-cell telemetry fields
│
├── data/                     dataset layer
│   ├── loader.py             MMLongBench / LongDocURL rows -> Question; answerable split
│   ├── annotations.py        read + validate the hand-labelled doc table (bin/scan/visual)
│   ├── binning.py            stamp bin_label / scan_label onto questions from the table
│   └── render.py             PDF page -> cached PNG + embedded-text spans
│
├── tools/                    per-page channel builders (one channel each)
│   ├── text.py               cheap embedded PyMuPDF text (the T channel)
│   ├── parser.py             parser markdown for TL/TLV, read from a disk cache;
│   │                         warms the cache by running a parser in its isolated env
│   ├── parser_worker.py      subprocess entry the isolated parser env runs (no project imports)
│   └── visual.py             page-image parts + vision-token estimation from the resolution preset
│
├── retrievers/               page retrievers arranged as cost rungs
│   ├── __init__.py           Retriever ABC + shared ranking helpers + memoization
│   ├── text.py               BM25 / BGE-M3 / Qwen3-Embedding-4B
│   ├── vision.py             ColModernVBERT / ColQwen2.5 / ColQwen3-4B
│   └── joint.py              deduplicated union of a text + a vision page set
│
├── models/                   reasoner backends + classifier, behind a registry
│   ├── __init__.py           ModelSpec parser + get_reasoner (the model-family swap point)
│   ├── qwen3vl.py            Qwen3-VL backend (prompt modes, quantization, prefill/decode timing)
│   ├── internvl.py           InternVL backend (family replication)
│   ├── classifier.py         first-pages modality-bin classifier (routing side tool)
│   └── payload.py            backend-neutral ModelInput + chat/local adapters
│
├── pipeline/                 the five frozen stages of ONE cell
│   ├── conditioner.py        page selection: oracle / retrieved-topk / similarity / full
│   ├── representation.py     the T/TL/TLV/V composer (cost-ordered, parser text, no bounding boxes)
│   ├── reasoner.py           Reasoner ABC (+ per-cell prompt instruction)
│   ├── judge.py              Judge interface + Stub / GPT-4o-mini / Gemini scorers
│   └── orchestrator.py       composes the stages; owns the two caches + telemetry capture
│
├── scoring/                  cached cells -> numbers
│   ├── accuracy.py           document-level accuracy + bootstrap CIs
│   ├── cost.py               token / latency (prefill+decode) / peak-VRAM aggregation
│   ├── frontier.py           the sufficiency-frontier rule
│   ├── retrieval.py          page precision / recall / F1 vs gold pages
│   ├── abstention.py         abstention/refusal detection
│   └── agreement.py          judge-human Cohen's kappa
│
├── experiments/              what runs, how, and on what
│   ├── tasks/                the single spec-driven Task + the base ABC (task_name is a label)
│   ├── engine/               driver (generate/judge loop, robustness, failed-only),
│   │                         side-artifact writers, cache/table paths + cell keys
│   ├── corpus/               question-set resolver + sampling, flat YAML spec loader
│   └── registry.py           any task_name label -> the unified Task
│
├── reporting/                judged rows -> tables
│   ├── build.py              task->table routing, cell grouping, build-time routing assembly
│   └── tables/               content-named builders (headline, parser, resolution, matched_cross,
│                             kdepth, retrieval_accuracy, hallucination, routing, scale, composition)
│
├── ops/                      entry points + operational tooling
│   ├── generate.py judge.py build.py   the three role entry points
│   ├── kaya/                 SLURM sync/submit runner + config.json + cluster guides
│   ├── specs/                YAML run specs (template + saved configs)
│   └── scripts/              standalone utilities (prestage, annotate_docs, resolution_probe, ...)
│
├── docs/                     authored prose + docs/generated/ script outputs
├── tests/                    pytest suite (invariants + plumbing) + fixtures/
└── old/            [reference]  v3 snapshot, deleted once v4 is fully production-green
```

Containment: everything heavy (envs, model weights, datasets, all caches) lives
under the repo root, gitignored and rsync-excluded, so nothing lands in `$HOME`.
Caches live under `results/cache/<run_tag>/`, one namespace per run so cells from
unrelated runs never collide.

---

# The experiment

## 1. The cell

Every number in the paper comes from one **cell**: a single
`(question, page-selection, representation, model)` combination, scored by a
judge. Answering a cell runs five stages, each a frozen interface so a piece can
be swapped without touching the rest:

1. **Conditioner** (`pipeline/conditioner.py`) picks *which* pages the model sees.
2. **Render** (`data/render.py`) rasterizes those pages and pulls their text.
3. **Representation** (`pipeline/representation.py`) turns the pages into a
   text/vision payload for the chosen rung of the ladder.
4. **Reasoner** (`models/`) feeds the payload to the MLLM and generates an answer.
5. **Judge** (`pipeline/judge.py`) scores the answer against the gold answer.

The orchestrator runs these in order and caches results in two layers: a
**prediction cache** keyed without the judge (one reasoner run, reusable by any
judge) and a **result cache** keyed with it. Cell keys are a SHA-256 over the
cell identity only (question, doc, condition, representation, model spec, page
indices) and nothing machine-dependent, so a re-run on another box produces the
same key and completes the same file.

### What a run writes to disk

The **generate** phase writes one jsonl file; the **judge** phase writes the second,
both under `results/cache/<run_tag>/<mode>/<task>/`:

- **`predictions.jsonl`** (written by `ops.generate`) is one row per cell,
  **including failures**: the model's answer text, which pages it saw, the document
  bin/identity fields, cost telemetry (text/visual tokens, output tokens, the
  prefill/decode latency split, peak VRAM), and `status`/`skipped_reason` (a cell
  that OOM'd or errored still writes a row, marked `oom` or `error`). It carries no
  judging, so `ops.judge` can score it with any judge and never re-run the model.
  This is the file `check_run` reads.
- **`results.jsonl`** (written by `ops.judge`) is a **strict superset**: every
  `predictions.jsonl` row plus the judge verdict (`result_key`, `judge_spec`,
  `score`, `correct`, `abstained`). Failed cells pass through unscored (score 0),
  so it stays one row per cell. This is the file the table builders read.

Both files share one schema across every run, so a G2 row looks like a G1 row.
What changes per run is the cells plus the **side artifacts** the run's config asks
for: benchmark files that never touch the reasoner. A run with retrieval methods
configured writes **`retrieval.jsonl`** (page precision/recall/F1 per question x
retriever x k) as **stage 1, before** the reasoner cells: it ranks every method
once, persists the rankings to `results/cache/<run_tag>/retrieval/`, and the
inference arms reuse those rankings to pick the reasoner's pages — so
`retrieval.jsonl` and the predictions share one ranking instead of computing it
twice. A run with a classifier configured writes `classifier.jsonl` after inference
(the modality-bin classifier priced once over the answerable docs). In the
canonical runs those are G2 (retrieval) and G3 (classifier).

The **stub judge** (`ops.judge --judge-spec stub`, the default) is a cheap offline
scorer, not a no-op: it counts a cell correct when the gold answer appears
(case-insensitively) in the model's text and the model did not abstain, and for
unanswerable questions when it *does* abstain. It runs through the same judge path
as the real judges with no API call, so `ops.judge --spec <spec> --judge-spec stub`
produces a complete `results.jsonl` offline that you can later re-judge with
Gemini/GPT (a different `--judge-spec` writes its own rows, keyed by judge).

## 2. The representation ladder (cost-ordered, not cumulative)

Four representations of increasing compute cost. The headline question is *which
rung is worth its cost, per document type?*

| Rung | What it feeds | Built by |
|---|---|---|
| `T` | cheap embedded text (digital-born only; empty on scans by design) | PyMuPDF via `tools/text.py` |
| `TL` | a PDF parser's layout-rich **markdown**, which *replaces* the embedded text | `tools/parser.py` |
| `TLV` | parser markdown **+** the page image | `tools/parser.py` + `tools/visual.py` |
| `V` | the page image only (parser-independent reference point) | `tools/visual.py` |

Two things to state honestly. First, the ladder is **not cumulative**: `TL`'s
parser text replaces `T`'s embedded text rather than adding to it, so `T ⊄ TL`.
It is ordered by **compute cost**, not by strict inclusion. Second, the "L" is
vestigial: there is no separate layout channel and **no bounding-box JSON
anywhere** (that was the token-heavy channel that caused truncation in earlier
work; it is gone). The `T/TL/TLV/V` names are kept for table and cache continuity.

**Parsers under comparison** (the `TL`/`TLV` text source; the parser comparison
swaps this and holds everything else): PyMuPDF is the free `T` floor;
**PaddleOCR-VL** (0.9B, default), **MinerU 2.5** (~1.2B), and **Unlimited-OCR**
are the three VLM parsers. Each parser is a heavy, separately-pinned stack, so it
runs in its **own isolated env** and its markdown is **pre-warmed to a disk
cache**; the parser VLM never co-resides with the reasoner on the GPU. The
reasoner path only ever *reads* that cache (`parser_markdown`), and a cold cache
raises `ParserCacheMiss` so the cell records a failure row rather than pulling a
parser model into the reasoner process.

## 3. Document bins (the deployable axis)

> ⚠ PENDING v5 — the binning source is being finalised, so this section describes
> a moving part. The bin **axis** is stable: *which modality dominates a document's
> information content*, document-level, ordered text → visual. How a document *gets*
> its bin is what is changing. The v5 direction is to bin by **representative
> document domains** or by `evidence_source`; the manual per-document annotation
> pass (below) is **currently optional and not the working default** — it proved
> hard to apply cleanly and the completeness requirement was relaxed. Until the v5
> decision lands in `docs/DECISIONS.md`, do not treat any single label source as
> final.

The thesis axis names three bins — **text-dominant** (information is linguistic; a
scanned handwritten note is text-dominant even though it is image-based on disk),
**mixed-modality** (needs both; e.g. a text-dense paper with a few figures), and
**visual-dominant** (information lives in the imagery/design) — deliberately *not*
MMLongBench's native `doc_type`, which encodes domain rather than modality.

Two mechanisms exist in the code. (1) **Manual annotation** — optional labels in
`annotations/doc_labels.csv` (`bin_label`, `scan_label` = digital/scanned,
exploratory multi-valued `dominant_visual`), read by `data/annotations.py` and
stamped onto each `Question` by `data/binning.py::stamp_bins`. A present sheet is
validated (columns checked, blank rows skipped), but completeness is **opt-in**
(`--require-complete-annotations`); an absent or partial sheet degrades to blank
bins (rendered `(unlabeled)`) so dev and smoke runs proceed. (2) **`doc_type`
sampling** — corpus sampling is by native `doc_type`, which is unaffected by the
bin labels, so runs proceed regardless of annotation coverage. A blind-subset
Cohen's-kappa flow (`ops/scripts/annotate_docs.py`) supports a second-annotator
reliability check against the same 0.75 gate as the judge, if the manual pass is
resumed.

## 4. Page selection (conditioners)

- **oracle** feeds exactly the gold evidence pages (the reasoning ceiling).
- **retrieved-topk** feeds a retriever's top-k pages (the RQ2 retrieval arm).
- **similarity** feeds a few similarity-ranked pages, used by the hallucination
  study where there is no oracle arm (unanswerable questions have zero gold pages).
- **full** feeds every page (the feed-everything baseline).

Oracle is also available as a `Retriever` (`retrievers/oracle.py`) for uniformity,
though the reasoner's oracle cells select via the conditioner (all gold pages). The
conditioner name carries its parameters (e.g. `retrieved_vision_k5__none` — the k
and the prompt mode), so each (k, prompt) lands in its own cached cell.

## 5. Visual resolution

Vision-token volume (resolution x page count) is the binding memory constraint, so
resolution is a named preset = a per-page pixel cap:
`full/high/med/low/min` in `config.VISUAL_RESOLUTION_PRESETS`. It is the **one
representation parameter held identical across machines** (a lower-res image is a
genuinely different, lossier input, so pooling different resolutions would compare
different representations). One preset (`DEPLOYMENT_RESOLUTION`, currently a `med`
placeholder pending the resolution probe) is fixed study-wide; only the scientific
resolution sweep varies it, and it does so under its own `run_tag` so the caches
never collide.

## 6. Reasoners, decoding, quantization

Backends resolve through `models.get_reasoner` from a `family-size-backend[-quant]`
spec. Primary: **Qwen3-VL** at 2B/4B/8B/32B; family replication: **InternVL3-8B**.
Decoding is greedy (`do_sample=False`, capped `max_new_tokens`). Optional
bitsandbytes `-4bit`/`-8bit` is appended to the spec so a quantized run gets its
own cache rows. There is **no input-token cap**: cells run at full sequence
length (the truncation telemetry is kept only as a canary that should read zero).
Anything that still OOMs a small GPU is completed on the bigger one via the retry
(section 11).

## 7. Prompting

The reasoner prompt has a swappable instruction preamble (`config.PROMPT_MODES`):
`none` (no guidance), `generic`, or `targeted` (abstention-targeted). Answerable
runs set `prompt_modes: [none]`; the hallucination run sweeps all three. The mode
rides on the conditioner name (the prediction key has no prompt field), so each
prompt condition is its own cached cell.

## 8. Judging, and the answerable / unanswerable split

A `Judge` maps `(question, prediction)` to a comparable `Score`. The real judge is
a **different model family** from the reasoner (Gemini-flash or GPT-4o-mini) so it
is an independent scorer for the kappa >= 0.75 validation gate; `StubJudge` keeps
offline tests runnable. Of the 1091 questions, ~250 are **unanswerable**. They are
removed from RQ1/RQ2 (so those accuracies are cleanly "accuracy on answerable
questions") and used **only** in the RQ3 hallucination study, where correct
behaviour is abstention. Each run declares its pool as a spec variable
(`corpus.pool`: answerable / unanswerable) — the G1/G2 runs answerable, G3
unanswerable — applied in `experiments/corpus/resolve.py::filter_by_pool`.

## 9. Telemetry (fixed schema, collected every run)

Every cell writes **exactly one** `predictions.jsonl` row regardless of outcome (and
one `results.jsonl` row after judging), so a failure is data, not a hole. The schema
is fixed once so results across every run
are comparable and placement (main paper vs appendix) can be decided after the
data exists.

- **Identity / provenance:** `bin_label`, `scan_label`, `condition`,
  `representation`, `model_spec`, `page_indices`, `machine`, and `status`
  (`ok` / `oom` / `error`) + `skipped_reason` (so a missing cell is
  distinguishable as OOM-skipped vs never-run).
- **Tokens:** `total_text_tokens`, `total_visual_tokens`, `text_tokens_fed`,
  `output_tokens`. With the cap gone `text_tokens_fed == total_text_tokens`, so
  `tokens_dropped` is a **canary that must read zero** (a nonzero value is a bug).
- **Latency:** end-to-end `latency_s` plus a `prefill_latency_s` / `decode_latency_s`
  split (prefill isolates the cost of *ingesting* the representation, the thing the
  ladder changes; likely a headline column).
- **Memory:** `peak_vram_bytes` per cell.

Per-run environment (`gpu_model`, `cuda`, `torch`, parser/retriever ids, seeds,
the resolution preset, `git_commit`) is recorded once in the run manifest, not
stamped on every row. The retrieval accuracy benchmark is a separate side artifact:
one row per (question, method, k) with page precision/recall/F1 per bin, covering
every retrieval method including ones never fed to the reasoner.

## 10. Retrieval methods as cost rungs

RQ2 mirrors RQ1's cost story: retrieval is laid out cheapest to most expensive.

| Axis | Cheap | Mid | Expensive |
|---|---|---|---|
| **Text** | BM25 (lexical) | BGE-M3 (dense) | Qwen3-Embedding-4B |
| **Vision** | ColModernVBERT (~250M) | ColQwen2.5 | ColQwen3-4B |

**Joint retrieval** is free: the deduplicated union of one already-computed text
page set and one vision page set (matched tiers), no new retrieval and no score
fusion. Single-method k sweeps `{1,3,5,7,10}`; joint uses `{1,3,5}` per method so
the union stays under 10 pages. With no input cap, a high-k accuracy drop is an
honest distractor effect, not a truncation artifact.

## 11. One pipeline, three canonical runs

There is **one** generation pipeline (§1). A run is defined entirely by its config,
and `task_name` is just a label (the cache namespace + parallel job), not a
different mechanism — the parser, resolution, family, dataset, quantization, and
model-size "experiments" are list-valued axes of a run, not new tasks. The three
canonical labels below are the same pipeline with different variable values; each
keeps a `G[num]_[name]` handle stating the mechanism, not an RQ or table number.

- **`G1_oracle_ladder`**: oracle pages x `{T,TL,TLV,V}`, answerable-only, primary
  reasoner. Feeds the cost-ordered headline and the per-bin frontier. Its YAML
  variants (one field changed each) produce the parser comparison, the resolution
  sweep, the InternVL family replication, the held-out dataset replication, and the
  quantization / model-size cost sweeps.
- **`G2_retrieval`**: retrieved pages x `TLV`(/`V`) x method x k. One retrieval
  pre-pass feeds two scorers: the reasoner (matched-vs-cross, k-depth) and the
  page-F1 side artifact (the retrieval benchmark, every method).
- **`G3_hallucination`**: unanswerable-only x similarity pages x `TLV` x the three
  prompt modes. Correct = abstention; no oracle arm exists. G3 also carries the
  optional one-shot modality-bin classifier as its side artifact (`classifier.jsonl`):
  when a classifier is configured it prices one first-pages pass per distinct
  document over G1's answerable doc set (latency + VRAM), else it is skipped.

Routing is not a task: the routing table is assembled at **build time** from G1's
ladder rows plus G3's classifier price (gold-bin ceiling when the classifier is
skipped).

## 12. Research questions

- **RQ1 (recipe by document type):** the cost-ordered ladder x bin headline with
  the sufficiency frontier marked; plus the parser comparison and resolution sweep;
  plus family (InternVL) and dataset (held-out MMLongBench) replications.
- **RQ2 (retrieval):** matched-vs-cross across all bins, the k-depth sweep, the
  per-bin retrieval-accuracy benchmark, and free joint retrieval.
- **RQ3 (deployment):** routing policies (oracle / predicted / uniform-cheapest /
  uniform-strongest, predicted priced with the classifier's own latency); the
  hallucination x prompting study; and the quantization (4/8/16-bit) and model-size
  (2B/4B/8B/32B) cost-frontier sweeps.

Accuracy is doc-level with 1000-resample bootstrap CIs (resampling over documents
to preserve within-document correlation); the sufficiency frontier picks the
cheapest rung whose CI upper bound reaches within a pre-registered 3-point margin
of the strongest rung.

## 13. Hardware: the machine split is just the retry

There is **no machine-split code**. Kaya (V100s) runs every task at full sequence
and the fixed resolution; a cell either succeeds or writes an `oom`/`error` row.
The set of failed rows *is* the supervisor's work queue: it re-runs the same task
in `--failed-only` mode on the bigger GPU and upgrades those rows **in place** in
the same jsonl (`experiments/engine/driver.py::merge_failed_only`). Because cell
keys and the resolved corpus are machine-independent, a supervisor re-run completes
the *same* file rather than a parallel one, so pooling is a file copy, not a merge.
`machine` is recorded per row as provenance only; it drives nothing.

---

## Where to go next

Three docs, each with one job (see `CLAUDE.md` for the discipline that keeps them
from contradicting each other):

- **`docs/AGENT_GUIDE.md`** — the coding agent's reference: repository structure,
  frozen interfaces, invariants, and the per-layer implementation reference.
- **`docs/DECISIONS.md`** — the only place history lives: every pivot, superseded
  design, and deviation, newest first.
- **`ops/kaya/KAYA.md`** — cluster operations: SLURM submission, offline-cache
  setup, sync, flags.

Generated (never hand-edited): `docs/REPO_STRUCTURE.md` (auto per-file map) and
`docs/generated/` (script outputs).