# Codebase guide

The single reference for a collaborator who **plans experiments and drafts the
paper without ever running the code** (e.g. a web-chat LLM with no shell). It
answers two kinds of question:

- **Operational** — "can the current pipeline do prompt × rung, or is that a code
  change?", "what fields are on a result row so I can plan a cross-tab?", "which
  tables already exist vs. need a new builder?", "which cells actually exist and
  are judged today?". That layer is **Part A** below.
- **Scientific** — the exact models, prompts, decoding, judge protocol, dataset,
  resolution, and retrieval settings the runs used. That layer is **Part B**
  (the former methods appendix), unchanged and authoritative for those facts.

Everything is the *actual* configuration, with `file:line` citations so each fact
is checkable. Where the code and a written summary disagree, the code / run spec
wins and the discrepancy is flagged inline.

## How to use this guide

| You want to know… | Go to |
| --- | --- |
| Can I express sweep X as a spec, or does it need code? | Part A §A1 |
| Which cells exist and are judged right now? | Part A §A2 |
| What a result row contains / where results live | Part A §A3 |
| Which tables exist, and how to add one | Part A §A4 |
| Which existing numbers to trust | Part A §A5 |
| Models, prompts, decoding, judge, dataset, retrieval (the science) | Part B §1–§9 |
| Frozen interfaces, cache-key contract, how a run executes | `docs/AGENT_GUIDE.md` |

Placement note: this is a working reference, not one of the repo's three canonical
docs (README / AGENT_GUIDE / DECISIONS). It leans on them but does not replace them.

---

# Part A — Operational layer

How the pipeline is driven and what it emits. The *science* each knob controls
lives in Part B; this part points there rather than restating it.

## A1. What you can change without a code change

A run is one YAML spec in `ops/specs/*.yaml` (see Part B §8 for the current specs
and the full axis vocabulary). An axis is **spec-only** (just write/edit a YAML)
when it is already part of the cell cache key or an existing cell-set option;
it **needs code** when it introduces a value or mechanism the pipeline can't yet
name. The cache key is `sha256(question_id, doc_id, condition, representation,
model_spec, page_indices, visual_resolution)` (`experiments/engine/paths.py:29`);
`condition` encodes retriever+k+prompt-mode (see §A3).

| Change | Spec-only? | Why / what it touches |
| --- | --- | --- |
| Reasoner size / family (2B/4B/8B/32B, InternVL3-8B) | ✅ spec | `reasoner_spec(s)`; specs already registered (Part B §1) |
| Quantization bf16 / 8bit / 4bit | ✅ spec | rides as a `model_spec` suffix, own cache rows |
| Visual resolution low / med / high | ✅ spec | `visual_resolution(s)`; part of the cell key |
| Prompt mode (none / grounded / abstain / abstain_balanced / cot / extract_cot; legacy aliases generic=grounded, targeted=abstain) | ✅ spec | `prompt_modes`; rides in `condition` suffix |
| Per-mode decode budget / final-answer delimiter | ✅ spec | `decode_budget` / `final_answer_delimiter`; run_tag-scoped, never in the cell key (a `run_settings.json` sidecar refuses mixing within one tag) |
| Retrieval depth k, representation rungs, parser, dataset, pool, scan filter, sampling | ✅ spec | all are config axes / cell-set options |
| Which retriever feeds generation (text/vision/joint arm) | ✅ spec | `inference_text_retriever` / `inference_vision_retriever` / `inference_joint` |
| A **new prompt string** (a 4th mode) | ❌ code | add to `config.PROMPT_MODES` (`config.py:61`) + `G3_PROMPT_MODES` |
| A **new reasoner / retriever / parser model** | ❌ code | add to the backend registry (`models/qwen3vl.py:24`, `models/internvl.py:15`, `retrievers/*.py`, `tools/parser.py:23`) |
| A **new page-selection policy** (beyond oracle / retrieved / joint / similarity / full) | ❌ code | new `InputConditioner` (`pipeline/conditioner.py`) |
| A **new covariate** to group by (not already on the row, §A3) | ❌ code | add to `Question`/`ResultRow` + backfill |
| A **new cross-tab** no current builder emits | ❌ code | new builder + plan entry (§A4) — the intended path, not a standalone script |

Prompt × rung specifically: **already supported as a spec** — `prompt_modes` and
`representations` are independent cell dimensions (`experiments/tasks/task.py:62`),
so a spec listing several of each produces the full cross. What does *not* exist
yet is a *builder* that tabulates prompt × rung (the closest is `hallucination`,
prompt-only); that table is the code change, not the run.

The Stage-3 frozen interfaces (`schema.py`, the pipeline/covariate ABCs,
`ModelInput`, the cache key + `ResultRow` shape) bound what stays spec-only:
anything that would change one of those is a checkpoint conversation, not a silent
edit (`docs/AGENT_GUIDE.md`).

## A2. Current data state (what exists and is judged)

Per source run_tag, from the live prediction/result caches (the build's own
`generation report`, regenerated 2026-07-18). "Cells" collapses re-runs to one row
per cell; `oom`/`error` are recorded as rows, never dropped (§A3). The run specs
behind each tag are in Part B §8.

| run_tag | task | cells | ok | oom | error | judged? | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g1-representation-full | G1 ladder (headline) | 3388 | 3143 | 245 | 0 | ✅ | the 8B baseline, scan:any |
| g1-reasoner-full / -scanned | G1 size/family | 7884 / 2280 | 7525 / 2239 | 359 / 41 | 0 | ✅ | 2B/4B/InternVL3-8B; digital + scanned halves merge at build |
| g1-resolution-full / -scanned | G1 resolution | 3942 / 1140 | 3549 / 1071 | 393 / 69 | 0 | ✅ | low/med/high on TLV,V |
| g1-quantization-full / -scanned | G1 quant | 5256 / 1520 | 4961 / 1488 | 295 / 32 | 0 | ✅ | 8bit/4bit |
| g1-parser-full-mineru | G1 parser | 1694 | 1466 | 228 | 0 | ✅ | TL,TLV |
| g1-parser-full-unlimited | G1 parser | 1694 | 1585 | 109 | 0 | ✅ | TL,TLV |
| g3-hallucination-full | G3 abstention | 2928 | 2698 | 230 | 0 | ✅ | none/generic/targeted; classifier side-artifact present |
| **g2-retrieval-full** | G2 retrieval | 5703 | 4435 | 1268 | 0 | ⚠ **partial** | see below |
| **all** | | 37429 | 34160 | 3269 | 0 | | 8.7% OOM overall |

\⚠ **G2 retrieval is the constrained arm.** The retrieval *benchmark* side-artifact
(page P/R/F1 per method, feeds `retrieval_accuracy*` and `retrieval_dpi`) is
complete. The retrieval-fed *generation* is **partial**: ~36% of the inference
cells were pulled locally before the compute cluster was migrated
(`docs/HANDOFF.md`), judging was still in flight, and the remainder is stranded on
the wiped cluster. So `matched_cross` and `kdepth` (which read G2 generation rows)
are built on a partial pool — treat them as provisional until a re-run.

\⚠ The `oom` cells (parser 337, g3 230, g2 ~1.27k, plus the G1 sweeps) are genuine
reasoner OOMs on the 16 GB V100, recorded as status rows. They exist as data but
carry no accuracy; recovering them needs an A100/H100 `--failed-only` sweep. Any
accuracy table silently excludes them (only `status==ok` rows score, §A3), so a
high-OOM cell (e.g. a long-document V rung) is measured on fewer questions — read
the per-column `n` footer.

## A3. Where results live and what a row contains

### Physical layout

```
results/cache/<run_tag>/<full|smoke>/<task_name>/
    predictions.jsonl   # one PredictionRow per cell (all statuses), no judge verdict
    results.jsonl       # one ResultRow per cell = prediction + judge verdict
    retrieval.jsonl     # G2 side-artifact: per (retriever,k,doc) page P/R/F1
    classifier.jsonl    # G3 side-artifact: per-doc predicted bin
    <phase>_status.json # per-phase run outcome
```

`<task_name>` ∈ {`G1_oracle_ladder`, `G2_retrieval`, `G3_hallucination`}. The
run_tag isolates one run's cells (`experiments/engine/paths.py:experiment_paths`).
One row = one cell. `results.jsonl` is a **strict superset** of `predictions.jsonl`
(same rows, plus the judge fields) — the judge phase reads predictions, scores,
and writes results, so build reads `results.jsonl` for accuracy and
`predictions.jsonl` only for OOM/status tables (`schema.py:254`).

**Join / group keys.** A cell's identity is `IDENTITY_FIELDS = (question_id,
doc_id, condition, representation, model_spec, visual_resolution)`
(`reporting/build.py:29`). Grouping on these collapses a cell's multi-judge or
re-run history to one row (the `ok` row wins, `_common.load_ok_rows`).

### The `condition` string grammar (this is where k and prompt-mode hide)

`condition = "<base>__<prompt_mode>"` (`experiments/tasks/task.py:30`). Split it
with `_common.split_condition`. Bases:

| base | meaning | example condition |
| --- | --- | --- |
| `oracle` | gold evidence pages | `oracle__none` |
| `retrieved_text_k<k>` | top-k from the text arm | `retrieved_text_k3__none` |
| `retrieved_vision_k<k>` | top-k from the vision arm | `retrieved_vision_k5__none` |
| `retrieved_joint_k<k>` | dedup union of text+vision top-k | `retrieved_joint_k3__none` |

So **k is inside the base**, **prompt-mode is the suffix**, retriever arm is in the
base. `provenance` (`oracle`/`retrieved`/`similarity`/`full`) is the coarser
selector class. G3's "similarity (bm25, k=3)" runs as `retrieved_text_k3__<mode>`
with `provenance=retrieved` (the `similarity` label in Part B §8 / `config.BASELINE`
is descriptive; the emitted base is `retrieved_text_k3`) — flag when citing.

### ResultRow fields (what you can filter / group on)

`ResultRow` (`schema.py:382`); `PredictionRow` (`schema.py:254`) is the same minus
the judge block. **Materialised on the row** (written at generation/judge time):

| group | fields | notes |
| --- | --- | --- |
| keys | `result_key`, `prediction_key` | sha256 cell ids (§A1) |
| axes | `condition`, `representation` (T/TL/TLV/V), `model_spec` (incl. `-4bit`/`-8bit`), `visual_resolution`, `provenance`, `page_indices` | the swept dimensions |
| covariates | `doc_type`, `bin_label`, `scan_label`, `hop` (none/single/multi), `is_unanswerable`, `evidence_sources` (tuple, a Q can cite several), `doc_id`, `question_id` | grouping axes for cross-tabs |
| judge | `score` (0/1), `correct`, `abstained`, `judge_spec` | verdict; `judge_spec` e.g. `gemini-flash` / `gpt4o-mini-judge` |
| status | `status` (ok/oom/error), `skipped_reason`, `oom_occurred` | failed cells are rows, not gaps |
| answer | `answer` | raw model text |
| tokens | `total_text_tokens`, `total_visual_tokens`, `text_tokens_fed`, `output_tokens`, `tokens_dropped`, `truncation_occurred` | `tokens_dropped`/`truncation_occurred` are a zero-canary (no input cap) |
| latency | `latency_s`, `prefill_latency_s`, `decode_latency_s` | prefill ≈ ingestion cost; end-to-end/decode inflated ~20× by the 256-token verbosity (Part B §3, §9) |
| memory | `peak_vram_bytes` | device 0 only, a lower bound — see Part B §9 |
| provenance | `machine`, `note`, `metadata.source_dataset` | `machine` drives nothing |

**Derived at build time** (not trustworthy off the raw row, recomputed by the loaders):

| field | how it's derived | where |
| --- | --- | --- |
| `scan_label` | blank rows backfilled from `annotations/auto_scan.csv` by doc_id | `reporting/tables/_load.py:_backfill_scan` (runs inside `load_ok`) |
| `doc_type` (retrieval side rows) | backfilled by doc_id from the corpus when blank | `reporting/build.py:_enrich_retrieval_rows` |
| `dpi` (retrieval side rows) | backfilled from `config.dpi` when 0 | same |
| `hop` | derived from evidence-page count at load | `schema.derive_hop` (materialised on `Question`, Part B §5) |
| accuracy + CIs | doc-level bootstrap over the grouped `ok` rows | `scoring/accuracy.py` (Part B §9) |

One real row (subset):

```json
{"question_id":"mmlongbench:000000","doc_id":"PH_2016.06.08_Economy-Final.pdf",
 "doc_type":"Research report / Introduction","condition":"oracle__none",
 "provenance":"oracle","representation":"T","model_spec":"qwen3vl-8b-local",
 "visual_resolution":"med","page_indices":[4],"status":"ok","correct":true,
 "abstained":false,"score":1.0,"judge_spec":"gemini-flash","hop":"single",
 "is_unanswerable":false,"evidence_sources":["Chart"],"scan_label":"digital",
 "prefill_latency_s":5.05,"decode_latency_s":6.27,"output_tokens":117,
 "peak_vram_bytes":8028019712}
```

## A4. The table build system (the primary output path)

All tables come from one plan-driven build; new analyses are **added to the build**,
not written as standalone scripts.

**Flow.** `reporting/plan.py` declares one `AnalysisTable` per output (its source
run_tag(s), the swept axis, the builder `"<module>.<fn>"`, whether it reads
`results`/`predictions`/`side:*`, and its caption/summary). `python -m ops.build`
→ `reporting.build.assemble_from_plan` runs each builder over its loaded rows,
attaches the baseline caption (swept axis + held-fixed values from
`config.BASELINE`, so each table explains itself), and `write_tables` emits **one
CSV per table** plus a combined **`results/tables/all_tables.md`** headed by the
generation report + baseline preamble. A builder that raises is logged and skipped,
so one bad table never sinks the run (`reporting/build.py:_safe`).

**Current inventory** (`reporting/plan.py`; ✓ = also emits a doc_type-collapsed
`<key>_summary`, markdown-only):

| key | shows | source run_tag(s) | reads | summary |
| --- | --- | --- | --- | --- |
| headline | ladder accuracy T/TL/TLV/V × doc_type | g1-representation-full | results | ✓ |
| parser | accuracy × parser (paddle/mineru/unlimited), TL/TLV | g1-representation + 2 parser tags | results | ✓ |
| resolution | TLV/V accuracy × resolution × doc_type | g1-resolution-full/-scanned | results | ✓ |
| scale | accuracy/VRAM/latency × reasoner spec | g1-representation + g1-reasoner-full/-scanned | results | – |
| quantization | accuracy + VRAM delta × quant × doc_type | g1-representation + g1-quantization-full/-scanned | results | ✓ |
| scan_vs_digital | ladder accuracy digital vs scanned × doc_type | g1-representation-full | results | ✓ |
| composition | accuracy × evidence_source × rung | g1-representation-full | results | – |
| routing | 4 routing policies: accuracy vs latency | g1-representation-full + G3 classifier | results + side | – |
| prefill_cost | prefill ms + input tokens × rung × doc_type | g1-representation-full | results | ✓ |
| vram_headroom | peak VRAM × spec/rung/resolution | all G1 run_tags (pooled) | results | ✓ |
| oom_frontier | OOM rate × rung/resolution/pages | all G1 run_tags (pooled) | **predictions** | ✓ |
| matched_cross | retrieval accuracy matched vs cross modality | g2-retrieval-full | results | – |
| kdepth | retrieval accuracy × depth k | g2-retrieval-full | results | – |
| retrieval_accuracy | page P/R/F1 × retriever × k (detail hidden from md) | g2-retrieval-full | **side:retrieval.jsonl** | ✓ |
| retrieval_accuracy_overall | page P/R/F1 × retriever × k (pooled) | g2-retrieval-full | side:retrieval.jsonl | – |
| retrieval_dpi | page P/R/F1 × render dpi × retriever × k | g2-retrieval-full | side:retrieval.jsonl | – |
| hallucination | abstention rate × prompt mode | g3-hallucination-full | results | – |
| abstention_by_doctype | abstention rate × prompt mode × doc_type | g3-hallucination-full | results | – |

**Cross-run merges.** `reporting/tables/_load.py` concatenates rows across a plan
entry's `run_tags` — that is how the digital+scanned halves merge (`load_ok` over
both tags), how `parser` compares three parser run_tags (`parser_by_tag` labels each
tag), and how `scale`/`quantization` line up the baseline 8B run against the sweep
tags. Because a merged/multi-spec cache can mix reasoners, single-reasoner tables
call `restrict_to_primary_spec` (`_common.py:108`) to keep `DEFAULT_REASONER_SPEC`
(or the most common spec) and **warn** rather than silently average 2B/4B/8B/32B
into one cell; `scale` and `quantization` deliberately skip it so they can compare
specs.

**How to add a table** (the supported extension path):
1. Write `reporting/tables/<name>.py` with `build(rows, *, margin_points=...) -> Table`
   (a builder may omit `margin_points`; the runner passes it only if present).
   Reuse the helpers: `_common` (`rows_for_condition`, `unanswerable_rows`,
   `restrict_to_primary_spec`, `group_by`, `acc_cell`, `doc_type_of`,
   `ordered_doc_types`, `frontier_rung`, cost helpers), `_load`
   (`load_ok`/`load_predictions`/`load_side`, `column_n_footer`), `_markdown`
   (report rendering).
2. Register an `AnalysisTable` in `reporting/plan.py` (key, task, `run_tags`,
   `swept`, `builder="<name>.build"`, `reads=`, optional `summary=`,
   `sweeps_key=` so the caption drops the swept axis, `detail_md=False` to hide a
   huge detail table from the md but keep the CSV).
3. Conventions: return a `Table` with a **per-column `n` footer** (`column_n_footer`)
   and let the plan attach the caption; `-` where a metric column has no meaningful n.
4. Run `python -m ops.build`; the table lands as `results/tables/<key>.csv` and in
   `all_tables.md`.

## A5. Aggregation issues and which numbers to trust

Two classes of silent aggregation bug have bitten this build before; both are
**fixed and verified in the current tree** (`python -m ops.build` is clean — no
pooling warning, no `(unlabeled)`/`(unknown)` column):

- **Scan-vs-digital backfill.** `mined_scan_vs_digital` groups by `scan_label`;
  cells generated before the auto-scan pass had a blank label and would form an
  `(unlabeled)` column. `load_ok` now backfills every blank `scan_label` from
  `annotations/auto_scan.csv` (`_load.py:_backfill_scan`), which labels all 135
  docs, so no `(unlabeled)` column appears. **Verified: absent.**
- **Primary-spec pooling.** Merged run_tags can carry several `model_spec`s; the
  single-reasoner tables now pass through `restrict_to_primary_spec` and warn on a
  pool. **Verified: no warning fires** (the merges that reach single-spec tables are
  8B-only).

Fixed in this change (same class — silent filter fallback made visible):

- The oracle-condition filters used the idiom `[... == "oracle"] or list(rows)`,
  which **silently pooled every condition** if the condition format ever drifted
  (e.g. `oracle` → `oracle__none` without a matching `base_condition` update). All
  such filters now go through `_common.rows_for_condition` / `unanswerable_rows`,
  which **log a warning and name the drift** before falling back. Behaviour is
  unchanged today (oracle rows exist, so no fallback fires and `all_tables.md` is
  byte-identical); a future drift now surfaces instead of quietly mixing conditions.
  Touched: `headline`, `composition`, `routing`, `resolution`, `scale`,
  `mined_prefill_cost`, `mined_scan_vs_digital`, `mined_quant_sensitivity`,
  `hallucination`, `mined_abstention_by_doctype`.

Trust summary: the **G1 tables** (headline, parser, resolution, scale, quantization,
scan_vs_digital, composition, routing, prefill_cost, vram_headroom, oom_frontier)
and **G3 tables** (hallucination, abstention_by_doctype) read complete, judged
caches. The **retrieval benchmark tables** (retrieval_accuracy*, retrieval_dpi) are
complete. The **G2 generation tables** (matched_cross, kdepth) are built on the
partial, possibly-unfinished-judging G2 pool (§A2) — provisional. Every accuracy
grid carries a per-column `n`; a small `n` (heavy OOM, or the partial G2) is the
signal to discount a cell.

---

# Part B — Exact experimental details (methods reference)

The scientific configuration the runs used. Part A points here for any science
fact rather than restating it. Everything below is unchanged from the methods
appendix and remains authoritative for models, prompts, decoding, the judge,
the dataset, resolution, and retrieval.

## 1. Exact model version strings / HF IDs and revisions

**No revisions are pinned anywhere in the code.** Every model is pulled at the
Hub default branch (`main`) at download time (`ops/scripts/download_hf.py` accepts
a `--revision` but the specs never pass one). So the "checkpoint" is whatever was
current when the weights were first cached. For the models still cached on this
machine I list the actual snapshot commit that was used; the rest were staged on
Kaya (now wiped by the migration) and are only recoverable as "default `main` as
of ~June–July 2026".

### Reasoner VLMs (`models/qwen3vl.py:24`, `models/internvl.py:15`)

| spec (internal) | HF ID | cached commit used |
| --- | --- | --- |
| `qwen3vl-2b-local` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `qwen3vl-4b-local` | `Qwen/Qwen3-VL-4B-Instruct` | (Kaya, default `main`) |
| `qwen3vl-8b-local` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| `qwen3vl-32b-local` | `Qwen/Qwen3-VL-32B-Instruct` | (Kaya, default `main`) |
| `internvl3-8b-local` | `OpenGVLab/InternVL3-8B` | (Kaya, default `main`) |

- **Baseline reasoner = `Qwen/Qwen3-VL-8B-Instruct`** (the `-Instruct` variant, not
  `-Thinking`), loaded via `Qwen3VLForConditionalGeneration` in bf16 (`torch_dtype="auto"`).
- The 2B/4B/8B/32B set is the model-size sweep (`config.scaling_specs`); InternVL3-8B
  is the cross-family comparison at 8B.

### Parsers (the TL/TLV text channel) (`tools/parser.py:23`)

| spec | HF ID | notes |
| --- | --- | --- |
| `paddleocrvl` (baseline) | `PaddlePaddle/PaddleOCR-VL` | PaddleOCR-VL-**0.9B**, pipeline `pipeline_version="v1"` explicitly selected (`tools/parser_worker.py:46`). Runs through the `paddleocr` 3.7 Python package, which pulls its own model bundle (PP-DocLayoutV2, PP-OCRv5_server det/rec, PP-LCNet orientation, UVDoc). Point release = whatever `pipeline_version="v1"` maps to in the installed paddleocr 3.7. |
| `mineru` | `opendatalab/MinerU2.5-2509-1.2B` | MinerU 2.5 (2509 release, 1.2B), a Qwen2-VL served through the vision auto-classes. |
| `unlimited` | `baidu/Unlimited-OCR` | DeepSeek-OCR-style; own `model.infer(...)` entry point. Pinned to `transformers==4.57.1` + `torch==2.10.0` in its isolated env (`docs/DECISIONS.md:556`). |

### Retrievers

Text (`retrievers/text.py:25`):
- BM25: `rank_bm25.BM25Okapi` (no model / no HF ID).
- `bge-m3` → `BAAI/bge-m3`, cached commit `5617a9f61b028005a4858fdac845db406aefb181`, loaded via `FlagEmbedding.BGEM3FlagModel`, dense vecs only.
- `qwen3-embedding` → `Qwen/Qwen3-Embedding-4B` (the **4B**), via `sentence_transformers.SentenceTransformer`.

Vision (ColPali-family, late-interaction) (`retrievers/vision.py:26`):
- `colmodernvbert` → `ModernVBERT/colmodernvbert` (+ base `ModernVBERT/colmodernvbert-base` for the text config; ~250M).
- `colqwen2.5` (baseline vision) → `vidore/colqwen2.5-v0.2`.
- `colqwen3` → `OpenSearch-AI/Ops-Colqwen3-4B` (loaded with `dims=2560`, fp16).

### Judges (`config.py:179`)
- `gpt-4o-mini` (OpenAI, `JUDGE_GPT_MODEL`).
- `gemini-2.5-flash` (Google, `JUDGE_GEMINI_MODEL`). API models, so no local revision.

### Domain classifier (`models/classifier.py:22`)
- `qwen3vl-2b-local` (= `Qwen/Qwen3-VL-2B-Instruct`), prompt version `bin-classifier-v1`,
  first `max_pages=2` pages, `V` (image-only) representation. Priced by G3 as a
  side-artifact only (predicted-domain routing is measured, not used to re-route).

---

## 2. Prompt templates (verbatim)

### Reasoner prompt (both backends identical)

Template ids: `qwen3vl-v1` (`models/qwen3vl.py:17`) and `internvl3-v1`
(`models/internvl.py:21`). They share the exact same header/body strings:

```
PROMPT_HEADER = "You are answering a question about a document."
PROMPT_BODY   = "Question:\n{question}\n\nDocument evidence:\n{context}\n\nAnswer:"
```

Assembly (`render_prompt`, `models/qwen3vl.py:77`): the instruction preamble (the
"prompt mode", see below) is appended to the header on its own line only when
non-empty; then a blank line, then the body.

- **With no instruction (mode `none`)** the full prompt string is:
  ```
  You are answering a question about a document.

  Question:
  {question}

  Document evidence:
  {context}

  Answer:
  ```
- **With an instruction (mode `generic` / `targeted`)** the header becomes two lines:
  ```
  You are answering a question about a document.
  {instruction}

  Question:
  {question}
  ...
  ```
- `{context}` is the composed representation (see §7 / §9). If empty it is replaced
  by the literal `(no document evidence was provided)`. Images are interleaved into
  `{context}` by replacing `<image>` placeholder tokens with real image blocks
  (Qwen chat-template image blocks; `<image>` tags for InternVL's `chat()`).
- There is **no input-token cap** — the entire text context is fed (`text_tokens_fed
  == total_text_tokens`, a zero-canary; `schema.py:40`).

### The prompt modes (instruction preambles) (`config.PROMPT_MODES`)

The registry holds six composed modes (`none`, `grounded`, `abstain`,
`abstain_balanced`, `cot`, `extract_cot`) plus two frozen legacy aliases:
`generic` is byte-identical to `grounded`, `targeted` byte-identical to
`abstain`. The strings the completed runs used:

```python
"none":     ""
"generic":  "Use only the provided document evidence and keep the answer concise."   # = grounded
"targeted": ("Use only the provided document evidence. If the evidence does not "
             "contain the answer, answer exactly: Not answerable.\n"
             "Keep the answer concise.")                                             # = abstain
```

**Which mode each completed task actually ran (from the run specs, not the defaults):**
- **G1 (oracle ladder), all sweeps** — `prompt_modes: [none]` → empty instruction, header only.
- **G2 (retrieval)** — `prompt_modes: [none]`.
- **G3 (hallucination / abstention)** — `prompt_modes: [none, generic, targeted]`, i.e. the
  three-way prompting comparison; `none` is the baseline arm.
- The six-mode faithfulness sweeps (`g3-faithfulness-full`, `g4-faithfulness-full`
  specs) are **planned, not yet run**; they add per-mode decode budgets
  (cot/extract_cot at 1024) and judge-time `Answer:` extraction, both run_tag-scoped.

> Caveat: `config.py:69` sets `DEFAULT_PROMPT_MODE = "targeted"` and a stale comment
> claims "targeted is the one every answerable (G1/G2) cell uses". That default is only
> the library fallback when a caller passes `instruction=None`; the **run specs pin
> `none` for G1/G2** (`ops/specs/kaya_g1_*_full.yaml`, `kaya_g2_full.yaml`) and
> `config.BASELINE` records `prompt_mode: none` for G1/G2. So the answerable runs used
> the empty prompt, matching your recollection.

### Domain-classifier prompt (`models/classifier.py:129`, id `bin-classifier-v1`)

```
Classify this document by which modality dominates its information content.
Use only the first pages provided. Answer with exactly one label from this list:
- text-dominant
- mixed-modality
- visual-dominant

Modality label:
```

### Parser OCR prompts
- PaddleOCR-VL: no free-text prompt (fixed pipeline).
- MinerU (`tools/parser_worker.py:84`): `"Convert this document page to clean Markdown. Output only the Markdown."`
- Unlimited-OCR (`tools/parser_worker.py:147`): `"<image>document parsing."`

---

## 3. Decoding parameters

### Reasoner (Qwen3-VL and InternVL3)
- **Greedy, `do_sample=False`.** No `temperature`, no `top_p`, no `top_k`, no
  `repetition_penalty` are set — the `generate` call passes only `max_new_tokens`,
  `do_sample=False`, and a streamer (`models/qwen3vl.py:397`; InternVL sets
  `generation_config={"max_new_tokens": ..., "do_sample": False}`, `models/internvl.py:177`).
- **`max_new_tokens = 256`** for the real runs (`config.DEFAULT_MAX_TOKENS = 256`,
  `config.py:54`). This is the value that was bumped from **64 → 256** and produced the
  verbose answers; `docs/DECISIONS.md:43` notes decode latency is inflated ~20× as a result.
- **Smoke runs cap at 64** (`SMOKE_MAX_TOKENS = 64`, applied only when `smoke=True`).
- `max_new_tokens` per model is recorded in each cell's metadata.

### Parser VLMs (not reasoner answers, but for completeness)
- MinerU: `max_new_tokens=4096`, `do_sample=False` (`tools/parser_worker.py:135`).
- Unlimited-OCR: `max_length=32768`, `no_repeat_ngram_size=35`, `ngram_window=128`
  (`tools/parser_worker.py:187`); gundam crop-mode tiling with base/base-640 fallbacks.

### Judges
- `temperature=0`, JSON-only response format (`response_format={"type":"json_object"}`
  for GPT-4o-mini, `response_mime_type="application/json"` for Gemini)
  (`pipeline/judge.py:210`, `:263`).

---

## 4. Judge rubric / prompt (how correctness is adjudicated)

Adjudication is **LLM-judge, not exact match.** The judge sees the question, the
gold answer, the unanswerable flag, and the model answer, and returns a structured
verdict.

### System prompt / rubric (`config.py:181`, `JUDGE_SYSTEM_PROMPT`)

```
You judge answers to document questions.
Return only JSON with keys:
- verdict: one of correct, incorrect, abstained
- extracted_answer: the answer extracted from the model response, or empty string
- rationale: a short reason

Mark correct when the model answer is semantically equivalent to the gold answer.
For unanswerable questions, mark correct only when the model abstains.
```

### User payload (`pipeline/judge.py:145`)
A JSON object (sorted keys):
```json
{"question": ..., "gold_answer": ..., "is_unanswerable": ..., "model_answer": ...}
```

### Extraction procedure for the verbose answers
- The judge itself does the extraction: it returns `extracted_answer` = the answer
  pulled out of the (possibly verbose) model response, plus a `verdict` and a short
  `rationale`. There is no separate regex/heuristic extractor for the generation cells.
- The judge's JSON is parsed leniently (`_extract_json_object`, `pipeline/judge.py:126`):
  strips ```` ```json ```` fences, else grabs the first `{...}` block.
- Verdict → score (`_score_from_verdict`, `pipeline/judge.py:159`):
  `correct` iff `verdict == "correct"`; `abstained` iff `verdict == "abstained"`
  **or** the answer matches an abstention surface form. For an unanswerable question,
  abstaining is forced to `correct = True`.
- **Abstention detection** (`scoring/abstention.py` + `config.ABSTENTION_FORMS`,
  `config.py:158`) matches these casefolded substrings against the answer:
  `not answerable`, `cannot be answered`, `can not be answered`, `cannot answer`,
  `unanswerable`, `insufficient information`, `not enough information`, `no answer`,
  `unknown from the document`, `not mentioned`, `not provided`.
- **Fallbacks:** if Gemini returns empty/non-JSON (e.g. a safety block) after 3
  tries, that one cell falls back to a heuristic `StubJudge` (substring gold-in-pred),
  so a single bad response can't abort a run (`pipeline/judge.py:272`). Multi-key
  failover on quota (`GEMINI_API_KEY` → `GOOGLE_API_KEY` → `GEMINI_API_KEY_SECONDARY`).
- **Two judges were used** (`gpt-4o-mini` and `gemini-2.5-flash`) — different families,
  which supports a judge-agreement check (`scoring/agreement.py`). The primary reported
  numbers are judged; the `StubJudge` is only offline plumbing / the per-cell fallback.

---

## 5. Dataset specifics

**Dataset: MMLongBench-Doc** (the only dataset actually run; a LongDocURL loader
exists in `data/loader.py` but was not used for the reported tables). Loaded from a
single staged parquet shard `.data/mmlongbench/data/train-00000-of-00001.parquet`
over **135 PDF documents**.

**Exact counts (recomputed from the staged data on 2026-07-18):**

| split | count |
| --- | --- |
| **Total questions** | **1091** |
| Answerable | **847** |
| Unanswerable (gold = "Not answerable") | **244** |
| Documents | 135 |
| Native `doc_type` labels | 7 |

- Answerable/unanswerable split is derived, not from a column: a question is
  unanswerable iff its gold answer casefolds to exactly `"not answerable"`
  (`schema.is_not_answerable`, `schema.py:34`). So **847 / 244**, not the
  approximate 841 / 250.
- **No questions are dropped.** Every row becomes a `Question`; failed reasoner
  cells are recorded as rows with `status=oom/error` rather than removed
  (`schema.py:18`). The manual modality-bin join is an *enrichment* only — docs
  without a hand label just get an empty `bin_label`, the question still runs.

**Answer-format distribution:** `Int` 290, `Str` 250, `None` (unanswerable) 244,
`Float` 160, `List` 147.

**Evidence-hop distribution** (derived from evidence-page count, `schema.derive_hop`):
single-page 487, multi-page 358, none (unanswerable / no gold pages) 246.
(The 246 "none" ≠ 244 unanswerable: two answerable questions carry no evidence pages.)

**Native `doc_type` (question counts):** Research report / Introduction 293,
Academic paper 204, Guidebook 156, Tutorial/Workshop 139, Financial report 117,
Brochure 101, Administration/Industry file 81.

### Evidence-page normalization (`data/loader.py:138`)
- Source `evidence_pages` are **1-based**; converted to **0-based** indices
  (`source_page - 1`, floored at 0), de-duplicated, order preserved.
- `page_count_required = max(evidence_pages) + 1` (`schema.py:86`); a validator
  checks gold pages fall inside the resolved PDF (`data/render.validate_gold_pages`).
- Unanswerable questions have no gold pages; for rendering, page 0 is used as a
  cheap sanity page (`data/render.py:165`).

### Modality bins (the thesis axis) — hand-labelled (`annotations/doc_labels.csv`)
Three ordered bins text→visual: `text-dominant`, `mixed-modality`, `visual-dominant`
(`config.DEFAULT_BINS`). Labels are per-document, joined onto questions
(`data/binning.stamp_bins`). Distribution over the 1091 questions:

| bin | questions | documents |
| --- | --- | --- |
| mixed-modality | 513 | 61 |
| visual-dominant | 279 | 36 |
| text-dominant | 69 | 10 |
| (unlabelled) | 230 | 28 |

28 of the 135 docs (230 questions) have no manual bin label; they still run, just
excluded from bin-grouped tables.

### Scan / digital split
- Per-document label from `annotations/auto_scan.csv` (auto-detected via PyMuPDF)
  plus manual `scan_label`. A page counts as "text" if it has ≥ 20 extracted chars
  (`SCANNED_MIN_CHARS_PER_PAGE = 20`); a doc is "scanned" when none of its first 5
  sampled pages carries a real text layer (`data/render.classify_scanned:47`).
- Distribution over questions: digital 672, scanned 189, unlabelled 230
  (docs: 84 digital, 23 scanned, 28 unlabelled).

---

## 6. Resolution presets (exact `max_pixels`)

`config.VISUAL_RESOLUTION_PRESETS` (`config.py:130`). Value = per-page pixel cap =
`tokens_per_page × 28 × 28` (Qwen packs one vision token per 28×28 patch), attached
as `max_pixels` on each image block so `qwen_vl_utils` downscales the page before
tokenizing:

| preset | tokens/page | `max_pixels` |
| --- | --- | --- |
| `high` | ~960 | 960 × 28 × 28 = **752,640** px |
| `med` (deployment default) | ~640 | 640 × 28 × 28 = **501,760** px |
| `low` | ~400 | 400 × 28 × 28 = **313,600** px |

- **Deployment resolution = `med`** (`config.DEPLOYMENT_RESOLUTION`), used by every
  table except the scientific resolution sweep. Note in code: this is still flagged a
  **PLACEHOLDER** pending the operational resolution-probe job `1017226`, whose verdict
  never landed before Kaya went down (`config.py:192`, `docs/DECISIONS.md:667`). In
  practice all runs used `med`; the resolution sweep (`kaya_g1_resolution_full.yaml`)
  runs `[low, med, high]` on the TLV and V rungs.
- These caps bound the VLM's *view* of the page. They are independent of the render
  DPI (200, see §9); a page is rasterized at 200 DPI, then the VLM downsamples to the
  `max_pixels` budget. Resolution is the one representation parameter held identical
  across machines (a lower-res image is a genuinely different, lossier input).

---

## 7. Retrieval indexing details

**Retrieval unit = the page.** Every retriever ranks the document's pages and
returns page indices; there is no sub-page chunking. This matches the survey's
page-level framing. `k ∈ {1,3,5,7,10}` for similarity/retrieved conditions
(`config.k_values`); the G2 benchmark scores `k ∈ {1,3,5}` and joints at `{1,3}`.

### BM25 (lexical, cheap rung) (`retrievers/text.py:29`)
- Library: `rank_bm25.BM25Okapi`, one index **per document** (page = document unit),
  built once and reused across every question and `k`.
- **Tokenizer** (`retrievers/__init__.py:21`): `re.findall(r"[A-Za-z0-9]+",
  text.casefold())` — lowercase alphanumeric runs, no stemming, no stopword removal.
- Page text is the PyMuPDF embedded text layer (not the parser markdown).
- Scores are min-max normalized to [0,1] before ranking; ties broken by page order.
- Fallback pure-python BM25 (only if `rank_bm25` import fails) uses `k1=1.5, b=0.75`
  (`simple_bm25_scores`, `retrievers/__init__.py:81`); the real runs use BM25Okapi's
  own defaults (`k1=1.5, b=0.75, epsilon=0.25`).

### Dense text rungs (`retrievers/text.py:90`)
- Cosine similarity of a query embedding vs. per-page embeddings; page embeddings
  cached once per document.
- **`bge-m3`** (`BAAI/bge-m3`): dense vecs, **1024-dim**, loaded fp16 on GPU
  (`FlagEmbedding.BGEM3FlagModel`), encode batch 8. (Its wrapper exposes no
  tokenizer, so no truncation telemetry.)
- **`qwen3-embedding`** (`Qwen/Qwen3-Embedding-4B`): **2560-dim** (model default),
  fp16, **`max_seq_length` capped to 4096 tokens** and **encode batch = 1** to fit a
  16 GB V100 with no FlashAttention (`config.QWEN3_EMBEDDING_MAX_SEQ_LEN = 4096`,
  `QWEN3_EMBEDDING_ENCODE_BATCH = 1`, `config.py:141`). Only the rare very-long page is
  truncated; truncation is recorded per page in the retrieval memo (24/840 questions
  had one page over the cap in the full run).
  > Deviation to flag: queries are encoded **without the model's instruction prompt**
  > (`retrievers/text.py:_encode` calls plain `encode()`, no `prompt_name="query"`),
  > so the `Instruct: ... \nQuery: ...` prefix the model card recommends is omitted.
  > The card estimates a 1–5% retrieval gain from the instruction; cite the arm as
  > "no-instruction". Rankings were otherwise verified consistent end-to-end
  > (memo ↔ scored rows byte-exact, no silent bm25 fallback in the full run).

### Vision rungs (ColPali-family late interaction) (`retrievers/vision.py:38`)
- Multi-vector (ColBERT-style) late interaction over **rendered page images**;
  relevance is MaxSim via `processor.score_multi_vector(query, page_embeddings)`.
  Not a single dense vector — one vector per image patch / query token.
- Page images rendered at the run DPI (200); page-image embeddings cached per doc
  (LRU over 8 docs), embed batch 2 (ColQwen3 batch 1).
- `colqwen2.5` = `vidore/colqwen2.5-v0.2` (baseline vision arm); `colqwen3` =
  `OpenSearch-AI/Ops-Colqwen3-4B` (loaded `dims=2560`, fp16); `colmodernvbert` =
  `ModernVBERT/colmodernvbert`.

### Joints (`retrievers/joint.py`, `config.joints = "matched"`)
- Text × vision unions paired by cost rung: cheap|cheap, mid|mid, expensive|expensive
  (bm25|colmodernvbert, bge-m3|colqwen2.5, qwen3-embedding|colqwen3), scored at `k ∈ {1,3}`.

### Retrieval used for generation (G2 inference stage)
Per the run spec (`ops/specs/kaya_g2_full.yaml`), the single arms feeding the reasoner
were: **text = `bge-m3`, vision = `colqwen2.5`, plus the joint union**, over the
**TLV and V** rungs.
> Discrepancy to flag: `config.BASELINE["G2_retrieval"]` captions this as
> "bm25 text / colqwen2.5 vision / joint" and `config` defaults `inference_text_retriever
> = "bm25"`, but the actual `g2-retrieval-full` spec overrode it to **`bge-m3`**. Confirm
> which you cite. The retrieval *benchmark* side-artifact (P/R/F1 per method) scored all
> three text + three vision + joints regardless.

---

## 8. The experiment matrix (what each run actually swept)

From `ops/specs/*.yaml`. All runs: dataset MMLongBench, `parser_dpi: 200`,
`visual_resolution` med unless swept, bf16 unless swept, `prompt_mode none` unless
swept (G3), Qwen3-VL-8B unless swept.

| run_tag | task | pool / scan | swept axis | reasoner | rungs |
| --- | --- | --- | --- | --- | --- |
| `g1-representation-full` | G1 oracle ladder (headline) | answerable / any (847 q) | representation | 8B | T,TL,TLV,V |
| `g1-reasoner-full` | G1 | answerable / **digital** | model family/size | 2B,4B,internvl3-8b | T,TL,TLV,V |
| `g1-parser-full` | G1 | answerable / any (847 q) | parser (mineru, unlimited; paddle is baseline) | 8B | TL,TLV |
| `g1-resolution-full` | G1 | answerable / **digital** | resolution low/med/high | 8B | TLV,V |
| `g1-quantization-full` | G1 | answerable / **digital** | quant 8bit/4bit (bf16 baseline) | 8B | T,TL,TLV,V |
| `g2-retrieval-full` | G2 retrieval | answerable / any | text×vision retrievers, k | 8B | TLV,V |
| `g3-hallucination-full` | G3 abstention | **unanswerable** / any (244 q) | prompt mode none/generic/targeted | 8B | T,TL,TLV,V |

- G1 headline and parser use **scan: any** (all 847 answerable); the reasoner-size,
  resolution, and quant sweeps use **scan: digital** only.
- G3 page selection: similarity retrieval, **bm25, k=3** (`retrieval_representation: [T]`,
  `k_values: [3]`, `inference_text_retriever: bm25`); classifier `qwen3vl-2b-local`
  priced as a side-artifact.

### Quantization (`models/qwen3vl.py:307`)
- bf16 (baseline, no quant).
- `4bit`: bitsandbytes NF4, `bnb_4bit_compute_dtype=bfloat16`, double-quant on (~7 GB for 8B).
- `8bit`: bitsandbytes `load_in_8bit` (~10 GB for 8B).
- A quantized run appends `-4bit`/`-8bit` to the spec so it gets its own cache rows.

---

## 9. Additional details (regardless of relevance)

### Representation ladder T / TL / TLV / V (`pipeline/representation.py`)
- Five cost-ordered rungs; **not cumulative**. T = PyMuPDF embedded text. TL =
  parser markdown text (replaces T's text, read from the warmed parser cache). TLV
  = parser markdown text + page images. TLVi = the same channels and the same token
  cost as TLV, ordered per page. V = page images only (image-only reference).
- The "L" (layout) is historical — there is no separate bounding-box/layout channel.
- **Ordering differs between TLV and TLVi.** TLV emits the text channel as a single
  labelled block `[text]\n...` holding every page's text joined by blank lines, then
  every image after it, so on a multi-page cell nothing associates a text chunk with
  an image. TLVi emits `[page N]` + that page's text, then that page's image, per
  page, so adjacency carries the pairing (`N` is the 1-based document page number).
  Both bind images to ordered `<image>` placeholders the same way.
- **TLVi is opt-in.** It is in `REPRESENTATION_LADDER` (so it is valid anywhere a
  rung is, and appears in the tables whenever a run produced it) but not in
  `DEFAULT_REPRESENTATIONS`, so a spec must list it. `representation` is a cache-key
  component, so TLVi cells never collide with TLV cells.
- `<image>` collision fix: literal `<image>` inside document text is rewritten to
  `[image]` before real sentinels are inserted (`models/payload.py`, `docs/DECISIONS.md:50`).

### Rendering / DPI
- **Render DPI = 200** for everything (`config.dpi = 200`, `parser_dpi: 200` in all
  specs) — the resolution the OCR/parser and the vision retrievers see; the VLM then
  downsamples images to the `max_pixels` budget (§6). Rendering via PyMuPDF
  (`zoom = dpi/72`).
- Text retrievers extract the PyMuPDF text layer (render_images=False), so DPI doesn't
  affect them; it affects parser input and vision retrieval.

### Sampling
- Reported full runs use `sampling: full` (whole answerable/unanswerable pool). The
  config also supports `per_bin` (default 100 q/bin, doc-coherent), `per_doc_type`,
  `limit`, and explicit `ids` subsets; these were for iteration, not the headline tables.

### Cost / latency telemetry per cell (`schema.py`, `models/qwen3vl.py:411`)
- `latency_s` (end-to-end generate), split into `prefill_latency_s` (time-to-first-token,
  via a streamer — a proxy for ingesting the representation) and `decode_latency_s`.
  Note: this split is only available for Qwen3-VL; InternVL's `chat()` records
  end-to-end only (prefill/decode = 0).
- `peak_vram_bytes` (`torch.cuda.max_memory_allocated`), `total_text_tokens`,
  `total_visual_tokens` (from Qwen `image_grid_thw`), `output_tokens`, `text_tokens_fed`.
- ⚠ **`peak_vram_bytes` is device 0 only, and is a lower bound.** Cells ran on 2× V100
  and the reasoner loads with `device_map="auto"`, which shards across both GPUs for
  **every** spec (`_max_memory_map`, `models/qwen3vl.py:285`, triggers on GPU count,
  not model size). But `reset_peak_memory_stats()` / `max_memory_allocated()` are
  called with no device argument (`models/qwen3vl.py:388,404`), so only the current
  device is measured. The evidence is in the data: reported minima sit at about half
  each model's bf16 weight size (2B 2.15 GB, 4B 4.45 GB, 8B 7.82 GB, against ~4/8/16 GB
  of weights), and 2B reads *higher* than 8B in `scale` because 8B's other shard is
  invisible. Device 1's peak was never written to any row, so this is **missing data,
  not a reporting bug**, and it is not recoverable from the cache — only a re-run with
  per-device accounting would fix it. Recording is deliberately left as-is so the
  34k existing rows stay internally comparable; the VRAM tables carry the caveat via
  `_common.SINGLE_DEVICE_VRAM_NOTE`.
- Cost metric = `latency_bs1` (batch-1 latency). Reminder: decode latency is inflated
  ~20× by the 256-token verbosity change, so prefill_ms is the cleaner cost column.
- InternVL vision-token accounting: one 448px tile = (448/14)² = **1024 tokens/page**,
  one tile per page (`models/internvl.py:26`).

### Attention / memory workarounds (V100, no FlashAttention-2)
- Reasoner forces the memory-efficient SDPA kernel (EFFICIENT → FLASH → MATH) so long
  visual sequences don't OOM materializing the full score matrix (`models/qwen3vl.py:349`).
- Multi-GPU loads reserve 5 GiB/GPU headroom via a `max_memory` map so activations fit
  (`models/qwen3vl.py:285`).
- Parser worker sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and falls back
  gundam → base-1024 → base-640 on OOM.

### Evaluation / statistics (`config.py:145`, `scoring/`)
- Accuracy = judge correctness. **Document-level bootstrap CIs**: `N_BOOTSTRAP = 1000`
  resamples, `BOOTSTRAP_SEED = 0`, 95% interval (2.5% / 97.5% quantiles). Resampling is
  at the document level (not question level) to respect doc-coherent structure.
- Pre-registered **sufficiency margin = 3.0 accuracy points** (`config.sufficiency_margin`).
- Frontier / ladder analysis over rung order `("T","TL","TLV","V")` (`scoring/frontier.py`).
- Judge agreement between GPT-4o-mini and Gemini computed in `scoring/agreement.py`.

### Caching / determinism
- Every cell is content-addressed (`prediction_key` / `result_key` = SHA-256 over the
  identity tuple; `docs/DECISIONS.md:688`); the parser tool is *not* in the prediction
  cache key, so each parser lands in its own `run_tag`. Greedy decoding + fixed seeds
  make cells reproducible.

### Known data-state caveats at handoff (`docs/HANDOFF.md`)
- Some cells are genuine reasoner OOMs on the 16 GB V100 (parser 109, g3 230, g2 ~1.25k),
  recorded as `oom` rows, not dropped — they need an A100/H100 `--failed-only` sweep.
- G2 inference was only ~36% pulled locally before Kaya's migration; the retrieval
  *benchmark* (P/R/F1) and G1/G3 are complete and judged.

---

## What this guide deliberately does not cover

- **The science behind each knob** — why the ladder is T/TL/TLV/V, what a bin means,
  the judge κ bar, bootstrap/frontier definitions, sampling semantics: those live in
  Part B §1–§9 above (and are owned there, not restated in Part A).
- **The frozen interfaces and the caching contract** — the exact ABCs, `ModelInput`,
  the cache-key freeze rules, and how a grouped run executes phase by phase: see
  `docs/AGENT_GUIDE.md`. Part A cites the cache-key composition and the swap points
  but does not reproduce that contract.
- **History / rationale of past changes** (pivots, superseded designs): `docs/DECISIONS.md`.
- **Live numbers.** Counts here (§A2, §A3 example) are a snapshot regenerated from the
  caches on 2026-07-18; re-run `python -m ops.build` for the current generation report
  rather than trusting these if the caches have moved on.

If Part A and `AGENT_GUIDE.md`/`README.md` ever disagree on an operational fact,
the code wins and the disagreement is a bug to flag, not a choice to make silently.
