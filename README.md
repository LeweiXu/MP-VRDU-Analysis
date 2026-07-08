# MP-VRDU: multi-page visually-rich document understanding

This repo measures **which document representation is actually needed** to answer
questions over long, visually-rich PDFs (plain text, text+layout, text+layout+vision,
or vision only), and **what each choice costs** on deployable local models. The
benchmark is MMLongBench-Doc; the primary reasoner is Qwen3-VL.

The rest of this file explains how the document-understanding pipeline is
actually built. If you just want to run it, see the quick start, then
`docs/USER_GUIDE.md` (local) and `kaya/KAYA_USER_GUIDE.md` (cluster).

> **v4 pivot note.** The project has adopted the **v4 plan** (`docs/pivot_v4.md`):
> a cost-ordered ladder (`T` = PyMuPDF embedded text, `TL` = parser-derived
> markdown that replaces `T`, **no bbox JSON**), a parser comparison, manual
> `bin_label` binning, four generation tasks (G1-G4), a fixed telemetry schema, and
> a Kaya/supervisor hardware split. **This README still describes the current code,
> which implements the v3 mechanisms** (Marker text + bbox-JSON layout, six tasks
> G1/G2/G3/G5/G6). The migration to v4 is pending. For the adopted intent see
> `docs/pivot_v4.md`, `docs/PROJECT_SPEC.md`, and `docs/AGENT_GUIDE.md` ("v4 pivot:
> adopted, code migration pending"); for what runs today, read on.

## Quick start

```bash
# 1. env
python3.11 -m venv envs/mpvrdu
envs/mpvrdu/bin/python -m pip install -r requirements.txt

# 2. sanity check
envs/mpvrdu/bin/python -m pytest

# 3. generate (GPU) -> judge (needs a Gemini/OpenAI key in .env) -> build tables
envs/mpvrdu/bin/python -m cli.generate --spec specs/smoke_generation.yaml
envs/mpvrdu/bin/python -m cli.judge    --run-tag yaml-smoke
envs/mpvrdu/bin/python -m cli.build
```

The three phases are split on purpose: **generate** is GPU-only and offline
(runs on the cluster), **judge** and **build** are local and need API keys.
Everything is cached and resumable. Full run commands, cluster submission, flags,
and outputs live in `docs/USER_GUIDE.md` and `kaya/KAYA_USER_GUIDE.md`.

---

# How the model is implemented

## 1. The pipeline and the "cell"

Every number in the paper comes from one **cell**: a single
(question, page-selection, representation, model) combination. Answering a cell
runs five stages, each a frozen interface so pieces can be swapped without
touching the rest:

1. **Conditioner** (`pipeline/conditioner.py`) picks *which* pages the model sees.
2. **Render** (`data/render.py`) rasterizes those pages and pulls their text.
3. **Representation** (`pipeline/representation.py`) turns the rendered pages into
   a text/layout/vision payload for the chosen rung of the ladder.
4. **Reasoner** (`models/`) feeds the payload to the MLLM and generates an answer.
5. **Judge** (`pipeline/judge.py`) scores the answer against the gold answer.

The orchestrator (`pipeline/orchestrator.py`) runs these in order and caches the
result keyed on the whole tuple, so re-runs only fill missing cells.

## 2. Page selection (which pages reach the model)

MMLongBench documents are long (tens to hundreds of pages), so a cell first
decides which pages to feed. The conditioners:

- **Oracle** (`OracleConditioner`): exactly the gold evidence pages. This is the
  reasoning *ceiling* (perfect retrieval), and it's what the sufficiency ladder
  (G1) uses. Native-unanswerable questions have no gold pages, so page 0 is fed
  as a sanity anchor.
- **Retrieved top-k** (`RetrievedTopK`): the k highest-ranked pages from a real
  retriever. Two retrievers (`covariates/retriever.py`):
  - **Text**: BM25 over page text plus dense BGE similarity (`BAAI/bge-small-en-v1.5`).
  - **Vision**: ColQwen late-interaction over rendered page images (`vidore/colqwen2.5-v0.2`).
  This is what the matched-vs-cross study (G5) uses: *matched* = vision-retrieval
  feeding vision reasoning, *cross* = text-retrieval feeding vision reasoning. G5
  sweeps k over `config.k_values` (default `(1, 3, 5, 7, 9)`), so both retrievers
  are run at every k.
- **Full doc** (`FullDoc`) and **buried oracle** (`BuriedOracle`, gold pages plus
  deterministic distractor pages) are the feed-everything and distractor
  baselines.

The conditioner returns a `PageSet` (zero-based indices + provenance), which is
part of the cache key.

## 3. Rendering

`data/render.py` uses PyMuPDF to rasterize each selected page to a PNG at a
**fixed 144 DPI**, and separately pulls line-level text spans (each with a
bounding box) straight from the PDF. Renders are cached deterministically under
`results/cache/renders/<stem>__dpi144/`, so a page is only rasterized once. The
144 DPI base image is the *source* for the vision layer; the actual resolution
fed to the model is capped later (see section 5), not here.

## 4. The three layers (the T / TL / TLV / V ladder)

The representation ladder has four rungs, built by composing three independent
channels (`tools/`). Each channel produces one string or image *per page*.

- **Text layer** (`tools/text.py`): the primary text channel is selected per
  document. Digital-born PDFs use **Marker** (`tools/layout.py::marker_text`),
  which converts each page to markdown. Scanned PDFs use PaddleOCR PP-OCRv5
  (`tools/text.py::ocr`) instead. The scanned/digital decision comes from
  `annotations/doc_labels.csv`: a filled human `scan_label` wins, otherwise the
  auto-seeded `auto_scan` value is used. If the sheet has no row for a document,
  the embedded-text scanned heuristic is used as a fallback.
- **Layout layer** (`tools/layout.py::marker_bbox_json`): Marker's JSON output,
  flattened to a list of blocks (block type + bounding box + short text) and
  **serialized as a JSON string**. This is important: layout is fed to the model
  as *text* describing where things are, not as an image. Fallback is PyMuPDF
  line-level spans.
- **Vision layer** (`tools/visual.py`): one **full-page** image per page. There
  are no in-page crops because MMLongBench has no in-page evidence boxes, so the
  region-crop path deliberately falls back to the whole page.

The rungs combine these:

| Rung | Text | Layout (JSON) | Page images |
|------|------|---------------|-------------|
| `T`   | yes | no  | no  |
| `TL`  | yes | yes | no  |
| `TLV` | yes | yes | yes |
| `V`   | no  | no  | yes |

Only `TLV` and `V` carry images; `Payload.__post_init__` re-checks this so a bug
can't leak an image into a text-only rung.

**Parse pre-pass.** Marker (and its Surya sub-models) and PaddleOCR can run on the
GPU. To avoid the parser/OCR stack and the reasoner fighting over a 16 GB V100,
the generate loop warms the text/layout cache for every cell *first* (reasoner not
yet loaded), writes the results to disk (`results/cache/.../marker/` and
`results/cache/.../ocr/`), then frees the GPU before loading the reasoner. On a
warm cache the reasoner phase never loads Surya or PaddleOCR.

### OCR handling

OCR is **not** a separate ladder rung. The active design keeps the four existing
rungs (`T` / `TL` / `TLV` / `V`) and chooses the text extractor by document:
Marker for digital-born PDFs, PaddleOCR for scanned PDFs. This makes `T` the
best available text channel for the document while preserving all existing cache
keys, frontier logic, and table shapes.

The older five-rung idea (`T / TO / TOL / TOLV / V`) is deliberately superseded.
That design would have run OCR on every page as a separate prompt block. The
implemented design instead runs OCR only for documents marked scanned in
`annotations/doc_labels.csv`, using `scan_label` when filled and `auto_scan`
otherwise.

**Why this is safe re: OOM.** OCR adds *text* tokens, not vision tokens, and only
for scanned documents. The input-token cap (`_truncate_context`, section 7)
hard-bounds the combined text before it reaches the O(seq^2) V100 attention.

**Current caveats.** OCR changes the text channel for scanned documents without
changing rung names, so use a fresh `--run-tag` when comparing against older
Marker-for-everything caches. `TL`/`TLV` still include the existing layout channel;
`V` stays vision-only to keep the "can vision alone answer" signal.

## 5. Visual resolution and downscaling

The knob that controls vision cost is **`max_pixels` per page**, not the render
DPI. Qwen packs one vision token per 28x28 patch, so a per-page pixel cap is a
per-page token cap: `max_pixels = tokens_per_page * 28 * 28`.

Downscaling happens at tokenization, not on disk. Each image block in the chat
message carries a `max_pixels` value; `qwen_vl_utils` "smart-resizes" the 144 DPI
PNG down to that budget, preserving aspect ratio and rounding to 28px patch
multiples. The full-resolution PNG is untouched on disk.

`--visual-resolution` picks a preset that overrides the cap for every model in the
run (`config.py::VISUAL_RESOLUTION_PRESETS`):

| Preset | Tokens/page | Pixels/page |
|--------|-------------|-------------|
| full | 1280 | 1,003,520 |
| high | 768  | 602,112   |
| med  | 512  | 401,408   |
| low  | 320  | 250,880   |
| min  | 224  | 175,616   |

If the flag isn't set, the cap is size-aware (`MAX_PIXELS_BY_SIZE`): the 8B gets
~768 tokens/page, the 32B ~520, smaller models the ~1280 default. Bigger models
keep more weights resident, so they get a tighter vision budget to stay inside the
V100. Lowering resolution is also the main lever for the O(seq^2) attention that
otherwise OOMs a many-page cell (see section 7).

## 6. How the layers are fed to the MLLM

This is the part people usually ask about, so here it is exactly.

`pipeline/representation.py` builds the payload parts in this order:

1. one `[text]` block = **all pages' text concatenated** (joined with blank lines),
2. one `[layout]` block = **all pages' layout JSON concatenated**,
3. then the **page images, one per page, in page order**.

So it is **all text, then all layout, then all images** - *not* interleaved per
page. Text and layout are each a single block spanning every page; the images
follow as a contiguous run at the end, ordered by page.

`models/payload.py` then renders this for the backend. For a local model,
`to_local_prompt()` joins the parts into one string and replaces each image with
an `<image>` placeholder (kept in order). `models/local_vlm.py` wraps that in the
frozen prompt template:

```
You are answering a question about a document. ...
Question:
<the question>

Document evidence:
[text] <all pages' text>
[layout] <all pages' layout JSON>
<image><image>...   (one per page, in page order)

Answer:
```

Two things to note: the **question comes before the evidence**, and the **images
come after the text and layout**, as a contiguous block. `messages_from_rendered_prompt`
splits on the placeholders and rebuilds an interleaved text/image chat message, so
each `<image>` binds to the right page's pixels through the processor. Within a
page, vision tokens follow Qwen's grid raster order (the `t*h*w` image grid).

A hosted API backend would get the identical ordering through
`to_chat_messages()`, which emits the same parts as text and base64 `image_url`
blocks. InternVL (`models/internvl.py`, the second model family) consumes the same
`ModelInput` but binds images through its own processor.

## 7. Sequence budget and the V100 constraints

Kaya's V100s are Volta (sm_70): no FlashAttention-2. The Qwen backend does request
PyTorch's memory-efficient (cutlass) SDPA kernel, but probe 1004834 found it does
not reliably engage for these VLMs, so attention effectively runs the O(seq^2) math
kernel that materializes the full `[heads, seq, seq]` score matrix (InternVL's
`chat()` path never requests the efficient kernel at all). A long multi-page
sequence then tries to allocate tens of GiB and OOMs even after the weights are
quantized (the score matrix is an activation, so 4-bit weights don't help). So the
thing that actually keeps a cell inside 16 GB is **bounding the input length**.

**Input-token cap.** Each reasoner has a `max_input_tokens` cap on the combined
text + vision sequence (`config.py`, `MAX_INPUT_TOKENS_BY_SIZE`). It is size-aware,
because a bigger model holds more resident weight and leaves less room for the
score matrix:

| reasoner | cap (tokens) |
| --- | --- |
| 2B / 4B | 8192 |
| 8B (Qwen3-VL **and** InternVL) | 4096 |
| 32B | 3072 |

At 4096 tokens the 8B score matrix is ~2 GiB (`32 heads x 4096^2 x 4 B`); a raw
~30k-token `TL` layout dump would be ~100 GiB and OOM any single GPU. Sharding
across 2x V100 does not help, because attention runs per-GPU (the weights split,
the score matrix does not).

**Truncation.** When a cell's context exceeds the cap, both backends'
`_truncate_context` (in `models/local_vlm.py` and `models/internvl.py`) trims the
*free text* to the budget left after reserving for the images (`n_images x
per_image_tokens`) and a small template reserve. It keeps every image placeholder
and puts the images first, then the trimmed text, so **images are never dropped**.
Because the payload is serialized text, then layout, then images, the tail that
gets cut is the **layout** channel first (its bbox JSON is the token-heavy part),
and text only if it still overflows.

This is not free, and it is worth knowing before reading the frontier. Measured on
the full-run cache with the 8B tokenizer, the share of oracle cells that hit
truncation is roughly **T ~1%, TL ~29%, TLV ~34%** (`TLV` truncates most because
its page images eat into the text budget). The dropped content is almost all
layout, so the "does layout help" (`T` vs `TL`) contrast is the most affected. On
A100-class hardware with FlashAttention the cap could be raised or removed; on the
V100s it is load-bearing. Treat a heavily-truncated `TL`/`TLV` cell as a partial
view of that rung.

**Many-page guards.** Questions whose gold evidence spans more than 10 pages are
dropped up front (`experiments/corpus.py::MAX_EVIDENCE_PAGES`), and any cell that
still OOMs is logged and skipped rather than aborting the run
(`--continue-on-error`).

## 8. Models and decoding

All reasoners sit behind one `Reasoner` ABC and consume the backend-neutral
`ModelInput`, so the pipeline never knows which model it's talking to:

- **Qwen3-VL** 2B / 4B / 8B / 32B local (`models/local_vlm.py`) - the primary
  family. 2B is the smoke model, 8B the main full-run model.
- **InternVL3-8B** local (`models/internvl.py`) - the second family, used to check
  whether the sufficiency findings replicate across architectures.
- **API VLMs** - not implemented, but the seam is real: a hosted
  OpenAI/Gemini/Anthropic reasoner would be a new `get_reasoner` registry entry that
  consumes `ModelInput.to_chat_messages()` behind the same ABC, so adding one doesn't
  touch the pipeline.

Weights load in bf16 by default. For single-GPU iteration and the quantization
appendix, 4-bit NF4 (double-quant) and 8-bit are available via bitsandbytes so the
8B fits one V100 (~7 GB at 4-bit vs ~16 GB bf16). Multi-GPU loads use
`device_map="auto"` with ~5 GiB/GPU reserved for activations, so a long-sequence
cell doesn't tip one GPU into OOM. Decoding is greedy (`do_sample=False`) with a
per-mode `max_new_tokens`.

## 9. Judging and token accounting

The judge is a separate model *family* from the reasoner (Gemini 2.5 Flash by
default, GPT-4o-mini as the paid alternative) so it's an independent scorer. It's
text-only: it sees the question, the gold answer, the unanswerable flag, and the
model's answer, and returns a `correct` / `incorrect` / `abstained` verdict on
semantic equivalence. Unanswerable questions count as correct only when the model
abstains.

Each cell records `input_text_tokens` (from the tokenizer), `input_visual_tokens`
(from Qwen's `image_grid_thw`), `output_tokens`, and latency. Those feed the
cost/latency columns in the tables, which is how a representation's *price* is
compared against its *accuracy* to find the frontier (the cheapest rung that's
still statistically sufficient).

# The generation tasks

The GPU work is organized by **generation task**, not by paper table. Each task
is one file in `experiments/` (`G1_sufficiency` .. `G6_classifier`) subclassing
`GenerationTask` with up to four hooks: `model_specs` (which reasoners to run),
`resolve_questions` (which corpus), `generation_cells` (the cells to answer), and
`run_side` (extra non-reasoner GPU work). `experiments/driver.py` runs them.

## Inference ordering inside a task

The whole point of the ordering is that the parser/OCR/retriever model stacks and
the reasoner never sit on the same 16GB V100 at once. `driver.py::generate` does
this for one task:

1. **Resolve the corpus** (`resolve_questions`) and open the two per-task caches
   in `results/cache/<run_tag>/<mode>/<task>/`: `predictions.jsonl`
   (`PredictionCache`) and `generate_results.jsonl` (`ResultCache`). Build the two
   retrievers *lazily* (`real_retrievers`): the objects exist but no BGE/ColQwen
   weights load until something calls `retrieve`.
2. For each reasoner spec (every current task has 0 or 1):
   - **Build the ordered cell list** (`generation_cells`). A cell is
     `(question, conditioner, rung)`.
   - **Parse pre-pass.** A throwaway orchestrator built with a *spec-only*
     reasoner (`_SpecOnlyReasoner`, no weights) walks every cell and runs only the
     A->B path: `conditioner.condition` -> `render_pages` ->
     `representation.build`. That is what forces the expensive preprocessing to
     happen now, with no reasoner on the GPU: page PNGs get rendered, Marker/Surya
     produce text + layout, PaddleOCR produces scanned-page text, and (for G5) the
     retrievers rank pages. Everything lands in the disk caches below. Cells whose
     prediction is already cached are skipped (nothing left to parse).
   - **Tear down the parser/OCR/retriever state.** Unload the retrievers
     (`retrievers.text.unload()` / `.vision.unload()` drop the BGE/ColQwen weights
     but keep the in-memory rankings), `reset_ocr_engine()` drops the shared
     PaddleOCR engine, then `free_gpu()`. After this the GPU is clear.
   - **Load the reasoner** (`reasoner_for`) and answer every cell **in list
     order**. Weights load lazily on the first `answer()`. On a warm cache the
     representation step is all disk reads, so Marker/Surya/PaddleOCR never reload
     and never share VRAM with the reasoner. Each answer is appended to
     `predictions.jsonl` immediately (see "Fields recorded per instance"), so the
     presence of that file is the signal that inference has actually started. A
     cell that raises is logged and skipped when `--continue-on-error` is set
     (with a `free_gpu()` to recover activation memory), instead of aborting.
   - **Free the reasoner** (`reasoner.free()` + `free_gpu()`) before the next spec.
3. **Run side work** (`run_side`), then `free_gpu()` again.

Note the order flipped in mid-2026: the reasoner used to be constructed at the top
of the spec loop, before the pre-pass. It is now constructed after, so the "reasoner
not loaded during the pre-pass" invariant holds in code, not just in the comment.

**Cell order.** For the ladder tasks (G1, G2, G3) it is **question-major,
rung-minor**: the model answers question 1 at `T`, then `TL`, then `TLV`, then `V`,
then moves to question 2, and so on. It is *not* all questions at `T` first. The
reasoner loads once and walks the whole list. For G5 the cells are question-major,
k-minor: for every k in `config.k_values`, each question contributes a matched
(vision-retrieval) then a cross (text-retrieval) cell, all at `TLV`; with the
default `(1, 3, 5, 7, 9)` that is 10 consecutive cells per question.

## What each rung preprocesses, and what gets cached

A cell's rung decides which channels `representation.build(pages)` composes, and
each channel is a different preprocessing path (all of this runs in the pre-pass):

- **`T` (text).** `text_channel(pages)` routes per document using
  `annotations/doc_labels.csv`: digital-born docs go through **Marker** (markdown
  text), scanned docs go through **PaddleOCR PP-OCRv5**. PyMuPDF embedded text is
  the cheap fallback. Emitted as one `[text]` string block.
- **`TL` (text + layout).** `T`, plus `layout_channel` = Marker `bbox` JSON
  (per-page block type + bounding boxes, serialized). Strings only, `[layout]`
  block.
- **`TLV` (text + layout + visual).** `TL`, plus `visual_channel` = one full-page
  `ImagePart` per rendered page.
- **`V` (visual).** Page images only, no text/layout.

The modality boundary is structural and re-checked in `Payload.__post_init__`:
only `TLV` and `V` ever attach images.

The disk caches these paths fill are keyed by content so re-runs are idempotent.
`--run-tag` is folded into `config.paths.cache_dir` (so it becomes
`results/cache/<run_tag>`), and the parse artifacts live directly under it, shared
across every mode and task in that run:

- `results/cache/<run_tag>/renders/<stem>__dpi<N>/page_XXXX.png`: rendered page
  images at the run's dpi (`config.dpi`, 144 in full). Text spans are re-extracted
  each call (cheap) rather than cached.
- `results/cache/<run_tag>/marker/<stem>__pXXXX__text.md` and `__bbox.json`: Marker
  text and layout, one file per page. The PyMuPDF fallback is deliberately *not*
  cached, so a later real Marker pass can replace it.
- `results/cache/<run_tag>/ocr/<stem>__dpiN__pXXXX.txt`: PaddleOCR text per scanned
  page. Empty/fallback results are not cached, so a blank page is retried rather
  than frozen as blank.
- Retrieval rankings are memoized **in memory** per `(question, page_count, k)` by
  `MemoizedRetriever`, so a k that repeats across cells is computed once; the memo
  survives the retriever unload into the inference phase.

The per-cell reasoner output is task-scoped, one level deeper under
`results/cache/<run_tag>/<mode>/<task>/`: `predictions.jsonl` and the throwaway
stub-scored `generate_results.jsonl` (see "Fields recorded per instance").

## The tasks, one by one

Every task caches per-cell records and (optionally) a side artifact as described
above; the per-task notes below only call out what is specific to each.

### G1_sufficiency

- **Purpose.** The core measurement. Feed the *gold* evidence pages (perfect page
  selection) and ask which rung of the ladder is *sufficient* per document bin,
  and what each rung costs. This is the reasoning ceiling: it isolates the
  representation question from the retrieval question.
- **Corpus & reasoner.** The shared per-bin sample (~100 questions per Option-A
  bin, drawn by whole document at `sample_seed=0`); the config's primary reasoner
  (Qwen3-VL-8B in full, 2B in smoke).
- **Cells & run.** `oracle` pages x `{T, TL, TLV, V}`, question-major (each
  question answered at all four rungs before the next). One reasoner spec, so the
  model loads once and walks the list. No side work.
- **Data.** Standard `predictions.jsonl` -> `results.jsonl` per cell.
- **Feeds.** Tables 1 (headline frontier), 2 (by question type), 5 (composition /
  mediation), and 7 (routing accuracy).

### G2_family

- **Purpose.** Cross-family replication: does the sufficiency frontier hold on a
  *different model architecture*? Runs the exact same oracle ladder with
  InternVL3-8B so Table 3 can check whether each bin's frontier matches Qwen's
  qualitatively.
- **Corpus & reasoner.** Same shared per-bin sample as G1; reasoner is
  `internvl3-8b-local`. Full only: in smoke `model_specs` returns nothing (one
  family), and Table 3 reuses G1.
- **Cells & run.** Identical oracle ladder to G1. InternVL binds images through
  its own processor but consumes the same `ModelInput`. (This is the task that
  needs `timm` installed, since InternVL's vision tower loads through it.)
- **Data.** Same fields as G1, tagged `model_spec=internvl3-8b-local`.
- **Feeds.** Table 3 (family replication), alongside G1's primary-family rows.

### G3_dataset

- **Purpose.** Dataset replication: does the per-bin recipe hold on a *disjoint set
  of documents*? Guards against the frontier being an artifact of the specific
  documents G1 happened to sample.
- **Corpus & reasoner.** `sample_table4_replication` draws ~100 questions per bin
  for text_heavy and in_between from documents **not** in G1's subset (the seed is
  matched to G1 so "G1's subset" is exactly the documents G1 ran). visual_heavy is
  too thin (only ~15 docs) to hold out, so it is excluded here; SlideVQA is the
  planned visual-heavy replication and is out of scope. Reasoner is the primary
  (full only; smoke reuses G1).
- **Cells & run.** The same oracle ladder, on the held-out corpus.
- **Data.** Standard per-cell records; the held-out document set is the only
  difference from G1.
- **Feeds.** Table 4 (dataset replication).

### G4_scale (planned, not implemented)

- **Purpose.** Scale sanity: is the sufficiency frontier stable across model
  *size*? Run the oracle ladder on the Qwen3-VL size series (2B / 4B / 8B / 32B)
  and check whether each size lands on the same per-bin frontier.
- **How it would be built.** A new `experiments/G4_scale.py` subclassing
  `GenerationTask` whose `model_specs` returns the size ladder
  (`qwen3vl-2b-local` .. `qwen3vl-32b-local`) and whose `generation_cells` reuses
  `oracle_ladder_cells`. No new engine work is needed: `driver.py` already loops
  `for spec in model_specs`, so it would run the full oracle ladder once per size
  (spec-major: all cells at 2B, then 4B, and so on), caching predictions tagged by
  `model_spec`. Registering it in `experiments/registry.py` and pointing Table 8's
  source at it is the rest. Table 8's builder already derives `scale_family` and
  size from `model_spec` and reports each size's frontier plus whether it matches
  the primary.
- **Why it is deferred.** The 32B model does not fit Kaya's V100s; it needs the
  supervisor's A100. Table 8 is currently gated off (see the build gate) so it is
  never produced from placeholder data.
- **Feeds.** Table 8 (scale sanity), once implemented.

### G5_retrieval

- **Purpose.** Does retrieval have to use the *same modality* as reasoning? Under
  real (imperfect) retrieval at a vision-bearing rung, compare *matched*
  (vision-retrieval feeding vision reasoning) against *cross* (text-retrieval
  feeding vision reasoning). Also records how good each retriever is.
- **Corpus & reasoner.** Shared corpus; primary reasoner; all cells run at `TLV`.
- **Cells & run.** A full **top-k sweep**: for every k in `config.k_values`
  (default `(1, 3, 5, 7, 9)`), per question two cells `retrieved_vision_k{k}` and
  `retrieved_text_k{k}` (question-major, k-minor). The `k` lives in the conditioner
  name so each k gets its own prediction-cache row. Page selection runs in the
  pre-pass: **text** = BM25 over page text plus dense BGE similarity
  (`bge-small-en-v1.5`, reads page text only, no image render); **vision** =
  ColQwen2.5 late-interaction over rendered page images at the run's dpi. Both are
  memoized by `(question, page_count, k)`, so a ranking is computed once even
  though several cells reuse it, and the memo outlives the retriever unload. In the
  generate phase the driver passes the real retrievers; in the judge phase it
  passes guards that raise if called, so every retrieved cell must be a
  prediction-cache hit.
- **Data.** Standard per-cell records (with `condition` = `retrieved_{modality}_k{k}`),
  plus the side artifact `retrieval.jsonl`: one `RetrievalEvalRow` per (question,
  retriever modality, **k**) with `retriever`, `modality`, `k`, `retrieved_pages`,
  `gold_pages`, and page `precision` / `recall` / `f1`.
- **Feeds.** Table 6 (matched vs cross), which now reports **each k separately**
  (a `k` column), so you can read matched-vs-cross as a function of retrieval depth.

### G6_classifier

- **Purpose.** Price the *predicted-routing* policy. Routing chooses a
  representation recipe per document, but at inference you do not know the
  document's type, so a classifier predicts it. This task measures the
  classifier's bin accuracy and latency so Table 7 can fold that cost into
  predicted routing (routing *accuracy* itself reuses G1's ladder rows).
- **Corpus & reasoner.** Shared corpus, but there are **no reasoner cells**
  (`model_specs` is empty). The only GPU work is the classifier in `run_side`.
- **Cells & run.** `run_side` runs `QwenDocTypeClassifier` once per *distinct
  document* (not per question, deduped by `doc_id`): it renders the first two pages
  at the run's dpi as a `TLV` input and asks Qwen3-VL-2B
  (`CLASSIFIER_REASONER_SPEC`) to pick one of the native MMLongBench document
  types, then maps that to an Option-A bin. Because this is a plain `run_side`
  task with no reasoner cells, it never enters the pre-pass/inference loop above;
  the classifier is the only thing that loads a model.
- **Data.** No `predictions.jsonl`. The side artifact `classifier.jsonl` holds one
  record per document: `doc_id`, gold and predicted `doc_type`, gold and predicted
  `bin`, `correct_bin`, `confidence`, `latency_s`, and the classifier name.
- **Feeds.** Table 7 (routing), as the predicted-routing cost/accuracy input.

## Fields recorded per instance

A reasoner cell produces two records:

**`predictions.jsonl`** (`CachedPrediction`) is the durable GPU output, written in
the generate phase and keyed *without* the judge. Per cell:

- `prediction_key` (hash of question_id + doc_id + condition + representation +
  model_spec + dpi),
- `question_id`, `doc_id`,
- `condition` (conditioner name, e.g. `oracle`, `retrieved_vision_k1`),
- `representation` (`T`/`TL`/`TLV`/`V`),
- `model_spec`,
- `provenance` (page-selection provenance, e.g. `oracle`, `retrieved`),
- `page_indices` (the pages actually fed), `note` (e.g. `k=1`),
- `text` (the model's answer),
- `input_text_tokens`, `input_visual_tokens`, `output_tokens`, `latency_s`.

(The generate phase also writes `generate_results.jsonl`, a throwaway `ResultRow`
scored by a stub judge, only to drive the run loop. The real scoring happens in
the judge phase.)

**`results.jsonl`** (`ResultRow`) is written in the judge phase by re-reading the
cached prediction and scoring it with the real judge (no GPU, no PDFs). It carries
everything above plus `doc_type`, `hop`, `is_unanswerable`, `evidence_sources`,
`judge_spec`, `score`, `correct`, `abstained`, and `metadata` (note,
source_dataset), under a `cache_key` that *includes* the judge spec.

That split of `prediction_key` (no judge) versus `cache_key` (with judge) is
deliberate: one GPU prediction can be re-scored by any number of judges without
re-running the model.

Side artifacts are one record per unit, not per cell:

- **`retrieval.jsonl`** (G5): one record per (question, retriever modality, k)
  (every k in the sweep is logged), with the retriever name, modality, k, and
  page-retrieval precision / recall / F1.
- **`classifier.jsonl`** (G6): one record per distinct document with `doc_id`,
  gold and predicted `doc_type`, gold and predicted `bin`, `correct_bin`,
  `confidence`, `latency_s`, and the classifier name.

# Repository map

- `config.py` - paths, resolution presets, per-size caps.
- `schema.py` - frozen data contracts (`Question`, `Page`, `Payload`, `Prediction`, `Score`).
- `data/` - dataset loaders, Option-A binning, PDF rendering.
- `tools/` - the text / layout / visual channel implementations.
- `pipeline/` - conditioners, representations, reasoner/judge ABCs, orchestrator.
- `models/` - backend registry, Qwen3-VL / InternVL reasoners, `ModelInput`.
- `covariates/` - retrievers and the doc-type classifier.
- `metrics/` - accuracy, cost, frontier, retrieval, abstention.
- `experiments/` - generation only: YAML spec loading, the generate+judge engine
  (`driver.py`), the `G*_*.py` tasks + registry, corpus/smoke resolvers, and the
  shared side-artifact writers (`side_artifacts.py`).
- `reporting/` - table building: the `tables/` package (one `T*_*.py` module per
  paper table, mirroring the `G*` task naming, with `__init__` as the aggregation
  entry point) and the table -> source-task routing that writes the CSVs + markdown
  (`build.py`).
- `gates/` - Section-2 go/no-go gates (`core.py`) plus the shared cached-cell viewer
  (`viewer.py`); run as `python -m gates`.
- `cli/` - the three experiment roles only: `generate` (GPU, YAML-first), `judge`, `build`.
- `specs/` - YAML generation templates, including `full_generation.yaml`.
- `scripts/` - standalone utilities: `run_probe` (feasibility probes),
  `inspect_results` (view a cached inference cell), `annotate_docs` (per-document
  manual labels), `split_docs_by_type`, staging.
- `kaya/` - cluster sync/submit runner and setup scripts.
- `docs/` - user guide, project spec, agent/implementation notes.
