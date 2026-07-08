# Pivot v4: deployment-framed rework of the ladder, binning, RQs, tasks, and telemetry

Status: decision record (a *pivot*, not an implementation spec). It states **what
changes and why**, and defers **how** (cache-key edits, interface changes, task
code) to the agent doing the documentation/codebase update. Where this file and
the existing docs disagree, **this file is the newer intent** and the older docs
(`PROJECT_SPEC.md`, `README.md`, `USER_GUIDE.md`, `AGENT_GUIDE.md`) should be
reconciled to it. Where this file is silent on a mechanism, the existing docs
still hold.

This supersedes the v3 spec on the five areas below. It does **not** change the
one-line thesis: *the representation an MP-VRDU system needs is a function of
document type.* It sharpens everything around that thesis toward a
**deployment / practical-use** framing.

---

## 0. The two governing principles

Every decision below serves one of two mandates. They are in tension with the
old "one table per experiment" habit, and the tension is resolved in favour of
these two:

1. **Few tasks, full coverage.** Organise generation tasks so the *smallest*
   number of runs collects *all* the data every planned table needs. Many things
   that read like separate experiments (parser choice, resolution, quantization,
   model size, model family, dataset replication) are the *same cell grid with
   one field changed*, so they are **YAML runs of one task**, not new tasks.
2. **Collect all cheap data, every run.** Every generation task records the full
   telemetry schema (§6), not just its own experiment's target metric. Uniform
   telemetry across every run is what lets us decide *after* the data exists
   which results go in the main paper, the appendix, or are cut. Non-uniform
   telemetry is nearly as useless as none, so the schema is fixed here, once.

A corollary the whole plan leans on: **we run a lot, collect a lot, and select
late.** Replication and aggregation tables most likely land in the appendix; the
main paper carries the three core experiments.

---

## 1. The representation ladder is redefined (and reframed as cost-ordered)

### 1.1 What changes

The `T` and `TL` rungs are redefined to fix the truncation / input-sequence
problem and to make each rung a clean deployment choice.

| Rung | Old (v3) | New (v4) |
|---|---|---|
| `T` | Marker text (Surya OCR under the hood) | **Cheap embedded-text extraction** (PyMuPDF). Digital-born only; by design returns nothing useful on scanned docs. |
| `TL` | text + serialized bbox-JSON layout | **Parser-derived layout-rich text** (markdown from a PDF-parser). The parser output **replaces** the cheap text layer; **bounding-box input is abandoned entirely.** |
| `TLV` | text + layout + page image | **Parser text + page image.** (Parser text, not embedded text.) |
| `V` | page image only | Unchanged: page image only. |

Bounding-box JSON is dropped everywhere. It was the token-heavy channel that
drove truncation, and it did not help the reasoner (no fine-tuning to exploit
coordinates). Removing it is the primary fix for the TL/TLV sequence-length and
truncation problems.

### 1.2 The ladder is now cost-ordered, NOT additive/cumulative

This must be stated honestly in the paper. Because `TL`'s parser text *replaces*
`T`'s embedded text (rather than adding to it), the ladder is **no longer
cumulative**: `T ⊄ TL`. It is instead **ordered by computational cost**, each
rung strictly more expensive to produce and feed than the last:

```
T  (cheap extract)  <  TL (parser)  <  TLV (parser + image)  <  V is the
image-only reference point (parser-independent)
```

The headline framing (Table 1) becomes: *four representations of increasing
compute cost; which is worth its cost, per document type?* The "L" label is now
vestigial — there is no separate layout channel; "TL" mechanically means
"parser-derived text." We keep the T/TL/TLV/V labels for table-shape and
cache-key continuity, and record here that the names are historical so nobody
"restores" bounding boxes later. This is the same "keep the names, change the
mechanism" discipline already used for OCR in the current README.

### 1.3 PDF-parsers under comparison (the `TL`/`TLV` text source)

"PDF-parser" is the umbrella term (OCR is one mechanism a parser may use; PyMuPDF
uses none). The parser only varies at `TL`/`TLV` (where parser text is used); `T`
and `V` are held fixed across parser runs.

| Tier / role | Tool | Notes |
|---|---|---|
| Floor (the `T` extractor) | **PyMuPDF** embedded text | Free, instant, digital-only. The cheap baseline. |
| Parser A | **PaddleOCR-VL** (0.9B) | End-to-end VLM parser; NaViT-style native-res encoder + ERNIE-4.5-0.3B. Vendor-reported OmniDocBench SOTA. |
| Parser B | **MinerU 2.5** (~1.2B) | Layout-detect + VLM pipeline; different paradigm to A. |
| Parser C | **Unlimited OCR** | Baidu open-weight; 40+ pages in one pass; ~93.9 OmniDocBench v1.6. |

**Why this is a contribution, not just a tool-swap.** The headline OmniDocBench
scores for these parsers are vendor self-reported and not independently
reproduced. An independent comparison scored by **downstream QA accuracy on
MMLongBench-Doc** (not edit-distance) is genuinely novel.

**Environment risk (flagged, not solved here).** These three parsers are heavy,
separately-pinned VLM stacks. They very likely will **not** co-exist with the
pinned reasoner env (transformers 4.57.x / torch 2.7 exact-pinned by vLLM). Treat
each parser as an **isolated env** whose output is **pre-warmed to the parser
cache in the pre-pass**, so the parser VLM never co-resides with the reasoner on
the GPU (same discipline already used for Marker/Surya/PaddleOCR).

**Marker.** Dropped from the comparison set. Existing Marker caches may be kept as
an appendix/continuity point; not required.

---

## 2. Binning moves from `doc_type` to manual annotation

### 2.1 Why

MMLongBench's native `doc_type` encodes *domain*, not *dominant modality*. Domain
does not map to our bins: each domain mixes scanned and digital docs, and on
visual inspection many docs the old mapping called "text-heavy" are actually
in-between or visual-heavy. The bin axis is the entire thesis, so it must be
labelled directly.

### 2.2 The annotation

A fast (<1 hr) manual pass over all **135 documents**, producing three labels:

- **`bin_label`** — document-level dominant modality. Provisional names adopted:
  **text-dominant / mixed-modality / visual-dominant**.
- **`scan_label`** — `digital` / `scanned`.
- **`dominant_visual`** — {tables, charts, figures, photos, none}, multi-valued.
  **Exploratory only:** collected opportunistically while annotating; analysed if
  it turns up something, otherwise discarded. No committed pipeline depends on it.

### 2.3 The bins are document-level dominant modality

Bins describe *which modality dominates the document's information content*, not
scan status and not page count:

- A scanned, mostly-handwriting document is **text-dominant** (its information is
  linguistic), even though it is scanned and image-based on disk.
- A text-sparse magazine cover is **visual-dominant** (information lives in the
  imagery/design).
- A text-dense academic paper with a few figures/tables is **mixed-modality**
  (requires human judgement).

The defensible axis is **modality dominance**; density is only how we
operationalise "dominant." "Text-dominant" defends the handwriting case better
than "text-dense" would (a sparse handwritten note is text-dominant but not
text-dense).

### 2.4 Reliability (recommended, not yet scheduled)

The `mixed-modality` bin needs human judgement and the whole thesis rests on the
bin axis. Recommended cheap insurance: a second annotator independently labels a
20–30 doc subset; report inter-annotator agreement (Cohen's κ, same bar as the
judge gate). On the record so it is not improvised later; not blocking.

---

## 3. Research questions, reworked

RQ1 is kept and strengthened; RQ2 and RQ3 are substantially expanded. **The RQ
framing is not set in stone** — e.g. the hallucination study could sit under RQ1.
What matters is that all data is collected, correctly and uniformly. The
answerable/unanswerable split (§3.3) governs which questions feed which RQ.

### RQ1 — Recipe by document type (reasoning / representation)

- Rework the headline (Table 1) into the **cost-ordered additive-modality
  comparison** of §1.2 on oracle pages: four rungs × three bins, frontier marked,
  answerable-only.
- **New: parser comparison** (§1.3) — swap the `TL`/`TLV` parser, hold everything
  else. One table.
- **New: image-resolution sweep** — vary the per-page vision-token budget at
  `TLV`/`V`. One table. (Supervisor hardware — see §5.)
- Replications retained: model **family** (InternVL) and **dataset** (held-out
  MMLongBench subset).
- Total RQ1 tables: **3 core** (headline, parser, resolution) + replications.

### RQ2 — Retrieval (mechanism + retrieval-side modality)

- **Matched vs cross, all three bins.** Does retrieval modality have to match
  reasoning modality? Run at `TLV` (with `V` available via YAML for the
  vision-retrieval→vision-reasoning contrast). Preliminary results already show
  all-modality `TLV` is the strongest reasoning rung, so RQ2 inference uses
  `TLV`/`V` only — not the whole ladder — which keeps the cell count tractable and
  matches "the representation you'd actually deploy."
- **Top-k sweep** with retrieval methods framed as **cost rungs** (§4). Tests
  whether larger k helps or hurts, at k ∈ {1,3,5,7,10} for single-method
  retrieval. Runs on supervisor hardware (no cap pressure → a high-k accuracy
  drop is a real distractor effect, not a truncation artifact).
- **Retrieval-accuracy benchmark** (page precision/recall/F1 vs gold, **per
  bin**), covering all retrieval methods incl. ones never fed to the reasoner.
  This is a **by-product of the same retrieval pass** (§5.2), not a separate GPU
  task.

### RQ3 — Deployment (routing + hallucination + prompting, plus cost sweeps)

- **Routing** retained: four policies (oracle routing, predicted routing,
  uniform-cheapest, uniform-strongest); predicted-routing cost includes the
  classifier's own latency. Routing accuracy reuses RQ1's ladder rows.
- **New: hallucination × prompting** (studied together). Uses the ~250
  unanswerable MMLongBench questions **pulled out of RQ1/RQ2**. Because
  unanswerable questions have **zero gold pages by construction**, there is *no
  oracle arm* — the only coherent page-selection is **similarity-retrieved 2–3
  pages** (lexical/semantic/visual). Feed those under **three prompt conditions**:
  no prompt / generic prompt / hallucination-targeted prompt. Correct behaviour =
  abstention. This inherently ties the hallucination study to RQ2's retrieval
  machinery (it reuses the retrieval cache at a fixed small k).
- **New: cost sweeps** — quantization (4/8/16-bit) and model size
  (2B/4B/8B/32B). Framed as **cost-frontier** studies (accuracy-per-VRAM,
  accuracy-per-latency), not accuracy studies — the telemetry *is* the point.
  Placement (main vs appendix) decided post-hoc by significance. Supervisor
  hardware for the size/quant/resolution sweeps.

### 3.3 The answerable / unanswerable split

The ~250 unanswerable questions (of 1091) are **removed from RQ1 and RQ2** so
those accuracies are cleanly "accuracy on answerable questions." They are used
**only** in the RQ3 hallucination study. Consequence to state in the paper: RQ1/RQ2
say nothing about hallucination; the dedicated study carries the entire abstention
story, and it is necessarily a *retrieval-fed* study (no oracle arm exists for
zero-gold-page questions). Confirm the judge's abstention scoring and the κ ≥ 0.75
gate still hold on the answerable-only set.

---

## 4. Retrieval methods as cost rungs

Both retrieval axes are laid out cheapest → most expensive, mirroring the
representation ladder so RQ2 parallels RQ1's cost story.

| Axis | Cheap | Mid | Expensive |
|---|---|---|---|
| **Text** | BM25 (lexical) | BGE-M3 (dense; MIT, self-hosted workhorse) | Qwen3-Embedding-4B (SOTA-class) |
| **Vision** | ColModernVBERT (~250M; within ~0.6 pts of ColPali) | ColQwen2.5-v0.2 (current default; reuses caches) | ColQwen3-4B (ViDoRe SOTA-class) |

Notes:
- Qwen3-Embedding **4B** is the "expensive text" rung (clearly pricier than
  BGE-M3, still self-hostable). The 8B is reserved for the accuracy-only benchmark
  if wanted, not the deployable rung.
- ColQwen2.5-v0.2 stays as the mid vision rung specifically to reuse existing
  retrieval caches.

### 4.1 Joint retrieval (free, post-hoc union)

Every retrieval method emits a **retrieved page set** per (question, k). Joint
retrieval is the **deduplicated union** of two already-computed page sets — no new
retrieval, no score fusion (union, *not* RRF, which would need scores and would
not be free).

- **Pairs:** representative **matched-tier** unions only — cheap+cheap
  (BM25 ∪ ColModernVBERT), mid+mid (BGE-M3 ∪ ColQwen2.5), expensive+expensive
  (Qwen3-Embedding ∪ ColQwen3).
- **k:** joint uses **k ∈ {1,3,5} per method**, so the union is always **< 10
  pages** (≤ 5+5, minus overlap). Joint is its **own condition family**, not a
  point on the single-method k-axis (that axis goes to k=10).
- Purpose: does combining sparse + dense (text + vision) beat either alone, for
  free?

---

## 5. Hardware split and the evidence-page partition

### 5.1 Two machines, marked per run

We run on **Kaya** (2× V100 16 GB, 32 GB RAM, sm_70, **no FlashAttention-2**).
The supervisor has **A100 / H100** access. Some experiments **cannot** run on
Kaya and must be marked as supervisor-only:

- Model-size sweep (esp. 32B), quantization sweep, image-resolution sweep — all
  need headroom Kaya lacks.
- The RQ2 k-sweep (up to k=10) runs on the supervisor so token caps never bite.

**Every YAML run carries its target machine** (`machine: kaya | supervisor`), and
the machine is recorded in the run manifest (and denormalised per-cell for
audit). Implementation must therefore express the machine in the YAML itself.

### 5.2 The evidence-page partition (exploiting the data)

Per `dataset_stats.md`, the vast majority of MMLongBench questions have **0–2
evidence pages**; only ~50 questions across the corpus have >2. We exploit this:

- **Kaya** runs questions with **≤ 2 evidence pages**. Because these are short,
  the input-token **cap can be raised**, reducing text truncation.
- **Supervisor** runs the **> 2 evidence-page** remainder (~50 questions per
  experiment) at the same agreed cap (its GPUs fit that cap where Kaya's do not).

**No confound.** The token cap is a **fixed experimental constant**; it is the
same number on both machines. Kaya simply cannot *fit* that cap on some questions
without OOM, while the supervisor can. Machine is just the executor, not a
different condition. To keep the pooled full-corpus numbers auditable, each cell
records `machine` and `cap_used`, and each run's manifest records its
`evidence_page_filter` (`≤2` / `>2` / `all`). Tables that report a full-corpus
number pool the two slices at build time.

---

## 6. Telemetry schema (fixed here, collected by every task)

Telemetry has **two scopes**, recorded at different granularities. Do **not**
stamp constant environment fields on every cell.

### 6.1 Per-cell (one row per cell, in the prediction record)

**Identity & config** (mostly existing): `prediction_key`, `question_id`,
`doc_id`, `condition`, `representation`, `model_spec`, `page_indices`,
`provenance`, `note`. New: `bin_label`, `scan_label` (stamped so tables need no
join to the annotation CSV); `machine`, `cap_used` (per-run in nature,
denormalised per-cell for pooling audit).

**Tokens** (the truncation story, simplified — vision is guaranteed fed, only
text is trimmed):
- `total_text_tokens` — text tokens before any truncation.
- `total_visual_tokens` — vision tokens (always fed in full).
- `text_tokens_fed` — text tokens actually fed to the reasoner
  (`= total_text_tokens` when no truncation).
- `output_tokens`.
- Derived (free): `tokens_dropped = total_text_tokens − text_tokens_fed`;
  `truncation_occurred = tokens_dropped > 0`; truncation severity =
  `tokens_dropped / total_text_tokens`.

**Latency** (deployment's primary metric):
- `latency_s` — end-to-end wall clock, batch=1 (headline, existing).
- `prefill_latency_s`, `decode_latency_s` — **collected.** Prefill isolates the
  cost of *ingesting the representation* (what the ladder changes) from decode
  (roughly constant, output-length noise). Likely a headline column: representation
  cost ≈ prefill cost.
- `tokens_per_sec_decode` — derived, free once the split exists.

**Memory:**
- `peak_vram_bytes` — **collected**, per-cell via
  `max_memory_allocated()` reset each cell. Genuinely cell-varying (scales with
  sequence length), deployment-relevant.
- Per-cell host RAM — **dropped** (noisy, rarely actionable). A single per-run
  host-RAM peak is not tracked (RAM not a binding constraint).

**Skips / errors:**
- `oom_occurred`, `skipped_reason` — record *why* a cell is missing so the build
  distinguishes "OOM-skipped" from "never-run." Turns skips into data (e.g. "TLV
  OOMs at k=10 on X% of cells").

### 6.2 Per-run (once, in `experiment_manifest.json`)

`gpu_model`, `gpu_count`, `cuda_version`, `torch_version`, `driver_version`;
`quantization`, `visual_resolution_preset`, `max_input_tokens`, `max_pixels`,
`dpi`; `parser_tool`, `retriever_text`, `retriever_vision` (which are active this
run); `git_commit`, `run_tag`, `mode`, `sample_seed`, `per_bin_sample`;
`machine`; `evidence_page_filter` (`≤2` / `>2` / `all`).

### 6.3 Retrieval side-artifact (one row per question × method × k)

No reasoner. `question_id`, `bin_label`, `retriever`, `modality`
(text/vision/joint), `k`, `retrieved_pages`, `gold_pages`, `precision`, `recall`,
`f1`; plus **retrieval cost** so the cost-rung story is honest:
`retrieval_latency_s` (per-query) and `index_build_amortized_s` (recorded once
per method×corpus, noted as amortized). This artifact **is** the RQ2
retrieval-accuracy benchmark and covers every method, including those never fed to
the reasoner.

---

## 7. Generation tasks, reworked

Consolidated from six (G1–G6) to **four**, per the "few tasks" mandate. The big
reduction is that the parser, resolution, family, dataset, quantization, and
model-size "experiments" are **YAML runs over G1**, not separate tasks. Each run
is tagged with its machine (§5.1). Every task collects the full §6 telemetry.

### G1 — Oracle ladder (reasoning core + all its sweeps)

- **Cells:** oracle pages × {T, TL, TLV, V}, answerable-only, primary 8B reasoner.
- **Feeds:** Table 1 (cost-ordered headline) and the per-bin frontier.
- **Reused by these YAML runs (one field changed each; not new tasks):**
  - parser comparison — vary parser at `TL`/`TLV`.
  - resolution sweep — vary `--visual-resolution` at `TLV`/`V` (supervisor).
  - family replication — reasoner → InternVL3-8B.
  - dataset replication — corpus → held-out MMLongBench subset.
  - quantization sweep — reasoner spec → `-4bit`/`-8bit`/bf16 (supervisor).
  - model-size sweep — reasoner → 2B/4B/8B/32B (supervisor; 32B A100-only).
- Split by evidence pages into two runs (≤2 Kaya, >2 supervisor), **pooled at
  build time** using per-cell `machine`/`cap_used` and per-run
  `evidence_page_filter`.

### G2 — Retrieval (inference + accuracy benchmark, one pass, two scorers)

- **Retrieval pre-pass** computes, once per (question, method, k), the retrieved
  page set for all six methods + the three joint unions (§4). Cached page sets are
  the shared substrate.
- **Scorer A (accuracy benchmark, no GPU reasoner):** the §6.3 side-artifact —
  page precision/recall/F1 per bin, for **every** method incl. non-inference ones.
  (This is the old "G3" folded in — it is a side-artifact, not a task.)
- **Scorer B (inference):** reasoner at `TLV` (and `V` via YAML) on the chosen
  page sets — matched vs cross across all three bins, and the k-sweep
  (single-method k ∈ {1,3,5,7,10}; joint k ∈ {1,3,5} per method).
- **Machine:** supervisor (headroom → k-sweep truncation-free).
- **Feeds:** matched-vs-cross table (per k), k-depth study, retrieval-accuracy
  benchmark.

### G3 — Hallucination / prompting (unanswerable subset)

- **Cells:** unanswerable-only (~250 q) × similarity-retrieved 2–3 pages (reuses
  G2's retrieval cache at fixed small k) × `TLV` × {no prompt, generic,
  hallucination-targeted}, 8B. Prompt condition is the swept variable.
- Correct = abstention. No oracle arm (zero gold pages by construction).
- **Feeds:** hallucination table + prompting comparison (together).

### G4 — Routing / classifier (deployment)

- The doc-type classifier pricing (the old G6) + routing-policy assembly. Routing
  accuracy reuses G1's rows; G4 adds only the classifier's own cost. Unchanged in
  spirit.
- **Feeds:** routing table.

### 7.1 Task → result map

| Result | Task | Notes |
|---|---|---|
| Cost-ordered headline (Table 1) | G1 | base run |
| Parser comparison | G1 | parser-varied run |
| Resolution sweep | G1 | resolution-varied run (supervisor) |
| Family replication | G1 | InternVL run |
| Dataset replication | G1 | held-out corpus run |
| Quantization sweep | G1 | quant-varied run (supervisor) |
| Model-size sweep | G1 | size-varied run (supervisor) |
| Matched vs cross (3 bins) | G2 | inference scorer |
| k-depth sweep | G2 | inference scorer |
| Retrieval-accuracy benchmark (per bin) | G2 | side-artifact scorer |
| Hallucination + prompting | G3 | |
| Routing | G4 | + reuses G1 |

---

## 8. Open items carried forward (not blocking this pivot)

- Inter-annotator agreement on a 20–30 doc subset (§2.4) — recommended, unscheduled.
- Isolated envs for the three PDF-parsers (§1.3) — an environment task to size.
- Confirm judge abstention scoring + κ ≥ 0.75 gate on the answerable-only set (§3.3).
- Qwen3-Embedding 8B in the accuracy-only benchmark — optional, decide if wanted.
- Final "expensive text" rung is Qwen3-Embedding-4B; revisit if 4B proves
  impractical to self-host at index time.

---

## 9. What this pivot deliberately keeps from v3

- The one-line thesis and the deployable (doc type) vs analytical (question type)
  axes.
- Document-level bootstrap CIs (1000 resamples over docs), the 3-point sufficiency
  margin, and the judge-family-≠-reasoner-family rule with the κ gate.
- The frozen pipeline shape (conditioner → render → representation → reasoner →
  judge), the two-layer cache (prediction key without judge; result key with
  judge), and the YAML-generate / artifact-judge / build role split.
- T/TL/TLV/V rung **names** and table shapes (mechanism changes; names stay).