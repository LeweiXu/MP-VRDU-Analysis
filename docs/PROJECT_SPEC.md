# Project spec: a doc-type recipe for multi-page document QA (v4)

The authoritative statement of *what* the paper claims and *why*. It reflects the
**v4 pivot** (`docs/pivot_v4.md`), which reframes the whole study toward a
deployment / practical-use story. It is kept in sync with the implementation:

- `README.md` is the ground truth for *how* the pipeline is actually built today
  (the cell, the ladder, the tasks, the repo map). The v4 mechanism changes
  (PyMuPDF `T`, parser-derived `TL`, four generation tasks, the new telemetry
  schema) are the adopted plan but the code migration is still pending, so where
  README describes the current mechanism it wins on "what the code does now" and
  this file wins on "what the paper claims and where we are heading."
- `docs/USER_GUIDE.md` restates this thesis and adds the local runbook.
- `docs/AGENT_GUIDE.md` holds the fixed decisions, frozen interfaces, and the
  implementation reference, plus the v4-vs-current-code deltas.

This spec supersedes the v3 spec. It does **not** change the one-line thesis; it
sharpens everything around it toward deployment.

---

## 1. One-line thesis

**The representation an MP-VRDU system needs is a function of document type, not a
single property of the model.** We measure that function on a cost-ordered
representation ladder, turn it into a deployment recipe indexed by document type,
and explain the recipe through evidence composition.

Venue target: **EACL long paper** (8 pages). Everything that does not serve the
thesis moves to the honours thesis or is cut. Two governing principles run through
the whole plan: **few tasks, full coverage** (the smallest number of runs collects
all the data every planned table needs, so many "experiments" are YAML runs of one
task with a single field changed), and **collect all cheap data, every run** (every
run records the full telemetry schema, so we can select late which results land in
the main paper, the appendix, or the cutting-room floor).

## 2. Motivation

Sensitive documents (medical records, contracts, financial filings, compliance
reports) cannot be sent to a cloud API; they have to be processed on-premise with
a self-hosted open MLLM, realistically 3B-32B. Under that constraint the expensive
design choices get made without evidence:

- Do you need to feed pages as images at all? Vision encoding is the most
  expensive thing on the GPU; if cheap text suffices for your documents, the
  saving is large.
- Does the answer depend on the *kind* of document? A contract corpus and a
  slide-deck corpus are not the same problem; one recipe cannot serve both without
  waste.
- Without doc-type labels, is it worth classifying each document? Routing adds a
  classifier's own cost; if a uniform policy is nearly as good, routing is dead
  weight.

The contribution is a **recipe indexed by document type**, plus the mechanism that
explains it, plus a set of honest cost-frontier readings (which representation,
parser, retriever, resolution, quantization, and model size is worth its cost).

## 3. Central construct: the cost-ordered representation ladder

The cost knob is *how evidence is represented*. Holding the gold pages fixed, page
content is fed from cheapest to most expensive, and we find the cheapest form that
still works, separately per document type.

| Rung | Content | Role |
|---|---|---|
| `T`   | cheap embedded-text extraction (PyMuPDF); digital-born only | cheapest reference |
| `TL`  | parser-derived layout-rich text (markdown from a PDF parser) | mid |
| `TLV` | parser text + native-resolution page image | expensive |
| `V`   | page image only | parser-independent reference |

Two things changed at v4 and both must be stated honestly in the paper:

- **Bounding-box JSON is dropped everywhere.** It was the token-heavy channel that
  drove truncation and it did not help the reasoner (no fine-tuning to exploit
  coordinates). At `TL` the parser's markdown text *replaces* `T`'s cheap embedded
  text; it does not add a separate layout channel.
- **The ladder is cost-ordered, not cumulative.** Because `TL` replaces rather than
  adds, `T ⊄ TL`. The rungs are ordered by compute cost: `T` (cheap extract) < `TL`
  (parser) < `TLV` (parser + image), with `V` as the parser-independent image-only
  reference. The headline framing is *four representations of increasing compute
  cost; which is worth its cost, per document type?*

The `T/TL/TLV/V` labels are kept for table-shape and cache-key continuity, but the
"L" is now vestigial: there is no separate layout channel, and "TL" mechanically
means "parser-derived text." Same "keep the names, change the mechanism" discipline
already used for OCR.

**Parsers under comparison (the `TL`/`TLV` text source).** "PDF parser" is the
umbrella term (OCR is one mechanism a parser may use). The parser only varies at
`TL`/`TLV`; `T` and `V` are held fixed across parser runs.

| Tier / role | Tool | Notes |
|---|---|---|
| Floor (the `T` extractor) | **PyMuPDF** embedded text | Free, instant, digital-only. |
| Parser A | **PaddleOCR-VL** (0.9B) | End-to-end VLM parser; NaViT-style native-res encoder + ERNIE-4.5-0.3B. |
| Parser B | **MinerU 2.5** (~1.2B) | Layout-detect + VLM pipeline; different paradigm to A. |
| Parser C | **Unlimited OCR** | Baidu open-weight; 40+ pages in one pass. |

The parser comparison is a contribution, not a tool-swap: the headline
OmniDocBench scores for these parsers are vendor self-reported, and an independent
comparison scored by **downstream QA accuracy on MMLongBench-Doc** (not
edit-distance) is genuinely novel. Marker is dropped from the comparison set
(existing caches may be kept as continuity, not required). Each parser is a heavy,
separately-pinned VLM stack that will not co-exist with the pinned reasoner env, so
each runs as an **isolated env** whose output is pre-warmed to the parser cache in
the pre-pass (same discipline already used for Marker/Surya/PaddleOCR).

## 4. Deployable vs analytical axes

- **Document type is deployable.** A lightweight classifier can predict it from the
  first pages (RQ3), so recipes may be indexed by it.
- **Question type is analytical only.** It is used to *explain* the recipe
  (mechanism), never in a deployment recommendation, because a practitioner does
  not know a question's type in advance.

## 5. Research questions

The RQ framing is not set in stone (for example the hallucination study could sit
under RQ1); what matters is that all data is collected correctly and uniformly. The
answerable / unanswerable split (§5.4) governs which questions feed which RQ.

**RQ1 - recipe by document type (what to build).** Given the correct pages, what is
the cheapest representation that lets an 8B MLLM reason to an answer, and does that
frontier depend on document type? *Deliverables:* the cost-ordered headline (Table
1, four rungs x three bins, frontier marked, answerable-only); a **parser
comparison** (swap the `TL`/`TLV` parser, hold everything else); an
**image-resolution sweep** (vary the per-page vision-token budget at `TLV`/`V`,
supervisor hardware). Replications: model **family** (InternVL) and **dataset**
(held-out MMLongBench subset). Three core tables (headline, parser, resolution)
plus replications.

**RQ2 - retrieval (mechanism + retrieval-side modality).** Does retrieval modality
have to match reasoning modality? *Deliverables:* **matched vs cross across all
three bins**, run at `TLV` (with `V` available via YAML for the
vision-retrieval->vision-reasoning contrast); RQ2 inference uses `TLV`/`V` only, not
the whole ladder, because preliminary results already show all-modality `TLV` is the
strongest reasoning rung, and that matches the representation you would actually
deploy. A **top-k sweep** with retrieval methods framed as cost rungs (§6), at
k in {1,3,5,7,10} for single-method retrieval, on supervisor hardware so a high-k
accuracy drop is a real distractor effect and not a truncation artifact. A
**retrieval-accuracy benchmark** (page precision/recall/F1 vs gold, per bin,
covering every retrieval method including ones never fed to the reasoner), which is
a by-product of the same retrieval pass, not a separate GPU task.

**RQ3 - deployment (routing + hallucination + prompting + cost sweeps).**
*Deliverables:* **routing**, four policies (oracle routing, predicted routing,
uniform-cheapest, uniform-strongest), predicted-routing cost including the
classifier's own latency, routing accuracy reusing RQ1's ladder rows. **Hallucination
x prompting** (studied together) on the ~250 unanswerable questions pulled out of
RQ1/RQ2: because unanswerable questions have zero gold pages by construction, there
is *no oracle arm*, so the only coherent page selection is similarity-retrieved 2-3
pages, fed under three prompt conditions (no prompt / generic / hallucination-
targeted), with correct behaviour = abstention. This ties the hallucination study to
RQ2's retrieval machinery at a fixed small k. **Cost sweeps**: quantization
(4/8/16-bit) and model size (2B/4B/8B/32B), framed as cost-frontier studies
(accuracy-per-VRAM, accuracy-per-latency), not accuracy studies; placement (main vs
appendix) decided post-hoc by significance. Supervisor hardware for the
size/quant/resolution sweeps.

### 5.4 The answerable / unanswerable split

The ~250 unanswerable questions (of 1091) are **removed from RQ1 and RQ2**, so those
accuracies are cleanly "accuracy on answerable questions." They are used **only** in
the RQ3 hallucination study. Consequence stated in the paper: RQ1/RQ2 say nothing
about hallucination; the dedicated study carries the entire abstention story, and it
is necessarily a retrieval-fed study (no oracle arm exists for zero-gold-page
questions). The judge's abstention scoring and the κ ≥ 0.75 gate are confirmed on
the answerable-only set.

## 6. Retrieval methods as cost rungs

Both retrieval axes are laid out cheapest -> most expensive, mirroring the
representation ladder so RQ2 parallels RQ1's cost story.

| Axis | Cheap | Mid | Expensive |
|---|---|---|---|
| **Text** | BM25 (lexical) | BGE-M3 (dense; self-hosted workhorse) | Qwen3-Embedding-4B (SOTA-class) |
| **Vision** | ColModernVBERT (~250M) | ColQwen2.5-v0.2 (current default; reuses caches) | ColQwen3-4B (ViDoRe SOTA-class) |

Qwen3-Embedding-4B is the deployable "expensive text" rung (the 8B is reserved for
an accuracy-only benchmark if wanted). ColQwen2.5-v0.2 stays as the mid vision rung
specifically to reuse existing retrieval caches.

**Joint retrieval** is a free post-hoc **deduplicated union** of two already-computed
page sets (no new retrieval, no score fusion, so union rather than RRF). Pairs are
representative matched-tier unions only (cheap+cheap, mid+mid, expensive+expensive);
joint uses k in {1,3,5} per method so the union is always < 10 pages, and it is its
own condition family, not a point on the single-method k-axis (which goes to k=10).

## 7. Pre-registered setup

Every choice below is fixed before the main runs.

- **Primary cost metric:** latency per question at batch=1. Prefill and decode
  latency are split out (prefill isolates the cost of ingesting the representation,
  which is what the ladder changes). Text and vision tokens, peak VRAM, and
  truncation are reported alongside. The full telemetry schema (`docs/pivot_v4.md`
  §6) is collected on every run, uniformly.
- **Sufficiency margin:** accuracy drop <= 3 points relative to the strongest
  representation. Sensitivity for margin in {2, 3, 5} in the Appendix.
- **Doc-type binning by manual annotation (not native `doc_type`).** MMLongBench's
  native `doc_type` encodes *domain*, not *dominant modality*, and each domain mixes
  scanned/digital and visual-heavy/text-heavy docs, so the bin axis (the whole
  thesis) is labelled directly. A fast (<1 hr) manual pass over all **135 documents**
  produces three labels: **`bin_label`** (document-level dominant modality:
  **text-dominant / mixed-modality / visual-dominant**), **`scan_label`**
  (`digital` / `scanned`), and **`dominant_visual`** ({tables, charts, figures,
  photos, none}, multi-valued, exploratory only). Bins describe which modality
  dominates the document's information content, not scan status and not page count:
  a scanned handwritten document is text-dominant (its information is linguistic); a
  text-sparse magazine cover is visual-dominant; a text-dense paper with a few
  figures is mixed-modality. Recommended cheap insurance (recommended, not blocking):
  a second annotator labels a 20-30 doc subset and we report inter-annotator
  agreement (Cohen's κ, same bar as the judge gate).
- **Ladder implementation:** `T` = PyMuPDF embedded text (digital-born only); `TL` =
  parser-derived markdown text (parser under comparison); `TLV` = parser text +
  native-resolution page image; `V` = page image only. No bounding-box channel.
- **Reasoner:** Qwen3-VL-8B primary. InternVL3-8B replicates the RQ1 headline table
  only. The quantization sweep (4/8/16-bit) and the size sweep (2B/4B/8B/32B) are
  cost-frontier studies on supervisor hardware (32B is A100-only).
- **Retrieval:** the cost-rung tables in §6 (text: BM25 / BGE-M3 / Qwen3-Embedding-4B;
  vision: ColModernVBERT / ColQwen2.5 / ColQwen3-4B), plus the free joint unions. RQ2
  inference runs at `TLV`/`V` only; the k-sweep runs on the supervisor.
- **Judge:** a *different family* from the reasoner. Gemini 2.5 Flash is the default
  (free tier); GPT-4o-mini is the paid alternative. Judge-human agreement on 200
  hand-labelled questions; **Cohen's kappa >= 0.75 required** on the answerable-only
  set before any main-run number is trusted.
- **Confidence:** every headline number carries a bootstrap 95% CI, resampled at the
  **document level** (1000 draws over docs), because questions cluster within 135
  docs. A frontier claim requires the cheaper representation's CI upper bound to
  reach within 3 points of the strongest representation's point estimate.
- **Hardware split, marked per run.** We run on **Kaya** (2x V100 16 GB, sm_70, no
  FlashAttention-2) and on the supervisor's **A100 / H100**. Every YAML run carries
  `machine: kaya | supervisor`. The size/quant/resolution sweeps and the RQ2 k-sweep
  (up to k=10) are supervisor-only. See §8.

## 8. Experiments and the hardware/evidence-page partition

The GPU work is organized as four generation tasks (`experiments/G*_*.py`), per the
"few tasks, full coverage" mandate. The parser, resolution, family, dataset,
quantization, and model-size "experiments" are **YAML runs over G1**, not separate
tasks. Every task collects the full telemetry schema.

- **G1 - oracle ladder (reasoning core + all its sweeps).** Oracle pages x
  {T, TL, TLV, V}, answerable-only, primary 8B. Feeds the cost-ordered headline
  (Table 1) and the per-bin frontier. Reused by the parser, resolution, family,
  dataset, quantization, and model-size YAML runs (one field changed each).
- **G2 - retrieval (inference + accuracy benchmark, one pass, two scorers).** A
  retrieval pre-pass computes the retrieved page set once per (question, method, k)
  for all six methods + the three joint unions. Scorer A is the retrieval-accuracy
  side-artifact (page P/R/F1 per bin, every method). Scorer B is the reasoner at
  `TLV` (and `V` via YAML) on the chosen page sets: matched vs cross across all three
  bins, and the k-sweep. Supervisor hardware.
- **G3 - hallucination / prompting (unanswerable subset).** Unanswerable-only (~250
  q) x similarity-retrieved 2-3 pages (reuses G2's retrieval cache at fixed small k)
  x `TLV` x {no prompt, generic, hallucination-targeted}, 8B. Correct = abstention;
  no oracle arm.
- **G4 - routing / classifier (deployment).** The doc-type classifier pricing +
  routing-policy assembly. Routing accuracy reuses G1's rows; G4 adds only the
  classifier's own cost.

**Hardware / evidence-page partition.** Per `dataset_stats.md` the vast majority of
MMLongBench questions have 0-2 evidence pages; only ~50 across the corpus have >2.
We exploit this: **Kaya** runs the <= 2 evidence-page questions (short, so the input
cap can be raised, reducing truncation); the **supervisor** runs the > 2
evidence-page remainder at the same agreed cap. The token cap is a fixed
experimental constant, identical on both machines; Kaya simply cannot *fit* that cap
on some questions without OOM while the supervisor can, so machine is the executor,
not a condition. Each cell records `machine` and `cap_used`, each run's manifest
records its `evidence_page_filter` (<=2 / >2 / all), and tables that report a
full-corpus number pool the two slices at build time.

### 8.1 Task -> result map

| Result | Task |
|---|---|
| Cost-ordered headline (Table 1) | G1 (base run) |
| Parser comparison | G1 (parser-varied run) |
| Resolution sweep | G1 (resolution-varied run, supervisor) |
| Family replication | G1 (InternVL run) |
| Dataset replication | G1 (held-out corpus run) |
| Quantization sweep | G1 (quant-varied run, supervisor) |
| Model-size sweep | G1 (size-varied run, supervisor) |
| Matched vs cross (3 bins) | G2 (inference scorer) |
| k-depth sweep | G2 (inference scorer) |
| Retrieval-accuracy benchmark (per bin) | G2 (side-artifact scorer) |
| Hallucination + prompting | G3 |
| Routing | G4 (+ reuses G1) |

## 9. Go / no-go gates

Gate tooling is `gates/core.py`, exposed via `python -m gates`.

- **Gate 1 - RQ1 frontier divergence (F1).** Run the cost-ordered headline on 8B,
  oracle pages, answerable-only. **Go** if >=2 of 3 bins have different sufficiency
  frontiers. **No-go** -> doc-type is not a useful axis; reframe around evidence
  composition alone.
- **Gate 2 - judge-human agreement (F2).** Hand-label 200 questions. **Go** if the
  judge reaches kappa >= 0.75 on the answerable-only set. **No-go** -> iterate the
  judge prompt or fall back to a stronger judge before any main run.
- **Gate 3 - classifier feasibility (F3).** On a 100-doc pilot, run the first-two-page
  Qwen3-VL-2B classifier. **Go** if top-1 bin accuracy >= 70%. **No-go** -> upgrade
  the classifier or scope RQ3 to the oracle-routing upper bound only.

## 10. Known risks fixed by data

- **Visual-dominant bin may be thin.** It is the most likely Gate-1 casualty and
  will carry the widest CIs. If it cannot be separated at the 3-point margin, the
  fallbacks, in order, are: (i) collapse to a two-bin contrast (text-dominant vs
  rest); (ii) recruit a visual-heavy dataset (SlideVQA) as the visual anchor. The
  manual `bin_label` annotation is precisely the insurance against a bad bin axis.
- **Bin-axis reliability.** The `mixed-modality` bin needs human judgement and the
  whole thesis rests on the bin axis; the recommended second-annotator κ on a 20-30
  doc subset (§7) is on the record so it is not improvised later.
- **Parser environments.** The three comparison parsers are heavy, separately-pinned
  VLM stacks that will not co-exist with the reasoner env. Each is an isolated env
  whose output is pre-warmed to the parser cache; sizing that is an open environment
  task.
- **Sampling correlation.** Questions cluster within documents (135 docs, 1091 Q).
  All subsetting and all CIs are handled at the **document level**.
- **V100 hardware limits.** Kaya's V100s are Volta (sm_70): no FlashAttention-2, so
  attention can fall back to an O(seq^2) kernel that OOMs long multi-page sequences,
  and the 8B does not fit one V100 in bf16. Mitigations are baked in (efficient
  kernel, per-size input-token and vision-pixel caps, per-cell skip on OOM, 2x V100
  sharding or 4-bit). The size/quant/resolution sweeps and the k-sweep run on the
  supervisor's A100/H100.

## 11. What this pivot keeps from v3

The one-line thesis and the deployable (doc type) vs analytical (question type)
axes; document-level bootstrap CIs (1000 resamples over docs), the 3-point
sufficiency margin, and the judge-family-≠-reasoner-family rule with the κ gate; the
frozen pipeline shape (conditioner -> render -> representation -> reasoner -> judge),
the two-layer cache (prediction key without judge; result key with judge), and the
YAML-generate / artifact-judge / build role split; the `T/TL/TLV/V` rung names and
table shapes (mechanism changes, names stay).

## 12. What was cut (and where it went)

Cut from the paper, retained for the honours thesis / future work: the full
distractor-burying sweep beyond the RQ2 k-depth study; scaling as a *story* (kept
only as a cost-frontier sanity check); the multi-dataset robustness suite beyond the
held-out replication. These are real but do not serve the single thesis at 8 pages.
