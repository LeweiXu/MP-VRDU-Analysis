# Project Specification — A Doc-Type Recipe for Multi-Page Document QA (v3)

> The authoritative statement of *what* the paper claims and *why*. The companion
> `implementation_plan.md` governs *how* the codebase is built. This spec mirrors the v3
> experimental plan and supersedes all earlier multi-topic / nine-RQ / three-topic specs.
> Where an older doc disagrees, this file is current.

---

## 1. One-line thesis

**The representation an MP-VRDU system requires is a function of document type, not a single
property of the model.** We measure this function on a controlled representation ladder, turn it
into a deployment recipe indexed by document type, and explain the recipe through evidence
composition.

Venue target: **EACL long paper** (8 pages). Everything not serving the thesis is moved to the
honours thesis or explicitly cut.

## 2. Motivation

Sensitive documents (medical records, contracts, financial filings, compliance reports) cannot be
sent to a cloud API; they must be processed on-premise with a self-hosted open MLLM, realistically
3B–32B. Under that constraint the expensive design choices are made without evidence:

- Do you need to feed pages as images at all? Vision encoding is the most expensive thing on the
  GPU; if cheap text suffices for your documents, the saving is large.
- Does the answer depend on what kind of document you have? A contract corpus and a slide-deck
  corpus are not the same problem; one recipe cannot serve both without waste.
- Without doc-type labels, is it worth classifying each document? Routing adds a classifier's own
  cost; if a uniform policy is nearly as good, routing is dead weight.

The contribution is a **recipe indexed by document type**, plus the mechanism that explains it.

## 3. Central construct: the representation ladder

The cost knob is *how evidence is represented*. Holding gold pages fixed, page content is fed from
cheapest to most expensive; we find the cheapest form that still works, separately per document
type.

| Rung | Content | Role |
|---|---|---|
| `T` | raw text (Marker-extracted) | reference (cheap) |
| `T+L` | text + serialized bbox layout (JSON) | cumulative |
| `T+L+V` | text + layout + native-resolution page image | cumulative |
| `V` | page image only | parser-independent reference |

The ladder is cumulative on `T`/`T+L`/`T+L+V` (marginal value of each added modality); `V` is the
parser-independent reference.

## 4. Deployable vs analytical axes

- **Document type is deployable.** A lightweight classifier can predict it from the first pages
  (RQ3), so recipes may be indexed by it.
- **Question type is analytical only.** It is used to *explain* the recipe (mechanism), never in a
  deployment recommendation, because a practitioner does not know a question's type in advance.

## 5. Research questions

**RQ1 — Recipe by doc type (what to build).** Given the correct pages, what is the cheapest
representation that lets an 8B MLLM reason to an answer, and does that frontier depend on document
type? *Deliverable:* a 3-row headline table (text-heavy / in-between / visual-heavy) × four
representations, sufficiency frontier marked; replicated on a second model family and a second
dataset.

**RQ2 — Mechanism behind the recipe (why it looks that way).** What explains the doc-type effect,
and does retrieval require the same modality as reasoning? *Deliverable:* (a) the doc-type effect
re-expressed as an evidence-composition effect (the recipe is what it is because a text-heavy
corpus is X% pure-text evidence); (b) matched vs cross (text-retrieval + vision-reasoning)
pipelines, with cross wins explained via a locate–reason modality divergence — a page can be
text-locatable but vision-reasoned.

**RQ3 — Routing under uncertainty (what to do without labels).** Without gold doc-type labels,
does running a lightweight classifier and dispatching to the RQ1 recipe beat a uniform policy,
once the classifier's own cost is counted? *Deliverable:* corpus-level accuracy and total cost of
four policies — oracle routing, predicted routing, uniform-cheapest, uniform-strongest; classifier
latency is added into predicted-routing cost, not hidden.

## 6. Pre-registered setup

Every choice below is fixed before the main runs.

- **Primary cost metric:** latency per question at batch=1 on a single A100 80GB. Text and vision
  tokens reported separately as secondary. (Local deployment cares about response time; token
  counts across modalities are not FLOPs-equivalent.)
- **Sufficiency margin:** accuracy drop ≤ 3 points relative to the strongest representation.
  Sensitivity for margin ∈ {2, 3, 5} in the Appendix.
- **Doc-type binning (Option A, fixed):** MMLongBench-Doc native `doc_type` categories aggregated
  by semantic domain into three bins:
  - **Text-heavy** = Administration/Industry file + Academic paper + Research report/Introduction
    (**578 Q / 54 docs**).
  - **In-between** = Financial report + Guidebook + Tutorial/Workshop (**412 Q / 50 docs**).
  - **Visual-heavy** = Brochure (**101 Q / 15 docs**).
  Data-driven clustering by evidence-modality distribution is reported in the Appendix as
  robustness (and is the fallback if the visual-heavy bin proves too thin; see §9). Semantic
  aggregation is practitioner-interpretable; data-driven grouping would leak the effect being
  studied, so it is validator, not primary.
- **Ladder implementation:** `T` = Marker raw text; `T+L` = Marker text + serialized bbox JSON;
  `T+L+V` = Marker text + native-resolution page image; `V` = page image only. Parser swap
  (Marker vs PyMuPDF) in the Appendix.
- **Reasoner:** Qwen3-VL-8B primary. InternVL3-8B replicates the RQ1 headline table only.
  Qwen3-VL-2B / 32B for scale sanity in the Appendix.
- **Retrieval:** BM25 + BGE-large (text), ColQwen (vision). RQ2 compares *matched* (retrieval
  modality = reasoning modality) vs *cross* (text retrieval + vision reasoning). Vision-retrieval +
  text-reasoning is not tested (no practical rationale; inflates the comparison surface).
- **Judge:** GPT-4o-mini (different family from Qwen and InternVL). Judge–human agreement on 200
  hand-labelled questions; **Cohen's κ ≥ 0.75 required** before any main-run number is trusted.
- **Confidence:** every headline number carries a bootstrap 95% CI (1000 resamples over
  questions). A frontier claim requires the cheaper representation's CI upper bound to reach within
  3 points of the strongest representation's point estimate.

## 7. Experiments

- **Exp 1 · RQ1 — Recipe by document type.** Sweep the ladder on oracle pages with Qwen3-VL-8B;
  fill the 3×4 headline table (Table 1), mark the frontier; re-slice by question type into the
  analytical 3×4 (Table 2, not for deployment); replicate the headline on InternVL3-8B (Table 3)
  and LongDocURL (Table 4).
- **Exp 2 · RQ2 — Mechanism.** (a) Evidence-composition mediation: decompose each doc-type bin
  into shares of text/table/chart/figure/layout evidence; show per-modality frontier + composition
  predicts the doc-type frontier (Table 5). (b) Retrieval-side modality: on cells where RQ1 says
  vision is needed, compare matched vs cross pipelines under real retrieval on accuracy and latency
  (Table 6); cross wins explained by locate–reason divergence in one paragraph + one qualitative
  figure.
- **Exp 3 · RQ3 — Routing.** Four policies on the full corpus: oracle routing, predicted routing
  (Qwen3-VL-2B few-shot classifies the first pages, then recipe), uniform-cheapest (`T`
  everywhere), uniform-strongest (`T+L+V` everywhere). Predicted-routing total latency includes the
  classifier's own latency (Table 7).
- **Exp 4 · Appendix — Scale sanity.** Re-run the RQ1 headline on Qwen3-VL-2B and 32B (Table 8).
  Main text cites one sentence: "the recipe is qualitatively stable across 2B–32B," or names the
  bins where the frontier moves. No scaling headline is claimed.

## 8. Go / no-go gates (Weeks 1–2)

- **Gate 1 · RQ1 frontier divergence.** Run Exp 1's headline table on 8B, oracle pages, full
  MMLongBench-Doc. **Go** if ≥2 of 3 doc-type rows have different sufficiency frontiers. **No-go**
  if all three land on the same rung → doc-type is not a useful axis; reframe around evidence
  composition alone.
- **Gate 2 · Judge–human agreement.** Hand-label 200 questions across doc-type × question-type
  strata. **Go** if GPT-4o-mini reaches κ ≥ 0.75. **No-go** → iterate the judge prompt or fall back
  to GPT-4o full before any main run.
- **Gate 3 · Classifier feasibility.** On a 100-doc pilot, run Qwen3-VL-2B few-shot doc-type
  classification from the first two pages. **Go** if top-1 ≥ 70%. **No-go** → upgrade the
  classifier or scope RQ3 to the oracle-routing upper bound only.

## 9. Known risks fixed by data

- **Visual-heavy bin is thin (101 Q / 15 docs).** It is the most likely Gate-1 casualty and will
  carry the widest CIs. If it cannot be separated from the other bins at the 3-point margin, the
  fallbacks, in order, are: (i) adopt the Appendix evidence-composition (data-driven) binning as
  primary; (ii) collapse to a two-bin contrast (text-heavy vs rest); (iii) recruit a visual-heavy
  dataset (SlideVQA) as the visual anchor. Recorded so the choice is pre-committed, not improvised.
- **Sampling correlation.** Questions cluster within documents (135 docs, 1091 Q). Any subsetting
  and all CIs are handled at the **document level** (draw documents, take their questions) so
  precision is not overstated.
- **Qwen3-VL API availability.** `transformers==4.53.2` did not expose the Qwen3-VL model class at
  Stage 1; resolving the load path (transformers upgrade within the vLLM/colpali window, or a
  confirmed vLLM path) is on the critical path before Gate 1, since every number needs a working
  8B reasoner.

## 10. What was cut (and where it went)

Cut from the paper, retained for the honours thesis / future work: the full retrieval-sufficiency
and distractor-burying sweep; scaling as a *story* (kept only as an Appendix sanity check);
fail-safe abstention; the multi-dataset robustness suite beyond one replication. These are real but
do not serve the single thesis at 8 pages.