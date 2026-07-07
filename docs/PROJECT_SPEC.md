# Project spec: a doc-type recipe for multi-page document QA (v3)

The authoritative statement of *what* the paper claims and *why*. It is kept in
sync with the implementation:

- `README.md` is the ground truth for *how* the pipeline is actually built (the
  cell, the ladder, the tasks, the repo map). Where this spec and the README
  disagree on a mechanism, the README wins and this file should be corrected.
- `docs/USER_GUIDE.md` restates this thesis and adds the local runbook.
- `docs/AGENT_GUIDE.md` holds the fixed decisions, frozen interfaces, and the
  implementation reference.

This spec supersedes all earlier multi-topic / nine-RQ / three-topic specs.

---

## 1. One-line thesis

**The representation an MP-VRDU system needs is a function of document type, not a
single property of the model.** We measure that function on a controlled
representation ladder, turn it into a deployment recipe indexed by document type,
and explain the recipe through evidence composition.

Venue target: **EACL long paper** (8 pages). Everything that does not serve the
thesis moves to the honours thesis or is cut.

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
explains it.

## 3. Central construct: the representation ladder

The cost knob is *how evidence is represented*. Holding the gold pages fixed, page
content is fed from cheapest to most expensive, and we find the cheapest form that
still works, separately per document type.

| Rung | Content | Role |
|---|---|---|
| `T`   | text only | reference (cheap) |
| `TL`  | text + serialized bbox layout (JSON) | cumulative |
| `TLV` | text + layout + native-resolution page image | cumulative |
| `V`   | page image only | parser-independent reference |

`T`/`TL`/`TLV` is cumulative (the marginal value of each added modality); `V` is
the parser-independent reference. Only `TLV` and `V` carry images; the modality
boundary is enforced structurally in `Payload`.

The text channel is chosen **per document**: digital-born PDFs use Marker text,
scanned PDFs use PaddleOCR, routed by `annotations/doc_labels.csv` (a filled human
`scan_label` wins, else the auto-seeded `auto_scan`). OCR is *not* a separate rung;
it just makes `T` the best available text for that document while keeping every
cache key and table shape unchanged.

## 4. Deployable vs analytical axes

- **Document type is deployable.** A lightweight classifier can predict it from the
  first pages (RQ3), so recipes may be indexed by it.
- **Question type is analytical only.** It is used to *explain* the recipe
  (mechanism), never in a deployment recommendation, because a practitioner does
  not know a question's type in advance.

## 5. Research questions

**RQ1 - recipe by doc type (what to build).** Given the correct pages, what is the
cheapest representation that lets an 8B MLLM reason to an answer, and does that
frontier depend on document type? *Deliverable:* a 3-row headline table
(text-heavy / in-between / visual-heavy) x four representations, sufficiency
frontier marked; replicated on a second model family and a held-out document set.

**RQ2 - mechanism behind the recipe (why it looks that way).** What explains the
doc-type effect, and does retrieval need the same modality as reasoning?
*Deliverable:* (a) the doc-type effect re-expressed as an evidence-composition
effect (the recipe is what it is because a text-heavy corpus is X% pure-text
evidence); (b) matched vs cross (text-retrieval + vision-reasoning) pipelines under
real retrieval, with cross wins explained via a locate-reason modality divergence
(a page can be text-locatable but vision-reasoned).

**RQ3 - routing under uncertainty (what to do without labels).** Without gold
doc-type labels, does running a lightweight classifier and dispatching to the RQ1
recipe beat a uniform policy, once the classifier's own cost is counted?
*Deliverable:* corpus-level accuracy and total cost of four policies - oracle
routing, predicted routing, uniform-cheapest, uniform-strongest; classifier latency
is added into predicted-routing cost, not hidden.

## 6. Pre-registered setup

Every choice below is fixed before the main runs.

- **Primary cost metric:** latency per question at batch=1 on a single A100 80GB.
  Text and vision tokens reported separately as secondary. (Local deployment cares
  about response time; token counts across modalities are not FLOPs-equivalent.)
- **Sufficiency margin:** accuracy drop <= 3 points relative to the strongest
  representation. Sensitivity for margin in {2, 3, 5} in the Appendix.
- **Doc-type binning (Option A, fixed), the single source of truth in
  `data/binning.py`:**
  - **Text-heavy** = Administration/Industry file + Academic paper + Research
    report/Introduction (**578 Q / 70 docs**).
  - **In-between** = Financial report + Guidebook + Tutorial/Workshop
    (**412 Q / 50 docs**).
  - **Visual-heavy** = Brochure (**101 Q / 15 docs**).

  Data-driven clustering by evidence-modality distribution is the Appendix
  robustness swap (and the fallback if visual-heavy proves too thin; see §9).
  Semantic aggregation is practitioner-interpretable; data-driven grouping would
  leak the effect being studied, so it is validator, not primary.
- **Full-run subset:** a full MMLongBench run defaults to ~100 questions per
  Option-A bin (drawn by *whole document* at `sample_seed=0`) instead of all 1091,
  so a run clears the cluster queue in a couple of hours. Bins below the target
  stay whole (visual-heavy keeps all 101 Q / 15 docs). Questions whose gold
  evidence spans more than 10 pages are dropped up front (a V100 attention limit,
  7 questions on the full corpus). The F1 frontier gate wants the whole corpus, so
  record whether a verdict came from the subset or the full run.
- **Ladder implementation:** `T` = Marker text (PaddleOCR for scanned docs);
  `TL` = text + serialized bbox JSON; `TLV` = text + layout + native-resolution
  page image; `V` = page image only. Parser swap (Marker vs PyMuPDF/Docling) in the
  Appendix.
- **Reasoner:** Qwen3-VL-8B primary. InternVL3-8B replicates the RQ1 headline table
  only. Qwen3-VL-2B / 32B for scale sanity in the Appendix (2B is also the smoke
  model). Main numbers are bf16 on 2xV100; 4-bit is single-GPU iteration and a
  possible appendix quant-sensitivity row.
- **Retrieval:** BM25 + BGE (`BAAI/bge-small-en-v1.5`) for text, ColQwen
  (`vidore/colqwen2.5-v0.2`) for vision. RQ2 compares *matched* (retrieval modality
  = reasoning modality) vs *cross* (text retrieval + vision reasoning), swept over
  k in {1, 3, 5, 7, 9}. Vision-retrieval + text-reasoning is not tested (no
  practical rationale; inflates the comparison surface).
- **Judge:** a *different family* from the reasoner. Gemini 2.5 Flash is the default
  (free tier); GPT-4o-mini is the paid alternative. Judge-human agreement on 200
  hand-labelled questions; **Cohen's kappa >= 0.75 required** before any main-run
  number is trusted.
- **Confidence:** every headline number carries a bootstrap 95% CI, resampled at
  the **document level** (draw documents with replacement, take all their
  questions; 1000 draws), because questions cluster within 135 docs / 1091 Q. A
  frontier claim requires the cheaper representation's CI upper bound to reach
  within 3 points of the strongest representation's point estimate.

## 7. Experiments

The GPU work is organized as generation tasks (`experiments/G*_*.py`), each feeding
one or more tables. See README "The generation tasks" for the per-task mechanics.

- **Exp 1 · RQ1 - recipe by document type (G1).** Sweep the ladder on oracle pages
  with Qwen3-VL-8B; fill the 3x4 headline table (Table 1), mark the frontier;
  re-slice by question type into the analytical 3x4 (Table 2, not for deployment);
  replicate the headline on InternVL3-8B (Table 3, task G2) and on a held-out
  MMLongBench document subset (Table 4, task G3).
- **Exp 2 · RQ2 - mechanism.** (a) Evidence-composition mediation: decompose each
  doc-type bin into shares of text/table/chart/figure/layout evidence; show
  per-modality frontier + composition predicts the doc-type frontier (Table 5,
  derived from G1). (b) Retrieval-side modality (G5): on cells where RQ1 says vision
  is needed, compare matched vs cross under real retrieval on accuracy and latency,
  swept over k (Table 6, one row per k); cross wins explained by locate-reason
  divergence in one paragraph + one qualitative figure.
- **Exp 3 · RQ3 - routing (G6 + G1).** Four policies on the corpus: oracle routing,
  predicted routing (Qwen3-VL-2B classifies the first two pages, then recipe),
  uniform-cheapest (`T` everywhere), uniform-strongest (`TLV` everywhere).
  Predicted-routing total latency includes the classifier's own latency (Table 7).
  Routing accuracy reuses G1's ladder rows; G6 only prices the classifier.
- **Exp 4 · Appendix - scale sanity (G4, planned).** Re-run the RQ1 headline on
  Qwen3-VL-2B and 32B (Table 8). Main text cites one sentence ("the recipe is
  qualitatively stable across 2B-32B", or names the bins where the frontier moves).
  No scaling headline is claimed. G4 is not yet implemented; the 32B needs the
  supervisor's A100, so Table 8 is gated off until then.

## 8. Go / no-go gates (Weeks 1-2)

Gate tooling is `gates/core.py`, exposed via `python -m gates`.

- **Gate 1 · RQ1 frontier divergence (F1).** Run Exp 1's headline table on 8B,
  oracle pages, full MMLongBench. **Go** if >=2 of 3 doc-type rows have different
  sufficiency frontiers. **No-go** if all three land on the same rung -> doc-type is
  not a useful axis; reframe around evidence composition alone. (Pending: needs the
  full 8B run.)
- **Gate 2 · judge-human agreement (F2).** Hand-label 200 questions across doc-type
  x question-type strata. **Go** if the judge reaches kappa >= 0.75. **No-go** ->
  iterate the judge prompt or fall back to a stronger judge before any main run.
- **Gate 3 · classifier feasibility (F3).** On a 100-doc pilot, run Qwen3-VL-2B
  few-shot doc-type classification from the first two pages. **Go** if top-1 bin
  accuracy >= 70%. **No-go** -> upgrade the classifier or scope RQ3 to the
  oracle-routing upper bound only.

## 9. Known risks fixed by data

- **Visual-heavy bin is thin (101 Q / 15 docs).** It is the most likely Gate-1
  casualty and will carry the widest CIs. If it cannot be separated from the other
  bins at the 3-point margin, the fallbacks, in order, are: (i) adopt the Appendix
  evidence-composition (data-driven) binning as primary; (ii) collapse to a two-bin
  contrast (text-heavy vs rest); (iii) recruit a visual-heavy dataset (SlideVQA) as
  the visual anchor. Recorded so the choice is pre-committed, not improvised.
- **Sampling correlation.** Questions cluster within documents (135 docs, 1091 Q).
  All subsetting and all CIs are handled at the **document level** (draw documents,
  take their questions) so precision is not overstated.
- **V100 hardware limits.** Kaya's V100s are Volta (sm_70): no FlashAttention-2, so
  attention can fall back to an O(seq^2) kernel that OOMs long multi-page
  sequences, and the 8B does not fit one V100 in bf16. Mitigations are baked in
  (efficient-attention kernel, per-size input-token and vision-pixel caps, the
  >10-evidence-page drop, per-cell skip on OOM, 2xV100 sharding or 4-bit). The 32B
  is out of scope on our hardware; it needs the supervisor's A100.

## 10. What was cut (and where it went)

Cut from the paper, retained for the honours thesis / future work: the full
retrieval-sufficiency and distractor-burying sweep; scaling as a *story* (kept only
as an Appendix sanity check); fail-safe abstention; the multi-dataset robustness
suite beyond the held-out replication. These are real but do not serve the single
thesis at 8 pages.
