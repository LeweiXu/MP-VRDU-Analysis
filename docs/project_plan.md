# Multi-Page Document QA — Empirical Analysis: Project Plan & Implementation Spec

> Single source of truth for the study. Covers motivation, the experimental design,
> datasets, metrics, implementation architecture, and execution order. Written for
> both human collaborators and agentic implementation.

---

## 1. One-line summary

An empirical analysis of **multi-page visually-rich document QA (MP-VRDU)** centred on
**reasoning / understanding** as the primary axis: given evidence in some representation,
can the model actually understand and answer? We measure how reasoning performance varies
with three cross-cutting variables — **document type**, **question type**, and
**representation mechanism** — and we measure **retrieval quality as a supporting
covariate** that (a) characterises where *locating* evidence is hard and (b) attributes the
Oracle-vs-Retrieved gap to locating versus reasoning. The deliverable is a deployment guide:
for a given document type and an affordable local model size, *where should effort go, and
is vision needed?*

The study is deliberately a **document-understanding** study, not a retrieval/RAG benchmark.
Retrieval is an instrument we measure, not the object of investigation. If scope allows, the
retrieval covariate can be expanded into a co-equal axis (deeper retriever/k sweeps, per-modality
retrieval-ceiling analysis) — that expansion is a strict superset of the covariate work, so
nothing built now is wasted.

This is the empirical follow-up to our MP-VRDU survey, which identified "Evaluation
Methodology and Scope" as an open challenge — specifically that answer-only accuracy
masks retrieval failure and rewards correct answers reached through incorrect reasoning.
This study operationalises a fix for that gap. The scope has been agreed with the
supervisor as doable and a genuine research gap.

---

## 2. Motivation (the deployment framing)

A large share of real document work involves **sensitive files** — contracts, financial
statements, ESG/compliance reports, medical records — that cannot be sent to a cloud API
and must be processed **on-premise with open models** (realistically 3B–32B vision-language
models). Under that constraint, three practical decisions are expensive to get wrong:

- **Is vision encoding needed at all?** It is the most expensive thing on the GPU; if cheap
  text suffices for a document type, that saves enormous cost.
- **Is retrieval needed,** or can you feed the whole document? Retrieval is real engineering,
  worth it only past a certain document length.
- **How large a model must you buy GPUs for,** and does the right answer shift as you scale?

Today these are answered by guesswork. We answer them by measurement, and crucially the
answer depends on **what kind of document and what kind of question** you have.

---

## 3. The core idea

The spine is **reasoning / understanding**: given the right evidence, in some representation,
can the model understand it and answer correctly? This is where the novelty and the
deployment payoff live — it is about what the model does *with* evidence, not about finding it.

Two failure modes exist when a model answers wrong:

- **Reasoning failure** (the spine) — given the correct pages, it still answers wrong:
  misreads a table/chart, or fails to combine evidence across pages.
- **Locating failure** (the covariate) — the retriever never surfaces the evidence page.

Answer-only accuracy hides which occurred. We isolate **reasoning** by giving the model the
gold evidence pages (the Oracle condition), and we **measure retrieval quality (Page-F1)
separately** so we can both characterise where locating is hard and attribute the
Oracle-vs-Retrieved gap. Reasoning is studied in depth across three cross-cutting variables,
one of which (representation) is the cost knob for the "is vision needed?" question.

---

## 4. Experimental design

### 4.1 The reasoning spine and the retrieval covariate

Implemented via an **input condition** that decides which pages reach the model:

| Condition          | Pages fed to model                       | Role                                  |
|--------------------|------------------------------------------|---------------------------------------|
| **Oracle**         | Exactly the gold evidence pages          | The reasoning measurement (the spine) |
| **Retrieved top-k**| k pages from a real retriever (k=1/3/5)  | Realistic performance + decomposition |
| **Full document**  | All pages (where context allows)         | "Feed everything" baseline            |

**Reasoning (the spine).** All depth goes here. Run on **Oracle pages** so residual error is
attributable to understanding, not locating. Reasoning accuracy is measured across the three
cross-cutting variables (§4.2), all model sizes (§4.4), and the representation sweep (§4.3).
Test which question types need which reasoning style; for cross-page questions, measure how
far apart evidence pages can be before accuracy degrades.

**Retrieval quality (the covariate).** Measured directly as **Page-F1 of the retriever against
the gold evidence pages** under the Retrieved condition — no special protocol, no distractor
fabrication. Sliced by document type, question type, and representation modality, this single
measurement does two jobs:
1. **Characterises locating difficulty** for the deployment story (e.g. "retrieval Page-F1 is
   low for visual-heavy docs and on multi-hop questions, high for contracts").
2. **Attributes the decomposition.** Interpreting the Oracle-vs-Retrieved gap *through* Page-F1
   turns "there is a gap" into "here is whether it is locating or reasoning": low Retrieved
   accuracy + low Page-F1 ⇒ locating bottleneck; high Page-F1 but Retrieved still below Oracle
   ⇒ reasoning-over-imperfect-evidence, not locating.

**Decomposition (read through the covariate):**
- Oracle accuracy = reasoning ceiling.
- (Oracle − Retrieved), interpreted with Page-F1 = locating- vs reasoning-attributable error.
- (100% − Oracle) = reasoning-attributable error.
- (Oracle − Full-doc) = distraction / long-context penalty.

**Retrieval stays a covariate, not the object.** Report retrieval modality contrasts as "what
the modality affords," best-in-class + spread, hedged "as of these implementations" — never a
retriever horse race. This keeps the study about document understanding, not RAG.

**Optional future probe (NOT in main scope):** a distractor-robustness protocol — fix the gold
evidence and bury it in increasing numbers of irrelevant pages, recording where accuracy
collapses — could characterise long-context robustness directly. It is deliberately excluded
from the main study because it entangles locating with reasoning-under-distraction and probes a
long-context-robustness question rather than understanding. Listed here only as a possible
extension if the study expands toward retrieval as a co-equal axis.

**Note on oracle not always being a ceiling:** a model occasionally scores higher on retrieved
than gold pages (a retrieved page carries a hint the annotation missed). Report this as a
finding about annotation completeness; do not hide it.

### 4.2 Three cross-cutting independent variables

1. **Document type** — text-heavy / in-between / visual-heavy (see datasets, §5).
2. **Question type** — single-hop / multi-hop / extractive / unanswerable / etc.
   **MMLongBench-Doc is primary here** (it ships these labels + evidence-source tags).
3. **Representation mechanism** — two-level structure (§4.3).

Every reasoning result (and the retrieval-quality covariate) is sliced by these three, plus document length bins and
evidence modality (text/layout/table/chart/figure) where the dataset provides them.

### 4.3 Representation mechanism (two levels)

**Conceptual correction:** modality (what kind of information) and extraction method (how it
is recovered) are different. OCR vs embedded-text is not "more than" — it is a different
*extraction* of the same text modality. So representation has two levels:

**Level 1 — Modality (PRIMARY SWEEP, run across the full design):**
Three modalities — **raw text**, **layout/structure**, **visual**. Tested **cumulatively**
as the primary framing (deployment-relevant marginal value), with two **exclusive**
single-modality conditions as reference anchors:

| Condition (cumulative)        | Channels present                         | Type       |
|-------------------------------|------------------------------------------|------------|
| text                          | raw text only                            | reference  |
| text + layout                 | text + structure/geometry                | cumulative |
| text + layout + visual        | all three                                | cumulative |
| visual-only                   | rendered pages only, no extracted text   | reference  |

The marginal-value reads directly: does adding layout on top of text help? does adding
visual on top of both help? — per document type and question type.

**Level 2 — Extraction method (THREE TARGETED DRILLS only):**
Extraction method is varied **only where the method choice could plausibly change the
outcome** (the "fork-existence" rule). Each drill is pinned to the slice where its fork is
real; elsewhere the comparison is null by construction and is **deliberately not run** (must
be reported as such, with rationale, not as a missing cell).

- **Drill 1 — Text: embedded vs OCR.** Target: the **scanned / degraded-text-layer slice**
  (mostly within MMLongBench-Doc). Rationale: born-digital docs have clean embedded text, so
  OCR is moot there; the fork exists only when there is no clean text layer. Output: "OCR only
  matters when there's no text layer, and here is its accuracy cost when it does."
- **Drill 2 — Layout: parsed Markdown vs bounding-box geometry.** Both are the *layout*
  modality, compared head-to-head. Target: **table-evidence and layout-evidence questions** in
  the **in-between document type** (financial/ESG, where tables dominate). Rationale: pure
  prose has no layout to recover. This is the highest-value drill (it answers "can structured
  text recover a table, or do you need geometry/vision?").
- **Drill 3 — Visual: full-page vs region-crop vs resolution.** Target: **chart/figure-evidence
  questions** in the **visual-heavy type**. Rationale: where vision is already necessary, this
  is a cost-optimisation drill — cheapest visual representation that still works.

**Parser as a controlled variable, not a confound:** for the text-bearing conditions, fix one
strong PDF→Markdown/HTML parser for the main results; re-run **one** modality comparison with a
second parser to confirm the finding is not a tooling artifact. The **visual** condition has no
parser and is therefore the cleanest, parser-independent evidence.

### 4.4 Scaling story (the headline)

Run the whole diagnosis across **3B / 7B / 32B from a single model family** (so the comparison
reflects size, not architecture). Hypothesis: the bottleneck **moves** — small models fail more
at reasoning; as they grow, reasoning saturates and the bottleneck shifts toward locating. If
true: the right thing to build depends on the model size you can afford.

### 4.5 The deployment lens

Domain is **not** a separate study — it is the document-type slice (variable 1) read for
practitioners, gathered into a per-document-type recommendation table. Because document type
and evidence type are correlated (contracts are text-heavy *because* prose; slides visual-heavy
*because* visual), report **both** the document-type view (deployment-facing) and the
evidence-type view (mechanistic), and state the domain effect is largely **mediated by**
evidence-type composition.

### 4.6 Controlling the combinatorics (star design discipline)

The full cross-product (input-condition × modality × extraction × doc-type × question-type ×
model-size) is too large to run exhaustively. Discipline:
- Define a **center cell** and vary **one variable at a time** away from it for most sweeps.
- The **modality sweep** runs across the full design (it is the primary representation result).
- **Extraction-method** is only the three targeted drills above, never a full grid.
- Fill a full 2D grid only for the one interaction of genuine interest:
  **modality × evidence-type** (where "is vision needed" is actually decided).

**Center cell:** Qwen2.5-VL-7B · Oracle pages · text+layout+visual (cumulative top) ·
ColQwen top-5 (for retrieved condition) · MMLongBench-Doc.

---

## 5. Datasets

### 5.1 General two-axis study

- **MMLongBench-Doc** — PRIMARY. 135 docs, avg 47.5 pages, 1,082 expert questions; gold
  evidence pages, five evidence-source labels (text/layout/table/chart/image), single-page /
  cross-page (33.7%) / unanswerable (20.6%); low fine-tune↔benchmark contamination. Primary
  for the question-type variable. Schema is page-level + coarse modality tag (no region boxes,
  no cell values) — sufficient for the reasoning measurement + Page-F1 covariate; note this limit.
- **LongDocURL** — SECOND (robustness replicate). Cross-domain long-doc with understanding /
  reasoning / **locating** tasks (ships evidence localisation). General findings replicated here
  so conclusions are not one-off.

### 5.2 Domain sets — main plan (text → in-between → visual)

| Dataset   | Role               | Size / pages                                  | Evidence format                                            | Vision native | Status |
|-----------|--------------------|-----------------------------------------------|------------------------------------------------------------|---------------|--------|
| **CUAD**  | text-heavy (contracts) | 510 contracts, few–100+ pages, 9,283 pp total | SQuAD-style clause spans → map to page; full PDFs          | yes (PDFs)    | peer-reviewed; CC BY 4.0 |
| **DocFinQA** | in-between (financial) | 801 SEC filings, avg ~123k words / often 150+ pages, 7,437 Q | golden context chunk → page; numeric+tabular | no (markdown; render for vision) | peer-reviewed |
| **SlideVQA** | visual-heavy (slides) | 2,619 decks, ~20 pages, 14,484 Q          | native evidence-page labels; single/multi-hop/numerical tags | yes         | peer-reviewed |

Each domain set runs the reasoning spine (Oracle pages) crossed with the modality sweep, plus the retrieval-quality covariate.
DocFinQA is text-only natively → its visual condition requires rendering filings to pages.

### 5.3 Alternatives & backups (if a main set is unworkable, or time permits)

| Dataset    | Role                | Notes |
|------------|---------------------|-------|
| **MMESGBench** | alt in-between (ESG) | 933 expert-validated Q, 45 docs, avg 157 pages (up to 2,000+); native PDFs + fine-grained evidence-modality labels; single/cross/unanswerable. Stronger native vision than DocFinQA; partly AI-generated → keep generator model out of judge loop. ACM MM '25. **Use as the multimodal-native second / robustness for in-between.** |
| **CiteVQA**    | alt vision / cross-domain | 711 PDFs, avg 40.6 pages, 1,897 Q; element-level bbox + page citations; cross-page; 7 domains. Fully synthetic pipeline + sample human validation; 2026 preprint → robustness check, not primary. |
| **TAT-DQA**    | backup financial    | 2,758 docs, ≤3 pages (85% single-page), 16,558 Q; table+text + bounding boxes. Too short to exercise retrieval meaningfully; useful for the reasoning spine + extraction-method depth on financial tables. CC BY 4.0. |
| **DUDE**       | backup cross-domain pool | 5,019 docs, avg 5.72 pages, 41,541 Q; rich diagnostic taxonomy, non-answerable + list answers. **Confirmed: no released per-question domain field or evidence-page list (extractive bbox only).** Use as a diagnostic/cross-domain pool, not a domain vehicle. |

---

## 6. Metrics

- **Answer accuracy** — LLM-as-judge following the MMLongBench-Doc protocol
  (response generation → answer extraction → rule-based scoring). Re-score *every* dataset
  with this one uniform scorer so cross-dataset numbers are comparable by construction;
  ignore each dataset's native metric.
- **Retrieval quality (covariate)** — page-level Recall / Precision / F1: for predicted page set
  `Ppred`, gold set `Pgt`: TP=|Ppred∩Pgt|, FP=|Ppred\Pgt|, FN=|Pgt\Ppred|.
- **Hallucination rate** — accuracy / abstention on unanswerable questions.
- **Cost (deployment-facing)** — token count and/or latency per query per representation, so
  the sufficiency frontier carries a real cost axis (the survey notes cost analysis is almost
  entirely absent).
- **Statistics** — report significance, effect size, and confidence intervals on all headline
  comparisons; replicate general findings on LongDocURL.

---

## 7. Implementation architecture

Clean but not fragmented: one cohesive module per concept. Stage 1 freezes interfaces; Stage 2
fills them. Config-driven; deterministic; full provenance per run.

### 7.1 Interface contracts (the spine)

```python
@dataclass
class Example:                       # normalised across all datasets
    qid: str
    question: str
    doc_id: str
    gold_answer: str
    gold_evidence_pages: list[int]   # 1-indexed; [] if unanswerable
    evidence_modalities: list[str]   # subset {text,layout,table,chart,figure}
    is_unanswerable: bool
    num_evidence_pages: int
    doc_num_pages: int
    question_type: str               # single_hop | multi_hop | extractive | ...
    document_type: str               # text_heavy | in_between | visual_heavy
    is_scanned: bool                 # gates Drill 1 (embedded vs OCR)

@dataclass
class EvidenceBundle:
    doc_id: str
    page_indices: list[int]
    modalities: list[str]            # which of {text,layout,visual} are present (cumulative)
    text_payload: list[str] | None
    layout_payload: list[dict] | None  # markdown OR bbox geometry (extraction-method dependent)
    image_payload: list[bytes] | None
    representation_name: str         # e.g. "text+layout(md)", "visual_only", ...

class Parser(ABC):                   # PDF -> per-page text/markdown/bbox + page images
    name: str
    def parse_document(self, doc_id: str) -> list[PageContent]: ...

class Representation(ABC):           # Example + pages + (modality set, extraction method) -> EvidenceBundle
    name: str
    def build(self, ex, pages, parsed) -> EvidenceBundle: ...

class Retriever(ABC):                # Example -> ranked page list
    name: str
    modality: str                    # text | vision | joint
    def retrieve(self, ex, doc) -> list[int]: ...

class Reasoner(ABC):                 # EvidenceBundle + question -> raw answer
    name: str
    def answer(self, ex, bundle) -> str: ...

class Judge(ABC):                    # raw answer vs gold -> score
    def score(self, ex, raw_answer) -> JudgeResult: ...
```

Input conditions are a function, not a class (orthogonality):

```python
def select_pages(ex, retriever, condition, k) -> list[int]:
    # oracle      -> ex.gold_evidence_pages
    # retrieved_k -> retriever.retrieve(...)[:k]
    # full_doc    -> list(range(1, ex.doc_num_pages + 1))
    # (optional, NOT main scope) locate_buried -> gold pages + N distractors (future robustness probe)
```

A run = (dataset) × (condition, k) × (representation: modality set + extraction method) ×
(retriever?) × (reasoner) × (judge). The orchestrator composes; no component knows the others.

### 7.2 Repo layout

```
.
├── PROJECT_PLAN.md            # this file
├── pyproject.toml
├── Makefile                   # make test | lint | run-demo
├── run.py                     # orchestrator entrypoint
├── core/
│   ├── types.py               # Example, EvidenceBundle, PageContent, JudgeResult
│   ├── interfaces.py          # the 5 ABCs
│   ├── conditions.py          # select_pages (oracle/retrieved/full; optional locate-buried as future probe)
│   ├── config.py              # pydantic config schema + loader
│   └── results.py             # JSONL result schema, writer, run manifest
├── data/                      # loaders -> Example (MMLongBench-Doc, LongDocURL, CUAD,
│                              #   DocFinQA, SlideVQA, + alt loaders)
├── parsing/                   # PDF -> embedded text / OCR / markdown / bbox / page images;
│                              #   primary + secondary parser
├── representation/            # modality builders (text, +layout, +visual, visual-only) and
│                              #   extraction-method variants (embedded/ocr, md/bbox, page/crop/res)
├── retrieval/                 # BM25, BGE, ColPali, ColQwen, joint fusion + Page-F1
├── inference/                 # Qwen2.5-VL 3B/7B/32B reasoners (one family) + optional Gemini
├── eval/                      # uniform LLM-as-judge + scoring + judge-agreement check
├── analysis/                  # decomposition tables, slices, modality sweep, the 3 drills,
│                              #   modality×evidence-type grid, stats (CI/effect size)
├── stubs/                     # Stage-1 deterministic stubs for all 5 interfaces
├── configs/                   # center_cell.yaml, demo.yaml, per-sweep/per-drill configs
├── tests/                     # unit + end-to-end demo on fixtures/
└── results/                   # run outputs (gitignored)
```

### 7.3 Operating principles

- Stage 1 freezes interface signatures; Stage 2 only fills them (flag any signature change).
- Config-driven: models, k, conditions, representations, drills are YAML, never hard-coded.
- Determinism + provenance: each run stamps resolved config + git SHA + versions + run ID.
- Fail loud at config/input validation; no silent fallbacks (missing model/parser → raise).
- Python 3.11+, type hints, ruff + mypy clean, pinned deps.
- pip installs use `--break-system-packages`.

---

## 8. Execution order

### Stage 1 — skeleton (no real models/data)
1. Repo scaffold + tooling (pyproject, Makefile, ruff/mypy/pytest).
2. Define the 5 interfaces + dataclasses + `select_pages` (oracle/retrieved/full; locate-buried stubbed as optional).
3. Config system (pydantic); ship center-cell + demo configs.
4. Stub implementations of all interfaces; synthetic fixture (3 docs, ~10 Q, all paths incl.
   unanswerable, scanned flag, each question/document type).
5. Orchestrator: load → select_pages → build representation → reason → judge → write result;
   batching, resume-on-crash, run manifest.
6. Result schema + JSONL writer; per-run summary.
7. Analysis math implemented for real against fixtures (decomposition, Page-F1, slices,
   modality sweep, modality×evidence-type grid, CI/effect-size).
8. Tests + `make run-demo` (full stub pipeline offline < 30s).

**Stage 1 gate (human):** demo runs; decomposition/Page-F1/slices/grid correct on fixture;
swapping a stub for a real impl needs no orchestrator change; resume works; lint+test+mypy green.

### Stage 2 — real implementations (fill stubs, validate each before next)
1. **MMLongBench-Doc loader** → Example; map evidence-source→modality, derive question/document
   type, length bins, answerable, scanned flag. Verify counts (135 docs, ~1,082 Q) and hand-check
   20 gold-evidence labels.
2. **Parser**: one strong PDF→Markdown/HTML + bbox + page-PNG rendering. Spot-check 5 docs.
3. **Representations**: text / text+layout(md) / text+layout(bbox) / +visual / visual-only.
   Unit-test each builds a valid EvidenceBundle for the same pages.
4. **Reasoner — Qwen2.5-VL-7B** (center). vLLM or HF + batching; multi-image + text input.
5. **Judge**: uniform MMLongBench-Doc protocol. **Judge-validation gate:** hand-label ~100 Q/A,
   report judge–human agreement before trusting any number.
6. **CENTER-CELL REPRODUCTION GATE (human):** Qwen2.5-VL-7B, full-doc, cumulative-top, on
   MMLongBench-Doc within ~1–2 pts of published. If not, stop and debug.
7. **Input conditions + retrieval covariate:** oracle / retrieved / full-doc; compute retriever
   Page-F1 against gold pages → first decomposition table, read through the covariate.
8. **Retrievers:** ColPali, ColQwen, BM25, BGE, joint fusion; Page-F1; slot into retrieved.
   (Also yields the visual-vs-text retrieval contrast, reported best-in-class + spread per
   evidence type, hedged "as of these implementations".)
9. **Modality sweep + modality×evidence-type grid** on the center model; the **3 targeted drills**
   (Drill 1 scanned slice; Drill 2 layout md-vs-bbox on table/layout questions in-between type;
   Drill 3 visual page/crop/res on chart/figure in visual-heavy type). Parser-robustness re-run.
10. **Domain sets:** CUAD, DocFinQA, SlideVQA loaders; run the reasoning spine + modality sweep + retrieval-quality covariate on each.
11. **Model sizes:** add 3B and 32B (same family); repeat key sweeps; full-doc on 32B only where
    runnable (confirm V100 feasibility — large multi-image inputs degrade past ~10–20 images).
12. **Hallucination + cost**; **LongDocURL replicate**; **statistics** (CI, effect size);
    assemble decomposition tables, degradation curves, sufficiency frontiers, deployment table.
    Optional: MMESGBench in-between robustness; CiteVQA cross-domain check.

**Stage 2 acceptance:** judge agreement reported & acceptable; center cell reproduces; full
decomposition + all three slices + modality sweep + grid + 3 drills populated; every
modality/extraction claim reported as best-in-class + spread + per-evidence-type + hedged;
null drill-cells reported as deliberately-not-run with rationale; adding LongDocURL needed only
a loader.

---

## 9. Rigour & safeguards

- **Uniform re-scoring** dissolves cross-dataset metric incomparability (we never compare
  published numbers; we re-run one scorer over each dataset's documents).
- **Confound disclosure:** for every domain set, report length distribution + question-type mix
  up front; match domain comparisons on length/question-type where possible, report descriptively
  (confounds stated) where not. Domain effect stated as mediated by evidence-type composition.
- **Memory-guessing check:** identify and set aside questions answerable without the document.
- **Generation ≠ evaluation:** for AI-built datasets (MMESGBench, CiteVQA), keep the generating
  model out of the evaluated-model and judge loops.
- **Fork-rationale for drills:** present extraction-method results as modality-sweep-everywhere +
  three targeted drills, each with an explicit "no scan → OCR moot / no table → layout moot /
  no vision-need → visual-rep moot" rationale; report null cells as deliberately-not-run.
- **Implementation-bounded claims:** modality/extraction/retrieval findings carry the
  "as of the implementations tested" hedge; claims bounded to "controlled evidence on
  MMLongBench-Doc (+ LongDocURL / domain sets) that ...", never "we prove in general that ...".

---

## 10. Open decisions to confirm before Stage 2

- **Model family + sizes:** one family across 3B / 7B / 32B (e.g. Qwen2.5-VL series). Confirm a
  clean trio exists in one family; confirm 32B full-doc is runnable on available V100s or scope
  it to retrieved/oracle pages.
- **In-between dataset:** DocFinQA primary; MMESGBench multimodal-native second/robustness.
- **Parser choice:** name the primary PDF→Markdown/HTML tool + version; name the secondary
  parser for the robustness re-run.
- **Reproduction target:** pin the exact published Qwen2.5-VL number used for the center-cell gate.