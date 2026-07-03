# Project Specification — Vision-Sufficiency and Representation Choice in Multi-Page Document Understanding

> Full standalone spec (v2). Read with `context.md` (decision history), `dataset_profile.md`
> (real per-dataset field availability), and `MP-VRDU_Survey.md` (the survey whose open
> challenges this study addresses). This document is authoritative for what to build and what
> to run. Dataset claims here are grounded in the actual profile report, not assumptions.
> This version supersedes v1: the four-RQ-plus-ablations structure is replaced by nine
> research questions in three groups, and routing and abstention are promoted to full RQs.

---

## 1. Background

Real-world documents (contracts, financial filings, ESG reports, technical manuals) routinely
span tens to hundreds of pages. Most visually-rich document understanding (VRDU) research
assumes a single page and reports a single answer-accuracy number that conflates two distinct
failures: failing to **locate** the answer-bearing pages, and failing to **reason** correctly
over them once found. This conflation hides *why* a system fails and therefore *what to fix*.

The cost of guessing is highest under a concrete deployment constraint: sensitive documents
cannot be sent to cloud APIs and must be processed **on-premise with small, locally hosted
vision-language models (VLMs)** in the 3B–32B range. There, the most expensive single design
choice is whether to encode pages as **images** at all (vision encoding dominates GPU cost),
yet that choice is currently made without evidence.

## 2. Research gap (grounded in the MP-VRDU survey)

The companion survey (`MP-VRDU_Survey.md`) organises MP-VRDU systems into four architectural
families (page-to-document encoders, MLLM-centric adaptation, retrieval-augmented pipelines,
adaptive-trajectory/agentic pipelines) and concludes that **progress repeatedly relocates the
bottleneck rather than removing it**, with no controlled, like-for-like measurement of where
difficulty actually lies. Three of the survey's five named open challenges are *measurement*
problems that a controlled empirical study is the right instrument for, and this study targets
exactly those:

- **Efficiency and deployability.** The survey notes the literature reports almost no
  inference-cost analysis (latency, token budgets, accuracy–cost Pareto curves are "largely
  absent") and explicitly flags compact models under on-device/privacy constraints as future
  work. → Addressed by RQ1–RQ5 (the deployment lens), with cost promoted to a first-class axis
  (RQ4).
- **Evaluation methodology and scope.** Answer-only accuracy masks retrieval failure and
  rewards correct answers reached by incorrect reasoning. → Addressed by the
  locating-vs-reasoning decomposition (RQ7) and the question-type/evidence-modality view (RQ6).
- **Supervision integrity and contamination.** Benchmarks reuse documents (MMLongBench-Doc
  draws from DUDE/SlideVQA/ChartQA) and are often judged by the same MLLM family. → Addressed
  by uniform re-scoring with a different-family judge, a memory-guessing check, and cross-suite
  overlap disclosure (Section 8).

The study is therefore positioned as **the controlled measurement layer the survey says the
field is missing**: it does not propose a new method, but produces per-document-type,
per-representation, cost-aware measurements that future method work (any of the four families)
can build on.

## 3. What the study is (and is not)

- **Is:** a document-**understanding** analysis. Primary axis = **reasoning** (accuracy given
  evidence in a controlled representation), studied across document type, question type,
  representation, model size, and **cost**, and extended to two deployment decisions a
  practitioner actually faces: whether to **route** by document type, and whether the system
  **abstains** when evidence is missing.
- **Covariates (measured, not studied):** retrieval quality (page-F1) and document-type
  classifier accuracy. Both are instruments that let us attribute error and bound the value of
  a deployment decision.
- **Is not:** a retrieval/RAG benchmark, a new method, a new dataset, or a per-domain
  architecture study. Routing is studied only at the level of the *representation* decision,
  not at the level of swapping models or pipelines per domain.

## 4. Experimental setup

### 4.1 Input conditions (control which pages reach the model)

| Condition | Pages fed | Role |
|---|---|---|
| **Oracle** | exactly the gold evidence pages | the reasoning measurement (primary) |
| **Retrieved top-k** | k pages from a real retriever (k∈{1,3,5}) | realistic performance + decomposition + abstention |
| **Full document** | all pages (where context allows) | feed-everything baseline |

Decomposition: Oracle = reasoning ceiling; (Oracle − Retrieved) read through retriever page-F1
= locating- vs reasoning-attributable error; (Oracle − Full-doc) = distraction/long-context
penalty.

### 4.2 Representation — two levels

**Modality (primary cumulative sweep, run across the design):**

| Condition | Channels | Type |
|---|---|---|
| `T` | raw text | reference |
| `T+L` | text + layout/structure | cumulative |
| `T+L+V` | text + layout + visual | cumulative |
| `V` | visual only | reference |

**Extraction method (representation-recovery ablations, RQ8–RQ9)** — varied only where the
method choice can change the outcome; null cells reported as deliberately-not-run with
rationale:
- **RQ8 (text):** embedded (PyMuPDF) vs OCR (PaddleOCR), read as the cost of always using OCR
  (born-digital control vs scanned/degraded slice).
- **RQ9 (layout):** geometry (PP-StructureV3 bbox) vs structure (Docling markdown) on
  table/layout-evidence questions; plus, parse held fixed, serialisation format
  Markdown vs HTML vs XML.
- **Visual granularity** (full-page vs region-crop vs resolution) is folded into the cost
  analysis (RQ4) rather than run as a standalone ablation, since it is inseparable from token
  cost.

**Modality-boundary rule:** text/layout extraction is **modular (non-VLM)** only; the VLM is
confined to the visual condition. Parser held fixed for main results (Docling); one re-run with
Marker confirms findings are not a parser artifact.

### 4.3 Models

Qwen2.5-VL **3B / 7B / 32B** (one family; 7B = center). vLLM/HF on Tesla V100s. Confirm 32B
full-document feasibility early; otherwise scope 32B to oracle/retrieved. Judge = a different
family; report judge–human agreement on ~100 hand-labelled items before trusting any number.

### 4.4 Tools per representation method

| Modality | Method | Tool | License |
|---|---|---|---|
| Text | embedded | PyMuPDF (raw) | BSD |
| Text | OCR | PaddleOCR classic PP-OCRv5 | Apache-2.0 |
| Layout | markdown | Docling (primary) | MIT |
| Layout | markdown (2nd parser) | Marker (robustness) | restricted* |
| Layout | bbox geometry | PaddleOCR PP-StructureV3 | Apache-2.0 |
| Visual | page image | Qwen2.5-VL native | Apache-2.0 |
| Doc-type classifier (RQ3) | — | cheap VLM/LLM pass (same family acceptable) | Apache-2.0 |
| Retrieval (vision) | — | ColQwen / ColPali | — |
| Retrieval (text) | — | BM25 + BGE | — |

*Primary recommendations (PyMuPDF, PaddleOCR, Docling, Qwen2.5-VL) are permissively licensed for
business use; Marker/Surya carry commercial-use restrictions and are used only internally.

### 4.5 Datasets — roles grounded in the profile report

**MMLongBench-Doc is the PRIMARY dataset and the basis of all initial experiments.** It is the
only dataset carrying *both* a document-type/domain label and evidence-modality labels, so the
full design (decomposition, domain slice, question-type slice, representation sweep, routing) is
built and validated here first. LongDocURL and the three domain sets provide robustness and
additional evidence.

| Dataset | Role | Key real fields (from profile) | Can do | Cannot do |
|---|---|---|---|---|
| **MMLongBench-Doc** | PRIMARY (general) | `doc_type` (7 domain classes), `evidence_sources` (text/layout/table/chart/figure), `evidence_pages`, `answer_format`; unanswerable via answer=`"Not answerable"`; PDFs via `doc_id` | domain slice, evidence-modality slice, derived-hop, locate/decomposition, representation sweep, routing, abstention | needs PDF rendering (no shipped images/text) |
| **LongDocURL** | robustness; question-type replicate; in-page boxes for crop | `task_tag`, `question_type` (9 classes), `evidence_pages`, `total_pages`, page `images`, in-page `<box>` coords in `detailed_evidences` | question-type robustness, locate/decomposition, vision, oracle-region crop (RQ4) | no domain label |
| **CUAD** | text-heavy domain backup | `clause_category` (41), `is_impossible` (real unanswerable), `answer_start/text` spans, `context` text | text reasoning, representation (text/layout), abstention | no evidence pages → no locate; no images |
| **DocFinQA** | in-between (financial) backup | `Question`, `Answer`, long `Context`, `Program` | text reasoning on long financial filings | no labels, no pages → no slicing, no locate; render needed for vision |
| **SlideVQA** | visual-heavy domain backup | native `evidence_pages`, per-slide `page_1..20` images, `arithmetic_expression` (numeric flag) | locate/decomposition, vision, numeric-question slice, abstention (retrieval-miss) | no question-type/domain label |

Cross-dataset facts: **hop is derived everywhere** from `len(evidence_pages)` (single=1,
multi≥2). Unanswerable is native only in CUAD (`is_impossible`) and via the answer string in
MMLongBench-Doc. Page images are shipped/pointed-to by MMLongBench-Doc (render), LongDocURL,
SlideVQA; CUAD/DocFinQA are text-only. In-page bounding boxes for oracle-region crops exist only
in LongDocURL.

### 4.6 Metrics & protocol

- **Answer accuracy:** one uniform LLM-as-judge (MMLongBench-Doc protocol: generate → extract →
  score), applied identically to every dataset so columns are commensurable; native metrics
  unused. Mean ± 95% CI, effect sizes on headline comparisons.
- **Retrieval covariate:** page Recall / Precision / F1 vs gold evidence pages; best-in-class +
  spread, hedged "as of the implementations tested."
- **Classifier covariate (RQ3):** document-type classification accuracy of the cheap predicted-
  routing pass, logged alongside routed accuracy.
- **Abstention (RQ5):** abstention rate and hallucination rate on (a) natively unanswerable
  questions and (b) answerable questions whose gold page the retriever missed (page-recall = 0).
- **Cost (RQ4):** tokens (text/visual) and latency per question per representation, reported as
  an accuracy–cost Pareto frontier per document-type group.
- **Sufficiency frontier:** cheapest condition within a pre-registered margin of, and not
  significantly worse than, the best.

## 5. Research questions & hypotheses

Nine RQs in three groups. **Deployment lens (RQ1–RQ5):** what a practitioner should build.
**Mechanistic view (RQ6–RQ7):** where difficulty originates, which the deployment answers are
mediated by. **Representation recovery (RQ8–RQ9):** how each non-visual modality is best
recovered.

- **RQ1 (headline, vision-sufficiency).** Is visual encoding required, and does it depend on
  document type? *H1a:* text-heavy frontier at `T`/`T+L`. *H1b:* visual-heavy `T+L+V` ≫ `T+L`,
  gap larger than text-heavy. *H1c:* in-between frontier genuinely uncertain.
- **RQ2 (scaling).** Does the frontier move with model size? *H2:* as size grows the frontier
  shifts cheaper and the bottleneck migrates reasoning→locating.
- **RQ3 (routing vs uniform).** Without document-type labels, is it better to route (classify,
  then apply the per-type frontier) or to apply one representation uniformly? *H3:* routing pays
  off only when per-type frontiers genuinely differ AND the classifier is accurate enough that
  misrouting does not erase the specialisation gain.
- **RQ4 (cost).** What is the accuracy–cost trade-off of each representation, and how should
  visual pages be encoded under a token budget? *H4:* the sufficiency frontier rarely coincides
  with the accuracy-maximising condition; region-crops/higher resolution beat full-page per
  visual token where vision is needed.
- **RQ5 (abstention).** When evidence is not retrieved, does the model abstain or hallucinate?
  *H5:* models abstain rarely; abstention depends on representation and document type.
- **RQ6 (question type).** Which question types need which representations? *H6:* multi-hop and
  numeric questions show larger reasoning gaps; chart/figure-evidence questions benefit most
  from vision regardless of document type.
- **RQ7 (decomposition).** When end-to-end accuracy is lost, is it locating or reasoning? *H7:*
  retriever page-F1 lower for visual-heavy docs and multi-hop questions → locating bottleneck;
  text-heavy is reasoning-dominated.
- **RQ8 (text recovery).** Is OCR a safe default for the text channel? *H8:* OCR ≈ embedded on
  born-digital, degrades gracefully elsewhere, making it a defensible uniform default.
- **RQ9 (layout recovery).** Does the representation of structure matter? *H9:*
  geometry-vs-structure matters most on table/layout evidence, at least one closing much of the
  gap to vision; serialisation format (MD/HTML/XML) of the same parse matters comparatively
  little.

## 6. Skeleton results tables (rows reflect what each dataset can actually support)

Headline and slicing tables run on **MMLongBench-Doc** (the only set with domain + modality
labels); robustness/extra-evidence rows come from the other datasets where their fields allow.
Dashes = to be filled. Frontier marked ★. Table numbering assumes a dataset-overview table as
Table 1.

| Table | Maps to | Content |
|---|---|---|
| Dataset overview | §4.5 | dataset roles & capabilities |
| **T1** | RQ1 | HEADLINE: vision-sufficiency by document type (7B, Oracle) |
| **T2** | RQ2 | scaling: frontier vs model size (3B/7B/32B, Oracle) |
| **T3** | RQ3 | routing vs uniform under unknown document type (+ classifier acc.) |
| **T4** | RQ4 | accuracy–cost: (a) modality cost, (b) visual granularity |
| **T5** | RQ5 | abstention vs hallucination (native unanswerable + retrieval-miss) |
| **T6** | RQ6 | question-type analysis (evidence modality + hop) |
| **T7** | RQ7 | decomposition: Oracle/Retrieved/Full-doc + page-F1 + bottleneck |
| **T8** | RQ8–RQ9 | representation-recovery ablations (text OCR-default; layout geom/struct + serialisation) |
| **T9** | robustness | Docling vs Marker, LongDocURL replicate, domain-backup corroboration |

(The exact LaTeX skeletons live in `tables/` and are the authoritative cell layout.)

## 7. Deployment lens (the synthesis the paper is built around)

The deployment group (T1–T5) combines into a per-document-type **deployment recommendation**:
- **Skip vision?** (T1, T4) If the frontier sits at `T`/`T+L` for a document type, visual
  encoding is unnecessary there — a direct GPU-cost saving, quantified against the cost axis.
- **Route or go uniform?** (T3) Per-document representation routing is worth its classifier
  overhead only if the per-type frontiers differ and the classifier is accurate enough.
- **Build retrieval?** (T7) Where page-F1 is low and the bottleneck is locating, retrieval is
  the binding constraint worth engineering; where reasoning-dominated, it is not.
- **Buy a bigger model?** (T2) Whether the recipe changes with affordable size.
- **Will it fail safely?** (T5) Whether the system abstains rather than hallucinates when
  evidence is missing.
- **Confound disclosure:** document type and evidence-modality are correlated; report both the
  document-type view (deployment-facing) and the evidence-modality view (mechanistic, T6), and
  state the domain effect is largely **mediated by** evidence-type composition.

## 8. Rigour & safeguards

- Uniform re-scoring dissolves cross-dataset metric incomparability (one scorer; no comparison
  of published numbers).
- Per-dataset length and question-mix profiles reported up front (confound disclosure); domain
  comparisons matched on length/question-type where possible, else reported descriptively.
- Memory-guessing check: identify and set aside questions answerable without the document.
- Generation ≠ evaluation: judge from a different family than evaluated models; for any
  AI-assisted data, keep the generator out of the judge/evaluated loop.
- Contamination disclosure: declare cross-suite document overlap (e.g. MMLongBench-Doc ⊃
  DUDE/SlideVQA/ChartQA documents) per the survey's supervision-integrity challenge.
- Representation-recovery null cells reported as deliberately-not-run, with the fork-existence
  rationale (no scan → OCR-default moot; no table → layout moot).
- Implementation-bounded claims: tool/representation/retrieval/classifier findings carry the
  "as of the implementations tested" hedge; conclusions bounded to "controlled evidence on
  MMLongBench-Doc (+ replicates) that …", never "we prove in general that …".
- Out-of-scope (inherited SP-VRDU gaps, named in Limitations, not addressed): multilingual
  layouts, summarisation/structured-extraction beyond VQA, streaming page-by-page inputs.

## 9. Open items to pin before main experiments

- Exact published Qwen2.5-VL number for the reproduction gate.
- Pre-registered sufficiency margin (e.g. within 2 points and not significantly worse).
- Pre-registered judge–human agreement bar.
- Confirm a clean Qwen2.5-VL 3B/7B/32B trio in one family; else pick the family that has it.
- MMLongBench-Doc `doc_type` → text/in-between/visual spectrum mapping for T1, agreed up front.
- **Confirm whether MMLongBench-Doc PDFs contain genuinely scanned documents** (decides whether
  RQ8's degraded slice is real or must be synthetically degraded; affects T8).
- **Confirm doc-type classifier choice for RQ3** (cheap same-family VLM pass is acceptable since
  the classifier is a covariate, not an evaluated model) and pre-register the abstention
  definition for RQ5 (what counts as an abstention vs a wrong answer).
