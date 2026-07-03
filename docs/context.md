# Context — MP-VRDU Representation & Deployment Study (conversation summary)

> Living context file for new chat sessions. Summarises how the study was scoped and the
> decisions that are now fixed. Read alongside `PROJECT_SPEC.md` (the full spec),
> `dataset_profile.md` (real dataset field availability), and `IMPLEMENTATION_PLAN.md` (the
> staged build plan). Where this file and older notes disagree, this file is current.

## What the study is

An empirical analysis of multi-page visually-rich document understanding (MP-VRDU). A system can
fail at two stages: **locating** the pages relevant to a question, and **reasoning** over them
once found. Answer-only accuracy conflates the two. The representation in which page content is
supplied to the model (raw text, `+`layout, `+`visual, visual-only) threads through both stages
and dominates cost on locally hosted models. The study measures, in a controlled way, what
representation each stage requires and what this implies for deployment.

Framing motivation: privacy-constrained, on-premise deployment with small local VLMs, where
encoding pages as images is the dominant cost and is currently chosen by guesswork. The headline
deliverable is a per-document-type **vision-sufficiency** result plus the deployment guidance that
follows from it.

## Study structure (now fixed: three topics, not a flat RQ list)

The paper is organised around **three topics**, each a `\subsection` under a `Study Design`
section, with an `Experimental Architecture` section of its own. The old "seven flat RQs" and the
intermediate "two lenses" framings are dropped; they collapsed into one umbrella (what
representation does each stage need, and what follows for deployment) with three facets.

- **Representation** (what *reasoning* requires). Oracle pages isolate reasoning. Facets:
  (a) sufficiency by document type — sweep `T`/`T+L`/`T+L+V`/`V`, locate the sufficiency frontier
  per document-type group (the headline); (b) evidence-modality mediation — re-slice by
  `evidence_sources` and hop count, showing the document-type effect is mediated by evidence-type
  composition; (c) cost of representation — accuracy vs text/visual tokens and latency, plus visual
  granularity (full-page / region-crop / resolution) at a matched token budget.
- **Retrieval** (what *locating* requires). Reports the *mechanism* only, not deployment
  thresholds. Facets: (a) retrieval-modality sufficiency — page R/P/F1 for text (BM25+BGE) vs
  vision (ColPali/ColQwen) retrievers, sliced by evidence modality; (b) retrieval–reasoning
  modality divergence — cross-tabulate the modality that locates against the modality that reasons;
  the off-diagonal (e.g. a chart answer retrieved via its caption) is the novel result.
- **Deployment** (synthesis). Reads the two mechanistic findings through a deployer's decisions,
  priced in cost/latency. Facets: (a) locate-vs-reason attribution (oracle/retrieved/full-doc gap
  read through page-F1); (b) bottleneck migration across model size; (c) sufficiency of retrieval —
  how good retrieval must be, incl. the precision–recall / distractor-burying sweep; (d) routing vs
  uniform under unknown document type; (e) fail-safe abstention under missing evidence.

**Scoping rule that resolves the ordering:** "how good must retrieval be" is a *deployment*
question and lives in Deployment, not Retrieval. Retrieval reports only the modality phenomenon.
This is why the order Representation → Retrieval → Deployment reads without leaking deployment
content into the middle section.

## Key conceptual decisions (fixed)

- **Two stages, one representation ladder threaded through both.** Representation asks the ladder of
  reasoning; Retrieval asks the same ladder of locating. The novel cross-cut is that the two can
  **diverge** — the modality sufficient to locate evidence need not match the modality sufficient to
  reason over it (caption example: text-locatable, vision-reasoned).
- **Locating is a covariate for the mechanism, a decision for deployment.** We measure retriever
  page-F1 directly and read the Oracle−Retrieved gap through it (attribution). 
- **Distractor-burying is now IN SCOPE.** *This reverses the earlier "do NOT bury" decision.* The
  earlier caution was that burying drifts into long-context/RAG-benchmark territory. The resolution
  is framing: burying is the **instrument** for the deployment question "how much irrelevant
  retrieved context can the model tolerate / at what length does retrieval stop being optional,"
  not a retrieval benchmark. Gold evidence is held present; distractor pages are drawn from the
  same corpus; Full-doc is the limiting case. Recorded here so this file and the spec agree.
- **Representation is cumulative** (`T` → `T+L` → `T+L+V`) with `V` as a parser-independent
  reference, giving the marginal value of each added modality.
- **Modality-boundary rule.** Text/layout are produced by modular, non-VLM tools only; the VLM is
  confined to the visual condition. Otherwise VLM-produced "OCR text" would embed visual
  understanding and could not be separated from `V`. Also matches the deployment story (modular =
  cheaper, auditable, permissively licensed).
- **Prompt sensitivity is not its own topic.** A single fixed prompt is held constant for
  commensurability; prompt variation appears only as the manipulation inside the Deployment
  abstention facet (neutral vs abstention-licensed) and as a robustness check on the headline.
- **Cost is object vs currency.** Cost is *interrogated* in Representation (what does each
  representation cost); in Deployment it is the *currency* the decisions are priced in (routing pays
  iff gain > classifier cost, etc.). No double Pareto analysis.

## Tool stack (fixed, deployable, mostly permissive)

- Text — embedded: **PyMuPDF** (BSD). OCR: **PaddleOCR classic PP-OCRv5** (Apache-2.0), not a VLM.
- Layout — markdown: **Docling** (MIT, primary) / **Marker** (restricted, internal robustness only).
  bbox geometry: **PaddleOCR PP-StructureV3** (Apache-2.0).
- Visual — **Qwen3-VL native** image input.
- Retrieval covariate — vision: **ColPali/ColQwen**; text: **BM25 + BGE**.
- Doc-type classifier (routing covariate) — a cheap model pass; a small model is fine since it is a
  covariate, not an evaluated model.

## Models (updated: Qwen3-VL, swappable family)

- Evaluated reasoner: **Qwen3-VL**, sizes **2B / 4B / 8B / 32B** (one family; 8B = center cell).
  *This replaces the earlier Qwen2.5-VL choice — Qwen3-VL offers more sizes for the scaling story.*
- The family must be **swappable** behind one `Reasoner` interface, via two backends: local-weights
  (vLLM/HF) and HTTP-API. This lets us later substitute open families (InternVL, Gemma) or closed
  APIs (GPT, Gemini) with no pipeline changes.
- **Closed models are for methodological comparison and as judges only, not for the deployment
  recommendation**, which stays bounded to locally-hostable open models per the privacy framing.
- Judge: a different family from the evaluated model; report judge–human agreement before trusting
  numbers.

## Datasets (updated: MMLongBench-Doc only for v1)

- **MMLongBench-Doc — the ONLY dataset used in v1.** It is the only benchmark with BOTH domain
  labels (`doc_type`, 7 classes) AND evidence-source/modality labels, plus gold evidence pages and
  a native unanswerable signal (answer = `"Not answerable"`). It alone supports every facet of all
  three topics, so the entire study is built and validated on it.
- **All other datasets are OUT OF SCOPE for v1** (optional later extension, time-permitting; they
  strengthen robustness but are not needed for the claims):
  - **LongDocURL** — headline replicate; also the only source of true in-page boxes (for a proper
    region-crop re-run of the granularity facet).
  - **CUAD** — text-heavy backup; real unanswerable (`is_impossible`); no evidence pages.
  - **DocFinQA** — in-between (financial) backup; no labels, no pages.
  - **SlideVQA** — visual-heavy backup; native evidence pages + per-slide images.
- Key cross-dataset fact retained: **hop is derivable everywhere** from evidence-page count
  (single = 1, multi ≥ 2).

## Implementation posture

- Staged build (`IMPLEMENTATION_PLAN.md`): each stage is self-contained, ends in a human
  checkpoint, then `/compact` before the next. Skeleton (interfaces + runnable stub) is frozen
  before any real tool or model is plugged in; tools and model backends fill implementations
  behind frozen interfaces.
- Clean-but-not-fragmented modular codebase whose file tree mirrors the paper's architecture
  (input-conditioning / representation / reasoning / scoring stages + retrieval and classifier
  covariates). Plain-Python config, small CLI scripts, no build systems or YAML.
- Cheapest-discovery-first: Stage 1 confirms the code-checkable open items (loader, scanned-vs-
  born-digital, model trio + backend swap, vision-retrieval feasibility on V100, unanswerable
  count, doc_type distribution) before heavier stages depend on them.
- DUDE was evaluated and rejected earlier (no per-question domain field or evidence-page list);
  MMLongBench-Doc is the primary for exactly the labels DUDE lacked.

## Paper

- Empirical-analysis paper (ACL/ARR). Sections: Intro → Related Work → Study Design (the three
  topics as subsections) → Experimental Architecture → Experimental Setup → Results (mirroring the
  three topics) → Discussion/Deployment → Limitations → Conclusion.
- Study Design subsections use bold run-in facet labels, not a bold-RQ format; a scannable
  enumeration of the three topics sits in the Study Design lead-in and/or the Introduction.
- Headline: per-document-type vision-sufficiency (Qwen3-VL, oracle pages), sufficiency frontier
  marked, vision-needed verdict per type; second headline is the retrieval–reasoning modality
  divergence; the bottleneck-migration-with-scale story is the deployment through-line.