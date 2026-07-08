mpvrdu/
├── README.md                     how the pipeline runs (mechanism ground truth)
├── CLAUDE.md                     coding-agent rules (+ "docstrings describe current function only")
├── __init__.py                   root package marker
├── config.py                     ExperimentConfig: paths, resolution presets, per-size token caps
├── schema.py                     frozen data contracts (Question, Page, Payload, Prediction, Score, telemetry)
├── requirements*.txt             env pins (Kaya main / local Blackwell / annotate)
│
├── data/                         dataset layer: load, label, render
│   ├── loader.py                 dataset rows → Question; answerable/unanswerable partitioning
│   ├── binning.py                bin_label lookup from manual annotations (per-document modality bin)
│   ├── annotations.py            read/validate the human doc-label table (bin/scan/dominant_visual)
│   └── render.py                 PDF → per-page PNG + embedded-text spans (PyMuPDF)
│
├── tools/                        per-page channel builders (the ladder's raw materials)
│   ├── text.py                   cheap embedded-text extraction (the T channel)
│   ├── parser.py                 PDF-parser layout-rich text for TL/TLV (PaddleOCR-VL / MinerU / Unlimited)
│   └── visual.py                 page-image channel + vision-token estimation (resolution lives here)
│
├── retrievers/                   page retrievers (formerly covariates/retriever.py)
│   ├── text.py                   lexical/dense text retrieval cost rungs (BM25 / BGE-M3 / Qwen3-Embedding)
│   ├── vision.py                 visual retrieval cost rungs (ColModernVBERT / ColQwen2.5 / ColQwen3)
│   └── joint.py                  post-hoc deduplicated union of a text + a vision page set
│
├── models/                       reasoner backends + the doc-type classifier, behind registries
│   ├── __init__.py               get_reasoner registry (name → backend)
│   ├── local_vlm.py              Qwen3-VL backend (size/quant/resolution variants)
│   ├── internvl.py               InternVL backend (second model family)
│   ├── classifier.py             doc-type/bin classifier (formerly covariates/classifier.py)
│   └── payload.py                backend-neutral prompt/image container (ModelInput)
│
├── pipeline/                     the five frozen stages of ONE cell (this name stays narrow)
│   ├── conditioner.py            page-selection policy (oracle / retrieved / similarity)
│   ├── representation.py         T/TL/TLV/V composer (cost-ordered; parser text; no bbox)
│   ├── reasoner.py               Reasoner ABC (backend-agnostic)
│   ├── judge.py                  scoring interface + API judges (answerable acc + abstention)
│   └── orchestrator.py           composes the five stages; owns the two cache layers + telemetry capture
│
├── scoring/                      turn cached cells into numbers (formerly metrics/ + live bits of gates/)
│   ├── accuracy.py               document-level accuracy + bootstrap CIs
│   ├── cost.py                   token / latency / VRAM aggregation (prefill/decode split)
│   ├── frontier.py               sufficiency-frontier rule over cost-ordered representations
│   ├── retrieval.py              page precision / recall / F1 (per bin, per method)
│   ├── abstention.py             abstention detection for the unanswerable/hallucination study
│   └── agreement.py              judge–human κ (retired F2's computation, kept as reported metric)
│
├── experiments/                  what runs, how it runs, what it runs on
│   ├── tasks/                    the generation tasks (mechanism-named, not G-numbered)
│   │   ├── base.py               GenerationTask ABC + shared cell factories
│   │   ├── oracle_ladder.py      oracle pages × {T,TL,TLV,V}; base grid for all RQ1 sweeps
│   │   ├── retrieval.py          retrieved pages × TLV/V × method × k; matched/cross + k-sweep
│   │   ├── hallucination.py      unanswerable × similarity pages × prompt condition
│   │   └── classifier_pricing.py side-only: prices the classifier (no reasoner cells) for routing
│   ├── engine/                   the run machinery
│   │   ├── driver.py             generate+judge loop: pre-pass, construction, cache writes, telemetry
│   │   ├── side_artifacts.py     shared side-artifact writers (retrieval benchmark, classifier logs)
│   │   ├── artifacts.py          artifact-driven judge/build helpers (settings persist across roles)
│   │   └── paths.py              cache/table path layout (keys include parser/res/model/quant/prompt)
│   ├── corpus/                   what-to-run-on resolution
│   │   ├── resolve.py            question-set resolver (answerable/unanswerable, replication subsets)
│   │   ├── smoke.py              reproducible doc-level subset for --smoke
│   │   └── yaml_spec.py          YAML spec → dynamic tasks (parser/res/quant/k/prompt/size sweeps)
│   └── registry.py               task-name → task collection (keeps YAML-first swapping)
│
├── reporting/                    judged rows → paper tables
│   ├── build.py                  explicit task→table routing (one task may feed many tables) + CSV/md write
│   └── tables/                   one builder per table, content-named (no T#)
│       ├── _common.py            shared helpers: bin order, representation order, telemetry columns, filters
│       ├── _markdown.py          markdown rendering of built CSVs
│       ├── headline.py           cost-ordered T/TL/TLV/V × bin (answerable-only)
│       ├── parser.py             parser comparison at TL/TLV
│       ├── resolution.py         image-resolution sweep
│       ├── matched_cross.py      retrieval matched-vs-cross across bins
│       ├── kdepth.py             top-k retrieval sweep (+ joint union condition)
│       ├── retrieval_accuracy.py page-F1 benchmark per bin/method (side-artifact scorer)
│       ├── hallucination.py      abstention × prompt condition
│       ├── routing.py            routing policies (reuses ladder rows + classifier price)
│       ├── scale.py              model-size / quantization cost-frontier
│       └── composition.py        evidence-source composition (secondary/appendix)
│
├── cli/                          the three runnable roles (only user entry points)
│   ├── generate.py               GPU generation from YAML specs (collects full telemetry)
│   ├── judge.py                  read predictions → judge → results
│   └── build.py                  route judged rows → table CSVs
│
├── ops/                          cluster + operational tooling (grouped to keep root legible)
│   ├── kaya/                     SLURM sync/submit runner + Kaya guides + config.json
│   ├── specs/                    YAML specs (template + saved run configs)
│   └── scripts/                  standalone utilities (see below)
│       ├── annotate_docs.py      manual document-label tool (writes the bin/scan/visual table)
│       ├── inspect_results.py    cached-cell viewer (absorbs the old gates/viewer.py)
│       ├── prestage.py           stage datasets/reasoners/retrievers/parsers (isolated parser envs)
│       ├── download_hf.py        HF snapshot/file staging helpers
│       ├── dataset_stats.py      descriptive dataset report generator
│       ├── profile_datasets.py   per-dataset table-readiness profiler
│       ├── run_probe.py          early feasibility probes
│       ├── gpu_test.py           minimal SLURM GPU sanity job
│       ├── kaya_status.py        read-only cluster status
│       ├── setup_env.py          login-node conda/env bootstrap
│       └── dump_docstrings.py    regenerates the per-file map in docs/
│
├── docs/                         authored prose only
│   ├── README-index.md           precedence order of the docs (who wins on what)
│   ├── PROJECT_SPEC.md           what/why: thesis, RQs, setup
│   ├── USER_GUIDE.md             how to run it locally
│   ├── AGENT_GUIDE.md            frozen interfaces + tree→paper map
│   ├── DECISIONS.md              changelog of pivots (pivot_v4 folded in here once applied)
│   ├── REPO_STRUCTURE.md         this tree + auto-generated per-file map
│   └── generated/                script outputs, NOT hand-written
│       ├── dataset_stats.md
│       ├── dataset_label_distributions.csv
│       ├── all_tables.md
│       └── questions.md
│
├── tests/                        pytest suite (unit + skeleton + e2e + docstring convention)
│
└── [gitignored]                  envs/ .cache/ .data/ results/ logs/ __pycache__/ .agents/ .vscode/ ...