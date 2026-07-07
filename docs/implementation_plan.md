# Implementation Plan — A Doc-Type Recipe for Multi-Page Document QA (v3)

> A staged build plan for a coding agent. This document is the single source of truth for
> *how* to build the codebase; `docs/USER_GUIDE.md` (the v3 mirror) is the source of truth for
> *what* the experiments are and *why*. Read both, plus `docs/AGENT_GUIDE.md` and
> `dataset_stats.md`, before starting any stage.

---

## 0. How to use this plan

**Audience.** A strong coding agent (~258k context) working under human supervision, on a
codebase where the skeleton, feasibility probes, data layer, frozen interfaces, and the Kaya
execution substrate are **already implemented** (logged in `docs/AGENT_GUIDE.md`). This plan governs
the remaining work, restructured into three sections: an MVP that proves the whole pipeline runs
fast, the full v3 runs, and optional extensions.

**Operating loop (mandatory).**
1. At the start of every stage, (re-)read this file in full and the relevant parts of
   `docs/USER_GUIDE.md`, `dataset_stats.md`, and the latest `docs/AGENT_GUIDE.md`.
2. Implement exactly the one stage named, nothing from later stages.
3. Produce the stage's **deliverables, docs, and tests**; update `docs/AGENT_GUIDE.md` with
   implementation-relevant deviations, migrations, operational notes, and findings from the run;
   run the tests; report results.
4. **Stop at the human checkpoint.** Summarise what was built, what the tests showed, and any
   surprises that should change later stages.
5. After the human approves, run `/compact`, then begin the next stage from step 1.

**Why staged + compact.** Each stage is self-contained and re-readable from this plan plus the
code already on disk. Do not rely on un-compacted conversational memory: if a decision, finding,
deviation, migration, or operational note matters for a later stage, write it into code comments
or `docs/AGENT_GUIDE.md`.

**Golden rules.**
- **Interfaces are frozen.** The Stage 3 contracts (`schema.py`, the pipeline ABCs, `ModelInput`)
  do not change. Pressure to change them is a checkpoint discussion recorded in `docs/AGENT_GUIDE.md`,
  never a silent edit. This invariant is what makes both the tooling and the model family
  swappable without rework.
- **Concise, not fragmented.** Prefer a handful of cohesive modules over many tiny files. No build
  systems, no YAML/Makefiles. Configuration is plain Python (a dataclass); runs are driven by
  small CLI scripts. Every Python file opens with a comprehensive module docstring that states:
  what the file does; why it exists in the experimental architecture; its main public entry points
  or data contracts; and, for runnable scripts/CLIs, the command form plus every accepted argument
  (or explicitly says there are no command-line arguments for import-only modules).
- **The code mirrors the paper.** The four pipeline stages and two covariates from
  `docs/USER_GUIDE.md` are first-class objects with the same names. A reader who knows the paper
  should recognise the architecture in the file tree.

---

## 1. Scope decisions fixed for this build (v3)

The scope narrowed to a single EACL long-paper thesis: **representation is a function of document
type.** The build serves three research questions, each producing paper tables:

- **RQ1 = Recipe by doc type** — the cheapest representation that lets an 8B MLLM reason to an
  answer, per document type; the sufficiency frontier (Tables 1–4).
- **RQ2 = Mechanism** — the doc-type effect re-expressed as evidence composition, plus
  matched-vs-cross retrieval and the locate–reason modality divergence (Tables 5–6).
- **RQ3 = Routing under uncertainty** — does classifying documents and dispatching to the recipe
  beat a uniform policy once the classifier's own cost is counted (Table 7)? Scale sanity is an
  appendix (Table 8).

**Fixed pre-registered choices (mirror of `docs/USER_GUIDE.md` §6):**
- **Cost metric:** latency/question at batch=1 on one A100 80GB (primary); text/vision tokens
  (secondary).
- **Sufficiency margin:** accuracy drop ≤ 3 points vs the strongest representation; sensitivity
  ∈ {2,3,5} in appendix.
- **Doc-type binning (Option A):** text-heavy = Administration/Industry file + Academic paper +
  Research report/Introduction (578 Q / 54 docs); in-between = Financial report + Guidebook +
  Tutorial/Workshop (412 Q / 50 docs); visual-heavy = Brochure (101 Q / 15 docs). Lives in ONE
  function so the Section-3 Option-B (data-driven) binning can swap it without touching experiment
  code.
- **Ladder tooling:** `T` = Marker raw text; `T+L` = Marker text + serialized bbox JSON;
  `T+L+V` = Marker text + native-resolution page image; `V` = page image only. **v3 makes Marker
  the primary parser; PyMuPDF becomes the appendix parser-swap.** This differs from the earlier
  Docling-primary note in the built `tools/layout.py`; reconcile at the Section-1 checkpoint.
- **Reasoner:** Qwen3-VL-8B primary; InternVL3-8B replicates the RQ1 headline only; Qwen3-VL-2B/32B
  for appendix scale sanity.
- **Retrieval:** BM25+BGE-large (text), ColQwen (vision). RQ2 compares matched vs cross;
  vision-retrieval + text-reasoning is not tested.
- **Judge:** GPT-4o-mini (different family). Cohen's κ ≥ 0.75 vs 200 hand-labels before any main
  number is trusted.
- **Confidence:** bootstrap 95% CI (1000 resamples) on every headline number; **sampling and CIs
  are handled at the document level** (draw documents, take their questions) because questions
  cluster within documents (135 docs, 1091 Q) and question-level CIs overstate precision — this
  matters most for the thin visual-heavy bin.

**What is cut from the paper** (retained in the tree, moved to Section 3 as optional): the
distractor-burying sweep (the `BuriedOracle` conditioner stays but is unused by the paper), the
full retrieval-sufficiency frontier, fail-safe abstention, scaling-as-a-story, and multi-dataset
robustness beyond one replication.

**Already built (do not re-implement).** The repository skeleton, Stage-1 feasibility probes, the
Kaya Python runner + static config, the Stage-2 data layer (loader + render), and the Stage-3
frozen interfaces (`schema.py`, pipeline ABCs, `ModelInput`, caching contract) exist and are logged
in `docs/AGENT_GUIDE.md`. This plan begins at the MVP, which fills those frozen interfaces with real
tools and models.

---
## 2. Target architecture (the contract all stages honour)

Data flows: a `Question` (from the data layer) is resolved to pages by an `InputConditioner`,
encoded by a `Representation`, answered by a `Reasoner`, and scored by a `Judge`. A `Retriever`
and a `DocTypeClassifier` are covariates that feed/annotate the flow.

**The working directory is the `mpvrdu/` repo root.** The tree below is rooted there. Everything
the project produces or downloads (datasets, model weights, HF cache, the conda env, results,
logs) lives under this root, so the whole project is one portable, self-contained directory: you
can rsync it to another machine and run. Documentation lives in `docs/`; standalone dataset
profiling utilities live in `scripts/`; Kaya-specific config, scripts, and guides live in `kaya/`.
The git-ignored artifact dirs below
(`.cache/`, `.data/`, `envs/`, `results/`, `logs/`) hold the heavy, machine-specific stuff and are
also excluded from the Kaya rsync (see section 2b), so each machine keeps its own native copy.

```
mpvrdu/                 # == the working directory / repo root
  .cache/              # HF_HOME + torch hub                  (git-ignored, rsync-excluded)
  .data/               # datasets + rendered pages            (git-ignored, rsync-excluded)
  envs/                # conda env(s) for the pipeline        (git-ignored, rsync-excluded)
  results/             # cached predictions + table CSVs      (git-ignored, rsync-excluded)
  logs/                # SLURM + application logs             (git-ignored)
  __init__.py
  config.py            # ExperimentConfig dataclass; root-relative paths, model spec, k values, margins
  schema.py            # Question, PageSet, Payload, Prediction, Score — the data contracts
  data/
    __init__.py
    loader.py          # load_mmlongbench() -> List[Question]  (single dataset in v1)
    render.py          # PDF -> page images / page text spans (shared by representation & retrieval)
  pipeline/
    __init__.py
    conditioner.py     # InputConditioner: Oracle / RetrievedTopK / FullDoc / BuriedOracle  (Stage A)
    representation.py  # Representation ABC + T, TL, TLV, V composers                        (Stage B)
    reasoner.py        # Reasoner ABC (backend-agnostic); backends live in models/           (Stage C)
    judge.py           # Judge ABC + uniform LLM-as-judge protocol                           (Stage D)
    orchestrator.py    # composes A->B->C->D for one (Question, condition); caching; run loop
  models/
    __init__.py        # model registry: name/spec -> Reasoner instance (the swap point)
    payload.py         # ModelInput: backend-agnostic text+image container + adapters
    local_vlm.py       # LocalVLMBackend: Qwen3-VL etc. via vLLM/HF
    api_vlm.py         # APIBackend: OpenAI / Gemini / Anthropic-style HTTP (chat + images)
  covariates/
    __init__.py
    retriever.py       # Retriever ABC + text (BM25+BGE) and vision (ColPali/ColQwen)
    classifier.py      # DocTypeClassifier ABC + cheap pass (RQ3 routing covariate)
  tools/
    __init__.py
    text.py            # embedded (PyMuPDF) + OCR (PaddleOCR)               -> text channel
    layout.py          # Marker text+bbox (v3 primary); Docling/PP-Struct = appendix swap
    visual.py          # page-image / region-crop / resolution variants     -> visual channel
  metrics/
    __init__.py
    accuracy.py        # mean +/- 95% CI, effect sizes
    retrieval.py       # page Recall/Precision/F1 vs gold
    abstention.py      # abstention rate, hallucination rate
    cost.py            # tokens (text/visual) + latency accounting
    frontier.py        # sufficiency-frontier rule (pre-registered margin)
  experiments/               # the library: one generation task per file
    __init__.py
    smoke.py           # frozen smoke corpus (doc ids)
    corpus.py          # config -> question set (smoke frozen corpus | full loader)
    paths.py           # shared cache/table path layout + phase status + logging
    base.py            # GenerationTask ABC + Cell/Retrievers + cell factories
    registry.py        # collects the G*_*.py tasks -> name/group resolve()
    G1_sufficiency.py  # one GenerationTask per file (add a file to add an experiment)
    G2_family.py       #   G2..G6; a scale task (G4) is out of scope for now
    G3_dataset.py
    G5_retrieval.py
    G6_classifier.py
    driver.py          # the generate (GPU) + judge (local) engine over tasks
    tables.py          # pure per-table aggregation functions (frontier, metric columns)
    reporting.py       # table -> source-task routing; builds CSVs + combined .md
  cli/                       # the three experiment roles ONLY (thin wrappers)
    generate.py        # GPU: cache predictions for task(s)  (a cluster submits this)
    judge.py           # local: score a task's cached predictions (no tables)
    build.py           # local: route source-task rows -> the 8 CSVs + .md
kaya/
  config.json          # static Kaya site/project/module/SLURM/path config
  kaya.py              # Python SSH/rsync/SLURM runner for login and GPU jobs
  download_hf.py       # login-node HF snapshot download + MMLongBench .data staging
  KAYA_AGENT_GUIDE.md  # definitive agent guide to Kaya operations
  KAYA_USER_GUIDE.md   # human quick guide for setup and run commands
scripts/               # standalone ops utilities (not imported by the package)
  profile_datasets.py  # table-readiness profile of the 5 datasets (existing)
  dataset_stats.py     # full per-dataset statistics -> md + csv (existing prep script)
  run_probe.py         # Stage 1 feasibility probes
  gates.py             # Section-2 go/no-go gate evaluation
  inspect_results.py   # inspect one/many cached inference results (doc + answers)
  annotate_docs.py     # per-document manual annotation sheet + scoring
  split_docs_by_type.py # copy the 135 docs into per-doc_type folders
docs/
  implementation_plan.md # this staged build plan + target tree + frozen-interface rules
  AGENT_GUIDE.md       # fixed decisions, tree-to-paper map, findings, models/data/tools/eval reference
  USER_GUIDE.md        # what/why (the v3 spec) + the run Runbook
  dataset_stats.md     # per-dataset statistics (+ dataset_label_distributions.csv)
tests/
  ...                  # one test module per stage
```

**Interface stability.** `schema.py`, the ABCs in `pipeline/` and `covariates/`, and the
`ModelInput` contract in `models/payload.py` are frozen at the end of Stage 3. Everything after
is an implementation behind one of these.

**The model swap point.** `models/__init__.py` maps a model spec (family, size, backend) to a
`Reasoner`. The pipeline never imports a concrete backend; it asks the registry for a `Reasoner`
and hands it a `ModelInput`. Adding InternVL/Gemma is a new local-backend entry; adding GPT/Gemini
is a new API-backend entry. No pipeline code changes. This is the concrete meaning of "swappable
family."

**Caching contract (decided Stage 3, honoured forever).** Every expensive call (representation
build, reasoner generation, retrieval, judge score) is keyed by a deterministic hash of its
inputs (including the model spec) and written to `results/cache/`. Re-running is idempotent and
resumable — the only way the multi-condition sweep is affordable.

**Root-relative paths (the thing that makes local == Kaya).** `config.py` derives `root` = the
repo root (walk up from `__file__`) and defaults every path under it: `hf_home=<root>/.cache`,
`data_dir=<root>/.data`, `results_dir=<root>/results`, `env_dir=<root>/envs`. No absolute machine
paths appear in code. Because the layout is identical on both machines, the same config runs
locally or on a Kaya compute node with no path edits, and the cache/results produced in either
place are mergeable.

---

## 2b. Local vs Kaya execution

The pipeline runs the same way in two places. Local is for editing and small/cheap runs; Kaya is
for the GPU-heavy grid. The two-machine model is the thing to internalise; the mechanics live in
`kaya/` and are driven by `envs/mpvrdu/bin/python -m kaya.kaya`.

**Two machines, two node types.**
- **Local** (this repo): edit and run here. Small samples, the API backend, or whatever local GPU
  you have. Everything is root-relative so the repo is fully self-contained.
- **Kaya login node** (`ssh kaya`): has internet, no GPU. Runs setup/staging scripts through
  `python -m kaya.kaya run scripts/setup_env.py` and `python -m kaya.kaya run scripts/prestage.py`.
  `module` loading needs a login shell; the Python CLI always uses `bash --login`.
- **Kaya compute node** (via `sbatch`/`srun`): has the GPU, no internet. SLURM jobs default to HF
  offline mode so HF reads the pre-staged `.cache`/`.data` instead of phoning home.

**The flow (driven entirely from Local via `kaya/kaya.py`).**
- `python -m kaya.kaya push` rsyncs the repo to the configured `remote_root` under `/group`,
  **excluding `.cache/ .data/ envs/ results/ logs/ .git`** — env, weights, and datasets are
  rebuilt/staged on Kaya, never copied (they are platform/CUDA-specific and huge). Only code
  crosses the wire.
- `python -m kaya.kaya run <file.py> -- <args...>` runs a Python script on the login node by
  default, or submits it to SLURM if the script header says `# kaya: target=gpu` or the caller
  passes `--target gpu`.
- `python -m kaya.kaya submit <file.py|file.sbatch> -- <args...>` submits a Python script through
  a generated sbatch wrapper, or submits an existing `.sbatch` file as-is. `pull` brings
  `results/` and `logs/` back into the local root.
- Task-specific operations are scripts in `kaya/` (`setup_env.py`, `prestage.py`, `gpu_test.py`,
  `run_probe.py`), not subcommands baked into `kaya.py`.

**Path mapping (same relative layout both sides).** The durable site values live in
`kaya/config.json`: SSH alias, remote root, module names, SLURM defaults plus optional
account/QOS, Hugging Face staging settings, local secret-forwarding rules, model/dataset IDs, and
rsync excludes. The default remote root is `/group/ems036/lxu/mpvrdu`. Artifact paths are
root-relative: `HF_HOME=<root>/.cache`, `data_dir=<root>/.data`, conda env at
`<root>/envs/mpvrdu`, results at `<root>/results`, logs at `<root>/logs`.

**Kaya rules to honor (non-negotiable).**
- Never hand-edit the remote mirror — `push` is `rsync --delete` and will overwrite/delete it.
- `logs/` must exist before `sbatch` (SLURM opens output/error files up front); `push` creates it.
- Existing `.sbatch` files are authoritative. They own their own `#SBATCH` directives, module/env
  setup, output paths, and command body unless explicit `kaya.py submit` SLURM overrides are
  supplied.
- Generated Python runs export `PYTHONPATH=<remote_root>`; hand-written `.sbatch` files must set
  their own `PYTHONPATH` or use module-style Python invocation.
- Anything touching `module` needs a **login shell** (`bash --login -lc`).
- Heavy artifacts (`.cache`, `.data`, `envs`) stay off the rsync path; download/build/stage them
  on Kaya.
- Keep all Kaya-specific source, config, and docs under `kaya/`; do not reintroduce `scripts/kaya/`
  or `docs/KAYA.md`.
- Confirm Kaya's module names, CUDA version, and GPU partition before relying on them (they
  drift); record what you find in `docs/AGENT_GUIDE.md`.

**Experiment execution model (generate on Kaya, judge locally, per experiment).**
Every paper table is one reusable `Experiment` (`experiments/T*_*.py`), run in two
phases split across machines because the reasoner/retrievers/classifier need a GPU
while the judge needs the internet:
- **Generate** on Kaya (GPU, offline): `kaya.kaya submit cli/generate.py -- --generation
  <task|group>`. One generation task per job keeps jobs small and fast-queueing.
  Predictions cache per task under `results/cache/<smoke|full>/<task>/`.
- **Pull** them back: `kaya.kaya pull` (rsyncs `results/`).
- **Judge** locally (no GPU, only an API key): `python -m cli.judge --generation <sel>`
  scores the cached predictions (no tables). **Build** locally: `python -m
  cli.build` routes each table's source-task rows into the eight CSVs +
  a combined `.md`. `--full` selects the full corpus/8B. Judge keys live only in
  the local `.env`; they are not forwarded to Kaya. This role split (generation
  tasks G1..G6, judge, build) supersedes the earlier per-table experiment model
  — the Section-2 stages below run these tasks.

---

# SECTION 1 — MVP: prove the whole pipeline runs, fast

**Purpose.** Before any full or GPU-heavy run, exercise *every* component the paper depends on —
prestaging, parsers, OCR, layout serialization, model load/generate, retrieval engines, the judge,
metrics, and every table builder — end to end on a **tiny fixed smoke corpus** (~6–10 documents
spanning all three bins). The MVP must produce every paper table shape (Tables 1–8) filled with
real-but-throwaway numbers. Its job is to surface integration and environment failures cheaply, on
the smallest model and a handful of questions, so the expensive Section-2 runs hit zero plumbing
bugs. The MVP is judged only by "did it run and produce the right-shaped artifact," never by
accuracy.

**MVP-wide conventions.**
- A single `--smoke` flag on every runnable path selects the frozen smoke corpus, the smallest
  model (Qwen3-VL-2B), and a low `max_tokens`, so a full MVP pass finishes in minutes on one GPU.
- Every MVP stage writes its artifact into the **same cache/format the full run uses**
  (`results/cache/`, keyed by input+model spec), so Section 2 is a scale-up, not a rewrite.
- Fill the frozen interfaces; never modify them. If a real tool cannot fit an interface, that is a
  checkpoint discussion, not a silent edit.

---

## Stage M1 — Smoke corpus, Option-A binning, config

**Goal.** A deterministic tiny corpus and the fixed Option-A binning, wired into config, so every
later MVP stage has a stable target.

**Build.**
- `experiments/smoke.py`: select ~6–10 MMLongBench documents covering all three bins (include ≥1
  Brochure so visual-heavy is present, ≥1 Financial report/Guidebook/Tutorial for in-between, ≥1
  Academic paper/Research report/Admin file for text-heavy) and their questions. Freeze the chosen
  `doc_id` list as a constant in the repo so the smoke set is reproducible across machines.
- `data/binning.py`: `doc_type_bin(doc_type) -> {"text_heavy","in_between","visual_heavy"}`
  implementing Option A. This is the **single source of truth** for binning; Section 3 Priority 1
  (Option B) replaces only this function's body behind the same signature.
- Extend `config.ExperimentConfig` with `smoke: bool`, `bins`, `sufficiency_margin=3`,
  `cost_metric="latency_bs1"`, `representations=("T","TL","TLV","V")`. Keep paths root-relative.

**Docs.** `docs/AGENT_GUIDE.md`: record the frozen smoke doc ids, the Option-A bin definitions with
their Q/doc counts, and the Marker-vs-Docling primary-parser reconciliation.

**Tests.** `tests/test_binning.py` — all 7 `doc_type` classes map to the correct bin; the smoke set
is non-empty in every bin; `ExperimentConfig(smoke=True)` resolves root-relative paths.

**Checkpoint.** Human confirms the smoke doc ids, the Option-A bin counts, and that Marker (not
Docling) is the primary parser for v3.

---

## Stage M2 — Prestage + tool smoke (parsers, OCR, layout, retrieval weights)

**Goal.** Prove every non-model tool loads and produces a well-formed artifact on the smoke corpus,
on Kaya, via a fast `--smoke` prestage path.

**Build.**
- Extend `scripts/prestage.py` with a `--smoke` mode that stages only what the smoke run needs
  (Qwen3-VL-2B, BGE-small or BGE-large per config, ColQwen, Marker weights, PaddleOCR warm) and
  verifies each import plus one tiny call. Most of this is wiring the existing config-driven
  prestage into a fast subset; do not add ad-hoc setup outside the config-driven path.
- `tools/text.py`: confirm `embedded()` (PyMuPDF) and `ocr()` (PaddleOCR PP-OCRv5) each return
  non-empty text on one smoke page.
- `tools/layout.py`: **make Marker the primary layout/text source** — `marker_text(pages)` and
  `marker_bbox_json(pages)` (serialized bbox layout for `T+L`). Keep the existing Docling/PP-Struct
  paths available for the appendix parser-swap, but the main path is Marker. Assert the bbox JSON
  is well-formed on one smoke page with a table.
- `tools/visual.py`: `full_page(pages)` and `resolution(pages, scale)` return page images with a
  token-cost estimate. `region_crop` degrades to page-level (no in-page boxes in MMLongBench per
  the Stage-1 verdict).

**Docs.** `docs/AGENT_GUIDE.md`: each tool's role, I/O, licence, and which table it serves; note Marker
primary / PyMuPDF+Docling appendix. Update the prestage inventory in `docs/AGENT_GUIDE.md`.

**Tests.** `tests/test_tools_smoke.py` — each tool returns a well-formed artifact on one smoke page;
Marker bbox JSON parses; prestage `--smoke` idempotency holds.

**Checkpoint.** Human reviews one extracted sample per tool, especially Marker bbox JSON quality
(this is the `T+L` channel's foundation).

---

## Stage M3 — Reasoner load/generate (critical-path unblock)

**Goal.** Resolve the Qwen3-VL load path flagged at Stage 1 and generate through the frozen
`Reasoner` ABC on the smoke corpus. **This gates every downstream number and is the single highest
risk in the whole build; do it first within the MVP.**

**Build.**
- Resolve the environment so `Qwen3-VL-2B` loads and generates: either upgrade `transformers`
  within the `colpali-engine`/`vllm` compatibility window so `Qwen3VLForConditionalGeneration`
  exists, or use a vLLM path confirmed to serve Qwen3-VL. Record the exact working versions and the
  chosen path in `docs/AGENT_GUIDE.md`.
- `models/local_vlm.py`: real `LocalVLMBackend(spec)` consuming `ModelInput.to_local_prompt()` for
  text-only and text+image inputs; record input/output tokens (text vs vision split) and batch=1
  latency into `Prediction`. One frozen prompt template, versioned in code.
- `models/__init__.py`: registry returns the real 2B local backend for the smoke spec.
- Smoke-generate one answer per representation (`T`/`T+L`/`T+L+V`/`V`) for a few smoke questions on
  Kaya GPU (`kaya.kaya submit`).

**Docs.** `docs/AGENT_GUIDE.md`: the working transformers/vLLM versions and load path, the frozen prompt
template, token accounting, and the note that closed models are comparison/judge only.

**Tests.** `tests/test_reasoner.py` — 2B answers a text-only and an image+text smoke question;
token/latency fields populated; the same frozen prompt is used across all four representations.

**Checkpoint.** Human confirms the Qwen3-VL load path and one generation per rung. **Hard gate for
the MVP** — nothing downstream is trustworthy until this passes.

---

## Stage M4 — Ladder + input conditioning, end to end (oracle)

**Goal.** The full A→B→C path (input-conditioning → representation → reasoner) on oracle pages, for
all four rungs, through the orchestrator's resumable cache.

**Build.**
- Wire `OracleConditioner` → each ladder composer (`T`/`TL`/`TLV`/`V`) → `ModelInput`, honouring
  the modality-boundary rule (only `TLV`/`V` attach images; `T`/`TL` are strings only). `T`/`TL`
  draw from Marker text + bbox JSON; `TLV`/`V` attach page images.
- Confirm the orchestrator caches one result row per (smoke question × rung), keyed including the
  model spec, and that re-running is a cache hit (idempotent, resumable).

**Docs.** Update `docs/ARCHITECTURE.md` if the composer wiring clarified anything; otherwise none.

**Tests.** `tests/test_ladder_e2e.py` — all four rungs produce valid cached rows for every smoke
question; `T`/`TL` payloads carry no image; a second run is a pure cache hit.

**Checkpoint.** Human reviews the cached smoke rows (four per question).

---

## Stage M5 — Judge + metrics + all eight table shapes

**Goal.** Score smoke rows with the real judge and emit **every** paper table shape (Tables 1–8),
filled with throwaway smoke numbers, proving the whole reporting path works before full runs.

**Build.**
- `pipeline/judge.py`: real GPT-4o-mini judge (generate→extract→score) via the API backend, applied
  identically across conditions; records verdict + extracted answer.
- `metrics/accuracy.py`: mean accuracy with **bootstrap 95% CI at the document level** (1000
  resamples over documents, not questions).
- `metrics/cost.py`: latency@batch=1 (primary) and text/vision token aggregation (secondary).
- `metrics/frontier.py`: the sufficiency-frontier rule (cheapest rung whose CI upper bound reaches
  within 3 points of the strongest rung's point estimate).
- `experiments/tables.py` + `experiments/reporting.py`: builders emitting CSVs matching Tables 1–8
  exactly — T1 headline (bin × 4 rungs + frontier + latency@frontier); T2 analytical
  (bin × question-type × 4 rungs); T3 family replication; T4 dataset replication; T5
  composition-mediation; T6 matched-vs-cross; T7 routing (4 policies); T8 scale sanity. On smoke
  data these are near-empty but correctly shaped.

**Docs.** `docs/AGENT_GUIDE.md`: judge protocol, the document-level bootstrap, the frontier rule.

**Tests.** `tests/test_judge_metrics.py` — judge returns a valid `Score`; document-level CI correct
on a toy set; frontier rule picks the right rung on constructed inputs; every table builder emits a
CSV whose columns/rows match the spec skeleton.

**Checkpoint.** Human confirms all eight table shapes render from a single smoke run.

---

## Stage M6 — Retrieval + classifier + policy smoke (covariates)

**Goal.** Prove both retrieval engines, the doc-type classifier, the matched-vs-cross pipelines, and
the four routing policies run end to end through their interfaces on the smoke corpus. Completes the
MVP: the entire paper is now runnable end to end, fast.

**Build.**
- `covariates/retriever.py`: text `BM25+BGE` and vision `ColQwen` return ranked smoke pages for
  `RetrievedTopK`; `metrics/retrieval.py` computes page R/P/F1 vs gold on the smoke set.
- `covariates/classifier.py`: Qwen3-VL-2B few-shot doc-type classification from the first two pages
  of a smoke doc; logs predicted bin vs gold.
- `experiments/T6_matched_cross.py`: the **matched vs cross** pipelines (matched =
  vision-retrieval + vision-reasoning; cross = text-retrieval + vision-reasoning) and
  `experiments/T7_routing.py`: the **four routing policies** (oracle routing, predicted routing with
  classifier latency folded in, uniform-cheapest `T`, uniform-strongest `T+L+V`).

**Docs.** Extend `docs/AGENT_GUIDE.md` with retrieval metric definitions (evidence-modality sliced),
the classifier's covariate role, and the routing-cost accounting (classifier latency counted).

**Tests.** `tests/test_covariates.py` — retriever returns k pages, F1=1 on a perfect toy;
per-modality slicing keys correctly; classifier returns a valid bin; each routing policy produces a
corpus-level row.

**Checkpoint.** Human reviews the retrieval signal and one classifier prediction. **End of MVP:**
the whole paper runs end to end on the smoke corpus and emits Tables 1–8. Green-light Section 2.

---
# SECTION 2 — Full v3 pipeline: the real runs and gates

**Purpose.** Scale the MVP-verified pipeline to the full dataset and execute the paper. No new
capability is built here beyond what Section 3 adds; Section 2 is orchestration, the three go/no-go
gates, and the eight tables at full size. Run only after the MVP passes end to end. Every run
follows the MVP's conventions minus `--smoke`: full corpus, Qwen3-VL-8B primary, document-level
CIs, resumable cache. Prefer `kaya.kaya submit` for the GPU grid; pull `results/` back and build
tables locally.

---

## Stage F1 — Gate 1: RQ1 frontier divergence (full headline, Table 1)

**Goal.** The paper's pivotal result and its first go/no-go decision.

**Build/run.**
- Full Exp 1 headline via the `G1_sufficiency` task (`cli.generate --generation
  G1_sufficiency --full`, then judge + build): Qwen3-VL-8B, `OracleConditioner`, full
  MMLongBench-Doc, all four rungs, all three Option-A bins → **Table 1** with frontier marks,
  per-cell document-level bootstrap CIs, and latency@frontier.
- Apply the gate: **Go** if ≥2 of the 3 bins have different sufficiency frontiers. **No-go** if all
  three land on the same rung → escalate; doc-type is not a useful axis and the story must reframe
  around evidence composition alone (Section 3 Priority 1) or collapse to a two-bin contrast.

**Sampling note.** Draw and report at the **document level**. Visual-heavy (Brochure, 101 Q / 15
docs) will carry the widest CIs; surface them honestly rather than hiding the imbalance.

**Docs.** `docs/USER_GUIDE.md (Runbook)`: the exact local and Kaya invocations for Table 1. Record the Gate-1
verdict and the per-bin frontiers in `docs/AGENT_GUIDE.md`.

**Tests.** `tests/test_frontier_gate.py` — the gate predicate correctly returns Go/No-go on
constructed frontier configurations (all-same → No-go; ≥2-differ → Go).

**Checkpoint.** **Human gate.** Do not proceed to replications until Gate 1 is Go, or the fallback
(Option B / two-bin) is chosen and recorded.

---

## Stage F2 — Gate 2: judge–human agreement

**Goal.** Validate the automated scorer before any main number is trusted.

**Build/run.**
- Tooling to sample 200 questions stratified over doc-type × question-type and record human labels
  (the labelling itself is human work; the tool produces the sheet and ingests it).
- Compute Cohen's κ of GPT-4o-mini vs the human labels. **Go** if κ ≥ 0.75; else iterate the judge
  prompt or fall back to GPT-4o full before any main run.

**Docs.** `docs/AGENT_GUIDE.md`: the κ result and the judge decision. Record in `docs/AGENT_GUIDE.md`.

**Tests.** `tests/test_agreement.py` — κ computed correctly on a toy labelled set; the stratified
sampler covers every non-empty doc-type × question-type cell.

**Checkpoint.** **Human gate.** No main-run number is trusted until κ ≥ 0.75 (or the stricter judge
is adopted).

---

## Stage F3 — Gate 3: classifier feasibility

**Goal.** Decide whether RQ3 predicted-routing is viable or must fall back to the oracle upper bound.

**Build/run.**
- 100-document pilot: Qwen3-VL-2B few-shot doc-type classification from the first two pages; report
  top-1 accuracy against gold `doc_type` (mapped through Option-A bins). **Go** if ≥70%; else
  upgrade the classifier (Qwen3-VL-8B or a small trained head) or scope RQ3 to reporting the
  oracle-routing upper bound only.

**Docs.** Record the classifier accuracy and the RQ3 scope decision in `docs/AGENT_GUIDE.md`.

**Tests.** `tests/test_classifier_gate.py` — the gate predicate returns Go/No-go correctly; the
pilot sampler draws 100 distinct documents.

**Checkpoint.** **Human gate.** Records whether RQ3 runs predicted routing or the oracle bound only.

---

## Stage F4 — Exp 1 replications (Tables 2–4)

**Goal.** Complete RQ1's robustness: question-type slice, family replication, dataset replication.

**Build/run.**
- **Table 2 (analytical):** re-slice the cached Table 1 runs by question type
  (single-hop text / table / chart-figure / multi-hop) per bin. No new generation — pure
  re-aggregation of cached rows. Marked analytical, not used for deployment recipes.
- **Table 3 (family replication):** re-run the RQ1 headline on `internvl3-8b-local` via the
  swappable backend; report whether each bin's frontier matches Qwen3-VL qualitatively.
- **Table 4 (dataset replication):** add a LongDocURL loader returning the frozen `Question` schema,
  then run the headline on LongDocURL, **doc-type layer only** (LongDocURL lacks evidence-modality
  labels for finer slicing). Report whether frontiers match MMLongBench.

**Docs.** `docs/AGENT_GUIDE.md`: the LongDocURL loader (fetch strategy, field mapping) per `dataset_stats.md`.

**Tests.** Extend `tests/test_data.py` for the LongDocURL loader (schema invariants hold);
`tests/test_runner_tables.py` — Tables 2–4 emit the expected cell sets.

**Checkpoint.** Human reviews family- and dataset-robustness of the recipe (is it Qwen3-VL-specific,
is it MMLongBench-specific).

---

## Stage F5 — Exp 2: mechanism (Tables 5–6)

**Goal.** RQ2 — explain the recipe, and test whether retrieval needs the same modality as reasoning.

**Build/run.**
- **Table 5 (composition mediation):** decompose each bin into evidence-modality shares from
  `evidence_sources` (% text / table / chart / figure / layout); compute the per-modality
  sufficient representation; show that composition × per-modality frontier **predicts** the Table 1
  frontier. This is the causal core of the paper.
- **Table 6 (matched vs cross):** on the bins where Table 1 required vision, run matched
  (vision-retrieval + vision-reasoning) vs cross (text-retrieval + vision-reasoning) under real
  retrieval, reporting accuracy and latency and Δ. Cross wins are explained by locate–reason
  modality divergence (a page text-locatable but vision-reasoned) in one paragraph plus one
  qualitative figure (the chart-with-caption case).

**Docs.** `docs/USER_GUIDE.md (Runbook)`: how to build Tables 5–6 and the qualitative figure.

**Tests.** `tests/test_mechanism.py` — the composition decomposition sums to 1 per bin; the
predicted-frontier computation matches a hand-worked toy; matched/cross rows are well-formed.

**Checkpoint.** Human reviews whether composition predicts the recipe — the mechanism claim stands
or falls here.

---

## Stage F6 — Exp 3: routing under classification cost (Table 7)

**Goal.** RQ3 — does routing beat uniform once the classifier's cost is counted?

**Build/run.**
- Four policies on the full corpus via `experiments/T7_routing.py`: oracle routing (gold doc-type →
  recipe), predicted routing (Qwen3-VL-2B classifies first pages → recipe), uniform-cheapest (`T`
  everywhere), uniform-strongest (`T+L+V` everywhere). **Predicted-routing total latency includes
  the classifier's own latency**, reported as a separate column, not hidden. Two uniform baselines
  prevent cherry-picking the comparison.

**Docs.** `docs/USER_GUIDE.md (Runbook)`: the routing run and its cost accounting.

**Tests.** `tests/test_routing.py` — predicted-routing total latency equals recipe latency plus
classifier latency; each policy yields a corpus-level accuracy+cost row.

**Checkpoint.** Human reviews whether predicted routing beats **both** uniform baselines on the
accuracy–cost trade; an honest "uniform-strongest is good enough" is a legitimate finding.

---

## Stage F7 — Appendix Exp 4 + robustness (Table 8 + margin/parser sensitivity)

**Goal.** The appendix items the main text cites in one line each.

**Build/run.**
- **Table 8 (scale sanity):** re-run the RQ1 headline on Qwen3-VL-2B and Qwen3-VL-32B. **32B is out
  of scope on our own hardware** (Kaya V100 16GB / 2×V100 32GB cannot hold it); run the 32B row on
  the supervisor's A100 account, or have him run that one job (see `docs/AGENT_GUIDE.md` "Hardware
  scope"). 2B runs on our V100s. Scope to oracle-only if still memory-bound. Main text cites one
  sentence: "the recipe is qualitatively stable across 2B–32B (Table 8)", or names the bins where
  the frontier moves.
- **Margin sensitivity:** recompute the Table 1 frontier at margin ∈ {2, 3, 5}.
- **Parser swap:** re-run the headline with PyMuPDF (and/or Docling) in place of Marker; report
  whether the frontier ordering is parser-stable even if absolute numbers shift.

**Docs.** `docs/USER_GUIDE.md (Runbook)`: appendix runs. Record the appendix sentences in `docs/AGENT_GUIDE.md`.

**Tests.** `tests/test_sensitivity.py` — margin sweep produces monotone frontier behaviour; parser
swap runs through the same table builder.

**Checkpoint.** Human confirms the one-line appendix claims the main text will cite. **End of the
paper's core evidence.**

---

# SECTION 3 — Additional experiments (if space/time allows)

**Purpose.** Extensions that strengthen the paper *if* room remains, in priority order. None is
required for the thesis. Each is self-contained and reuses the built pipeline behind frozen
interfaces. Do not start any until Section 2 is complete and the paper's core is drafted.

---

## Stage P1 — Option B: evidence-composition (data-driven) binning

**Goal.** Validate (or replace) the Option-A binning by clustering doc types on their evidence
composition. This is the pre-committed Gate-1 fallback and a paper-strengthening robustness result
even when Option A holds; it is cheap because it re-bins cached results with no new generation.

**Build.**
- In `data/binning.py`, add `doc_type_bin_datadriven()` behind the same signature: cluster the 7
  `doc_type` categories by their `evidence_sources` distribution (share of Figure+Chart vs
  Pure-text+Layout vs Table) into three bins. Swap it in via config; re-aggregate the cached Table 1
  results under the new bins.

**Docs.** `docs/AGENT_GUIDE.md`: the data-driven bin assignments and whether they agree with Option A.

**Tests.** `tests/test_binning.py` extended — the data-driven binner returns three non-empty bins
and is a pure function of the evidence-source distribution.

**Checkpoint.** Human compares Option A vs Option B: if they agree, report agreement as robustness;
if the visual-heavy bin was thin under A, decide whether B becomes primary.

---

## Stage P2 — Visual-heavy anchor (SlideVQA), only if Gate 1 was weak on the visual bin

**Goal.** If MMLongBench's visual-heavy bin (Brochure, 101 Q / 15 docs) cannot be separated at the
3-point margin, recruit SlideVQA as an unambiguous visual anchor to corroborate the visual-heavy row.

**Build.**
- SlideVQA loader → frozen `Question` schema (native `evidence_pages`, per-slide images). Re-run the
  RQ1 headline for the visual-heavy row only, clearly marked cross-dataset.

**Docs.** `docs/AGENT_GUIDE.md`: the SlideVQA loader per `dataset_stats.md`.

**Tests.** Reuse `tests/test_data.py` parameterised over the SlideVQA loader (schema invariants).

**Checkpoint.** Human decides whether the cross-dataset visual row is needed and sound.

---

## Stage P3 — Retrieval-sufficiency frontier

**Goal.** The "how good must retrieval be" companion to RQ3; cut for space but high-value.

**Build.**
- Sweep retriever quality (top-k ∈ {1,3,5}, the two retrievers) via `RetrievedTopK`; relate
  end-to-end accuracy to retriever page-F1 (from `metrics/retrieval.py`); locate the plateau.

**Docs.** `docs/USER_GUIDE.md (Runbook)`: the sweep and the accuracy-vs-page-F1 curve.

**Tests.** `tests/test_retrieval_sufficiency.py` — the curve builder produces monotone-keyed points.

**Checkpoint.** Human reviews the plateau and the "minimum retrieval quality worth engineering"
sentence.

---

## Stage P4 — Distractor-burying sweep

**Goal.** The precision–recall-tolerance instrument; the `BuriedOracle` conditioner is already built
and otherwise unused.

**Build.**
- Hold gold pages present, pad with same-corpus distractors at increasing counts via `BuriedOracle`;
  measure accuracy decay; full-doc is the endpoint.

**Docs.** `docs/USER_GUIDE.md (Runbook)`: the burying sweep.

**Tests.** `tests/test_burying.py` — distractor count increases monotonically; gold pages always
present in the conditioned set.

**Checkpoint.** Human reviews the accuracy-vs-#distractors curve.

---

## Stage P5 — Fail-safe abstention

**Goal.** Deployment-completeness; uses the 244 native-unanswerable questions and the Stage-1
abstention definition already logged.

**Build.**
- Neutral vs abstention-licensed prompt on native-unanswerable questions and on answerable questions
  whose gold page a retriever missed (page-recall 0); report abstention and hallucination rates via
  `metrics/abstention.py`.

**Docs.** `docs/AGENT_GUIDE.md`: the two-prompt manipulation and the abstention/hallucination
definitions.

**Tests.** `tests/test_abstention.py` — abstention/hallucination rates correct on a constructed
labelled set.

**Checkpoint.** Human reviews the abstention-vs-hallucination table.

---

## Appendix — Stage dependency summary

| Section | Stage | Builds | Gate |
|---|---|---|---|
| built | 0–3 | skeleton, probes, Kaya runner, data layer, frozen interfaces | done (see `AGENT_GUIDE.md`) |
| 1 MVP | M1 | smoke corpus + Option-A binning + config | — |
| 1 MVP | M2 | prestage + parser/OCR/Marker-layout/retrieval-weight smoke | — |
| 1 MVP | M3 | Qwen3-VL load/generate (critical-path unblock) | **MVP hard gate** |
| 1 MVP | M4 | ladder + input-conditioning end to end (oracle) | — |
| 1 MVP | M5 | judge + metrics + all 8 table shapes | — |
| 1 MVP | M6 | retrieval + classifier + matched/cross + routing smoke | end of MVP |
| 2 Full | F1 | Gate 1 — full RQ1 headline (Table 1) | **frontier divergence** |
| 2 Full | F2 | Gate 2 — judge–human κ | **κ ≥ 0.75** |
| 2 Full | F3 | Gate 3 — classifier feasibility | **top-1 ≥ 70%** |
| 2 Full | F4 | Exp 1 replications (Tables 2–4) | — |
| 2 Full | F5 | Exp 2 mechanism (Tables 5–6) | — |
| 2 Full | F6 | Exp 3 routing (Table 7) | — |
| 2 Full | F7 | Exp 4 appendix + sensitivity (Table 8) | — |
| 3 Extra | P1 | Option B data-driven binning | — |
| 3 Extra | P2 | visual anchor (SlideVQA) if needed | — |
| 3 Extra | P3 | retrieval-sufficiency frontier | — |
| 3 Extra | P4 | distractor-burying sweep | — |
| 3 Extra | P5 | fail-safe abstention | — |

**Invariant across every stage:** the Stage 3 interfaces (schema, ABCs, `ModelInput`) do not change.
The MVP proves the plumbing on a smoke corpus; Section 2 fills it at full scale under three gates;
Section 3 extends only if the paper has room. Any pressure to change a frozen interface is a
checkpoint conversation recorded in `docs/AGENT_GUIDE.md`, never a silent edit.
