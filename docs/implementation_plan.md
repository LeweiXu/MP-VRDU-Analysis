# Implementation Plan — MP-VRDU Representation & Deployment Study

> A staged build plan for a coding agent. This document is the single source of truth for
> *how* to build the codebase; `PROJECT_SPEC.md` is the source of truth for *what* the
> experiments are and *why*. Read both before starting any stage.

---

## 0. How to use this plan

**Audience.** A strong coding agent (~258k context) working under human supervision.

**Operating loop (mandatory).**
1. At the start of every stage, (re-)read this file in full and the relevant parts of
   `PROJECT_SPEC.md`, `dataset_profile.md`, and `context.md`.
2. Implement exactly the one stage named, nothing from later stages.
3. Produce the stage's **deliverables, docs, and tests**; update
   `docs/DECISIONS.md` with implementation-relevant deviations, migrations,
   operational notes, and findings from the run; run the tests; report results.
4. **Stop at the human checkpoint.** Summarise what was built, what the tests showed, and any
   surprises that should change later stages.
5. After the human approves, run `/compact`, then begin the next stage from step 1.

**Why staged + compact.** Each stage is self-contained and re-readable from this plan plus the
code already on disk. Do not rely on un-compacted conversational memory: if a decision,
finding, deviation, migration, or operational note matters for a later stage, write it into
code comments or `docs/DECISIONS.md`.

**Golden rules.**
- **Skeleton before tools.** Stages 1–3 define interfaces and a runnable end-to-end stub.
  Later stages plug real tools into those interfaces without changing them. If a real tool
  forces an interface change, that is a checkpoint discussion, not a silent edit.
- **Concise, not fragmented.** Prefer a handful of cohesive modules over many tiny files. No
  build systems, no YAML/Makefiles. Configuration is plain Python (a dataclass); runs are driven
  by small CLI scripts. Every file opens with a module docstring stating its role in the
  experimental architecture.
- **The code mirrors the paper.** The four pipeline stages and two covariates from
  `PROJECT_SPEC.md` are first-class objects with the same names. A reader who knows the paper
  should recognise the architecture in the file tree.

---

## 1. Scope decisions fixed for this build

**Paper structure this build serves.** The study is organised around **three topics**, not a flat
research-question list. This plan uses the shorthand RQ1/RQ2/RQ3 for them; the mapping is:

- **RQ1 = Representation** (what *reasoning* requires): sufficiency frontier by document type,
  evidence-modality mediation, and cost/granularity.
- **RQ2 = Retrieval** (what *locating* requires): retrieval-modality sufficiency and the
  retrieval–reasoning modality *divergence*. Reports the mechanism only.
- **RQ3 = Deployment** (synthesis): locate-vs-reason attribution, bottleneck migration with model
  size, how good retrieval must be (incl. the distractor-burying sweep), routing vs uniform, and
  fail-safe abstention.

The shorthand is for cross-referencing code to paper facets; the code names objects after the
pipeline stages (input-conditioning / representation / reasoning / scoring + covariates), not after
RQ numbers.

**Distractor-burying is in scope.** The `BuriedOracle` condition (gold pages held present, padded
with same-corpus distractor pages) is part of RQ3's "how good must retrieval be" facet. This
reverses an earlier "do NOT bury" note in `context.md`; the reversal is recorded there and in the
spec. Burying is the instrument for a deployment question, not a retrieval benchmark, so it stays a
document-understanding measurement.

Two further decisions narrow the v1 build. Both are deliberate and reduce surface area.

**Single dataset (v1): MMLongBench-Doc only.** It is the only benchmark carrying *both*
document-type labels and evidence-modality labels plus gold evidence pages, so it alone supports
every facet of all three RQs (representation frontier, evidence-modality slice, retrieval
modality/divergence, locate-vs-reason decomposition, routing, unanswerable/abstention). The other
datasets (LongDocURL, CUAD, DocFinQA, SlideVQA) are **out of scope for v1** and appear only as an
optional later stage (Stage 10) if time allows; they strengthen robustness but are not required
for the claims. Do not build multi-dataset machinery in v1: one loader, one schema, one render
path.

**Swappable model family.** The evaluated reasoner is **Qwen3-VL** (sizes 2B / 4B / 8B / 32B, one
family, 8B as the center configuration for single-model experiments). The family must be
**swappable** behind a stable interface so later testing can substitute other open families
(InternVL, Gemma) or closed APIs (GPT, Gemini). This is achieved with a single `Reasoner` ABC and
two backends (local-weights and HTTP-API); see Stage 3 and Stage 6. Closed models are for
methodological comparison and as judges, **not** for the deployment recommendation, which is
bounded to locally-hostable open models per the privacy framing.

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
    layout.py          # Docling markdown + PP-StructureV3 bbox             -> layout channel
    visual.py          # page-image / region-crop / resolution variants     -> visual channel
  metrics/
    __init__.py
    accuracy.py        # mean +/- 95% CI, effect sizes
    retrieval.py       # page Recall/Precision/F1 vs gold
    abstention.py      # abstention rate, hallucination rate
    cost.py            # tokens (text/visual) + latency accounting
    frontier.py        # sufficiency-frontier rule (pre-registered margin)
  experiments/
    __init__.py
    runner.py          # config -> cells; caches per (question, condition) result
    tables.py          # cached results -> the RQ result tables (CSV)
  cli/
    run_probe.py       # Stage 1 feasibility probes
    run_experiment.py  # main entry: config -> cached predictions
    build_tables.py    # cached predictions -> result CSVs
kaya/
  config.json          # static Kaya site/project/module/SLURM/path config
  kaya.py              # Python SSH/rsync/SLURM runner for login and GPU jobs
  download_hf.py       # login-node HF snapshot download + MMLongBench .data staging
  KAYA_AGENT_GUIDE.md  # definitive agent guide to Kaya operations
  KAYA_USER_GUIDE.md   # human quick guide for setup and run commands
scripts/               # standalone ops utilities (not imported by the package)
  profile_datasets.py  # table-readiness profile of the 5 datasets (existing)
  dataset_stats.py     # full per-dataset statistics -> md + csv (existing prep script)
docs/
  DECISIONS.md         # fixed decisions, stage findings, implementation notes (created Stage 0)
  ARCHITECTURE.md      # how the tree maps to the paper (created Stage 3)
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
- **Kaya login node** (`ssh kaya`): has internet, no GPU. Builds the conda env
  (`kaya.kaya setup-env`) and stages models/datasets (`kaya.kaya prestage`) into
  `<root>/.cache` and `<root>/.data`. `module` loading needs a login shell; the Python CLI
  always uses `bash --login -lc`.
- **Kaya compute node** (via `sbatch`/`srun`): has the GPU, no internet. Python-generated SLURM
  jobs default to HF offline mode so HF reads the pre-staged `.cache`/`.data` instead of phoning
  home.

**The flow (driven entirely from Local via `kaya/kaya.py`).**
- `python -m kaya.kaya push` rsyncs the repo to the configured `remote_root` under `/group`,
  **excluding `.cache/ .data/ envs/ results/ logs/ .git`** — env, weights, and datasets are
  rebuilt/staged on Kaya, never copied (they are platform/CUDA-specific and huge). Only code
  crosses the wire.
- `python -m kaya.kaya run-gpu -- <command...>` = push + generate sbatch + submit + wait + pull.
  `pull` brings `results/` and `logs/` back into the local root. One-time env/model setup is a
  login-node step via `python -m kaya.kaya setup-env` and `python -m kaya.kaya prestage`.
- `python -m kaya.kaya run-login -- <command...>` runs code directly on the login node, useful
  for cheap dataset checks and setup diagnostics.

**Path mapping (same relative layout both sides).** The durable site values live in
`kaya/config.json`: SSH alias, remote root, module names, SLURM defaults plus optional
account/QOS, Hugging Face prestage transport settings, model/dataset IDs, and rsync excludes. The
default remote root is `/group/ems036/lxu/mpvrdu`. Artifact paths are root-relative:
`HF_HOME=<root>/.cache`, `data_dir=<root>/.data`, conda env at `<root>/envs/mpvrdu`, results at
`<root>/results`, logs at `<root>/logs`.

**Kaya rules to honor (non-negotiable).**
- Never hand-edit the remote mirror — `push` is `rsync --delete` and will overwrite/delete it.
- `logs/` must exist before `sbatch` (SLURM opens output/error files up front); `push` creates it.
- Anything touching `module` needs a **login shell** (`bash --login -lc`).
- Heavy artifacts (`.cache`, `.data`, `envs`) stay off the rsync path; download/build/stage them
  on Kaya.
- Keep all Kaya-specific source, config, and docs under `kaya/`; do not reintroduce `scripts/kaya/`
  or `docs/KAYA.md`.
- Confirm Kaya's module names, CUDA version, and GPU partition before relying on them (they
  drift); record what you find in `docs/DECISIONS.md`.

---

## Stage 0 — Repository skeleton and conventions

**Goal.** Create the tree, dependency list, and decision log. No logic yet.

**Build.**
- Create the directory tree with `__init__.py` and a one-line docstring in each file stating its
  architectural role.
- `requirements.txt` pinned for the V100/CUDA target: torch, vllm or transformers, pymupdf,
  paddleocr + paddlepaddle, docling, rank-bm25, FlagEmbedding (BGE), colpali-engine, datasets,
  huggingface_hub, pillow, numpy, scipy, pandas, requests (API backend), pytest. Declare only.
- Conda env at `<root>/envs/mpvrdu`, created from `requirements.txt` (locally; recreated on Kaya
  by `python -m kaya.kaya setup-env`). The env lives under the root like every other artifact.
- `kaya/` — the pipeline's Kaya tooling and documentation:
  - `config.json`: static SSH alias, remote root, module, SLURM, model, dataset, and rsync config.
  - `kaya.py`: Python push/pull/login/GPU/probe runner over SSH, rsync, and SLURM.
  - `download_hf.py`: login-node HF snapshot download and MMLongBench `.data` staging.
  - `KAYA_AGENT_GUIDE.md` and `KAYA_USER_GUIDE.md`: canonical Kaya docs. No Kaya docs live under
    `docs/`, and no Kaya scripts live under `scripts/`.
- `docs/DECISIONS.md` seeded with the fixed decisions here and in `PROJECT_SPEC.md` (single
  dataset, Qwen3-VL family + sizes, swappable backends, modality-boundary rule, root-relative
  self-contained layout, local/Kaya execution model), plus an "Open items (Stage 1 confirms)"
  section. Every later run appends implementation-relevant changes, deviations from the plan,
  migrations, operational notes, and findings here.
- `.gitignore` (confirm `results/ .data/ .cache/ logs/` present; add `envs/`; plus model caches,
  `__pycache__`; store downloaded datasets/renders under `.data/`, not `data/`), short
  `README.md`.

**Docs.** `README.md`, `docs/DECISIONS.md`, `kaya/KAYA_AGENT_GUIDE.md`,
`kaya/KAYA_USER_GUIDE.md`.

**Tests.** `tests/test_skeleton.py::test_tree_imports` — every module imports without error;
Kaya config tests sanity-check `kaya/config.json` (paths resolve under the remote root, rsync
excludes cover the heavy dirs, and `scripts/kaya/` is absent).

**Checkpoint.** Human confirms tree, dependency choices, and the Kaya path mapping / rsync
excludes in `kaya/config.json`.

---

## Stage 1 — Feasibility probes (confirm open items)

**Goal.** Answer every code-confirmable open item before building on assumptions. Fail fast.
Findings written to `docs/DECISIONS.md`.

**Build `cli/run_probe.py`** with independently runnable probes, each printing a verdict:

1. **Loader smoke test.** Load a small sample of MMLongBench-Doc via its fetch strategy
   (`dataset_profile.md`). Confirm critical fields parse: `doc_type`, `evidence_sources`,
   `evidence_pages`, `answer`, `answer_format`, and the `"Not answerable"` signal. Confirm source
   PDFs resolve for rendering.
2. **Scanned vs born-digital probe (RQ1 robustness).** Sample PDFs; test for an extractable
   embedded text layer (PyMuPDF text length per page). Report the scanned fraction. Verdict: is
   the embedded-vs-OCR check real, or must degraded text be synthesised?
3. **In-page box probe (RQ1c crop / RQ2).** Check whether MMLongBench-Doc exposes any in-page
   evidence coordinates. If not (likely), record that region-crop is limited to page-level and
   note LongDocURL as the future source of true boxes (Stage 10). Do not block.
4. **Model-family probe (RQ3 scaling + swap).** Confirm Qwen3-VL 2B/4B/8B exist on HF and load and
   generate on the target hardware; record 32B feasibility (load? full-doc context? or
   oracle/retrieved only?). Then confirm the **swap** works: instantiate one model via the local
   backend and one trivial call via the API backend (a cheap model) through the *same* `Reasoner`
   interface, proving backend-agnosticism before the pipeline depends on it.
5. **Vision-retrieval feasibility probe (RQ2, critical path).** Confirm ColPali/ColQwen load and
   index a handful of pages within V100 memory, and that BM25+BGE text retrieval runs. RQ2's
   retrieval-modality analysis depends on both; surface any memory blocker now, not in Stage 8.
6. **Unanswerable/abstention probe (RQ3d).** Confirm `"Not answerable"` questions are present at
   usable count (~20% expected); a tiny generation to see whether the base model ever emits a
   refusal surface form. Draft an abstention definition to pre-register.
7. **doc_type distribution (RQ1/RQ3).** Print MMLongBench-Doc `doc_type` class counts; propose (for
   human approval) the text/in-between/visual spectrum mapping. Record the proposal; do not
   finalise in code.

**Where each probe runs.** The code-only probes (1 loader, 2 scanned-vs-born-digital, 3 in-page
boxes, 6 unanswerable count, 7 doc_type distribution) run locally. The GPU-dependent probes
(4 model-family load/generate + 32B feasibility, 5 vision-retrieval ColPali/ColQwen memory) run on
a Kaya **compute** node via `python -m kaya.kaya run-probe ... --target gpu`, following the Kaya
rules in section 2b (models pre-staged into `.cache` and MMLongBench staged into `.data` on the
login node first; generated compute jobs default to HF offline mode). Record the confirmed Kaya
module / CUDA / partition names in `docs/DECISIONS.md` alongside the probe verdicts.

**Docs.** "Stage 1 findings" in `docs/DECISIONS.md`: one verdict per probe, the Kaya
module/CUDA/partition names, and any consequent change to later stages.

**Tests.** `tests/test_probes.py` — each probe runs on a tiny sample without raising; verdict
objects have expected fields.

**Checkpoint.** Human approves: spectrum mapping, embedded-vs-OCR real-vs-synthetic decision,
abstention definition, 32B scoping, and confirmation that both model backends and both retrievers
are viable on the hardware.

---

## Stage 1.5 — Kaya operations consolidation

**Goal.** Make Kaya execution a maintainable Python-controlled substrate before later stages rely
on it for model, retrieval, and full experiment runs.

**Build.**
- Move all Kaya-specific source, config, and documentation under `kaya/`; remove the old standalone
  reference kit, old shell wrappers, stale SLURM files, pycache, and old logs. Do not leave Kaya
  docs under `docs/` or Kaya scripts under `scripts/`.
- `kaya/config.json`: static site/project config on disk (SSH alias, remote root, modules,
  artifact paths, rsync excludes, SLURM defaults/accounting, HF prestage transport settings, model
  IDs, dataset IDs). Normal operation should not depend on exported bash variables.
- `kaya/kaya.py`: Python CLI that can:
  - push source to Kaya with the configured rsync excludes;
  - pull `results/` and `logs/`;
  - run commands on the login node with the configured conda env;
  - generate and submit SLURM jobs for GPU nodes, wait for completion, pull logs/results, and print
    log tails;
  - run `cli.run_probe` on login or GPU targets.
- `kaya/download_hf.py`: login-node staging helper that downloads HF snapshots into `.cache/` and
  exposes MMLongBench as `.data/mmlongbench/{data,documents}` so local and Kaya loaders see the
  same layout. Kaya defaults disable HF Xet and download serially because the first live staging
  attempt hit partial/range-size failures with the Xet path.
- `kaya/KAYA_AGENT_GUIDE.md`: definitive agent-facing Kaya guide.
- `kaya/KAYA_USER_GUIDE.md`: concise user setup/run guide.

**Docs.** Update `docs/DECISIONS.md`, this plan, and `README.md` so every Kaya reference points to
`kaya/` and `python -m kaya.kaya`.

**Tests.** Extend skeleton tests so `kaya.kaya` and `kaya.download_hf` import, `kaya/config.json`
contains expected root-relative artifact paths and rsync excludes, the two guides exist, and
`scripts/kaya/` does not exist. Also run a harmless CLI parse/config check locally.

**Kaya validation.** With the VPN and SSH key active, run:

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
envs/mpvrdu/bin/python -m kaya.kaya run-login --no-activate --no-push -- pwd
envs/mpvrdu/bin/python -m kaya.kaya push
envs/mpvrdu/bin/python -m kaya.kaya setup-env
envs/mpvrdu/bin/python -m kaya.kaya prestage --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya run-probe loader --target login --json
envs/mpvrdu/bin/python -m kaya.kaya gpu-test
envs/mpvrdu/bin/python -m kaya.kaya run-probe model-family --target gpu --heavy --json --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya run-probe retrieval --target gpu --heavy --json
```

**Checkpoint.** Human confirms the static config values, remote setup, Python runner ergonomics,
and Kaya probe results before Stage 2.

---

## Stage 2 — Data layer (records + rendering)

**Goal.** One normalised question representation and the PDF→pages utilities shared by
representation and retrieval.

**Build.**
- `schema.py`: core dataclasses (frozen where sensible):
  - `Question` (id, doc_id, question, gold_answer, answer_format, doc_type, evidence_pages,
    evidence_sources, hop (derived from `len(evidence_pages)`: single=1, multi>=2),
    is_unanswerable (from `"Not answerable"`), raw_fields).
  - `PageSet` (page indices + provenance: oracle / retrieved / full / buried).
  - `Page` (index, lazy image handle, text spans) — produced by `render.py`.
  - Placeholders `Payload`, `Prediction`, `Score` (filled by their stages).
- `data/loader.py`: `load_mmlongbench(sample=None) -> List[Question]`, normalising real fields per
  the profile. Single function; no dataset-dispatch machinery.
- `data/render.py`: PDF → page images and page text spans, cached under `results/cache/`. This is
  the shared substrate for both the representation channels and the retrievers.

**Docs.** `schema.py` docstring enumerating every field and its MMLongBench-Doc source. Short
`docs/DATA.md`.

**Tests.** `tests/test_data.py` — load N questions; assert invariants (hop matches evidence-page
count; unanswerable flag correct; gold pages within page range; render yields a `Page` with image
or text).

**Checkpoint.** Human confirms the normalised schema before interfaces depend on it.

---

## Stage 3 — Pipeline skeleton (interfaces + runnable stub) **[freeze point]**

**Goal.** The architecturally central stage. Define all ABCs, the backend-agnostic `ModelInput`,
and a working end-to-end orchestrator on **stubs**, so the whole pipeline runs and produces
well-typed rows before any real tool or model exists.

**Build.**
- `pipeline/conditioner.py`: `InputConditioner` ABC + `OracleConditioner`, `RetrievedTopK` (takes
  a `Retriever`; stubbed for now), `FullDoc`, and `BuriedOracle` (gold pages + N same-corpus
  distractor pages; the RQ2 burying sweep). Returns `PageSet`.
- `pipeline/representation.py`: `Representation` ABC with `build(PageSet) -> Payload`; composers
  `T`, `TL`, `TLV`, `V` calling **stub channel functions**. Encodes the modality-boundary rule
  structurally: only `V` and `TLV` may attach images; `T`/`TL` attach strings only.
- `models/payload.py`: `ModelInput` — a backend-agnostic container of ordered text and image parts,
  with two adapters: `to_chat_messages()` (API: messages array, base64 image parts) and
  `to_local_prompt()` (local: chat template + image placeholders). **This is the contract that
  makes the family swappable.** The `Payload` from a `Representation` maps to a `ModelInput`.
- `pipeline/reasoner.py`: `Reasoner` ABC with `answer(question, model_input) -> Prediction`. Stub
  returns a fixed string and zeroed token/latency fields.
- `models/__init__.py`: `get_reasoner(spec) -> Reasoner` registry; returns the stub in this stage.
- `pipeline/judge.py`: `Judge` ABC with `score(question, prediction) -> Score`. Stub returns a
  heuristic score.
- `covariates/retriever.py`, `covariates/classifier.py`: ABCs + stubs.
- `pipeline/orchestrator.py`: config + Question + condition + representation -> one cached,
  well-typed result row. Implement the **caching contract** here (deterministic key including
  model spec; jsonl cache; resumable).
- `config.py`: `ExperimentConfig` (dataset fixed to MMLongBench-Doc; model spec; conditions;
  k in {1,3,5}; burying levels; representations; sufficiency margin). Paths are **root-relative**:
  a derived `root` (walk up from `__file__`) with `hf_home=<root>/.cache`, `data_dir=<root>/.data`,
  `results_dir=<root>/results`, `env_dir=<root>/envs` — no absolute machine paths, so the same
  config runs local or on Kaya (section 2b).
- `cli/run_experiment.py`: wire config -> orchestrator over a tiny sample end to end, on stubs.

**Docs.** `docs/ARCHITECTURE.md`: tree ↔ paper-stage mapping, ABC signatures, the `ModelInput`
contract and how it enables the swap, the caching contract, the root-relative path convention and
local/Kaya execution model (section 2b) with the `kaya/` Python runner layout, and an explicit **frozen
interfaces** list. (Per the global rule, once `ARCHITECTURE.md` exists, later structural changes to
it are a confirm-first step, not a silent edit.)

**Tests.** `tests/test_pipeline_skeleton.py` — orchestrator runs end-to-end on stubs for every
(condition × representation) combo and emits a valid `Score`; cache hit on re-run (idempotency);
modality-boundary enforced (T/TL payloads carry no image); `ModelInput` round-trips through both
adapters.

**Checkpoint.** Human signs off on the **frozen interfaces**, especially `ModelInput`. After this,
tool and model stages only fill implementations.

---

## Stage 4 — Text channel (embedded/OCR) and layout channel

**Goal.** Replace text/layout stubs with real modular, non-VLM tools.

**Build.**
- `tools/text.py`: `embedded(pages)` via PyMuPDF; `ocr(pages)` via PaddleOCR PP-OCRv5. Same return
  type, interchangeable behind `T`.
- `tools/layout.py`: `docling_markdown(pages)` (primary); `ppstructure_bbox(pages)` (the
  geometry-vs-structure fork feeding RQ1 robustness / RQ2 discussion). Wire into `TL`.
- Update `T`/`TL` composers to call these instead of stubs. **Do not touch the ABCs.**

**Docs.** `docs/TOOLS.md` started: each tool's role, I/O, licence, which analysis it serves. Note
Marker as an internal-only robustness parser, not wired to the main path in v1.

**Tests.** `tests/test_text_layout.py` — embedded and OCR both return non-empty text for a
born-digital page; layout markdown preserves table structure where a table exists; composers
respect the modality boundary.

**Checkpoint.** Human reviews extraction quality on a few real pages.

---

## Stage 5 — Visual channel

**Goal.** Replace the visual stub; implement the granularity variants RQ1c needs.

**Build.**
- `tools/visual.py`: `full_page(pages)`, `region_crop(pages, boxes)` (page-level if no in-page
  boxes per Stage 1 verdict), `resolution(pages, scale)`. Each image payload carries a token-cost
  estimate feeding `metrics/cost.py`.
- Update `TLV` and `V` composers to attach images via these. `V` = images only; `TLV` = strings +
  images. Payloads convert to `ModelInput` image parts.

**Docs.** Extend `docs/TOOLS.md` with visual variants and the token-cost estimation method.

**Tests.** `tests/test_visual.py` — full_page yields one image per page; resolution scaling changes
the token estimate monotonically; `V` payload carries images and no text; images survive the
`ModelInput` round-trip.

**Checkpoint.** Human confirms crop/resolution behaviour.

---

## Stage 6 — Reasoner backends (Qwen3-VL local + API) and cost

**Goal.** Replace the reasoner stub with real backends behind the frozen `Reasoner` ABC, proving
the family is swappable; turn on cost accounting.

**Build.**
- `models/local_vlm.py`: `LocalVLMBackend(spec)` serving Qwen3-VL (2B/4B/8B/32B) via vLLM/HF,
  consuming `ModelInput.to_local_prompt()`. One frozen prompt template (versioned in code).
  Records input/output tokens (split text vs visual) and latency into `Prediction`.
- `models/api_vlm.py`: `APIBackend(spec)` for OpenAI/Gemini/Anthropic-style chat+image HTTP,
  consuming `ModelInput.to_chat_messages()`. Same `Prediction` contract; token counts from the
  API response.
- `models/__init__.py`: registry maps specs to the right backend
  (e.g. `qwen3vl-8b-local`, `internvl-8b-local`, `gpt-…-api`, `gemini-…-api`).
- `metrics/cost.py`: aggregate tokens and latency per condition.
- Confirm 2B/4B/8B local paths; gate 32B per Stage 1.

**Docs.** `docs/MODELS.md`: how to register a model, the two backends, the `ModelInput` adapters,
the frozen prompt template (fixed across conditions for commensurability), token accounting, 32B
scoping, and the note that closed models are comparison/judge only, not deployment.

**Tests.** `tests/test_reasoner.py` — Qwen3-VL 2B answers a text-only and an image+text question;
token/latency populated; **swap test**: the same `ModelInput` produces a valid `Prediction`
through both a local backend and a (mocked) API backend, confirming the pipeline is
backend-agnostic; prompt template identical across representations.

**Checkpoint.** Human reviews sample generations, a first real accuracy–cost row, and the swap
test.

---

## Stage 7 — Judge, reproduction gate, accuracy/abstention metrics

**Goal.** The measurement layer. Real judge, validated against humans, plus the metrics that read
its output.

**Build.**
- `pipeline/judge.py`: uniform judge (generate→extract→score) from a **different family** than the
  evaluated reasoner, reusing the `models/` backends (the judge is just another `Reasoner` spec
  wrapped by the `Judge` protocol). Applied identically across all conditions.
- `metrics/accuracy.py` (mean ± 95% CI, effect sizes); `metrics/abstention.py` (abstention rate,
  hallucination rate per the Stage 1 definition).
- **Reproduction gate**: reproduce the published Qwen3-VL MMLongBench-Doc number within the
  pre-registered tolerance. Block progress if it fails.
- **Judge–human agreement**: score ~100 hand-labelled items; report against the pre-registered
  bar. Block if below.
- **Memory-guessing check**: a no-document condition; flag and set aside questions answerable
  without the document.

**Docs.** `docs/EVALUATION.md`: judge protocol, agreement result, reproduction-gate result,
abstention definition, memory-guessing handling.

**Tests.** `tests/test_judge_metrics.py` — judge returns valid `Score`; accuracy CI correct on a
toy set; abstention/hallucination correct on a constructed labelled set; agreement computation
correct on a toy set.

**Checkpoint.** Human reviews agreement + reproduction numbers. **Hard gate:** no full sweep until
both pass.

---

## Stage 8 — Covariates (retrieval RQ2, classifier RQ3)

**Goal.** Real implementations of the two covariates; the retrieval-modality analysis at the heart
of RQ2.

**Build.**
- `covariates/retriever.py`: text `BM25 + BGE` and vision `ColPali/ColQwen`, returning ranked
  pages for `RetrievedTopK`. `metrics/retrieval.py`: page R/P/F1 vs gold, sliced by evidence
  modality (this feeds RQ2's retrieval-modality-sufficiency and the retrieve-vs-reason divergence
  cross-tab).
- `covariates/classifier.py`: cheap `doc_type` classifier (a small model via the `models/`
  registry is fine — it is a covariate, not an evaluated model). Returns predicted class +
  confidence; log accuracy vs gold.

**Docs.** Extend `docs/EVALUATION.md` with retrieval metric definitions, the evidence-modality
slicing, and the classifier's covariate role, all with the "as of implementations tested" hedge.

**Tests.** `tests/test_covariates.py` — retriever returns k pages; R/P/F1 correct on a toy gold
set; perfect retriever scores F1=1; per-modality slicing keys correctly; classifier returns a
valid class and logs accuracy.

**Checkpoint.** Human reviews retrieval quality per modality (the RQ2 divergence signal) and
classifier accuracy (which bounds whether RQ3 routing can pay off).

---

## Stage 9 — Experiment runner and result tables

**Goal.** Compose everything into the RQ result tables. No new capability — orchestration and
aggregation over cached pieces.

**Build.**
- `experiments/runner.py`: expand an `ExperimentConfig` into the cells each topic needs —
  **Representation** (ladder × doc-type frontier; evidence-modality × hop slice; accuracy–cost
  Pareto + visual-granularity panel), **Retrieval** (retrieval R/P/F1 by modality × evidence type;
  the retrieval–reasoning modality divergence cross-tab), **Deployment** (oracle/retrieved/full-doc
  attribution; the same across 2B/4B/8B/32B for bottleneck migration; accuracy-vs-page-F1 retrieval
  sufficiency curve; accuracy-vs-burying curve with the full-doc endpoint; routing vs uniform +
  classifier accuracy; abstention vs hallucination under neutral vs licensed prompt). Rely on the
  cache so partial runs resume.
- `metrics/frontier.py`: the pre-registered sufficiency-frontier rule (used for both the
  representation frontier and the retrieval-sufficiency frontier).
- `experiments/tables.py` + `cli/build_tables.py`: aggregate cached results into CSVs matching the
  RQ table skeletons in `tables/`.

**Docs.** `docs/RUNBOOK.md`: exact commands to (re)produce each table, expected runtime/cost, and
which conditions feed which RQ. For each table-producing run, give both the local invocation and
the Kaya one (`envs/mpvrdu/bin/python -m kaya.kaya run-gpu -- python -m cli.run_experiment
<config>`); note that the resumable cache plus the root-relative `results/` dir make local and
Kaya runs interchangeable and their outputs mergeable (pull Kaya `results/` back and rebuild
tables locally). Cross-link `kaya/KAYA_USER_GUIDE.md` for user setup and
`kaya/KAYA_AGENT_GUIDE.md` for operational details.

**Tests.** `tests/test_runner_tables.py` — runner produces the expected cell set for a tiny config
(no missing/extra conditions); frontier rule picks the right column on constructed inputs; each
table builder emits a CSV whose shape matches its skeleton.

**Checkpoint.** Human reviews a tiny-sample full run of all RQ tables before scaling to the full
dataset.

---

## Stage 10 (optional, time-permitting) — Additional datasets

**Goal.** Extend the validated v1 pipeline to the robustness/domain datasets *only if time allows*.
Not required for the paper's claims.

**Build.**
- Add per-dataset loaders behind the existing `load_*` pattern (LongDocURL, CUAD, DocFinQA,
  SlideVQA), each returning the same `Question` schema. LongDocURL additionally supplies true
  in-page boxes for a proper region-crop re-run of RQ1c and a headline replicate of RQ1.
- No pipeline, model, metric, or table changes — only new loaders and new rows in existing tables.

**Docs.** Extend `docs/DATA.md` per dataset (fetch strategy, field mapping, what it can/cannot do
per `dataset_profile.md`).

**Tests.** Reuse `tests/test_data.py` parameterised over the new loaders (schema invariants hold).

**Checkpoint.** Human decides per dataset whether the marginal robustness is worth the run cost.

---

## Appendix — Stage dependency summary

| Stage | Builds | Depends on | Frozen after |
|---|---|---|---|
| 0 | tree, deps, decision log | — | — |
| 1 | feasibility probes (incl. backend swap + vision retrieval) | 0 | open-item verdicts |
| 1.5 | Kaya Python runner + static config | 1 | Kaya operational substrate |
| 2 | schema + loader + render | 1.5 | `schema.py` data contracts |
| 3 | ABCs + `ModelInput` + stub orchestrator + cache | 2 | **all interfaces incl. `ModelInput`** |
| 4 | text + layout tools | 3 | — |
| 5 | visual tools | 3 | — |
| 6 | Qwen3-VL local + API backends + cost | 3,4,5 | prompt template |
| 7 | judge + repro gate + metrics | 3,6 | judge protocol (hard gate) |
| 8 | retriever + classifier covariates | 3,7 | — |
| 9 | runner + RQ tables | all | — |
| 10 | extra datasets (optional) | 2–9 | — |

**Invariant across every stage:** the Stage 3 interfaces (schema, ABCs, `ModelInput`) do not
change. Any pressure to change them is a checkpoint conversation recorded in `docs/DECISIONS.md`,
never a silent edit. This invariant is what makes both the tooling and the model family swappable
without rework.
