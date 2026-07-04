# Decisions and Implementation Log

Fixed decisions, the tree-to-paper map, frozen interfaces, and condensed stage
findings. This file (with `docs/implementation_plan.md`) replaces a separate
`ARCHITECTURE.md` — see `CLAUDE.md`. Treat fixed decisions as binding unless a
checkpoint changes them.

## Fixed decisions (v3)

The study is one EACL thesis: **the representation an MP-VRDU system needs is a
function of document type.** `docs/implementation_plan.md` / `PROJECT_SPEC.md`
are v3; the old three-topic v1 plan is archived at
`docs/implementation_plan_old.md`. Where they disagree, v3 wins.

- **Dataset:** MMLongBench-Doc primary (only source with doc type, evidence-
  modality labels, gold pages, and the unanswerable signal). LongDocURL is the
  dataset replication (Stage F4); other datasets are optional.
- **Reasoner:** Qwen3-VL-8B primary; 2B/32B appendix scale sanity (4B unused);
  InternVL3-8B replicates the RQ1 headline only. All behind one `Reasoner` ABC
  with local-weight and HTTP-API backends. Closed models are for comparison/
  judging, not the deployment recommendation.
- **Ladder:** `T` (Marker text), `TL` (Marker text + serialized bbox JSON),
  `TLV` (text + page image), `V` (page image). **Marker is the primary parser;**
  PyMuPDF/Docling/PP-Structure are the appendix parser-swap. Only `TLV`/`V` may
  attach images (modality boundary, enforced structurally + in `Payload`).
- **Binning (Option A, semantic domain), the single source of truth in
  `data/binning.py`:** text_heavy = Administration/Industry file + Academic paper
  + Research report/Introduction (578 Q / **70 docs**); in_between = Financial
  report + Guidebook + Tutorial/Workshop (412 Q / 50 docs); visual_heavy =
  Brochure (101 Q / 15 docs). The plan text's "54 docs" for text_heavy is a typo;
  native doc counts are 10+26+34 = 70. Option B (data-driven) is the P1 swap
  behind the same signature.
- **Metrics:** cost = latency@batch1 on one A100 (primary), text/vision tokens
  (secondary); sufficiency margin 3 points (sensitivity {2,3,5}); **document-
  level** bootstrap 95% CIs (1000 resamples over docs, not questions — questions
  cluster within 135 docs / 1091 Q).
- **Judge:** a *different family* from the reasoner, gated by Cohen's κ ≥ 0.75 vs
  200 hand labels (Stage F2). Two API judges implemented: GPT-4o-mini (OpenAI,
  paid) and Gemini-flash (Google, free tier). `StubJudge` is offline plumbing.
- **Paths:** root-relative; `.cache/ .data/ envs/ results/ logs/` live under the
  repo root on both machines. Two-machine Kaya model (local edit + sync; login
  for staging; compute for offline GPU jobs); all Kaya source/config/docs in
  `kaya/`.

## Architecture (tree ↔ paper)

Data flow: `Question` → `InputConditioner.condition` → render pages →
`Representation.build` → `Payload` → `ModelInput.from_payload` → `Reasoner.answer`
→ `Prediction` → `Judge.score` → `Score` → `ResultRow` (cached). `Retriever` and
`DocTypeClassifier` are covariates.

| Path | Role |
|---|---|
| `schema.py` | Data contracts: `Question`, `PageSet`, `Page`, `Payload`, `Prediction`, `Score`, `TextPart`/`ImagePart`. |
| `config.py` | `ExperimentConfig` + root-relative `ProjectPaths`. |
| `data/{loader,render,binning}.py` | Loader, PDF→pages substrate, Option-A binning. |
| `pipeline/conditioner.py` | Stage A: `Oracle`, `RetrievedTopK`, `FullDoc`, `BuriedOracle`. |
| `pipeline/representation.py` | Stage B: `T`/`TL`/`TLV`/`V` ladder; modality boundary. |
| `pipeline/reasoner.py` | Stage C: `Reasoner` ABC (swap point). |
| `pipeline/judge.py` | Stage D: `StubJudge`, `GPT4oMiniJudge`, `GeminiJudge`. |
| `pipeline/orchestrator.py` | Composes A→D per cell; owns the two cache layers. |
| `models/payload.py`, `models/__init__.py` | `ModelInput` + adapters; `get_reasoner(spec)` registry. |
| `covariates/{retriever,classifier}.py` | Retrieval + doc-type classifier covariates. |
| `tools/{text,layout,visual}.py` | Non-VLM channel functions the composers call. |
| `metrics/*` | accuracy (doc-level CI), retrieval, cost, frontier, abstention. |
| `experiments/T*_*.py` | One reusable `Experiment` per paper table (T1-T8); smoke and full share them. |
| `experiments/{base,registry,driver,corpus,tables}.py` | Experiment contract, name→experiment map, two-phase runner, corpus resolver, table primitives. |
| `cli/experiments.py`, `kaya/generate.py` | Run experiments: generate on GPU, judge/build anywhere. |

**Frozen interfaces (Stage-3 freeze; change only via a checkpoint recorded
here):** `schema.py` contracts; `models/payload.py::ModelInput` + its
`from_payload`/`to_chat_messages`/`to_local_prompt`; `InputConditioner.condition(
question, page_count)`; `Representation.build(pages)`; `Reasoner.answer(question,
model_input)`; `Judge.score(question, prediction)`; `Retriever.retrieve(question,
page_count, k)`; `DocTypeClassifier.classify(question)`; the orchestrator cache
key + `ResultRow` shape. (`Representation.build` takes rendered `Page`s, not a
`PageSet`, so the composer stays a pure page-encoder.)

**Caching (two layers, both under `results/cache/`).** (1) `ResultCache` — one
`ResultRow` per cell keyed by SHA-256 over `{question_id, doc_id, condition,
representation, model_spec, judge_spec, dpi}`; idempotent + resumable from disk.
(2) `PredictionCache` (additive, optional) — the reasoner output keyed the same
way **minus judge_spec**, so one prediction is scored by any judge without
re-running the model. `k`/burying level are encoded in the conditioner name
(`retrieved_k3`, `buried_n10`). Model spec is in both keys, so scaling/family
sweeps produce distinct, mergeable rows.

**Swap point.** The pipeline never imports a backend; it asks
`get_reasoner(spec)` for a `Reasoner` and hands it a `ModelInput`. Adding a
Qwen size or InternVL/GPT/Gemini is a new registry entry, no pipeline change.

## Environment and dependencies

Kaya env: `envs/mpvrdu` (Python 3.11, `requirements.txt`). Local RTX 5070
(Blackwell, sm_120) env: `envs/mpvrdu-local-gpu` (`requirements-local-rtx5070.txt`,
torch 2.8+cu128, no vLLM) because the Kaya `torch==2.7.0+cu126` wheel has no
sm_120 kernels. Both `pip check` clean.

**Why the dependency churn (and yes, it is normal for GPU ML).** The repo pulls
four heavy, fast-moving stacks into one env — vLLM (serving), ColPali/ColQwen
(retrieval), Marker/Surya (parsing), PaddleOCR/PaddleX (OCR) — plus a bleeding-
edge model (Qwen3-VL, which only landed in `transformers` 4.57). Each stack pins
`transformers`/`torch`/`pillow` independently and they barely overlap:

- `transformers==4.57.6`: Qwen3-VL needs ≥4.57; colpali `<4.58`; surya `≥4.56.1`;
  marker `<5`; vLLM `≥4.51.1`. Usable window is essentially just 4.57.x.
- `torch==2.7.0` (+cu126): vLLM 0.9.2 pins it **exactly**; that pin dominates.
- `pillow==10.4.0`: marker/surya require `<11` (the repo's original 11.3.0 broke).
- `huggingface_hub==0.34.4` (transformers 4.57 needs ≥0.34); added `hf_xet`.
- `paddleocr==3.1.0` must pair with `paddlex 3.1.x` (3.7.x breaks its predictor).
- `openai` is **capped at ≤1.90.0 by vLLM 0.9.2** — do not raise it. `google-genai`
  is already transitive via Marker, so the Gemini judge adds no new dependency.

The mitigation already in place: isolate the one exact-pinned troublemaker (vLLM)
by keeping a separate local env without it.

## Stage findings (condensed)

**Stage 1 (local probes, `.data/mmlongbench`).** Loader: 1091 question records
with the required fields; unanswerable = `answer == "Not answerable"` (244 =
22.36%). Abstention = normalized refusal surface ("not answerable", "cannot be
answered", "insufficient information", "unknown from the document", …);
hallucination = substantive answer on unanswerable Qs or on retrieved conditions
with page-recall 0. In-page evidence boxes **absent** → `region_crop` degrades to
page-level. Scanned fraction ~0.25 (a real embedded-vs-OCR slice). doc_type
counts: Research report/Intro 293Q/34d, Academic 204/26, Guidebook 156/22,
Tutorial/Workshop 139/17, Financial 117/11, Brochure 101/15, Admin 81/10.

**Stage 2 (data layer).** Normalizes `evidence_pages` 1-based → 0-based (original
kept in `Question.raw_fields`); rendered PNGs cached under
`results/cache/renders/` (reproducible compute artifacts, not dataset source).

**Stage 3 (freeze).** Defined all ABCs, `ModelInput`, the caching orchestrator,
`ExperimentConfig`, and a stub CLI; the whole pipeline runs on stubs end to end.
Frozen list above.

**MVP M1–M6.** M1: Option-A binning + frozen 7-doc smoke corpus (one per
doc_type) + config knobs (`smoke`, `bins`, `cost_metric="latency_bs1"`,
`sufficiency_margin=3`, `max_tokens`). M2: Marker `marker_text`/`marker_bbox_json`
primary (PyMuPDF fallback for tests), PaddleOCR `ocr()`, visual channel, and a
`kaya/prestage.py --smoke` barrier; `docs/TOOLS.md`. M3: resolved the Qwen3-VL
load path (`transformers==4.57.6` exposes `Qwen3VLForConditionalGeneration`),
`LocalVLMBackend`, frozen prompt `m3-qwen3vl-v1`, token/latency accounting;
`docs/MODELS.md`. M4: oracle ladder end to end through the resumable cache. M5:
real judge, document-level bootstrap accuracy, cost, sufficiency frontier, and
all eight table builders + `cli/build_tables.py`; `docs/EVALUATION.md`. M6:
`BM25BGERetriever` + `ColQwenRetriever`, page R/P/F1 + evidence-modality slices,
`QwenDocTypeClassifier`, and matched/cross + four routing policies.

## Per-experiment runs (two-phase: generate on Kaya, judge locally)

Each paper table is one reusable `Experiment` (`experiments/T1_headline.py` …
`T8_scale.py`); the same object serves the tiny smoke run and the full run (only
the config's model/corpus differ). They run the **real** pipeline (Qwen3-VL
reasoner, BM25+BGE / ColQwen retrievers, Qwen classifier, Gemini/GPT judge). No
stub reasoners or injected scorers on this path.

- **Why per experiment.** Running one table per Kaya job keeps jobs small (fast
  backfill) and lets a single experiment be re-run in isolation after a change.
  `experiments/registry.py` maps names/groups (`all`, `rq1`, `rq2`, `rq3`,
  `appendix`); `experiments/driver.py` runs them. Each experiment caches under its
  own dir (`results/cache/<smoke|full>/<name>/`), so runs never collide and merge
  cleanly. Aggregation-only tables (T2/T5/analytical relabels, T3/T8 in smoke)
  declare `depends_on` and build from T1's rows with no new generation; T6 adds
  retrieval cells; T7 adds the classifier as GPU side work.
- **Why two phases, split across machines.** Reasoner/retrievers/classifier need a
  GPU; the judge needs the internet — on Kaya those never coexist. So generation
  runs on Kaya (GPU, offline) and the judge + table build run **locally** after a
  `kaya.kaya pull` brings the prediction cache back (`pull` already rsyncs
  `results/`). The local judge phase loads **no models** (prediction-cache hits
  only), which keeps the workstation responsive. The phase-2 `_GuardRetriever` /
  `_SpecOnlyReasoner` raise if a cell was missed in generation.
- **Commands.** Kaya generate: `kaya.kaya submit kaya/generate.py -- --experiment
  T1_headline` (or `all`), then `kaya.kaya pull`. Local judge+build:
  `cli.experiments --phase judge --experiment all`. One-machine (GPU + internet):
  `cli.experiments --phase all`. Add `--full` for the full corpus/8B. Judge
  defaults to `gemini` (`--judge gpt-4o-mini` / `stub`). Tables → `results/tables/
  <smoke|full>/`. Judge keys are **not** forwarded to Kaya (only `HF_TOKEN` is);
  keep `GEMINI_API_KEY`/`OPENAI_API_KEY` in the local `.env`.

**Orchestrator cache-selection fix (bug found during the refactor).** `ResultCache`
defines `__len__`, so an empty one is falsy; `Orchestrator.__init__` used
`self.cache = cache or ResultCache(default)`, which silently discarded a fresh
per-experiment cache and fell back to the shared default file (all experiments
would collide). Changed to `cache if cache is not None else …` (same for
reasoner/judge). Not a frozen-interface change — the cache key and `ResultRow`
shape are untouched.

## Kaya operational notes (hazards that recur)

- **Queue waits.** The GPU request (`--partition=gpu --gres=gpu:1`) never changed;
  what grew was the resource envelope. `slurm` defaults are now `cpus=4, mem=24G,
  time=00:30:00` (were `8/64G/02:00:00`). A long walltime is the main backfill
  killer — short jobs slot into gaps, a 2h job waits for a full slot. Raise
  per-job with `--time/--mem/--cpus-per-task` for the Section-2 grid.
- **`run` vs `submit`.** `run` executes on the login node (SSH, no SLURM) unless
  the `.py` header says `target=gpu`; `submit` always goes through SLURM (generated
  sbatch for `.py`, as-is for `.sbatch`). GPU resources come from `kaya/config.json`
  or `--partition/--gres/--cpus-per-task/--mem/--time/--account/--qos`.
- **Offline caches.** Compute jobs run HF-offline and must read root-relative
  caches: the runner exports `HF_HOME`/`HF_HUB_CACHE=<root>/.cache`, unsets
  inherited `TRANSFORMERS_CACHE`, and sets `MODEL_CACHE_DIR=<root>/.cache/datalab/
  models` (Marker/Surya) plus Paddle/Docling/Torch/Xet paths. `prestage.py`
  stages Qwen weights, BGE, ColQwen, Marker/Surya, PaddleOCR, Docling; it is
  idempotent (probes the Hub cache with `local_files_only` before any network).
- **Orphaned remote processes.** Long login-node runs use `ssh -tt` (pty) +
  keepalives + a `trap … HUP TERM INT` process-group kill so a local Ctrl-C tears
  down the remote tree instead of orphaning it (HF's blocking sockets have no read
  timeout and would hang forever). Never hand-edit the remote mirror — `push` is
  `rsync --delete`. `logs/` must exist before `sbatch`.
- **Live config to re-confirm (drifts):** modules `Anaconda3/2024.06`,
  `cuda/12.6.3`, partition `gpu` (nodes k[026-042], 34 GPUs, MaxTime 3d), GRES
  `gpu:1`; account/QOS blank (group membership grants access).
