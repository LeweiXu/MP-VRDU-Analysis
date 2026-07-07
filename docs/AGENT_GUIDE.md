# Agent guide (decisions, implementation log, reference)

The coding agent's single reference for how this repo is built. It holds the
fixed decisions, the tree-to-paper map, the frozen interfaces, the caching
contract, condensed stage findings, and (at the end) the implementation
reference for models, the data layer, tools, and evaluation. Together with
`docs/implementation_plan.md` (the staged build plan) it replaces a separate
`ARCHITECTURE.md`, see `CLAUDE.md`. The user-facing "what and why" plus the
run commands live in `docs/USER_GUIDE.md`. Treat fixed decisions as binding
unless a checkpoint changes them; you may edit this file directly to record
implementation-relevant decisions and deviations.

This file absorbed the former `MODELS.md`, `DATA.md`, `TOOLS.md`, and
`EVALUATION.md` (now the "Implementation reference" section below). The old
`context.md` was a pre-v3 conversation summary and was dropped as superseded by
`USER_GUIDE.md` / `implementation_plan.md`.

## Fixed decisions (v3)

The study is one EACL thesis: **the representation an MP-VRDU system needs is a
function of document type.** `docs/implementation_plan.md` / `docs/USER_GUIDE.md`
are v3; the old three-topic v1 plan is archived at
`docs/implementation_plan_old.md`. Where they disagree, v3 wins.

- **Dataset:** MMLongBench-Doc primary (only source with doc type, evidence-
  modality labels, gold pages, and the unanswerable signal). LongDocURL is the
  dataset replication (Stage F4); other datasets are optional.
- **Hardware scope (this team):** the only GPUs we can reach are Kaya V100 16GB
  (1 or 2 per node → 16 or 32GB). **The 32B model is out of scope for us** — it
  does not fit our V100s. When the 32B scale-sanity row (Table 8) is actually
  needed, run it on the supervisor's A100 account (his QOS/allocation), or have
  him run that job for us; it is the only piece that requires A100-class VRAM.
  The 8B primary runs on 2×V100 (32GB, `device_map="auto"` shards it) or on 1×
  V100 only via CPU-offload/quantisation (bf16 8B weights alone are ~16GB).
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
| `experiments/G*_*.py` + `base.py`, `registry.py` | One `GenerationTask` per file (G1..G6); base is the ABC + cell factories; registry collects them for `--generation`. |
| `experiments/driver.py` | The generate (GPU) + judge (local) engine over tasks; the phase-2 guards; `config_from_args`. |
| `experiments/{tables,reporting}.py` | Pure per-table aggregation functions; the table -> source-task routing that writes CSVs + `.md`. |
| `experiments/{paths,corpus}.py` | Shared cache/table layout + status/logging; corpus resolver. |
| `cli/{generate,judge,build}.py` | Thin runnable wrappers: generate on GPU (the half a cluster submits), judge/build locally. |

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
`scripts/prestage.py --smoke` barrier (tools reference below). M3: resolved the Qwen3-VL
load path (`transformers==4.57.6` exposes `Qwen3VLForConditionalGeneration`),
`LocalVLMBackend`, frozen prompt `m3-qwen3vl-v1`, token/latency accounting;
(models reference below). M4: oracle ladder end to end through the resumable cache. M5:
real judge, document-level bootstrap accuracy, cost, sufficiency frontier, and
all eight table builders + `experiments/reporting.py` (evaluation reference below). M6:
`BM25BGERetriever` + `ColQwenRetriever`, page R/P/F1 + evidence-modality slices,
`QwenDocTypeClassifier`, and matched/cross + four routing policies.

## Role split: generation tasks, judge, build

The study is organized by **generation task**, not by paper table, because most
tables are pure aggregations of the same generated predictions. The `experiments`
package is the library (task defs + engine + builders); the runnable roles are
thin wrappers `cli/{generate,judge,build}.py`. They run the **real** pipeline
(Qwen3-VL reasoner, BM25+BGE / ColQwen retrievers, Qwen classifier, Gemini/GPT
judge). No stub reasoners or injected scorers on this path.

- **Generation tasks (the only GPU work), one file each.** `experiments/G*_*.py`
  define `G1_sufficiency` (oracle ladder, primary 8B — the source rows for tables
  1, 2, 5, 7), `G2_family` (InternVL ladder → table 3), `G3_dataset` (held-out
  text_heavy+in_between ladder → table 4), `G5_retrieval` (matched/cross cells +
  retrieval R/P/F1 → table 6), and `G6_classifier` (doc-type classifier side work
  → table 7's routing price). Adding an experiment is just a new `G*_*.py` +
  a line in `experiments/registry.py`; a scale task (2B/32B → table 8) is out of
  scope for now. Each task caches under `results/cache/[<run-tag>/]<smoke|full>/<task>/`
  (run-tag namespaces the whole tree, so it sits *before* `<smoke|full>`), so jobs
  never collide. Groups: `all`, `reasoners`.
- **Table routing replaces `depends_on`.** `experiments/reporting.py`'s `TABLES`
  registry declares each table's source task(s): table1←G1, table3←G1+G2,
  table4←G3, table6←G5 (+retrieval side), table7←G1 (+classifier side), table8←G1.
  This matters because the `experiments/tables.py` builders mostly don't filter by
  model_spec, so a table is only correct when handed exactly its source tasks'
  rows (unioning all specs would corrupt table1's single-spec aggregate).
- **Why the roles split across machines.** Reasoner/retrievers/classifier need a
  GPU; the judge needs the internet — on Kaya those never coexist. So generation
  runs on Kaya (GPU, offline); `cli.judge` (scores predictions, no tables)
  and `cli.build` (aggregates into CSVs + `.md`) run **locally** after a
  `kaya.kaya pull`. Judge loads **no models** (prediction-cache hits only); the
  `_GuardRetriever` / `_SpecOnlyReasoner` raise (`CacheMiss`) if a cell was missed
  in generation, which `--continue-on-error` turns into a skip.
- **Commands.** Kaya generate: `kaya.kaya submit cli/generate.py --
  --generation G1_sufficiency` (or `all`), then `kaya.kaya pull`. Local:
  `python -m cli.judge --generation all` then `python -m cli.build`.
  Add `--full` for the full corpus/8B. Judge defaults to `gemini` (`--judge
  gpt-4o-mini` / `stub`). Tables → `results/tables/<smoke|full>[-<run-tag>]/`.
  Judge keys are **not** forwarded to Kaya (only `HF_TOKEN` is); keep
  `GEMINI_API_KEY`/`OPENAI_API_KEY` in the local `.env`.

**Default per-bin document-level subset for full runs.** A full mmlongbench run
now defaults to ~100 questions per Option-A bin instead of all 1091, so a full T1
clears the Kaya queue in a couple of hours (short walltime backfills) rather than
needing a 2-day slot. `ExperimentConfig.per_bin_sample` (default 100) +
`sample_seed` (default 0) drive it; `experiments/corpus.py::sample_questions_per_bin`
draws **whole documents** per bin (never splitting a document) until the bin
reaches the target, honoring the USER_GUIDE §9 document-level sampling rule.
Bins below the target are kept whole, so visual-heavy stays at all 101 Q / 15 docs
(it cannot be subset; SlideVQA is the planned visual robustness anchor). Default
subset = 309 Q (text_heavy 100/11 docs, in_between 108/13, visual_heavy 101/15).
A second `--sample-seed` gives a largely disjoint subset for a robustness rerun.
CLI: `--per-bin-questions N` (0 = whole corpus, the old behaviour), `--sample-seed
N`; an explicit `--questions N` global cap still overrides. Applies to mmlongbench
full runs only (smoke and LongDocURL ignore it). **Gate provenance:** the F1
frontier-divergence gate as specced wants the whole corpus, so a subset run is a
fast preview; record whether an F1 verdict came from the 100/bin subset or
`--per-bin-questions 0`. Not a frozen-interface change (additive config +
corpus-resolver logic; cache key/`ResultRow` untouched).

**Orchestrator cache-selection fix (bug found during the refactor).** `ResultCache`
defines `__len__`, so an empty one is falsy; `Orchestrator.__init__` used
`self.cache = cache or ResultCache(default)`, which silently discarded a fresh
per-experiment cache and fell back to the shared default file (all experiments
would collide). Changed to `cache if cache is not None else …` (same for
reasoner/judge). Not a frozen-interface change — the cache key and `ResultRow`
shape are untouched.

## Section-2 gates and full-stage tooling (F1-F6)

Gate tooling is implemented in `experiments/gates.py` and exposed through
`scripts.gates`; commands are recorded in `docs/USER_GUIDE.md` (Runbook).

- **F1 frontier divergence.** `scripts.gates frontier` reads the full Table-1 CSV and
  returns Go when at least two Option-A bins have different frontiers. The full
  Qwen3-VL-8B run has not been executed in this log, so the verdict is pending.
- **F2 judge-human agreement.** `scripts.gates agreement-sample` writes the 200-row
  human labelling sheet from judged rows; `agreement-score` computes Cohen's
  kappa over `correct` / `incorrect` / `abstained` and gates at 0.75. The human
  labels and kappa result are pending.
- **F3 classifier feasibility.** `scripts.gates classifier-pilot --full` samples 100
  distinct documents, runs the first-two-page Qwen3-VL-2B classifier, and gates
  top-1 Option-A bin accuracy at 0.70. The pilot result and RQ3 scope decision
  are pending.

**F4-F6 tooling.** The Section-2 full experiments are implemented but not yet
run at full scale in this log. F4 adds the Table-2 analytical slice, the
InternVL3-8B Table-3 replication backend, and the LongDocURL Table-4 loader /
experiment. LongDocURL annotations are read from staged
`.data/longdocurl/LongDocURL_public.jsonl` or a cached `dengchao/LongDocURL`
snapshot; PDFs must still be staged manually under
`.data/longdocurl/documents/<doc_no>.pdf` because the public annotation cache
only records source paths. F5 now computes evidence-composition mediation and
matched vision-vs-cross text-to-vision retrieval rows from real cached
predictions/retrieval side records. F6 now builds one corpus-level row for each
routing policy and amortizes classifier latency as total classifier time divided
by evaluated question rows.

The model registry now dispatches Qwen3-VL local sizes 2B/4B/8B/32B to the
Hugging Face local backend and `internvl3-8b-local` to
`OpenGVLab/InternVL3-8B`. Other non-Qwen families remain explicitly stubbed
until their replication stage.

## Vision-token cap (CUDA OOM fix)

The F1-F6 grid OOM'd on Kaya V100s (16GB) even for the 2B smoke model, allocating
6.4 GiB inside a single SDPA attention op. Root cause was **not** model size: it
was an uncapped visual sequence. `data/render.py` rasterizes pages at 144 DPI
(~1.9M px = ~2500 vision tokens/page after Qwen's 28x28 merge), and
`LocalVLMBackend` fed them through the processor with no `max_pixels`, so Qwen's
~12.8M px default never downscaled them. A multi-page oracle cell built a 5k-10k
token sequence; the V100 is Volta (sm_70) with no FlashAttention-2, so attention
falls back to the O(seq^2) SDPA math kernel and the score tensor blew past 16GB.

Fix: `ExperimentConfig.max_pixels` (default 1,003,520 = 1280*28*28, ~1300
tokens/page) plus a size-aware override `config.max_pixels_for_spec` /
`MAX_PIXELS_BY_SIZE` that tightens the cap for the bigger reasoners (8B ->
602,112 = 768*28*28, ~800 tok/page; 32B -> 401,408 = 512*28*28, ~520 tok/page;
2B/4B keep the 1.0M default). Threaded `get_reasoner(spec, max_new_tokens=,
max_pixels=)` -> `LocalVLMBackend`, applied via the per-image `max_pixels` key
that `qwen_vl_utils.process_vision_info` honors. Same wiring also fixed
`_reasoner_for` ignoring `config.max_tokens` (full runs were capped at 64 new
tokens). `cli/generate.py` now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
True` as a fragmentation mitigation. InternVL is untouched (fixed 448px tiling
already bounds its vision tokens). Not a frozen-interface change: `get_reasoner`
gained optional kwargs; `Reasoner.answer` and the cache key are unchanged.

**Verbose smoke tooling.** `experiments.paths.configure_logging(verbose)` sends
`mpvrdu.*` logs to stdout; smoke runs are verbose (DEBUG) by default (`--quiet`
opts out, `--verbose` forces it for full). Generation logs per-task banners
and per-cell start/result lines; the orchestrator logs each stage (conditioner ->
render -> representation -> reasoner -> judge) at DEBUG, so an OOM/crash points at
the exact cell and stage. On failure the full traceback is logged to stdout (the
SLURM log), not just the per-task `generate_status.json`. One
`kaya.kaya submit cli/generate.py -- --generation all --continue-on-error`
job runs every task with per-task isolation: a
failing task records its status and the run continues to the next.

## GPU memory management (parser/reasoner co-residence)

After the `max_pixels` cap, a full smoke run still OOM'd — but for a different
reason the verbose logs exposed: **multiple model stacks piling onto one 16GB
V100 and never being freed.** `create_model_dict()` in `tools/layout.py` reloads
the whole Surya stack for every page/cell (the ~170s stalls), and nothing in the
pipeline ever ran `torch.cuda.empty_cache()`. So the Marker/Surya parser, the
ColQwen retriever, and the Qwen reasoner/classifier all sat on the GPU at once,
and a single grouped job accumulated them until a late task started at ~15GB in use.

Fix (implemented): make the parser and the reasoner never share VRAM, and free
GPU memory between stages/experiments.

- **Marker disk cache** (`tools/layout.py`): `marker_text`/`marker_bbox_json`
  cache each page's artifact under `results/cache/marker/` (root derived from the
  page's render path, so tests use their tmp cache and prod uses the repo cache;
  the frozen `Representation.build(pages)` signature is untouched). Real Marker
  output is cached; the pymupdf fallback is not. On a warm cache the reasoner
  phase never loads Surya. Also kills the per-cell reparse.
- **Parse pre-pass** (`experiments/driver.py::generate` + `Orchestrator.prewarm_cell`):
  before the reasoner weights load, run condition→render→`build` for every cell to
  warm the Marker (and retrieval) caches, then `retriever.unload()` +
  `free_gpu()`. The reason pass then has the whole GPU; `prewarm_cell` skips cells
  whose prediction is already cached.
- **Explicit frees**: `free_gpu()` (gc + `empty_cache` + `synchronize`) runs after
  the pre-pass, after each spec's reason pass (`LocalVLMBackend.free()` drops the
  weights), and after `run_side`. Retrievers gained `unload()`
  (`ColQwenRetriever`/`BM25BGERetriever`/`MemoizedRetriever` keep their memoized
  rankings, drop the model). The T7 classifier now also receives `max_pixels`.
- **GPU count is controlled by `--gres`, not code.** `device_map="auto"` adapts:
  1×V100 → CPU-offload for 8B (slow), 2×V100 → shard (fits). Testing plan: run
  T1/T6/T7 with the 8B on `--gres gpu:v100:1` vs `gpu:v100:2` to see whether one
  GPU suffices after this memory management or two are required.

Not a frozen-interface change: additive caches + `unload`/`free`/`prewarm_cell`
methods; the cache keys and `ResultRow` are untouched.

## Single-16GB-GPU 8B feasibility (quantization option)

Probe `1003970` (`kaya/single_gpu_probe.py`, 2026-07-05) on one V100 16GB:
8B **bf16 does not fit** one V100 (OOM at load), but **4-bit NF4 fits at 7.1GiB
peak and 8-bit at 10.2GiB**, both generating correct answers. Quantization is not
faster (Volta has no FA2; image cells ~30s either way) — the payoff is
schedulability (1-GPU jobs backfill in minutes vs an overnight wait for a 2-GPU
node). Tradeoff: quantized weights are not the pre-registered bf16, so main/table
numbers stay on **2×V100 bf16**; 4-bit is for fast iteration and a possible
appendix quant-sensitivity row. `bitsandbytes==0.49.2` installed into the remote
`envs/mpvrdu` (rsync-excluded, persists). Running the pipeline quantized would need
a `quantization` kwarg threaded `get_reasoner -> LocalVLMBackend` plus
`quantization` added to the cache key (additive, not a frozen-interface change);
not wired yet. Full writeup: `SINGLE_GPU_8B_FEASIBILITY.md`.

## V100 has no efficient attention: reasoner input-token cap

Probe `1004834` (`kaya/attn_probe.py`) established on a real V100 that **Qwen3-VL
has no memory-efficient attention kernel available on Volta (sm_70)**: forcing
`SDPBackend.EFFICIENT_ATTENTION` raises "No available kernel" for both bf16 and
fp16, and there is no FlashAttention-2 on sm_70. So attention **always** runs the
math kernel, which materializes the full `[heads, seq, seq]` score matrix
(O(seq^2) memory). A single dense-page `TL` cell (serialized bbox-layout JSON is
~30k tokens) then tries to allocate ~105 GiB and OOMs — **on 2xV100 too**, since
attention runs per-GPU. This is a critical-path bug the 4-bit smoke surfaced
(the earlier runs never reached the reasoner, dying in the parse pre-pass).

Fix: a size-aware **input-token cap** (`config.max_input_tokens`, default 8192;
`MAX_INPUT_TOKENS_BY_SIZE` = 8B->5120, 32B->3072) threaded
`_reasoner_for -> get_reasoner -> LocalVLMBackend` (and into
`QwenDocTypeClassifier`). `LocalVLMBackend.render_prompt`/`_truncate_context`
trims the context text to the budget left after reserving for images + template,
**keeping every image placeholder** so image counts still match; truncated cells
put images first then trimmed text. 5120 keeps the 8B score matrix ~3.4 GiB,
which fits on one 16GB V100 (4-bit) and per-GPU on 2xV100 (bf16).

**Deviation recorded:** this truncates the text of very long `T`/`TL`/`TLV` cells
(dense pages) on the main runs, not just the smoke. It is forced by V100 hardware
(no O(seq) attention). The `TL` bbox-layout JSON is the main offender and is a
candidate for a more compact serialization later (would raise the effective text
budget). Not a frozen-interface change: additive optional kwargs; cache key +
ResultRow unchanged. `_sdpa_context` still prefers the efficient kernel when one
exists (e.g. an A100), and is a harmless no-op on the V100.

**Second OOM: 2xV100 weight-shard headroom.** With the input cap in place, the
bf16 8B on 2xV100 still OOM'd after ~50 real cells (`GPU 1: 13.0GiB in use, a
2.9GiB alloc, 2.76GiB free` — missed by 0.14GiB). Cause: `device_map="auto"`
fills each GPU with weights and leaves almost no room for the activation/KV/
attention peak, so a longer-text cell tips one GPU over. Fixes: (1)
`LocalVLMBackend._max_memory_map` reserves ~5GiB/GPU when sharding across >1 GPU
(caps each V100 at ~10GiB of weights, leaving ~5GiB free); single-GPU/4-bit loads
are untouched. (2) Tightened the 8B input cap 5120 -> 4096 (attention score
~3.4GiB -> ~2.1GiB). Both keep the peak comfortably inside 16GB per GPU. Verified
by a small `t1-memtest` on 2xV100 before resubmitting the full run.

**Third OOM: vision tokens are not total-capped (handled by per-cell skip).**
`max_input_tokens` truncates the *text* context but not the *vision* tokens, so a
question with many gold pages contributes vision tokens on top of the 4096 text
cap and the O(seq^2) SDPA math-attention on a V100 OOMs. This is a long-tail
problem: on the full corpus the oracle-page distribution is mostly 1-2 pages, but
a handful of cells have 9, 10, even **24 gold pages**. At those, `4.27GiB` gets
requested in SDPA and OOMs — **even at `--visual-resolution low`** (the two full
runs, jobs 1009634/1009635, both died at the same 911th G1 cell). Per-page
downscaling helps but can't save a 24-page cell; a true total-vision cap
(count vision tokens, shrink `max_pixels`/drop pages to a budget) is still the
"correct" fix and remains unimplemented.

**Mitigation (implemented): per-cell skip.** `experiments/driver.py::generate`
now takes `skip_failed_cells` (wired to `--continue-on-error`): a cell that raises
is logged, the GPU freed (`free_gpu`, recovers activation memory after a CUDA
OOM), and the loop continues instead of aborting the whole task. So a run
completes over the ~99.7% of cells that fit and simply omits the handful of
pathological many-page cells; the judge (`--continue-on-error`) then scores
what was generated. This turns a fatal task failure into a small, documented gap.

**Explicit filter (implemented): drop >10-evidence-page questions.**
`experiments/corpus.py::load_questions` now drops any question whose gold
evidence spans more than `MAX_EVIDENCE_PAGES` (=10) pages, applied *after* per-bin
sampling so it never perturbs which documents get drawn (the existing bf16 cache
stays consistent). On the full corpus this is 7 questions (12-24 pages; incl.
`mmlongbench:000855` at 24). This is a deterministic superset of the per-cell
skip: at `--visual-resolution low` only the 24-page question actually OOMs, so the
cutoff is conservative (it also drops 6 that generated fine at low res), but it's
resolution-independent and doesn't rely on a runtime OOM. Both phases resolve the
same filtered set, so judge/build agree. Chosen over a higher cutoff to keep the
rule simple; raise `MAX_EVIDENCE_PAGES` if the low-res 12-21 page cells are wanted
back.
**Also:** 4-bit weights are ~7GB vs bf16's ~13GB resident, but 4-bit is *slower*
per cell (dequant) and, at full resolution, timed out a 12h run; bf16 at
`--visual-resolution low` completed in 10h. See `docs/HANDOFF.md`.

## Quantized reasoner as a model-spec variant

To run the 8B on one 16GB V100 (see feasibility above), quantization is exposed
as a **spec suffix**, not a new cache-key field: `qwen3vl-8b-local-4bit` /
`-8bit`. `ModelSpec.parse` strips the trailing `-4bit`/`-8bit` into
`ModelSpec.quantization` while `name` keeps the full string, so quantized runs
get **their own cache rows** (the cache key uses the spec string, unchanged
structurally) and `size` still resolves to `8b` for the per-size pixel cap.
`get_reasoner` passes `quantization` to `LocalVLMBackend`, which builds a
bitsandbytes `BitsAndBytesConfig` (4-bit NF4 double-quant, or 8-bit) in
`_load_components`; `model_id_for_spec` strips the suffix to find the base
checkpoint. `config.quantization` (None/"4bit"/"8bit") appends the suffix to
`reasoner_spec`; `--quantization` on `cli.generate` sets
it (must match between generate and judge phases). Not a frozen-interface change:
`get_reasoner`/`LocalVLMBackend` gained an optional kwarg, cache key + ResultRow
unchanged. `bitsandbytes==0.49.2` is in the remote env only, so the config-build
test skips locally. Mains stay bf16; 4-bit is single-GPU iteration + a possible
appendix row.

## Judge-phase robustness (retry + partial cache)

Two fixes so the judge phase survives real runs against a flaky free-tier API
and a partial generate cache:

- **Transient-error retry.** `pipeline/judge.py` wraps both API judge calls
  (`GeminiJudge`, `GPT4oMiniJudge`) in `_with_retry`: exponential backoff on
  429/5xx (`ServerError`, connection/timeout), non-transient errors (400/401)
  still raise on the first try. Free-tier gemini flash returns sporadic 503s; one
  used to kill a whole ~800-cell judge run mid-corpus even though the scored rows
  are cached in `results.jsonl`.
- **Partial-cache tolerance.** The judge only re-scores cached predictions; a
  cell that generate never produced (e.g. after the vision-token OOM) hit the
  guard reasoner/retriever and hard-errored. The guards now raise `CacheMiss`
  (subclass of `RuntimeError`), and `--continue-on-error` makes the judge skip
  those cells and log the count, so a partial cache still builds a partial table
  (T1 built from 826/1236 cells, 538 skipped). Not a frozen-interface change:
  `run_cell`, the cache key, and `ResultRow` are untouched; `judge()`/`run_judge()`
  gained an optional `skip_uncached`/`continue_on_error` kwarg.

## Full-run knobs: visual resolution, run-tag, combined tables

Added for the two full 8B runs (4-bit current-res vs bf16 aggressive-downscale):

- **`--visual-resolution {full,high,med,low,min}`** (`config.visual_resolution`).
  Fixes the per-page vision-token budget for every reasoner via
  `config.max_pixels_for_resolution`, overriding the size-aware
  `max_pixels_for_spec` default. `high`≈768 tok/page is the current 8B default;
  `low`≈320, `min`≈224 downscale harder so a many-gold-page cell's O(seq^2)
  attention fits bf16 on 2xV100 (the "third OOM"). Not in the cache key, so clear
  or `--run-tag` the cache when changing it for one spec.
- **`--run-tag TAG`** (`config.run_tag`). Namespaces the whole cache tree under
  `results/cache/<TAG>/` (predictions, renders, marker, side records) and tables
  under `results/tables/<mode>-<TAG>/`. Needed because two concurrent
  `--generation all` jobs otherwise write the same paths, and both the render
  cache (check-then-write in `data/render.py`) and the prediction cache (plain
  append) corrupt under cross-node concurrent writes. Judge/build must pass the
  same tag. Implemented by rebuilding `config.paths.cache_dir` in
  `__post_init__`; the frozen prediction/cache *key* is untouched.
- **Combined markdown tables.** `cli.build` (and
  `experiments.tables.render_tables_markdown`) writes one `all_tables.md` with all
  eight tables alongside the CSVs; tables with no cached rows render a blank
  skeleton row so the report always shows all shapes.

## `cli/` is the three experiment roles only; utilities live in `scripts/`

`cli/` holds exactly `generate.py`, `judge.py`, `build.py` (the thin wrappers a
run submits). Every other runnable moved to `scripts/`: `run_probe.py`,
`gates.py`, plus the new inspection/annotation tools below. Invoke them as
`python -m scripts.<name>`. This is a deliberate structure decision (not a frozen
interface); the target tree in `docs/implementation_plan.md` reflects it.

**Stale `temp/` cache (diagnosis, resolved).** A gitignored `temp/cache/full/`
tree keyed by *table* (`T1_headline`, `T2_analytical`, ...) survived from before
the generation-task refactor and made it look like GPU output was still
table-organized. It was dead (no code referenced it) and has been deleted. The
live cache is task-keyed: `results/cache/[<run-tag>/]<smoke|full>/<task>/`
(run-tag namespaces the whole tree, so it sits *before* `<smoke|full>`; see
`experiments/paths.py::experiment_paths` + `config.py` `__post_init__`). The gate
CLI's old hardcoded `results/cache/full/T1_headline/...` defaults were repointed
to run-tag-aware resolution via `experiment_paths` (`scripts/gates.py`
`default_results_path` / `default_table1_path`, driven by `--run-tag` / `--full`).

**Inspection + annotation tooling.**
- `experiments/inspect.py` (+ `scripts/inspect_results.py`): join a task's
  `predictions.jsonl` and `results.jsonl` back to the `Question` + PDF, render the
  fed pages, and dump each cell into `./inspect/<slug>/` (copied PDF, page PNGs,
  and an `info.md` that lists *every* `CachedPrediction` and `ResultRow` field).
  Filters: question/doc/representation/condition/incorrect-only/abstained-only.
  Reuses `render_pdf` against the run's shared render cache (hits, not re-renders).
  Limitation: the judge's free-text rationale is not persisted (only the
  verdict/score in `ResultRow`), so it can't be shown.
- Gate F2 viewer: `scripts.gates agreement-sample --render` calls
  `experiments.gates.render_agreement_packet` (reusing `experiments/inspect.py`)
  to render the sampled cells' pages into a scrollable `agreement_view.md`
  alongside the CSV, so a human can label `human_label` while seeing the document.
  `agreement-score` (Cohen's kappa, gate 0.75) is unchanged.
- `scripts/annotate_docs.py`: interactive, resumable per-document annotation of the
  135 docs. `annotate` opens each PDF in turn and prompts a menu per field
  (text/visual bin, scanned vs digital, dominant visual element, multi-column),
  seeded with `doc_type_bin` + `data.render.classify_scanned` priors, writing
  `annotations/doc_labels.csv` (a committable dir, not gitignored `results/`) after
  every doc. `score` reports human-bin-vs-`auto_bin` agreement per doc_type (this
  tests the three-domain assumption) plus the scanned fraction. `classify_scanned`
  is the shared version of the Stage-1 scanned-vs-born-digital heuristic.

## Kaya operations (elsewhere)

The Kaya operational how-to, the `clear-cache` command, SLURM queue hazards,
offline-cache setup, and module/config drift live in `kaya/KAYA_AGENT_GUIDE.md`,
not here. This guide stays about the local pipeline and its decisions; the
hardware findings above name the V100 only as the context those fixes were made
for.

---

# Implementation reference

Condensed from the former `MODELS.md`, `DATA.md`, `TOOLS.md`, and
`EVALUATION.md`. The frozen contracts are in the decisions above; this is the
"how each layer actually behaves" reference.

## Models (reasoner backends + prompt)

- **Load path.** `transformers==4.57.6` (top of the colpali `>=4.53.1,<4.58.0`
  window) exposes `Qwen3VLForConditionalGeneration` / `...MoeForConditionalGeneration`
  / `Qwen3VLProcessor`, resolving the Stage-1 class gap without moving Marker,
  Surya, vLLM, or ColPali outside their windows.
- **Registry.** `qwen3vl-{2b,4b,8b,32b}-local` dispatch to the shared HF backend
  (`Qwen/Qwen3-VL-*-Instruct`); `internvl3-8b-local` -> `OpenGVLab/InternVL3-8B`
  via `models.internvl.LocalInternVLBackend` (same `Reasoner.answer` contract).
  A trailing `-4bit`/`-8bit` suffix selects a bitsandbytes-quantized load of the
  same checkpoint (see "Quantized reasoner" above). Other families stay stubbed.
- **Frozen prompt.** Qwen template `m3-qwen3vl-v1` (InternVL `f4-internvl3-v1`),
  one fixed template held constant across the four rungs;
  `ModelInput.to_local_prompt()` supplies `{context}` and each `<image>`
  placeholder becomes a Qwen image block in page order.
- **Accounting** per `Prediction`: `input_text_tokens` (tokenizer count, image
  placeholders stripped), `input_visual_tokens` (Qwen `image_grid_thw` estimate),
  `output_tokens` (generated ids after trimming), `latency_s` (batch=1 wall clock
  around `generate`), plus metadata (backend, model id, template version,
  `max_new_tokens`, `max_pixels`, `max_input_tokens`, `quantization`, image count).
- **Closed models** are comparison/judge only, behind the same ABC via
  `ModelInput.to_chat_messages()`; the pipeline never imports vendor SDKs.

## Data layer

- **Paths** (root-relative both machines): dataset `.data/mmlongbench`, parquet
  `.data/mmlongbench/data/*.parquet`, PDFs `.data/mmlongbench/documents/*.pdf`,
  render cache `results/cache/renders/<pdf-stem>__dpi<N>/page_XXXX.png`.
- **`load_mmlongbench()`** -> `Question`: `id` (`mmlongbench:000000`), `doc_id`,
  `question`, `gold_answer`, `answer_format`, `doc_type`, `evidence_pages`
  (normalised 1-based -> 0-based), `evidence_sources`, `hop` (from evidence-page
  count: none/single/multi), `is_unanswerable` (gold == "Not answerable"),
  `raw_fields` (+`source_dataset="mmlongbench"`).
- **`render_question_pages()`** resolves the PDF and renders the gold pages;
  unanswerable questions with no gold pages render page 0. Each `Page` carries the
  0-based index, PDF path, optional cached PNG, and PyMuPDF line spans.
- **LongDocURL loader** (`load_longdocurl()`) still exists (reads
  `.data/longdocurl/LongDocURL_public.jsonl` or the `dengchao/LongDocURL`
  snapshot; PDFs staged under `.data/longdocurl/documents/<doc_no>.pdf`), but
  **Table 4 no longer uses it** (see "Table 4" below); it is kept for a possible
  future replication.

## Tools (non-reasoner channels)

- **Primary ladder (Marker).** `tools.layout.marker_text(pages)` and
  `marker_bbox_json(pages)` feed `T`/`TL`/`TLV`; `tools.visual.full_page(pages)`
  and `resolution(pages, scale)` feed `TLV`/`V`. Marker (`marker-pdf==1.10.2`,
  GPL-3.0 code, Datalab OpenRail-M weights) is primary, run without LLM mode.
  Marker output is disk-cached under `results/cache/marker/` (the pymupdf fallback
  is not cached).
- **Appendix/fallback.** `tools.text.embedded` (PyMuPDF), `tools.text.ocr`
  (PaddleOCR PP-OCRv5), Docling parser-swap, and `tools.visual.region_crop` which
  degrades to full page (MMLongBench has no in-page boxes). The pymupdf fallback in
  `marker_bbox_json` exists only so local tests run before Marker is installed;
  `prestage --smoke` calls Marker with `allow_fallback=False`.
- **Prestage.** `scripts/prestage.py [--smoke]` stages Qwen weights, BGE, ColQwen,
  Marker/Surya, PaddleOCR, Docling (idempotent, offline-probing). `--local --smoke`
  uses local caches and CPU tool device.

## Evaluation

- **Judge.** Two API judges behind `Judge.score`: `GeminiJudge` (gemini, default,
  free tier) and `GPT4oMiniJudge` (OpenAI, paid); `StubJudge` is offline plumbing.
  Each returns verdict (`correct`/`incorrect`/`abstained`) + extracted answer +
  rationale; an abstaining verdict on a native-unanswerable question counts
  correct. Keys live in the local `.env` only, never forwarded to Kaya. Gate F2
  (`scripts.gates agreement-*`) computes Cohen's kappa vs 200 human labels, gated at
  0.75.
- **Accuracy.** `metrics.accuracy.accuracy_summary()` = mean correctness + 95%
  bootstrap CI, resampled at the **document level** (draw `doc_id`s with
  replacement, take all their rows), 1000 draws, seed 0.
- **Cost.** `metrics.cost.cost_summary()` = mean latency@batch1 (primary) + split
  text/vision/output token sums (secondary).
- **Frontier.** `metrics.frontier.sufficiency_frontier()` orders `T->TL->TLV->V`;
  the frontier is the cheapest rung whose upper CI reaches within the margin
  (default 3 points) of the strongest rung's point estimate. `scripts.gates frontier`
  gates F1 (Go when >=2 Option-A bins differ).
- **Retrieval.** `metrics.retrieval` scores page precision/recall/F1 vs gold
  `evidence_pages`, sliced by `<retrieval-modality>:<evidence-source>` (e.g.
  `text:table`) so matched/cross separates locating from evidence modality.
- **Composition (Table 5).** Each bin decomposed into normalized evidence-source
  shares (text/table/chart/figure/layout, summing to 1); predicted bin frontier =
  strongest per-modality frontier among modalities with >=10% share.
- **Classifier (Table 7 covariate).** `QwenDocTypeClassifier` renders the first
  two pages, builds `TLV`, asks Qwen3-VL-2B for a native doc_type, maps it through
  Option-A binning. Predicted routing counts classifier cost explicitly as total
  classifier latency / evaluated rows, reported as its own `classifier_latency_bs1_s`
  column. Gate F3 (`scripts.gates classifier-pilot`) gates top-1 bin accuracy at 0.70.
- **Tables 1-8** are emitted by `experiments.tables`; **Table 4 is now a held-out
  MMLongBench subset** (disjoint documents for text_heavy/in_between, reused
  visual_heavy), not LongDocURL, binned by the same three domains as Table 1.
