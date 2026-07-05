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

**Default per-bin document-level subset for full runs.** A full mmlongbench run
now defaults to ~100 questions per Option-A bin instead of all 1091, so a full T1
clears the Kaya queue in a couple of hours (short walltime backfills) rather than
needing a 2-day slot. `ExperimentConfig.per_bin_sample` (default 100) +
`sample_seed` (default 0) drive it; `experiments/corpus.py::sample_questions_per_bin`
draws **whole documents** per bin (never splitting a document) until the bin
reaches the target, honoring the PROJECT_SPEC §9 document-level sampling rule.
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
`cli.gates`; commands are recorded in `docs/RUNBOOK.md`.

- **F1 frontier divergence.** `cli.gates frontier` reads the full Table-1 CSV and
  returns Go when at least two Option-A bins have different frontiers. The full
  Qwen3-VL-8B run has not been executed in this log, so the verdict is pending.
- **F2 judge-human agreement.** `cli.gates agreement-sample` writes the 200-row
  human labelling sheet from judged rows; `agreement-score` computes Cohen's
  kappa over `correct` / `incorrect` / `abstained` and gates at 0.75. The human
  labels and kappa result are pending.
- **F3 classifier feasibility.** `cli.gates classifier-pilot --full` samples 100
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
tokens). `kaya/generate.py` now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
True` as a fragmentation mitigation. InternVL is untouched (fixed 448px tiling
already bounds its vision tokens). Not a frozen-interface change: `get_reasoner`
gained optional kwargs; `Reasoner.answer` and the cache key are unchanged.

**Verbose smoke tooling.** `experiments.driver.configure_logging(verbose)` sends
`mpvrdu.*` logs to stdout; smoke runs are verbose (DEBUG) by default (`--quiet`
opts out, `--verbose` forces it for full). The driver logs per-experiment banners
and per-cell start/result lines; the orchestrator logs each stage (conditioner ->
render -> representation -> reasoner -> judge) at DEBUG, so an OOM/crash points at
the exact cell and stage. On failure the full traceback is logged to stdout (the
SLURM log), not just the per-experiment `generate_status.json`. One
`kaya.kaya submit kaya/generate.py -- --experiment section2 --continue-on-error`
job runs all seven Section-2 tables (T1-T7) with per-experiment isolation: a
failing table records its status and the run continues to the next.

## GPU memory management (parser/reasoner co-residence)

After the `max_pixels` cap, a full smoke run still OOM'd — but for a different
reason the verbose logs exposed: **multiple model stacks piling onto one 16GB
V100 and never being freed.** `create_model_dict()` in `tools/layout.py` reloads
the whole Surya stack for every page/cell (the ~170s stalls), and nothing in the
pipeline ever ran `torch.cuda.empty_cache()`. So the Marker/Surya parser, the
ColQwen retriever, and the Qwen reasoner/classifier all sat on the GPU at once,
and a single `section2` job accumulated them until T7 started at ~15GB in use.

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
`reasoner_spec`; `--quantization` on `cli.experiments` and `kaya.generate` sets
it (must match between generate and judge phases). Not a frozen-interface change:
`get_reasoner`/`LocalVLMBackend` gained an optional kwarg, cache key + ResultRow
unchanged. `bitsandbytes==0.49.2` is in the remote env only, so the config-build
test skips locally. Mains stay bf16; 4-bit is single-GPU iteration + a possible
appendix row.

## clear-cache command

`kaya.kaya clear-cache` removes cached generation results on the remote to start
fresh. Default drops `results/cache/{full,smoke}` + `results/tables/{full,smoke}`
and keeps the expensive `renders`/`marker` parse caches; `--renders` also drops
those, `--all` nukes all of `results/cache` + `results/tables` + logs, `--logs`
empties `logs/` (keeps the dir, sbatch needs it), `--mode`/`--experiment` scope
it, `--local` mirrors the removals locally, `--dry-run`/`--yes` gate execution.
Paths are validated to stay inside `results/`/`logs/`.

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
