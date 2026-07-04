# Decision and Implementation Log

This log records fixed decisions, stage findings, implementation-relevant
deviations, migrations, operational assumptions, and follow-up hazards. Later
stages must treat fixed decisions as binding unless a checkpoint explicitly
changes them.

## Fixed for v1

- Dataset: MMLongBench-Doc only. Other datasets are deferred to optional Stage
  10 because MMLongBench-Doc is the only v1 source with document type,
  evidence-modality labels, gold evidence pages, and the unanswerable signal.
- Evaluated model family: Qwen3-VL at 2B, 4B, 8B, and 32B, with 8B as the
  center configuration. Stage 1 must confirm concrete checkpoint availability
  and 32B feasibility on Kaya before full runs depend on it.
- Model swap point: all reasoners sit behind one `Reasoner` interface, with
  local-weight and HTTP-API backends. Closed models are allowed for comparison
  and judging, not for the deployment recommendation.
- Pipeline stages mirror the paper: input conditioning, representation,
  reasoning, and scoring, with retrieval and document-type classification as
  covariates.
- Representation ladder: `T`, `TL`, `TLV`, and `V`. Text/layout channels must
  be produced by modular non-VLM tools; only `TLV` and `V` may attach images.
- Distractor-burying is in scope as a deployment instrument: gold pages remain
  present and same-corpus distractors test how much irrelevant context the
  model tolerates.
- Paths are root-relative. `.cache/`, `.data/`, `envs/`, `results/`, and
  `logs/` live under the repository root on both local and Kaya.
- Kaya execution uses a two-machine model: local edits and Python-driven sync;
  Kaya login for environment/model/data staging; Kaya compute for offline GPU
  jobs. All Kaya-specific source, config, and docs live under `kaya/`.

## Stage 0 implementation log

- Created the Stage 0 skeleton modules as docstring-only placeholders. No
  runtime interfaces or logic are frozen yet; those are Stage 2 and Stage 3
  work.
- `data/` is reserved for the importable Python package. Downloaded datasets,
  synthetic samples, and rendered pages live under `.data/` so artifact storage
  cannot conflict with importable code.
- Updated `.gitignore`, Kaya environment variables, sync exclusions, README,
  and runbook docs to treat `.data/` as the root-relative dataset/render
  artifact directory.
- Added pipeline-specific Kaya scripts. Stage 1.5 later moved all Kaya-specific
  source/config/docs under `kaya/` and replaced the shell workflow with a
  Python CLI.
- `requirements.txt` is a declaration only at Stage 0. Stage 1 must validate
  the pins against Kaya's actual module, CUDA, and GPU partition configuration.
- The old standalone reference/demo kit under `kaya/` has been removed.
  Pipeline operations use `kaya/kaya.py`.

## 2026-07-03 Environment install

- Created the local conda environment at `envs/mpvrdu` with Python 3.11,
  matching the Kaya setup script.
- Changed `requirements.txt` from `torch==2.7.1` to `torch==2.7.0` because
  `vllm==0.9.2` pins `torch==2.7.0`; the previous pair was not installable by
  pip's resolver.
- Changed `requirements.txt` from `colpali-engine==0.3.8` to
  `colpali-engine==0.3.13` because `0.3.8` required `transformers<4.48`, while
  `0.3.13` supports the pinned `transformers==4.53.2`.
- Added `chardet==5.2.0` to keep `requests==2.32.4` inside its supported
  compatibility window; `paddlex` otherwise resolved to `chardet 7.4.3` and
  emitted an import-time warning.
- Set the local env's site pip config (`envs/mpvrdu/pip.conf`, ignored) to use
  `/home/lingwei/mpvrdu/.cache/pip` as the cache directory, keeping pip cache
  writes inside the repo artifact root.

## Open items Stage 1 confirms

- MMLongBench-Doc fetch/render path and field parsing.
- Whether MMLongBench-Doc has a real scanned-document slice or requires
  synthetic degradation for text-recovery analysis.
- Whether in-page evidence boxes exist in v1; expected outcome is page-level
  crops only.
- Qwen3-VL 2B, 4B, 8B, and 32B availability and Kaya feasibility.
- Local-backend and API-backend instantiation through the same `Reasoner`
  contract.
- ColPali/ColQwen and BM25+BGE feasibility on target hardware.
- Native unanswerable count and the exact abstention definition.
- MMLongBench-Doc `doc_type` distribution and the text/in-between/visual
  spectrum mapping.
- Confirmed Kaya module names, CUDA version, GPU partition, and GPU request
  syntax.

## Stage 1 findings

Local probes were run on 2026-07-03 with `envs/mpvrdu/bin/python -m
cli.run_probe local --json` against `.data/mmlongbench`.

- **Loader smoke:** pass. The local parquet has 1,091 question records with the
  required fields (`doc_id`, `doc_type`, `question`, `answer`,
  `evidence_pages`, `evidence_sources`, `answer_format`). The first 64 records
  parsed cleanly, and 9/9 sampled source PDFs resolved under
  `.data/mmlongbench/documents`. MMLongBench's native unanswerable signal is
  `answer == "Not answerable"`.
- **Scanned vs born-digital:** pass. In a 12-PDF sample, 3 PDFs had no
  extractable text in the sampled pages and 9 had an embedded text layer
  (`scanned_fraction=0.25`). The embedded-vs-OCR check is therefore a real
  MMLongBench slice, not only a synthetic-degradation experiment.
- **In-page boxes:** pass, confirmed absent in v1. MMLongBench records expose
  page-level evidence only (`evidence_pages`, `evidence_sources`), so
  Stage 5 `region_crop` must degrade to page-level crops. True in-page crops
  remain a LongDocURL/Stage 10 extension.
- **Model family:** partially confirmed. Public Hugging Face model pages exist
  for `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`,
  `Qwen/Qwen3-VL-8B-Instruct`, and `Qwen/Qwen3-VL-32B-Instruct`; the model cards
  list the 8B/32B repositories as 9B/33B parameter models. Local load/generate
  was not run without Kaya GPU/cache access, but `model-family --run-heavy` now
  attempts a child-process vLLM generation smoke for each requested model id.
  The installed
  `transformers==4.53.2` does **not** expose
  `Qwen3VLForConditionalGeneration` or `AutoModelForMultimodalLM`, while the
  current Qwen3-VL model cards show those/newer APIs. Stage 6 must either
  upgrade `transformers` within the `colpali-engine`/`vllm` compatibility window
  or use a vLLM path confirmed to support Qwen3-VL.
- **Backend swap:** only a Stage 1 smoke exists. `cli.run_probe model-family`
  instantiates local/API echo implementations behind the same tiny
  `answer(prompt) -> str` shape to keep the requirement explicit. The production
  `Reasoner` ABC and real local/API backends are still Stage 3/Stage 6 work.
- **Vision retrieval:** partial. BM25 works locally on a tiny smoke corpus, and
  `FlagEmbedding` plus `colpali_engine` import successfully. BGE model
  load/search and ColPali/ColQwen page indexing were not run locally; they must
  be run on Kaya compute with pre-staged weights. `retrieval --run-heavy`
  attempts a BGE embedding smoke and a tiny ColQwen image/query scoring pass.
- **Unanswerable/abstention:** pass. MMLongBench has 244/1,091 natively
  unanswerable questions (22.36%). Proposed abstention definition: a prediction
  abstains if it contains a normalized refusal/no-evidence form such as
  "not answerable", "cannot be answered", "insufficient information",
  "not enough information", "no answer", or "unknown from the document".
  Hallucination is a substantive non-abstaining answer on native-unanswerable
  questions or on answerable retrieved conditions with page recall 0.
- **`doc_type` distribution:** pass. Full counts are:
  Research report / Introduction 293 questions / 34 docs; Academic paper 204 /
  26; Guidebook 156 / 22; Tutorial/Workshop 139 / 17; Financial report 117 /
  11; Brochure 101 / 15; Administration/Industry file 81 / 10.
- **Spectrum mapping proposal for approval:** rule-based on evidence labels,
  not final code. Proposed text-heavy: Administration/Industry file. Proposed
  in-between: Financial report. Proposed visual-heavy: Academic paper,
  Brochure, Guidebook, Research report / Introduction, Tutorial/Workshop. This
  is intentionally conservative and should be approved or revised at the Stage 1
  checkpoint.
- **Kaya live configuration:** not yet confirmed on the cluster. The configured
  defaults remain `Anaconda3/2024.06`, `cuda/12.6.3`, partition `gpu`, and GRES
  `gpu:1` from `kaya/config.json`, but Stage 1 did not verify live module names,
  CUDA, partition, or GPU request syntax on Kaya.

GPU follow-up commands after the repo, data, and models are staged on Kaya:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/run_probe.py -- model-family --run-heavy --json
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/run_probe.py -- retrieval --run-heavy --json
```

If the all-size model probe is too large for one job, pass `--model-id` to test
one Qwen3-VL size at a time.

## 2026-07-03 Stage 1 implementation log

- Replaced the `cli/run_probe.py` placeholder with Stage 1 feasibility probes.
  The local probes read `.data/mmlongbench` directly and reuse parser/stat
  helpers from `scripts/profile_datasets.py`; those existing profiling scripts
  were not edited.
- Added `ProbeVerdict`/`ProbeConfig` result objects and CLI groups:
  `loader`, `scanned`, `boxes`, `unanswerable`, `doc-type`, `model-family`,
  `retrieval`, `local`, and `all`. The hardware-dependent probes are safe by
  default and report `needs_hardware`/`partial` instead of downloading weights or
  requiring a GPU.
- Added real optional `--run-heavy` child-process attempts for Kaya:
  per-model vLLM generation for Qwen3-VL, BGE embedding, and ColQwen image/query
  scoring. These are not run by default and were not run locally because no CUDA
  device is visible.
- Added a Kaya probe wrapper; Stage 1.5 replaced this static sbatch wrapper with
  generated SLURM scripts from `kaya/kaya.py`.
- Added `tests/test_probes.py` with a tiny synthetic MMLongBench-like parquet
  and PDFs. The tests exercise field parsing, PDF resolution, scanned-vs-born
  digital detection, absence of boxes, unanswerable counting, doc-type mapping,
  and safe hardware-probe defaults.

## 2026-07-03 Stage 1.5 Kaya implementation log

- Moved Kaya ownership to `kaya/` and removed the old standalone Kaya demo kit,
  old shell wrappers, stale SLURM files, pycache, and old logs. The top-level
  `scripts/` directory is now only for non-Kaya standalone utilities such as
  dataset profiling.
- Replaced shell-environment-driven Kaya operations with `kaya/kaya.py`, a
  Python CLI that reads static site/project settings from `kaya/config.json`.
  Durable values such as SSH alias, remote root, modules, SLURM defaults, model
  IDs, dataset IDs, secret-forwarding rules, and rsync excludes now live on
  disk in that config file.
- Simplified `kaya/kaya.py` to a generic runner with `show-config`, `push`,
  `pull`, `run`, `submit`, and `watch`. Task-specific operations now live in
  runnable scripts under `kaya/`: `setup_env.py`, `prestage.py`,
  `gpu_test.py`, and `run_probe.py`.
- Python runnable scripts can declare `# kaya:` headers for default target,
  conda activation, HF offline mode, and job name. Existing `.sbatch` files are
  supported directly and are the foundation for future stage jobs; the `.sbatch`
  file owns its own SLURM directives and shell setup unless explicit
  `kaya.py submit` SLURM overrides are supplied.
- Generated login/GPU Python executions export `PYTHONPATH=<remote_root>` so
  repo packages import correctly when Python files are executed by path. Custom
  `.sbatch` files should set their own `PYTHONPATH` or use module-style Python
  invocation.
- Added optional `slurm.account` and `slurm.qos` fields. They are blank by
  default because the earlier Kaya partition check showed `gpu` access by group
  membership rather than a required explicit account flag; if Kaya rejects a job
  for accounting, set these fields or pass `--account`/`--qos`.
- Added `kaya/download_hf.py` for login-node Hugging Face staging. It uses
  `huggingface_hub.snapshot_download` for model snapshots, downloads cached
  files into `.cache/`, and stages MMLongBench-Doc file-by-file into
  `.data/mmlongbench/{data,documents}` using symlinks from the HF cache when
  possible. This matches the local loader/probe layout while making individual
  dataset file failures visible.
- Added `hf_xet` to requirements after Kaya reported that Xet storage was
  enabled but `hf_xet` was missing. The earlier `HF_HUB_DISABLE_XET=1` fallback
  is no longer the intended path. `HF_TOKEN` is read from the root `.env` file
  and forwarded only to online login-node runs; `.env` is excluded from rsync.
- Live Kaya validation completed before the runner simplification:
  `push`, old `run-login --no-activate --no-push -- pwd`, and old
  `setup-env --no-push` succeeded. The remote env was created at
  `/group/ems036/lxu/mpvrdu/envs/mpvrdu`; `pip check` reported no broken
  requirements; torch reported `2.7.0+cu126` with CUDA `12.6`.
- Live MMLongBench prestaging initially failed in Hugging Face's Xet/ranged
  download path with consistency, HTTP 416, and file-size mismatch errors on
  PDF files. A temporary Xet-disable workaround was superseded by the decision
  to install/use `hf_xet` and keep all downloads through the Hugging Face Hub
  package interface. A later rerun still exposed stale partial Hub/Xet state
  (`416 Client Error` and Xet file-size mismatch), so `kaya/download_hf.py` now
  prints the active `huggingface_hub`/`hf_xet` versions, token/cache settings,
  and exact repo/file being fetched. Model snapshots purge incomplete files,
  lock files, the repo-specific Hub cache, and likely Xet cache directories
  before a serial `force_download=True` retry. MMLongBench staging lists the
  repo and downloads parquet/PDF files one at a time with per-file retries; if a
  single file still fails Hub cache consistency checks, it streams that file via
  Hugging Face's filesystem interface into `.data/mmlongbench`.
- Kaya dataset-only prestaging now passes with
  `envs/mpvrdu/bin/python -m kaya.kaya run kaya/prestage.py -- --skip-models`.
  The live rerun staged 1 parquet and 135 PDFs. `mi_phone.pdf` still failed the
  normal Hub cache consistency check twice (`24877442` expected vs `31841340`
  bytes received), then staged successfully through the HfFileSystem fallback.
  A login-node loader verification also passed:
  `envs/mpvrdu/bin/python -m kaya.kaya run --no-push kaya/run_probe.py -- loader --json`
  loaded 1,091 records, resolved 9/9 sample PDFs, and reported no sample PDF
  misses.
- Stage 2 is now also the full setup/prestage barrier for later stages. In
  addition to Qwen3-VL reasoner weights and MMLongBench, `kaya/prestage.py`
  stages BGE (`BAAI/bge-small-en-v1.5`), ColPali/ColQwen
  (`vidore/colpali-v1.3`, `vidore/colqwen2-v1.0`,
  `vidore/colqwen2.5-v0.2`), warms PaddleOCR, and uses Docling's model
  downloader for layout/table-related models under `.cache/`. Later stages
  should not add ad hoc setup steps without updating this config-driven
  prestage path.
- PaddleOCR 3.1.0 must be paired with PaddleX 3.1.x. An unpinned transitive
  install pulled `paddlex==3.7.2`, whose `PaddlePredictorOption` constructor is
  keyword-only and breaks PaddleOCR 3.1.0's `PaddlePredictorOption(model_name,
  ...)` call during prestage warmup. `requirements.txt` now pins
  `paddlex[ie,multimodal,ocr]>=3.1.0,<3.2.0`, and `kaya/prestage.py` fails
  early with a version diagnostic if the old environment has not been rebuilt.
- The Stage 2 data layer normalises MMLongBench `evidence_pages` from one-based
  source page numbers to zero-based internal page indices. The original row is
  retained in `Question.raw_fields` for audit/debugging. Rendered PNGs are
  cached under `results/cache/renders/`, not `.data/`, so they are reproducible
  compute artifacts rather than dataset source files.
- Added `kaya/KAYA_AGENT_GUIDE.md` as the definitive agent-facing Kaya guide and
  `kaya/KAYA_USER_GUIDE.md` as the user-facing quick guide.
- Updated tests and docs so future stages invoke Kaya through
  `envs/mpvrdu/bin/python -m kaya.kaya run|submit|watch ...`, not
  `scripts/kaya/*.sh` or task-specific `kaya.py` subcommands.

## 2026-07-03 Prestage idempotency fix

- `kaya/prestage.py` previously called Hugging Face's network API on every
  rerun even when a model snapshot or MMLongBench file was already fully
  staged, which made re-running the full prestage command slow and
  unnecessary once a machine already had everything. Fixed in
  `kaya/download_hf.py`:
  - `snapshot()` (used for both reasoner and retrieval model weights) now
    first probes the Hub cache with `snapshot_download(local_files_only=True)`
    before doing anything else. That call raises if any file for the
    requested revision is missing, so a successful return proves the
    snapshot is complete and no HTTP request was made; only a cache miss
    falls through to the real (network) download path. `--force-download`
    bypasses the probe as before.
  - `stage_mmlongbench_from_hub()` now checks each parquet/PDF target with a
    new `target_is_staged()` helper (file or valid symlink, non-zero size)
    before downloading it, and skips per-file downloads that are already
    staged. The one remaining network call is the initial `list_repo_files`
    listing (filenames only, no data transfer), which is needed to know what
    the dataset currently contains.
  - Both skip paths are unit-tested by mocking `huggingface_hub.snapshot_download`
    and by exercising `target_is_staged()` against real/empty/missing/valid-symlink/
    broken-symlink files; no dedicated `tests/` module was added since this is
    a login-node-only script with no CI path to it, matching the existing
    `download_hf.py` test coverage level.

## 2026-07-03 Kaya orphaned-process / hang fix

- Diagnosed a real hang on `envs/mpvrdu/bin/python -m kaya.kaya run
  kaya/prestage.py -- --skip-dataset`: two stale processes from earlier Kaya
  sessions (a manual `download_hf.py --dataset ... --stage-mmlongbench` from
  Stage 1.5 live validation, and an interrupted `kaya.kaya run
  kaya/prestage.py -- --skip-models` from the same day) were still alive on
  the login node, hours after being started, each with sockets stuck in
  `CLOSE-WAIT` (the HF CDN peer had closed the connection but the client's
  blocking socket read never returned, since `huggingface_hub`'s requests
  session sets no read timeout). One held the MMLongBench dataset's Hub cache
  lock; its `.incomplete` blob was stalled at exactly 31,841,340 bytes, the
  same corrupted-size symptom already logged here for `mi_phone.pdf` during
  Stage 1.5, i.e. a recurring Xet/Hub consistency bug on Kaya's network, not
  new. Both processes had in fact already died on their own by the time they
  were killed, apparently reaped once the kernel's default TCP keepalive
  (~2h) finally noticed the dead peer.
- Root cause for why they survived a local interrupt at all: `kaya/kaya.py`'s
  `ssh_script()` ran every remote command as `ssh <alias> bash --login -s`
  with no pseudo-terminal. Without a pty, OpenSSH has no controlling-terminal
  session to hang up when the local client disconnects, so it does not
  signal the remote process group at all; a plain piped `bash --login -lc
  ...` orphan just keeps running on the login node indefinitely, regardless
  of whether the local `kaya.py run` was interrupted or the network dropped.
  Reproduced and confirmed directly: killing the local ssh client for a
  `sleep 60 &`-backgrounded remote script left the remote `sleep` process
  alive.
- Fixed in `kaya/kaya.py`:
  - `ssh_script()` gained an `interruptible: bool` flag. When set, the script
    is staged as a remote file (via the existing `write_remote_file`) and run
    as `ssh -tt` (forced pty) with `ServerAliveInterval=15
    ServerAliveCountMax=3` keepalives, instead of being piped over stdin.
    Running from a file rather than piping over stdin avoids the classic pty
    echo problem (a pty would otherwise echo piped stdin script text back
    into captured output). `handle_run` (the long-running, foreground,
    interruptible login-node execution path used by `prestage.py` and
    friends) now passes `interruptible=True`; `submit_remote_sbatch` and
    `ensure_remote_dirs` are left on the fast, non-pty, stdin-piped path
    since they are short, deterministic, and (for `submit_remote_sbatch`)
    depend on clean unmerged stdout for job-id parsing, which a pty session
    would risk polluting.
  - Added `guard_process_group()`, which prepends `trap 'kill -- -$$
    2>/dev/null || true' HUP TERM INT` (deliberately **not** `EXIT`) to every
    remote script as a backstop, so a hangup/interrupt signal reaching the
    remote shell kills its whole process group, including any synchronous
    foreground child. Trapping `EXIT` too was tried first and is wrong: it
    fires the same self-kill on a normal, successful exit, which sends the
    already-exiting shell a self-inflicted `SIGTERM` and turns a clean run
    into a spurious SSH exit-255 failure. Caught by reproducing manually
    before it shipped.
  - Added `ConnectTimeout=10` to the `wait_for_job` SLURM polling `ssh`
    calls so a dropped connection during a long `squeue`/`sacct` poll loop
    fails fast on the next iteration instead of hanging indefinitely.
  - Verified end to end: a `sleep 60 &`-backgrounded remote script started
    through the new `interruptible=True` path, interrupted locally with
    SIGTERM after 4s, left no remote `sleep` process behind. A real
    `kaya.py run kaya/prestage.py -- --skip-dataset` afterwards completed
    cleanly (all 7 model repos already cached, correctly skipped; dataset
    skipped as requested) and then correctly failed fast, not hung, on the
    pre-existing PaddleOCR/PaddleX version mismatch noted in the Stage 1.5
    log, which the remote env has not yet been rebuilt to pick up.

## 2026-07-03 Stage 3 implementation log (freeze point)

Stage 3 defined every pipeline ABC, the backend-agnostic `ModelInput`, a caching
orchestrator, the expanded `ExperimentConfig`, and a runnable stub CLI. The whole
pipeline now runs end to end on stubs and emits well-typed rows before any real
tool or model exists. `docs/ARCHITECTURE.md` was created and holds the
tree↔paper mapping and the authoritative **frozen interfaces** list.

- **Frozen interfaces (Stage 3 invariant).** `schema.py` contracts, the ABCs in
  `pipeline/` and `covariates/`, and the `ModelInput` contract in
  `models/payload.py` are frozen. Later stages fill implementations behind them;
  any change is a checkpoint conversation recorded here, never a silent edit.

- **Schema placeholders filled.** `Payload` now carries `modality` +
  ordered parts; `Prediction` carries `model_spec` and split
  text/visual/output token counts + latency (zeroed by the stub, filled in
  Stage 6); `Score` carries `value` + `correct`/`abstained`/`judge_spec`.
  Added shared `TextPart`/`ImagePart` in `schema.py` so both `Payload` and
  `ModelInput` compose the same part types without a layering cycle
  (schema ← models.payload ← pipeline.reasoner ← models registry).

- **`ImagePart` dual source.** An image is carried either by path
  (`image_path`, the render output) or inline (`data`, e.g. decoded from a chat
  data URI). `read_bytes()`/`data_uri()` hide the difference so the local and
  API adapters never branch on image origin. This is what lets
  `ModelInput.from_chat_messages` round-trip losslessly (image *bytes* survive;
  the on-disk path is intentionally not recovered).

- **Interface signature note (deviation from the plan's literal wording).** The
  plan wrote `Representation.build(PageSet) -> Payload`, but a `PageSet` carries
  only page indices + provenance, not a `doc_id`, so it cannot resolve pixels or
  text on its own. Frozen signature is therefore
  `Representation.build(pages: Sequence[Page]) -> Payload`: the orchestrator
  resolves the `PageSet` into rendered `Page`s (via `data/render.py`) and hands
  those to the composer. This keeps the representation a pure page-encoder and
  keeps rendering in one place. Recorded here as the deliberate freeze choice.

- **Conditioner signature.** `InputConditioner.condition(question, page_count)`.
  The orchestrator computes each document's total `page_count` once (cached per
  doc) so `FullDoc`/`BuriedOracle` know the page range without every conditioner
  re-opening the PDF. `OracleConditioner` falls back to page 0 for
  native-unanswerable questions (no gold pages) so the pipeline still has
  something to render. `RetrievedTopK` wraps a `Retriever`; `BuriedOracle` uses
  deterministic first-N non-gold distractors so the cache key is stable.

- **Model swap point.** `models/__init__.py::get_reasoner(spec)` parses a
  `family-size-backend` spec (or `stub`) via `ModelSpec.parse` and returns a
  `Reasoner`. Stage 3 resolves every spec to `StubReasoner`; Stage 6 will
  dispatch `local`→`LocalVLMBackend` and `api`→`APIBackend`. No pipeline code
  will change when it does.

- **Caching contract (frozen).** `make_cache_key` = SHA-256 over
  `{question_id, doc_id, condition, representation, model_spec, judge_spec,
  dpi}`; rows are appended as jsonl to
  `results/cache/orchestrator/results.jsonl` via `ResultCache`. `run_cell` is
  idempotent (cache hit returns the stored row) and resumable (a fresh
  orchestrator rebuilds its index from disk). The model spec is in the key, so
  the scaling sweep and any family swap produce distinct, mergeable rows. `k`
  and burying level are encoded in the conditioner `name`
  (`retrieved_k3`, `buried_n10`), so they are part of the condition dimension of
  the key.

- **Abstention definition promoted to a shared helper.** The pre-registered
  Stage 1 refusal-surface test now lives as `metrics/abstention.py::is_abstention`
  (used by the stub judge and, later, the Stage 7 abstention metrics) so there is
  one definition of "the model declined to answer".

- **Config.** `ExperimentConfig` holds the v1 knobs: dataset fixed to
  `mmlongbench`; center reasoner `qwen3vl-8b-local` + the 2B/4B/8B/32B scaling
  specs; conditions `oracle/retrieved/full/buried` with `k∈{1,3,5}` and burying
  levels `{10,25,50}`; representation ladder `T/TL/TLV/V`; sufficiency margin 2.0
  points; `dpi=144`. `ProjectPaths` stays overridable so tests point
  `data_dir`/`cache_dir` at a fixture with no pipeline change.

- **Tools are Stage 3 placeholders, not stubs-that-do-nothing.** `text_channel`
  returns the embedded text render already extracted; `layout_channel` echoes it
  (no structure yet); `visual_channel` returns one `ImagePart` per rendered page.
  Stages 4/5 replace these behind the same return types; the composers do not
  change.

- **Tests.** `tests/test_pipeline_skeleton.py` runs the orchestrator over every
  (condition × representation) cell on a tiny PDF fixture, asserts cache
  idempotency + resumability, that the cache key depends on the model spec, that
  the modality boundary is enforced both structurally and via `Payload`, and that
  `ModelInput` round-trips through both adapters. Full suite: 23 passed.

## 2026-07-04 v1 → v3 scope pivot (plan swap, before MVP Stage M1)

Stages 0–3 above were built against the **v1 plan**, now archived at
`docs/implementation_plan_old.md` (the three-topic study: RQ1/2/3 =
Representation / Retrieval / Deployment, multi-dataset-capable, distractor-burying
and fail-safe abstention and scaling-as-a-story all in scope). `PROJECT_SPEC.md`
and `docs/implementation_plan.md` are now **v3**: a single EACL long-paper thesis,
"the representation an MP-VRDU system requires is a function of document type."
Where the two disagree, v3 is current. The v3 build starts at the MVP (Stage M1);
this entry records what carries over and what each v3 stage must reconcile, so the
M1 start is not misled by v1-era notes elsewhere in this log.

**Carries over unchanged (do not re-implement).** The repo skeleton, Stage-1
feasibility probes, the Kaya Python runner + static config, the Stage-2 data layer
(loader + render), and the **Stage-3 frozen interfaces** (`schema.py`, the pipeline
and covariate ABCs, `ModelInput`, the caching contract). v3 fills these same
interfaces with real tools/models; the freeze still holds.

**Reconciliation items (each owned by a specific v3 stage, not done in this pivot):**

- **Doc-type binning.** v3 fixes **Option A** (semantic-domain bins): text-heavy =
  Administration/Industry file + Academic paper + Research report/Introduction
  (578 Q / 54 docs); in-between = Financial report + Guidebook + Tutorial/Workshop
  (412 Q / 50 docs); visual-heavy = Brochure (101 Q / 15 docs). This **supersedes**
  the Stage-1 "conservative proposal" above (which put Academic/Guidebook/Research/
  Tutorial in visual-heavy) — that proposal is dead. Option A lands in
  `data/binning.py` as the single source of truth at **Stage M1**; the data-driven
  Option B is the Section-3 P1 swap behind the same signature.
- **Primary parser.** v3 makes **Marker** the primary ladder text/layout source;
  PyMuPDF (and Docling/PP-Structure) become the appendix parser-swap. This reverses
  the v1 Docling-primary note baked into `tools/layout.py`. Implemented at **Stage
  M2**; confirmed with the human at the M1/M2 checkpoint. (Tool placeholder
  docstrings were updated in this pivot to point at Marker-primary / v3 stage names
  so they stop pointing coders at the v1 plan.)
- **Sufficiency margin.** v3 primary margin is **3** points (sensitivity {2, 3, 5}
  in the appendix); `config.sufficiency_margin` is still the v1 value **2.0**. M1
  updates it as part of the config extension below.
- **Config extension.** **Stage M1** adds `smoke: bool`, `bins`,
  `cost_metric="latency_bs1"`, and sets `sufficiency_margin=3`. The existing v1
  knobs stay as-is until then. `k_values` stay (retrieval, RQ2/RQ3). The
  `BuriedOracle` conditioner and `burying_levels` **stay in the tree but are unused
  by the paper** (moved to Section-3 P4); do not delete them. `scaling_specs` still
  lists 4B — v3 scale sanity uses **2B / 32B** only (8B is primary), so 4B is
  simply unused, not removed.
- **Judge.** v3 judge is **GPT-4o-mini** (different family), gated by Cohen's
  **κ ≥ 0.75** vs 200 hand-labels. `config.judge_spec` is still `"stub"`; the real
  judge is **Stage M5**, validated at **Stage F2**.
- **Confidence intervals.** v3 requires **document-level** bootstrap CIs (1000
  resamples over documents, not questions), because questions cluster within
  documents (135 docs / 1091 Q). Built in `metrics/accuracy.py` at **Stage M5**.
- **Reasoner / replication.** Qwen3-VL-8B primary; **InternVL3-8B** replicates the
  RQ1 headline only (Stage F4); Qwen3-VL-2B/32B are appendix scale sanity (Stage F7).
- **Cut to Section 3 (retained in tree, not on the paper's critical path):** the
  full retrieval-sufficiency frontier, the distractor-burying sweep, fail-safe
  abstention, scaling-as-a-story, and multi-dataset robustness beyond one LongDocURL
  replication. The `BuriedOracle` conditioner and `metrics/abstention.py` stay but
  are unused by the main paper.

**Stale-reference cleanup done in this pivot (docs/comments only, no interface
change):** `tools/text.py`, `tools/layout.py`, `tools/visual.py`, and
`cli/run_experiment.py` docstrings were updated to point at v3 stage names
(M2 / Section-2 runner) and Marker-primary instead of the v1 stage numbers
("Stage 4/5/9"), nine-RQ names ("RQ4"), and Docling-primary. `docs/ARCHITECTURE.md`
still carries one v1-era line (`covariates/retriever.py` labelled "RQ7
decomposition"); per the global architecture rule it is left for a confirm-first
edit rather than changed silently here.

## 2026-07-04 Stage M1 implementation log

Stage M1 added the deterministic v3 MVP smoke corpus, the fixed Option-A
document-type binning source, and the config knobs later MVP stages consume.

- **Option-A binning source.** Added `data/binning.py::doc_type_bin()` as the
  single source of truth for native MMLongBench-Doc `doc_type` labels:
  text-heavy = Administration/Industry file + Academic paper + Research report /
  Introduction; in-between = Financial report + Guidebook + Tutorial/Workshop;
  visual-heavy = Brochure. The function is intentionally the swap point for the
  Section-3 Option-B data-driven robustness check.
- **Option-A counts.** The raw staged parquet and `docs/dataset_stats.md` give:
  text-heavy 578 questions / **70 documents**, in-between 412 questions / 50
  documents, visual-heavy 101 questions / 15 documents. The v3 plan/spec text
  said text-heavy was 578 questions / 54 documents; that is an arithmetic/doc-count
  typo because the native-class document counts are 10 + 26 + 34 = 70. Code and
  tests use the data-derived 70-document count.
- **Frozen smoke corpus.** Added `experiments/smoke.py` with seven short
  MMLongBench documents, one per native `doc_type`, selected by low page count and
  PDF availability. The smoke set contains 54 questions across 7 documents:
  `2303.05039v2.pdf` (Academic paper, 8 Q, 9 pages),
  `7c3f6204b3241f142f0f8eb8e1fefe7a.pdf` (Administration/Industry file, 6 Q,
  15 pages), `BRO-GL-MMONEY.pdf` (Brochure, 6 Q, 16 pages),
  `f86d073b0d735ac873a65d906ba82758.pdf` (Financial report, 9 Q, 20 pages),
  `8dfc21ec151fb9d3578fc32d5c4e5df9.pdf` (Guidebook, 12 Q, 18 pages),
  `379f44022bb27aa53efd5d322c7b57bf.pdf` (Research report / Introduction, 6 Q,
  17 pages), and `0e94b4197b10096b1f4c699701570fbf.pdf` (Tutorial/Workshop, 7 Q,
  15 pages).
- **Config extension.** `ExperimentConfig` now has `smoke`, `bins`,
  `cost_metric="latency_bs1"`, `max_tokens`, and v3
  `sufficiency_margin=3.0`. `ExperimentConfig(smoke=True)` selects
  `qwen3vl-2b-local` and caps `max_tokens` at 64. Existing root-relative
  `ProjectPaths` behaviour is unchanged.
- **Runnable smoke hook.** `cli.run_experiment --smoke` now loads the frozen smoke
  corpus through `experiments.smoke.load_smoke_questions()` and uses the smoke
  config. Non-smoke `--sample` behaviour is preserved for the Stage-3 stub path.
- **Parser reconciliation.** v3 treats Marker as the primary `T`/`T+L` text and
  layout source; PyMuPDF/Docling/PP-Structure are appendix/parser-swap paths. M1
  records the decision and keeps implementation of Marker extraction itself scoped
  to Stage M2.
