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
