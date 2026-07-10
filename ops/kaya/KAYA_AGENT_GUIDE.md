# Kaya HPC Agent Guide

This is the canonical operational guide for agents working on MP-VRDU on UWA
Kaya. All Kaya-specific scripts, configuration, and documentation live in this
`ops/kaya` directory.

## Mental Model

Kaya work has three locations:

- **Local repo:** edit code, run cheap checks, and launch `kaya.py`.
- **Kaya login node:** reached with `ssh kaya`; has internet; should not run
  GPU/model workloads. Use it for sync checks, environment setup, and Hugging
  Face staging.
- **Kaya compute node:** reached through SLURM (`sbatch`/`srun`); has GPUs; may
  have no internet. Compute jobs should read pre-staged `.cache/` and `.data/`
  artifacts and default to Hugging Face offline mode.

The remote mirror is configured in `ops/kayaconfig.json` as
`/group/ems036/lxu/mpvrdu`. `/group` is shared storage visible from both login
and compute nodes, so a file staged on the login node can be read by a GPU job.

Root-relative artifacts on both machines:

- `.cache/`: Hugging Face, torch, and pip caches.
- `.data/`: downloaded datasets and rendered artifacts.
- `envs/mpvrdu`: conda environment on that machine.
- `results/`: experiment outputs pulled back from Kaya.
- `logs/`: SLURM stdout/stderr and generated runner scripts.

`push` excludes `.git/`, `.env`, `.cache/`, `.data/`, `envs/`, `results/`, and
`logs/`. Never depend on local datasets, weights, results, or secrets being
copied to Kaya.

## SLURM and sbatch

Kaya uses SLURM to schedule compute-node work. The core commands are:

- `sbatch job.sbatch`: submit a batch job.
- `squeue -u $USER`: show queued/running jobs.
- `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode`: show final
  accounting.
- `scancel <jobid>`: cancel a job.

An `.sbatch` file is a shell script with `#SBATCH` directives before commands.
Typical directives:

```bash
#!/bin/bash --login
#SBATCH --job-name=mpvrdu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Important details:

- `--partition=gpu --gres=gpu:1` requests one GPU on the configured GPU
  partition.
- `--account` and `--qos` are optional in `ops/kayaconfig.json`; leave blank unless
  SLURM rejects jobs for accounting.
- `logs/` must exist before submission because SLURM opens output files before
  executing the script.
- Anything using `module` should run in a login shell (`#!/bin/bash --login` or
  `bash --login`).
- Existing `.sbatch` files are authoritative. If a future stage needs a custom
  job, create an `.sbatch` file and submit it through `kaya.py`. Explicit
  `kaya.py submit --time ... --partition ... job.sbatch` options are passed to
  `sbatch` as deliberate overrides.
- `kaya.py` submits `.sbatch` files from the remote repo root. Custom `.sbatch`
  files should `cd "$SLURM_SUBMIT_DIR"` and set `PYTHONPATH="$SLURM_SUBMIT_DIR"`
  if they execute repo files by path.

## Static Config

Use `ops/kayaconfig.json` for durable site/project values:

- SSH alias and remote root.
- module names.
- root-relative artifact paths.
- conda Python version and PyTorch wheel index.
- SLURM defaults and optional account/QOS.
- Hugging Face download concurrency.
- model and dataset IDs.
- rsync excludes.
- local secret forwarding rules.

`HF_TOKEN` is read from the root `.env` file locally and forwarded only into
online login-node runs, such as `scripts/prestage.py`. `.env` is excluded from
rsync and should never be copied to Kaya.

## kaya.py Contract

`kaya.py` is intentionally small. It owns only common execution mechanics:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya show-config
envs/mpvrdu/bin/python -m ops.kaya.kaya push
envs/mpvrdu/bin/python -m ops.kaya.kaya pull
envs/mpvrdu/bin/python -m ops.kaya.kaya run <program> -- <args>
envs/mpvrdu/bin/python -m ops.kaya.kaya submit <file.py|file.sbatch> -- <args>
envs/mpvrdu/bin/python -m ops.kaya.kaya watch [job_id]
```

It does not contain task-specific subcommands. Setup, prestage, GPU smoke, and
probe commands are separate runnable scripts:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/setup_env.py
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py -- --skip-models
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py -- --skip-retrieval-models --skip-parsers --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --time 00:05:00 scripts/gpu_test.py
envs/mpvrdu/bin/python -m ops.kaya.kaya submit path/to/job.sbatch -- --job-arg value
```

Python runnable files can declare defaults in their header:

```python
# kaya: target=login
# kaya: env=true
# kaya: offline=false
# kaya: job-name=optional_name
```

Supported header keys:

- `target=login|gpu`: `run` executes on login or submits a generated GPU job.
- `env=true|false`: activate `envs/mpvrdu`.
- `offline=true|false`: set Hugging Face offline variables.
- `job-name=name`: default SLURM job name for generated Python jobs.

CLI flags override headers.

## Command Semantics

`push`:

- Creates remote root and artifact dirs.
- Runs `rsync -az --delete` from local repo root to Kaya remote root.
- Applies configured excludes, including `.env`, `.cache`, `.data`, `envs`,
  `results`, and `logs`.
- Remote-only source edits are not preserved. Treat local as authoritative.

`pull`:

- Pulls remote `logs/` and `results/` into local `logs/` and `results/`.
- Does not pull `.cache`, `.data`, or `envs`.

`run`:

- Pushes by default.
- Runs a repo-local `.py` file or a command on the login node by default.
- If a `.py` header or `--target gpu` selects GPU, it generates an sbatch
  wrapper and submits it.
- For online login jobs, forwards configured secrets such as `HF_TOKEN`.

`submit`:

- Pushes by default.
- For `.py` files, generates an sbatch wrapper with config/CLI SLURM defaults.
- For `.sbatch` files, submits the file with any explicit SLURM overrides. The
  file owns its own setup, environment activation, output paths, and offline
  mode. It should set `PYTHONPATH` if it runs repo files by path.
- Waits, pulls logs/results, and prints log tails unless `--no-wait` or
  `--no-pull` is provided.

`watch`:

- Waits on a supplied job id or `.kaya_last_job`.
- Pulls logs/results unless `--no-pull`.
- Prints matching `logs/*_<jobid>.out` and `.err` tails.

## Hugging Face Staging

Use `scripts/prestage.py` on the login node. It calls
`huggingface_hub.snapshot_download` for model snapshots and Hugging Face file
APIs for file-by-file MMLongBench staging; no direct URL downloader should be
added. The project depends on `hf_xet` so Xet-backed cache downloads use the
Hugging Face/Xet client. `HF_TOKEN` is read from local `.env` and exported only
for online login-node execution.

Stage 2 makes prestage the complete setup barrier for later work. The configured
inventory includes Qwen3-VL reasoners, BGE text retrieval, ColPali/ColQwen
vision retrieval, MMLongBench, and PaddleOCR/Docling cache warmups. Use
`--skip-reasoner-models`, `--skip-retrieval-models`, `--skip-tool-caches`,
`--model-id`, and `--retrieval-model-id` for cheap subset checks; do not add
one-off later-stage download commands without moving them into config/prestage.

MMLongBench is intentionally staged file-by-file because Kaya showed repeatable
Hub cache consistency errors on individual PDFs. The staging script prints the
dataset repo, file counts, each file path, and the active `huggingface_hub` /
`hf_xet` versions. If a single file still fails Hub cache consistency checks
after retry, it is streamed through Hugging Face's filesystem interface into
`.data/mmlongbench`.

Compute-node jobs should not download. Stage assets first, then run compute
jobs offline.

## Safety Rules

- Keep all Kaya-specific source/config/docs under `ops/kaya`.
- Do not reintroduce `scripts/kaya/` or `docs/KAYA.md`.
- Do not put code, secrets, or generated results in `.cache`, `.data`, `envs`,
  `results`, or `logs`.
- Do not hand-edit the remote mirror; `push` uses `rsync --delete`.
- Record persistent operational findings in `docs/AGENT_GUIDE.md`.

## Common Failures

- `module: command not found`: use a login shell.
- `conda activate` fails: run `scripts/setup_env.py` and check module names.
- HF asks to download on compute: run `scripts/prestage.py` on login first, then
  keep GPU jobs offline.
- Xet warning says `hf_xet` missing: rerun setup after requirements update.
- PaddleOCR `PaddlePredictorOption` TypeError during prestage: the remote env
  has an incompatible transitive PaddleX; rerun `scripts/setup_env.py` so the
  `paddlex>=3.1,<3.2` pin is applied.
- No SLURM logs: ensure `logs/` exists and the `.sbatch` output paths point
  there.
- SLURM rejects partition/GRES/account: inspect `sinfo`, `scontrol show
  partition gpu`, and update `ops/kayaconfig.json`.

## clear-cache command

`ops.kaya.kaya clear-cache` removes cached generation results on the remote to start
fresh. Default drops `results/cache/{full,smoke}` + `results/tables/{full,smoke}`
and keeps the expensive `renders`/`marker` parse caches; `--renders` also drops
those, `--all` nukes all of `results/cache` + `results/tables` + logs, `--logs`
empties `logs/` (keeps the dir, sbatch needs it), `--mode`/`--experiment` scope
it, `--local` mirrors the removals locally, `--dry-run`/`--yes` gate execution.
Paths are validated to stay inside `results/`/`logs/`.

## Operational notes (hazards that recur)

- **Queue waits.** The GPU request (`--partition=gpu --gres=gpu:1`) never changed;
  what grew was the resource envelope. `slurm` defaults are `cpus=4, mem=24G,
  time=00:30:00`. A long walltime is the main backfill killer, short jobs slot
  into gaps, a 2h job waits for a full slot. Raise per-job with
  `--time/--mem/--cpus-per-task` for the Section-2 grid. bf16 8B needs
  `--gres gpu:v100:2` (32GB); a 2-GPU node is scarce, so it queues. 4-bit fits
  `--gres gpu:v100:1` and backfills in minutes.
- **`run` vs `submit`.** `run` executes on the login node (SSH, no SLURM) unless
  the `.py` header says `target=gpu`; `submit` always goes through SLURM (generated
  sbatch for `.py`, as-is for `.sbatch`). GPU resources come from `ops/kayaconfig.json`
  or `--partition/--gres/--cpus-per-task/--mem/--time/--account/--qos`.
- **Offline caches.** Compute jobs run HF-offline and must read root-relative
  caches: the runner exports `HF_HOME`/`HF_HUB_CACHE=<root>/.cache`, unsets
  inherited `TRANSFORMERS_CACHE`, and sets `MODEL_CACHE_DIR=<root>/.cache/datalab/
  models` (Marker/Surya) plus Paddle/Docling/Torch/Xet paths. `scripts/prestage.py`
  stages Qwen weights, BGE, ColQwen, Marker/Surya, PaddleOCR, Docling; it is
  idempotent (probes the Hub cache with `local_files_only` before any network).
- **Orphaned remote processes.** Long login-node runs use `ssh -tt` (pty) +
  keepalives + a `trap … HUP TERM INT` process-group kill so a local Ctrl-C tears
  down the remote tree instead of orphaning it (HF's blocking sockets have no read
  timeout and would hang forever). Never hand-edit the remote mirror, `push` is
  `rsync --delete`. `logs/` must exist before `sbatch`.
- **Don't push while a job runs.** `push`/`submit` do `rsync --delete` on the
  code mirror; a running job already loaded its modules so it survives, but avoid
  it to prevent surprises.
- **A transient `squeue` blip reads as an empty result.** When polling job state,
  confirm a "gone" job's real end state with `sacct -j <id> -o State` (COMPLETED /
  FAILED / TIMEOUT) before concluding it finished.
- **Live config to re-confirm (drifts):** modules `Anaconda3/2024.06`,
  `cuda/12.6.3`, partition `gpu` (nodes k[026-042], 34 GPUs, MaxTime 3d), GRES
  `gpu:1`; account/QOS blank (group membership grants access).
