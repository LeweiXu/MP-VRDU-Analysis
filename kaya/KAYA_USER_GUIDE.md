# Kaya User Guide

This guide explains how to run MP-VRDU work on Kaya through `kaya.py`.

## Assumptions

- You are on the university VPN.
- `ssh kaya` works without a password prompt.
- You run commands from the local repo root.
- The local conda env exists at `envs/mpvrdu`.
- Root `.env` contains `HF_TOKEN=...` if Hugging Face auth is needed.

All fixed Kaya values are in `kaya/config.json`: SSH alias, remote path,
modules, env path, SLURM defaults, rsync excludes, model IDs, dataset IDs, and
secret forwarding rules.

Check the config:

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
```

## How Kaya Works

The login node is for setup and downloads. It has internet but no GPU. GPU work
must go through SLURM, which schedules jobs onto compute nodes.

An `.sbatch` file is the normal SLURM batch script format. It has `#SBATCH`
directives for resources and then shell commands to run. Example directives:

```bash
#SBATCH --job-name=my_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Useful SLURM commands:

```bash
ssh kaya 'squeue -u $USER'
ssh kaya 'sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode'
ssh kaya 'scancel <jobid>'
```

## First-Time Setup

Push the code mirror:

```bash
envs/mpvrdu/bin/python -m kaya.kaya push
```

Build/update the Kaya env:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run kaya/setup_env.py
```

Stage the dataset, all configured model weights, and tool caches on the login
node:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run kaya/prestage.py
```

For a smaller first pass:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run kaya/prestage.py -- --skip-models
envs/mpvrdu/bin/python -m kaya.kaya run kaya/prestage.py -- --skip-retrieval-models --skip-tool-caches --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya run kaya/prestage.py -- --skip-reasoner-models --retrieval-model-id BAAI/bge-small-en-v1.5
```

`prestage.py` uses the Hugging Face Python package. Models are downloaded as
Hub snapshots. This includes Qwen3-VL reasoners, BGE text retrieval, and
ColPali/ColQwen vision retrieval. MMLongBench is staged file-by-file so the logs
show the dataset repo, file counts, and each parquet/PDF path being fetched.
PaddleOCR and Docling caches are warmed under `.cache/`. `hf_xet` is installed
for Xet-backed cache downloads. `HF_TOKEN` is read from local `.env` and
forwarded to the remote login-node process. `.env` itself is never rsynced.

If prestage fails in PaddleOCR with a `PaddlePredictorOption` `TypeError`, rerun
`envs/mpvrdu/bin/python -m kaya.kaya run kaya/setup_env.py` first. The env needs
the PaddleX 3.1 pin from `requirements.txt`.

## kaya.py Commands

Runner options come before the program path. Everything after the program path,
usually separated with `--`, is forwarded to the script or `.sbatch` file.

### `show-config`

Prints `kaya/config.json` as resolved by the runner.

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
```

### `push`

Mirrors local source to the configured remote root with:

```text
rsync -az --delete
```

Excluded paths include `.git/`, `.env`, `.cache/`, `.data/`, `envs/`,
`results/`, `logs/`, and `__pycache__/`. Local source is authoritative; remote
source-only edits can be deleted by the next push.

```bash
envs/mpvrdu/bin/python -m kaya.kaya push
```

### `pull`

Pulls remote `logs/` and `results/` back to the local repo. It does not pull
datasets, model caches, or environments.

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
```

### `run`

Runs a Python file or command on the login node by default. If the Python file
declares `# kaya: target=gpu`, or you pass `--target gpu`, `run` generates and
submits a GPU sbatch wrapper.

```bash
envs/mpvrdu/bin/python -m kaya.kaya run kaya/run_probe.py -- loader --json
envs/mpvrdu/bin/python -m kaya.kaya run --target gpu --time 00:05:00 kaya/gpu_test.py
```

Options:

- `--target auto|login|gpu`: default `auto`; reads Python headers, otherwise
  uses login.
- `--env` / `--no-env`: activate or skip the configured conda env.
- `--offline` / `--online`: force Hugging Face offline or online mode.
- `--no-push`: skip the pre-run rsync.
- `--no-wait`: for GPU jobs, submit and return.
- `--no-pull`: for waited GPU jobs, skip pulling logs/results.
- `--tail-lines N`: log lines to print after a waited GPU job.
- SLURM options for generated GPU wrappers: `--job-name`, `--partition`,
  `--gres`, `--account`, `--qos`, `--cpus-per-task`, `--mem`, `--time`.

Everything after `--` is passed to the script or command.

### `submit`

Submits a repo-local `.py` or `.sbatch` file to SLURM.

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:05:00 kaya/gpu_test.py
envs/mpvrdu/bin/python -m kaya.kaya submit path/to/job.sbatch -- --arg-for-job value
```

For `.py`, `kaya.py` generates the sbatch wrapper and applies SLURM CLI/config
options. For `.sbatch`, the file's own `#SBATCH` directives and shell commands
are authoritative unless you deliberately pass explicit SLURM options such as
`--time` or `--partition`, which are forwarded to `sbatch` as overrides.
Custom `.sbatch` files are submitted from the remote repo root; use
`cd "$SLURM_SUBMIT_DIR"` and set `PYTHONPATH="$SLURM_SUBMIT_DIR"` if the script
runs repo files by path.

Options are the same lifecycle options as `run`: `--no-push`, `--no-wait`,
`--no-pull`, `--tail-lines`, plus SLURM options for generated Python wrappers.

### `watch`

Waits for a job and pulls/prints logs. If no job id is supplied, it uses
`.kaya_last_job`.

```bash
envs/mpvrdu/bin/python -m kaya.kaya watch
envs/mpvrdu/bin/python -m kaya.kaya watch <jobid> --tail-lines 200
```

Options:

- `--job-name`: improves exact log filename matching.
- `--no-pull`: skip pulling logs/results.
- `--tail-lines N`: number of log lines to print.

## Python Script Headers

Runnable Python files can declare Kaya defaults in the first few lines:

```python
# kaya: target=login
# kaya: env=true
# kaya: offline=false
# kaya: job-name=optional_name
```

Supported keys:

- `target=login|gpu`
- `env=true|false`
- `offline=true|false`
- `job-name=name`

CLI flags override these headers.

## Smoke Commands

Login-node data probe:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run kaya/run_probe.py -- loader --json
```

GPU smoke:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:05:00 kaya/gpu_test.py
```

Heavy Stage 1 probes:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:30:00 kaya/run_probe.py -- model-family --run-heavy --json --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:30:00 kaya/run_probe.py -- retrieval --run-heavy --json
```

Single-experiment reasoner smoke (generate the headline table's predictions):

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T1_headline
```

Experiments (all 8 tables, real models). Generation runs on Kaya (GPU); the
judge + table build run **locally** (no GPU, only an API key). Each table is its
own experiment, so you can run one or all. Put `GEMINI_API_KEY` (or
`OPENAI_API_KEY`) in your **local** `.env` — judge keys are not forwarded to Kaya.

```bash
# phase 1 on Kaya: generate + cache predictions on the GPU (one job per experiment)
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T1_headline
# ...or all experiments in one job:
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment all

# bring the prediction cache back
envs/mpvrdu/bin/python -m kaya.kaya pull

# phase 2 locally: judge + build the table CSVs (results/tables/smoke/*.csv)
envs/mpvrdu-local-gpu/bin/python -m cli.experiments --phase judge --experiment all
```

Add `--full` for the full corpus/8B run. Locally (a GPU + internet in one env)
you can also run both phases at once: `python -m cli.experiments --phase all`.

## GPU Allocation

The default config uses `--partition=gpu --gres=gpu:1`. `slurm.account` and
`slurm.qos` are blank, so jobs use your default eligible Kaya association. If
SLURM rejects a job for accounting, set those fields in `kaya/config.json`.

## Normal Loop

1. Edit locally.
2. Run quick local checks.
3. `envs/mpvrdu/bin/python -m kaya.kaya submit <script.py|job.sbatch>`.
4. Inspect local `logs/` after the runner pulls results.
5. Record durable findings in `docs/AGENT_GUIDE.md`.
