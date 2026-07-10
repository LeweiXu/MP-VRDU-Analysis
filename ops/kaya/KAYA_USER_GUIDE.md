# Kaya User Guide

This guide explains how to run MP-VRDU work on Kaya through `kaya.py`.

## Assumptions

- You are on the university VPN.
- `ssh kaya` works without a password prompt.
- You run commands from the local repo root.
- The local conda env exists at `envs/mpvrdu`.
- Root `.env` contains `HF_TOKEN=...` if Hugging Face auth is needed.

All fixed Kaya values are in `ops/kaya/config.json`: SSH alias, remote path,
modules, env path, SLURM defaults, rsync excludes, model IDs, dataset IDs, and
secret forwarding rules.

Check the config:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya show-config
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
envs/mpvrdu/bin/python -m ops.kaya.kaya push
```

Build/update the Kaya env:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/setup_env.py
```

Stage the dataset, all configured model weights, and tool caches on the login
node:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py
```

For a smaller first pass:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py -- --skip-models
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py -- --skip-retrieval-models --skip-tool-caches --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/prestage.py -- --skip-reasoner-models --retrieval-model-id BAAI/bge-small-en-v1.5
```

`prestage.py` uses the Hugging Face Python package. Models are downloaded as
Hub snapshots. This includes Qwen3-VL reasoners, BGE text retrieval, and
ColPali/ColQwen vision retrieval. MMLongBench is staged file-by-file so the logs
show the dataset repo, file counts, and each parquet/PDF path being fetched.
PaddleOCR and Docling caches are warmed under `.cache/`. `hf_xet` is installed
for Xet-backed cache downloads. `HF_TOKEN` is read from local `.env` and
forwarded to the remote login-node process. `.env` itself is never rsynced.

If prestage fails in PaddleOCR with a `PaddlePredictorOption` `TypeError`, rerun
`envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/setup_env.py` first. The env needs
the PaddleX 3.1 pin from `requirements.txt`.

## kaya.py Commands

Invocation is `python -m ops.kaya.kaya [--config PATH] <command> [options] [program] [-- forwarded args]`.
Runner options come before the program path; everything after `--` is forwarded
verbatim to the Python script or `.sbatch` file. The one global option is:

| Flag | Default | Meaning |
|---|---|---|
| `--config PATH` | `ops/kaya/config.json` | Path to the Kaya JSON config the runner resolves site/SLURM/path values from. |

The commands are `show-config`, `push`, `pull`, `run`, `submit`, `watch`,
`cancel`, and `clear-cache`.

### `show-config`

Prints `ops/kaya/config.json` as resolved by the runner. No options.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya show-config
```

### `push`

Mirrors local source to the configured remote root with `rsync -az --delete`. No
options. Excluded paths include `.git/`, `.env`, `.cache/`, `.data/`, `envs/`,
`results/`, `logs/`, and `__pycache__/`. Local source is authoritative; remote
source-only edits can be deleted by the next push.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya push
```

### `pull`

Pulls remote `logs/` and `results/` back to the local repo. No options. It does
not pull datasets, model caches, or environments.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya pull
```

### Shared job options (`run` and `submit`)

`run` and `submit` both take the same three groups of options: HF/env, SLURM
resources, and job lifecycle. They're listed once here and referenced by both
commands.

**HF / env** (from `add_common_run_args`):

| Flag | Default | Meaning |
|---|---|---|
| `--env` / `--no-env` | env on | Activate or skip the configured conda env (`envs/mpvrdu` per config). |
| `--offline` / `--online` | header/config | Force Hugging Face offline, or allow online access and forward configured secrets (`HF_TOKEN`) on login jobs. When unset, the script header (`# kaya: offline=...`) or the default target decides. |

**SLURM resources** (only used for generated `.py` wrappers; for a `.sbatch`
file these are passed only when set, as overrides). When a flag is omitted the
value falls back to `slurm.*` in `ops/kaya/config.json` (defaults: `partition=gpu`,
`gres=gpu:1`, `cpus_per_task=4`, `mem=24G`, `time=00:30:00`, blank
`account`/`qos`).

| Flag | Config default | Meaning |
|---|---|---|
| `--job-name NAME` | script header or `generate` | SLURM job name; sets the `logs/<name>_<id>.out` prefix. |
| `--partition P` | `gpu` | SLURM partition. |
| `--gres G` | `gpu:1` | Generic resource request, e.g. `gpu:v100:1` or `gpu:v100:2`. |
| `--account A` | blank | SLURM account/allocation (leave blank to use your default association). |
| `--qos Q` | blank | SLURM QOS. |
| `--cpus-per-task N` | `4` | CPUs per task. |
| `--mem M` | `24G` | Memory request, e.g. `64G`. |
| `--time T` | `00:30:00` | Wall time, e.g. `06:00:00`. |

**Lifecycle** (from `add_job_lifecycle_args`):

| Flag | Default | Meaning |
|---|---|---|
| `--no-push` | off | Skip the pre-run rsync (use the code already on the remote). |
| `--no-wait` | off | Submit and return immediately instead of blocking until the job ends. |
| `--no-pull` | off | Do not pull `logs/`+`results/` back after the job exits. |
| `--tail-lines N` | `120` | Number of log lines to print after a waited job. |

### `run`

Runs a Python file or command on the **login node** by default. If the Python
file declares `# kaya: target=gpu`, or you pass `--target gpu`, `run` generates
and submits a GPU sbatch wrapper (same path as `submit`).

| Flag | Default | Meaning |
|---|---|---|
| `--target auto\|login\|gpu` | `auto` | Where to run. `auto` reads the Python header and otherwise uses login. |
| `program` | required | Repo-local `.py` file, or a bare command name for login-node execution. |

Plus all [shared job options](#shared-job-options-run-and-submit).

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/dataset_stats.py
envs/mpvrdu/bin/python -m ops.kaya.kaya run --target gpu --time 00:05:00 scripts/gpu_test.py
```

### `submit`

Submits a repo-local `.py` or `.sbatch` file to SLURM.

| Flag | Default | Meaning |
|---|---|---|
| `program` | required | Repo-local `.py` (wrapped in a generated sbatch) or an existing `.sbatch` file. |

Plus all [shared job options](#shared-job-options-run-and-submit).

For `.py`, `kaya.py` generates the sbatch wrapper and applies the SLURM
CLI/config options above. For `.sbatch`, the file's own `#SBATCH` directives and
shell commands are authoritative unless you deliberately pass explicit SLURM
options (e.g. `--time`, `--partition`), which are forwarded to `sbatch` as
overrides. Custom `.sbatch` files run from the remote repo root; use
`cd "$SLURM_SUBMIT_DIR"` and set `PYTHONPATH="$SLURM_SUBMIT_DIR"` if the script
runs repo files by path.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --time 00:05:00 scripts/gpu_test.py
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --gres gpu:v100:1 --time 06:00:00 \
  cli/generate.py -- --spec specs/full_generation.yaml
envs/mpvrdu/bin/python -m ops.kaya.kaya submit path/to/job.sbatch -- --arg-for-job value
```

### `watch`

Waits for a job and pulls/prints logs. If no job id is supplied, it uses
`.kaya_last_job`.

| Flag | Default | Meaning |
|---|---|---|
| `job_id` | `.kaya_last_job` | SLURM job id to wait on (positional, optional). |
| `--job-name NAME` | none | Job name used to match `logs/<name>_<id>.out` exactly. |
| `--no-pull` | off | Do not pull `logs/`+`results/` before printing tails. |
| `--tail-lines N` | `120` | Number of log lines to print. |

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya watch
envs/mpvrdu/bin/python -m ops.kaya.kaya watch <jobid> --tail-lines 200
```

### `cancel`

Cancels your SLURM jobs. Give explicit ids, or use `--all`/`--job-name` (both
optionally narrowed by `--state`).

| Flag | Default | Meaning |
|---|---|---|
| `job_id ...` | none | Specific job id(s) to cancel (positional, zero or more). |
| `--all` | off | Cancel all of your jobs. |
| `--job-name NAME` | none | Cancel your jobs with this SLURM job name. |
| `--state S` | none | Restrict `--all`/`--job-name` to a SLURM state, e.g. `PENDING` or `RUNNING`. |

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya cancel 1009584
envs/mpvrdu/bin/python -m ops.kaya.kaya cancel --job-name t1-4bit
envs/mpvrdu/bin/python -m ops.kaya.kaya cancel --all --state PENDING
```

### `clear-cache`

Removes cached generation results (and optionally logs) on Kaya so a run starts
fresh. Remote by default; add `--local` to mirror the same removals in the local
repo. Prompts for confirmation unless `--yes`.

| Flag | Default | Meaning |
|---|---|---|
| `--mode full\|smoke` | both | Restrict to one cache mode under `results/cache/<mode>/`. |
| `--experiment NAME` | all | Restrict to one cache dir under the selected mode(s), e.g. a generation task `G1_sufficiency`. |
| `--renders` | off | Also drop the render/marker parse caches (kept by default so re-runs skip re-rendering). |
| `--logs` | off | Also empty the `logs/` directory (keeps the dir itself). |
| `--all` | off | Drop the whole `results/cache` + `results/tables` + `logs`. |
| `--local` | off | Mirror the same removals in the local repo, not just the remote. |
| `--dry-run` | off | Print the targets without removing anything. |
| `--yes` | off | Skip the confirmation prompt. |

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya clear-cache --mode full --experiment G1_sufficiency --dry-run
envs/mpvrdu/bin/python -m ops.kaya.kaya clear-cache --mode full --experiment G1_sufficiency --local --yes
```

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

CLI flags override these headers. The parser only scans the first 40 lines, so
put the `# kaya:` lines at the very top of the file (comments before the module
docstring are fine); a long docstring can push them past the cutoff and they get
silently ignored.

## Smoke Commands

Login-node data check:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run scripts/dataset_stats.py
```

GPU smoke:

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --time 00:05:00 scripts/gpu_test.py
```

Deployment-resolution probe (GPU):

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --target gpu --gres gpu:v100:2 --time 00:30:00 scripts/resolution_probe.py
```

Single YAML smoke (cache a small G1/G5 2B run):

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit cli/generate.py -- --spec specs/smoke_generation.yaml
```

Experiments now split by **role** (see `docs/USER_GUIDE.md`): the study is
organized by *generation task* (`G1_sufficiency`, `G2_family`, `G3_dataset`,
`G5_retrieval`, `G6_classifier`), not by table. Only generation needs a GPU;
judging and table building run **locally** (no GPU, only an API key). Put
`GEMINI_API_KEY` (or `OPENAI_API_KEY`) in your **local** `.env` — judge keys are
not forwarded to Kaya.

```bash
# 1. GENERATE on Kaya (GPU): cache predictions from a YAML spec
envs/mpvrdu/bin/python -m ops.kaya.kaya submit cli/generate.py -- --spec specs/full_generation.yaml

# 2. bring the prediction cache back
envs/mpvrdu/bin/python -m ops.kaya.kaya pull

# 3. JUDGE locally (scores predictions; no tables): reads manifests under the run-tag
python -m cli.judge --run-tag yaml-full

# 4. BUILD tables locally: routes each task's judged rows into the 8 CSVs + a .md
python -m cli.build --run-tag yaml-full
```

Use `specs/smoke_generation.yaml` / `--run-tag yaml-smoke` for the smoke template.

## GPU Allocation

The default config uses `--partition=gpu --gres=gpu:1`. `slurm.account` and
`slurm.qos` are blank, so jobs use your default eligible Kaya association. If
SLURM rejects a job for accounting, set those fields in `ops/kaya/config.json`.

## Normal Loop

1. Edit locally.
2. Run quick local checks.
3. `envs/mpvrdu/bin/python -m ops.kaya.kaya submit <script.py|job.sbatch>`.
4. Inspect local `logs/` after the runner pulls results.
5. Record durable findings in `docs/AGENT_GUIDE.md`.
