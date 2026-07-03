# Kaya HPC Agent Guide

This is the canonical operational guide for agents working on MP-VRDU with
Kaya. All Kaya-specific scripts, configuration, and documentation live in this
`kaya/` directory.

## Current Contract

- Local machine: edit code, run tests, and launch Kaya actions.
- Kaya login node: reachable as SSH alias `kaya`; has internet; no GPU. Use it
  for conda environment setup and Hugging Face dataset/model staging.
- Kaya compute node: reached through SLURM; has GPU; should run offline against
  pre-staged `.cache/` and `.data/` assets.
- Remote mirror: configured in `kaya/config.json` as
  `/group/ems036/lxu/mpvrdu`.
- Root-relative artifacts on both machines:
  `.cache/`, `.data/`, `envs/`, `results/`, `logs/`.
- Code sync excludes those artifacts. Do not rsync local data, model caches,
  environments, results, logs, or `.git/` to Kaya.

## Static Configuration

Use `kaya/config.json` for all site-specific values:

- `ssh_alias`
- `remote_root`
- module names
- SLURM partition/GRES/CPU/memory/time defaults
- optional SLURM account/QOS values
- conda Python version
- PyTorch wheel index
- Hugging Face prestage transport settings
- default model and dataset IDs
- rsync exclude patterns

Do not require users or agents to export shell variables for normal operation.
Temporary CLI overrides are acceptable for one command, but durable values
belong in `kaya/config.json`.

## Python CLI

Run all Kaya operations from the local repo root:

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
envs/mpvrdu/bin/python -m kaya.kaya push
envs/mpvrdu/bin/python -m kaya.kaya pull
```

The CLI uses SSH, rsync, and SLURM. It generates small remote shell scripts from
the static JSON config. Generated SLURM scripts are written under the remote
`logs/` directory and are not source files.

## Setup Flow

1. Push code:

   ```bash
   envs/mpvrdu/bin/python -m kaya.kaya push
   ```

2. Build/update the Kaya conda environment on the login node:

   ```bash
   envs/mpvrdu/bin/python -m kaya.kaya setup-env
   ```

3. Stage models and MMLongBench-Doc on the login node:

   ```bash
   envs/mpvrdu/bin/python -m kaya.kaya prestage
   ```

`prestage` downloads model snapshots into `.cache/` and stages MMLongBench into
`.data/mmlongbench/{data,documents}` using symlinks from the HF cache. This is
required because `cli.run_probe` and later loaders read the root-relative
`.data/` layout, not only Hugging Face's cache layout.

Kaya defaults to `hf.disable_xet=true` and `hf.max_workers=1` in
`kaya/config.json`. The first live MMLongBench staging attempt hit HF/Xet
partial download and range-size errors, so login-node prestaging now prefers
plain HTTP and serial downloads unless the config is deliberately changed.

For a smaller first pass, stage one model:

```bash
envs/mpvrdu/bin/python -m kaya.kaya prestage --model-id Qwen/Qwen3-VL-2B-Instruct
```

## Running Code

Login-node command:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-login -- python -m cli.run_probe loader --json
```

GPU command:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-gpu --job-name mpvrdu_probe --time 00:10:00 -- python -m cli.run_probe retrieval --json
```

Convenience probe command:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-probe loader --target login --json
envs/mpvrdu/bin/python -m kaya.kaya run-probe retrieval --target gpu --heavy --json
```

`run-gpu` and GPU-target `run-probe`:

- push code unless `--no-push` is provided;
- write a generated sbatch script under remote `logs/`;
- submit with `sbatch --parsable`;
- wait for the job unless `--no-wait` is provided;
- pull `results/` and `logs/` unless `--no-pull` is provided;
- print local log tails.

Funding/accounting: `kaya/config.json` currently leaves `slurm.account` and
`slurm.qos` empty, so jobs use the user's default eligible Kaya association on
the `gpu` partition. If SLURM rejects a job for accounting, set those config
fields or pass `--account ... --qos ...` on the command.

## Stage 1.5 Validation Commands

After `setup-env` and `prestage`, use:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-probe loader --target login --json
envs/mpvrdu/bin/python -m kaya.kaya gpu-test
envs/mpvrdu/bin/python -m kaya.kaya run-probe model-family --target gpu --heavy --json --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya run-probe retrieval --target gpu --heavy --json
```

If model probing all sizes in one job is too large, run one `--model-id` per
job. Record module/CUDA/partition findings and failures in `docs/DECISIONS.md`.

## Safety Rules

- Never hand-edit the remote mirror; `push` owns it and uses `rsync --delete`.
- Never store code under `.data/`, `.cache/`, `envs/`, `results/`, or `logs/`.
- Never depend on local `.data/` being copied to Kaya.
- Compute jobs should stay offline by default. Use `--no-offline` only for a
  deliberate diagnostic job.
- Anything that must survive future stages goes in source code,
  `kaya/config.json`, this guide, `kaya/KAYA_USER_GUIDE.md`, or
  `docs/DECISIONS.md`.

## Common Failure Modes

- `module: command not found`: the SSH command did not run as a login shell.
  The Python CLI uses `bash --login -lc`; do not replace that lightly.
- `conda activate` fails: run `setup-env`; check the Anaconda module in
  `kaya/config.json`.
- HF asset missing on compute: run `prestage` on the login node first.
- Probe cannot find PDFs: confirm `.data/mmlongbench/data` and
  `.data/mmlongbench/documents` exist on Kaya. `prestage` should create them.
- SLURM rejects partition/GRES: update `kaya/config.json` after confirming Kaya's
  current GPU partition and request syntax.
