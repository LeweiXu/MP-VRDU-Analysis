# Kaya User Guide

This is the short guide for running MP-VRDU work on Kaya.

## One-Time Local Assumptions

- You are on the university VPN.
- `ssh kaya` works without a password prompt.
- You are running commands from the repo root.
- The local conda env exists at `envs/mpvrdu`.

All fixed Kaya values are in `kaya/config.json`. Edit that file if your SSH
alias, project path, module names, or SLURM defaults differ.

Check the config:

```bash
envs/mpvrdu/bin/python -m kaya.kaya show-config
```

## First-Time Kaya Setup

Push the code mirror:

```bash
envs/mpvrdu/bin/python -m kaya.kaya push
```

Build the Kaya conda env:

```bash
envs/mpvrdu/bin/python -m kaya.kaya setup-env
```

Stage the dataset and models:

```bash
envs/mpvrdu/bin/python -m kaya.kaya prestage
```

`prestage` uses the Hugging Face settings in `kaya/config.json`. The default is
plain HTTP and serial downloads because that was more reliable than HF/Xet for
the MMLongBench PDF snapshot on Kaya.

For a smaller first test, stage only the smallest model:

```bash
envs/mpvrdu/bin/python -m kaya.kaya prestage --model-id Qwen/Qwen3-VL-2B-Instruct
```

## Smoke Tests

Run a login-node data probe:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-probe loader --target login --json
```

Run a GPU smoke test:

```bash
envs/mpvrdu/bin/python -m kaya.kaya gpu-test
```

Run Stage 1 heavy probes:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-probe model-family --target gpu --heavy --json --model-id Qwen/Qwen3-VL-2B-Instruct
envs/mpvrdu/bin/python -m kaya.kaya run-probe retrieval --target gpu --heavy --json
```

The command waits for SLURM, pulls `logs/` and `results/`, and prints the log
tails. Full logs are under local `logs/`.

## Running Arbitrary Commands

Login node:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-login -- python -m cli.run_probe local --json
```

GPU node:

```bash
envs/mpvrdu/bin/python -m kaya.kaya run-gpu --job-name mpvrdu_job --time 00:30:00 -- python -m cli.run_probe retrieval --json
```

Useful GPU options:

- `--time HH:MM:SS`
- `--mem 64G`
- `--cpus-per-task 8`
- `--partition gpu`
- `--gres gpu:1`
- `--account <account>` if Kaya requires an explicit SLURM allocation account
- `--qos <qos>` if Kaya requires a QOS for that allocation
- `--no-wait` to submit and return immediately
- `--no-pull` to skip pulling logs/results

## GPU Allocation

The current config submits GPU jobs with `--partition=gpu --gres=gpu:1` and no
explicit SLURM account. That matches the earlier Kaya notes for this project:
the job runs under your default eligible Kaya association/group allocation.

If Kaya rejects a probe for missing or invalid funding/accounting, set
`slurm.account` and, if needed, `slurm.qos` in `kaya/config.json`, or pass
`--account ... --qos ...` on one command.

## Normal Loop

1. Edit locally.
2. Run local tests.
3. Run a Kaya command through `envs/mpvrdu/bin/python -m kaya.kaya ...`.
4. Inspect local `logs/`.
5. Record important findings in `docs/DECISIONS.md`.
