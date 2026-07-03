# Kaya Runbook

This is the canonical Kaya workflow for the MP-VRDU pipeline. The older
`kaya/` directory is a standalone reference kit and is not the operational path
for this codebase.

## Roles

- Local machine: edit code, run small samples, push jobs, and pull results.
- Kaya login node: has internet and no GPU. Build `envs/mpvrdu` and pre-stage
  Hugging Face models/datasets into `.cache/`.
- Kaya compute node: has GPU and no internet. Jobs source `scripts/kaya/env.sh`
  and call `set_offline` before running the pipeline.

## One-time configuration

Set these if the defaults in `scripts/kaya/env.sh` do not match your account:

```bash
export KAYA_USER=lxu
export KAYA_PROJECT=ems036
export KAYA_SSH_ALIAS=kaya
```

The remote mirror defaults to:

```text
$MYGROUP/mpvrdu
```

with root-relative artifacts:

```text
$KAYA_REMOTE_DIR/.cache
$KAYA_REMOTE_DIR/envs/mpvrdu
$KAYA_REMOTE_DIR/.data
$KAYA_REMOTE_DIR/results
$KAYA_REMOTE_DIR/logs
```

## Sync model

Run sync commands from the local repository:

```bash
bash scripts/kaya/sync_kaya.sh push
bash scripts/kaya/sync_kaya.sh pull
```

`push` uses `rsync --delete` for code, while excluding `.git/`, `.cache/`,
`envs/`, `.data/`, `results/`, and `logs/`. The importable `data/` package is
code and is synced normally.

Never hand-edit the remote mirror; the next push owns it.

## Login-node setup

After a push, run login-node setup through SSH:

```bash
ssh kaya "cd /group/ems036/lxu/mpvrdu && bash --login scripts/kaya/setup_env.sh"
ssh kaya "cd /group/ems036/lxu/mpvrdu && bash --login scripts/kaya/prestage.sh"
```

Use the actual remote path for your account if you changed `KAYA_USER`,
`KAYA_PROJECT`, or `KAYA_REMOTE_DIR`.

## Compute-node jobs

Smoke-test CUDA first:

```bash
bash scripts/kaya/sync_kaya.sh run scripts/kaya/gpu_test.sbatch
```

Pipeline jobs use the same local command shape:

```bash
bash scripts/kaya/sync_kaya.sh run scripts/kaya/run_experiment.sbatch --help
```

`run` pushes code, submits the job, waits for it to leave the queue, and pulls
`results/` and `logs/` back into the local root.

## Rules

- Anything touching `module` must run in a login shell.
- Compute-node jobs must call `set_offline` before loading Hugging Face assets.
- `logs/` must exist before `sbatch`; `sync_kaya.sh` creates it on the remote.
- Do not copy `.cache/`, `.data/`, `envs/`, or dataset artifacts between machines with
  rsync. Build or download them natively on Kaya.
- Record confirmed module names, CUDA version, partition, and GPU request syntax
  in `docs/DECISIONS.md` during Stage 1.
