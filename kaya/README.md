# Kaya General Setup/Testing (Qwen demo)

A from-scratch guide to UWA Kaya: login, modules, conda, SLURM basics, and a
small worked example running Qwen text and vision-language models. This is
independent of the mpvrdu pipeline — for that project's setup see
`docs/kaya_cheatsheet.md`. Everything here is self-contained in this `kaya/`
directory; commands below assume your shell's cwd is `kaya/`.

Kaya module names, partitions, and CUDA/driver versions drift over time.
Confirm them yourself before relying on anything below.

## Mental model — read this first

There are **two machines** and **two kinds of node**, which trips people up:

```text
LOCAL (your laptop/WSL)              KAYA
┌───────────────────────┐            ┌─────────────────────────────────────┐
│ kaya/  (this folder)   │  rsync     │ $KAYA_REMOTE_DIR  (= a copy of kaya/)│
│  - edit files here     │ ─push──>   │  - never hand-edit; push overwrites │
│  - logs/, results/     │ <──pull─   │    it (rsync --delete)              │
└───────────────────────┘            │                                       │
                                      │  lives under $MYGROUP = /group/...   │
                                      │  (network filesystem)                │
                                      │                                       │
                                      │  ┌───────────────┐  ┌──────────────┐ │
                                      │  │ LOGIN node     │  │ COMPUTE node │ │
                                      │  │ `ssh kaya`     │  │ via sbatch/  │ │
                                      │  │ has INTERNET   │  │ srun         │ │
                                      │  │ NO GPU, shared │  │ has GPU      │ │
                                      │  │                │  │ NO INTERNET  │ │
                                      │  └───────┬────────┘  └──────┬───────┘ │
                                      │          └─────────┬────────┘         │
                                      │            both see the SAME          │
                                      │            $MYGROUP filesystem        │
                                      └─────────────────────────────────────┘
```

Key point: **"offline" describes the *node*, not a directory.** `$KAYA_REMOTE_DIR`
(under `/group`) is mounted on and identical from both the login node and
compute nodes — `cd $KAYA_REMOTE_DIR` works the same either way. The
difference is whether *that node* can reach the internet:

- **Login node** (plain `ssh kaya`): has internet. Use it to download
  models/datasets (§5) and create the conda env (§4).
- **Compute node** (anything run via `sbatch`/`srun`): has the GPU but no
  internet. `set_offline` (in `env.sh`) makes `transformers`/`huggingface_hub`
  read from the already-downloaded `$HF_HOME` cache instead of trying to fetch.

So step 4/5 below ("on Kaya, login node") and step 6/7 ("via sbatch, compute
node") can both `cd $KAYA_REMOTE_DIR` — what changes is *how* you got there
(interactive ssh vs. a queued job) and whether `set_offline` is active.

## Environment variable reference

Everything is defined in `kaya/env.sh` with `export VAR="${VAR:-default}"` —
i.e. **edit the right-hand side of the `:-` to change a default**. All scripts
(`sync_kaya.sh`, `*.sbatch`, `setup_env.sh`, `prestage.sh`) `source env.sh`
themselves, so editing `env.sh` and re-running a script always picks up the
new value automatically — *except* in a shell where you previously ran
`source env.sh` yourself (see "gotcha" below).

| Variable | Meaning | Default | Used by |
|---|---|---|---|
| `KAYA_HOST` | Kaya's hostname | `kaya.hpc.uwa.edu.au` | `sync_kaya.sh` (only to derive nothing now — informational/`ssh-copy-id`) |
| `KAYA_USER` | Your Kaya username | `lxu` | informational, ssh-copy-id |
| `KAYA_PROJECT` | Your `/group` project code | `ems036` | derives `MYGROUP` |
| `KAYA_SSH_ALIAS` | `~/.ssh/config` `Host` alias for passwordless ssh | `kaya` | every `ssh`/`rsync` call in `sync_kaya.sh` |
| `MYGROUP` | Your project+user dir on `/group` | `/group/$KAYA_PROJECT/$KAYA_USER` | base for everything below |
| `KAYA_REMOTE_DIR` | Where this `kaya/` folder is mirrored on Kaya | `$MYGROUP/kaya_test` | `sync_kaya.sh push/pull/submit/run` |
| `KAYA_ENV` | Path to the conda environment | `$MYGROUP/conda_environments/qwen_demo` | `setup_env.sh`, `activate_env` |
| `HF_HOME` | Hugging Face cache (models + datasets) | `$MYGROUP/hf_cache` | `prestage.sh`, `download_hf.py`, `qwen_infer.py` |
| `KAYA_CUDA` | CUDA module to load | `cuda/12.6.3` | `load_modules` |
| `KAYA_ANACONDA` | Anaconda module to load | `Anaconda3/2024.06` | `load_modules` |
| `KAYA_TORCH_INDEX_URL` | pip index for the torch wheel matching `KAYA_CUDA` | `.../whl/cu126` | `setup_env.sh` |

`env.sh` also defines three **functions** (not variables), used inside
`.sbatch` scripts and interactively on Kaya:

- `load_modules` — `module load` Anaconda + CUDA (needs a **login shell**, see §3).
- `activate_env` — `conda activate "$KAYA_ENV"`.
- `set_offline` — sets `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` (compute nodes only).

### Gotcha: `source env.sh` doesn't "update" an already-exported variable

`export VAR="${VAR:-default}"` only fills in `default` if `VAR` is **currently
unset/empty**. If you previously ran `source env.sh` in a shell, `VAR` is now
exported with the *old* value, and re-running `source env.sh` in that *same
shell* will keep reusing it (the `:-` sees a non-empty `VAR` and does nothing).

- If you only ever run things as `bash sync_kaya.sh ...` / `sbatch foo.sbatch`
  (each spawns a **fresh** process that sources `env.sh` itself), you're
  unaffected — edits to `env.sh` always take effect.
- If you've run `source env.sh` interactively and then edit `env.sh`, either
  open a new shell, or:
  ```bash
  unset KAYA_HOST KAYA_USER KAYA_PROJECT KAYA_CUDA KAYA_ANACONDA KAYA_TORCH_INDEX_URL \
        MYGROUP KAYA_SSH_ALIAS KAYA_REMOTE_DIR KAYA_ENV HF_HOME
  source env.sh
  ```

### Verifying everything is set up correctly

`check_env.sh` prints every resolved variable and runs sanity checks
appropriate to where it's run:

```bash
# Locally: checks rsync is installed and `ssh kaya` is passwordless
bash check_env.sh

# On Kaya (after pushing, see §2): checks $MYGROUP/$KAYA_REMOTE_DIR/$KAYA_ENV/
# $HF_HOME exist, and whether this node has internet
source env.sh   # so $KAYA_REMOTE_DIR expands below — needed once per local shell
ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && bash check_env.sh'"
```

Run this any time something seems off, or after editing `env.sh`.

> **Note:** in every `ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && ...'"`
> one-liner below, `$KAYA_REMOTE_DIR` (and `$MYGROUP`) must expand on your
> **local** shell before being sent to ssh — i.e. you need `source env.sh`
> locally first (once per shell). If you instead escape it (`\$KAYA_REMOTE_DIR`)
> it expands on the *remote* shell, where it's unset (unless that remote shell
> already sourced `env.sh`), and `cd` silently goes to `$HOME` instead.

## 0. One-time setup

### 0a. Configure kaya/env.sh

Edit the top of `kaya/env.sh` (the `--- EDIT THESE ---` block) for your
account — `KAYA_USER`, `KAYA_PROJECT`, etc. See the variable table above for
what each one does and what it derives.

### 0b. SSH key (avoid passwords / interactive logins)

One-time, from your **local machine**:

```bash
ssh-keygen -t ed25519 -C "$(whoami)@$(hostname)-kaya" -f ~/.ssh/kaya_ed25519
ssh-copy-id -i ~/.ssh/kaya_ed25519.pub "$KAYA_USER@$KAYA_HOST"   # one last password prompt
```

Add an alias to `~/.ssh/config` so `ssh kaya` is passwordless:

```sshconfig
Host kaya
    HostName kaya.hpc.uwa.edu.au
    User lxu
    IdentityFile ~/.ssh/kaya_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
```

Test with `ssh kaya 'squeue -u $USER'` (single quotes — `$USER` must expand on
Kaya, not locally) — if that runs without a prompt, every `sync_kaya.sh`
command below is also passwordless. `sync_kaya.sh` uses this `kaya` alias
(`KAYA_SSH_ALIAS` in `env.sh`) for every ssh/rsync call.

With this in place you rarely need an interactive shell on Kaya at all:
`sync_kaya.sh push|submit|watch|run|pull` do everything via one-shot ssh
commands and rsync. Use plain `ssh kaya` for the rare ad-hoc command (model
downloads, checking quota, etc — see §5 and §8).

### 0c. Verify

```bash
bash check_env.sh
```

Expect `OK ssh kaya works without a password` and `OK rsync found`. Fix
anything marked `MISS` before continuing.

## 1. Orientation (background, no commands needed yet)

This account's GPU partition (`ssh kaya 'scontrol show partition gpu'`):

```text
PartitionName=gpu  Nodes=k[026-042]  TotalNodes=17  TRES: gres/gpu=34
MaxTime=3-00:00:00  AllowGroups=kaya-users,kaya-admins
```

i.e. `--partition=gpu --gres=gpu:1` (used by the sbatch scripts here) is valid
for this account. Other partitions: `ondemand`, `ondemand-gpu`, `amdgpu`,
`work` (default).

## 2. Sync this directory to Kaya (LOCAL)

No GitHub needed — push `kaya/`'s contents straight to `$KAYA_REMOTE_DIR` over
rsync. Run from inside `kaya/` (pushes `.` by default; pass a path to push
something else):

```bash
bash sync_kaya.sh push
```

This is `rsync --delete`, so it mirrors local -> remote (excludes `.git`,
`.cache`, `logs`, `results`, `__pycache__`, `.kaya_last_job`). Re-run after
every local edit — `submit`/`run` already do this for you. **Never edit files
directly in `$KAYA_REMOTE_DIR`** — the next `push` deletes/overwrites them.

`sync_kaya.sh` creates `$KAYA_REMOTE_DIR` (and its `logs/` dir, needed because
sbatch opens `logs/%x_%j.{out,err}` before running anything — a missing
`logs/` means a job fails with no log at all) automatically.

Verify:

```bash
ssh kaya "ls $KAYA_REMOTE_DIR"
```

You should see `env.sh`, `setup_env.sh`, `*.sbatch`, etc.

## 3. Discover modules (on Kaya, login node)

`module` is defined by shell init scripts that only run in a **login** shell,
so non-interactive `ssh kaya '...'` needs `bash -lc`:

```bash
ssh kaya bash -lc "'module avail Anaconda 2>&1 | tail -5; module avail cuda 2>&1 | tail -5'"
```

(The same applies to every other ad-hoc `ssh kaya '...'` one-liner below that
calls `module`/`load_modules` — wrap it in `bash -lc "'...'"`. Plain
`sbatch`/`squeue`/`sacct`/`scontrol`/`quota` don't need this.)

Update `KAYA_CUDA`, `KAYA_ANACONDA`, `KAYA_TORCH_INDEX_URL` in `kaya/env.sh` to
match what's available, then `bash sync_kaya.sh push` again.

## 4. Create the conda environment (on Kaya, login node — needs internet)

```bash
ssh kaya
cd $KAYA_REMOTE_DIR
bash setup_env.sh
exit
```

This creates `$KAYA_ENV` (`$MYGROUP/conda_environments/qwen_demo`) and
installs torch + transformers + accelerate + `qwen-vl-utils`. **This step is
slow — often 20-40+ minutes** for two reasons, both expected:

1. The CUDA-enabled torch wheel for `KAYA_TORCH_INDEX_URL=cu126` bundles the
   *entire* CUDA 12.6 toolkit (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
   `cuda-toolkit`, etc.) — typically 5-8 GB total, vs ~200 MB for a CPU-only
   build.
2. `/group` is network-mounted storage; conda/pip environments contain tens of
   thousands of small files, and unpacking that many files onto network
   storage is much slower than local disk. This is the dominant cost on most
   HPC clusters.

It's a one-time cost — let it finish. To check it's still progressing (not
hung) from another terminal:

```bash
ssh kaya 'ps aux | grep pip'
```

Pick a `KAYA_TORCH_INDEX_URL` that matches the CUDA module / driver on Kaya —
mismatches show up later as "CUDA available: False" or undefined-symbol
import errors (see §9).

Verify when done:

```bash
ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && bash check_env.sh'"
```

should now show `OK conda env exists`.

## 5. Download models / datasets (on Kaya, login node — needs internet)

Anything from the Hugging Face Hub (or any other dataset) must be fetched
**from the login node** (it has internet) into `$HF_HOME`
(`$MYGROUP/hf_cache`) — a directory under `/group` that compute nodes can also
read, just not write-via-download themselves.

**Option A — the bundled script**, which stages the two demo Qwen models:

```bash
ssh kaya
cd $KAYA_REMOTE_DIR
source env.sh && load_modules && activate_env
bash prestage.sh
exit
```

It downloads:

- `Qwen/Qwen2.5-1.5B-Instruct` — text-only, runs comfortably on a single GPU.
- `Qwen/Qwen2.5-VL-3B-Instruct` — vision-language, takes an image + text prompt.

Re-run anytime; `snapshot_download` resumes/skips already-downloaded files.
Check space first with `ssh kaya 'quota -s'`.

**Option B — ad-hoc, one-off downloads** (any model or dataset), without an
interactive shell, using the ssh key set up in §0b:

`download_hf.py` (in this directory, so it's already on Kaya after `push`)
wraps `snapshot_download` so the ssh command stays simple — `bash -lc` gives a
login shell so `module`/`load_modules` work:

```bash
# A different/extra model
ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && source env.sh && load_modules && activate_env && python download_hf.py Qwen/Qwen2.5-7B-Instruct'"

# A dataset
ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && source env.sh && load_modules && activate_env && python download_hf.py yubo2333/MMLongBench-Doc --dataset'"
```

Both land under `$HF_HOME` and are picked up automatically by `transformers`
(`from_pretrained('Qwen/Qwen2.5-7B-Instruct')`) once `set_offline` is active on
the compute node — no path wrangling needed, because `$HF_HOME` is the same
path on both node types.

To download a **non-HF dataset** (e.g. via `wget`/`curl` from some other URL),
just run it on the login node and put it under `$MYGROUP`, e.g.:

```bash
ssh kaya "mkdir -p $MYGROUP/data && cd $MYGROUP/data && curl -LO <url>"
```

Verify:

```bash
ssh kaya bash -lc "'cd $KAYA_REMOTE_DIR && bash check_env.sh'"
```

should now show `OK HF_HOME exists` with a non-trivial size.

## 6. GPU smoke test (LOCAL, drives Kaya via ssh)

```bash
bash sync_kaya.sh run gpu_test.sbatch
```

This pushes `kaya/`, submits the job (to a **compute node** — has the GPU, no
internet), blocks until it finishes, prints the final job state, and pulls
`logs/` + `results/` back. Check `logs/qwen_gpu_test_<jobid>.out` locally — it
should show `CUDA available: True` and a GPU name. If it prints `False` or
errors, the CUDA module / driver / torch wheel combination is wrong; fix this
(in `kaya/env.sh`, then re-push) before going further.

## 7. Run Qwen inference (LOCAL, drives Kaya via ssh)

Text model:

```bash
bash sync_kaya.sh run run_qwen.sbatch \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain what a SLURM job scheduler does, in two sentences."
```

Vision-language model (image must already be on Kaya — it'll be pushed
automatically if it's inside `kaya/`, otherwise `rsync` it into `$MYGROUP`
first):

```bash
bash sync_kaya.sh run run_qwen.sbatch \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --prompt "Describe this image." \
  --image "$MYGROUP/some_image.png"
```

The model's reply ends up in `logs/qwen_infer_<jobid>.out`, synced to your
local `kaya/logs/` directory automatically.

If you'd rather not block your terminal, split it up:

```bash
bash sync_kaya.sh submit run_qwen.sbatch --prompt "hello"   # prints + saves jobid
bash sync_kaya.sh watch                                       # blocks until done
bash sync_kaya.sh pull                                        # fetch logs/results
```

For interactive debugging (rare — only when a job fails and you need to poke
around). This `srun` step lands you on a **compute node** (GPU, no internet):

```bash
ssh kaya
srun --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --time=0:30:00 --pty /bin/bash -l
cd $KAYA_REMOTE_DIR
source env.sh
load_modules
activate_env
set_offline
python qwen_infer.py --prompt "hello"
exit
```

## 8. Monitor jobs and pull results back (LOCAL)

```bash
# Note the single quotes: $USER and <jobid> must expand on Kaya, not locally.
ssh kaya 'squeue -u $USER'
ssh kaya 'sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES'
ssh kaya 'scancel <jobid>'
bash sync_kaya.sh pull                       # syncs logs/ and results/
bash sync_kaya.sh pull-path logs/qwen_infer_12345.out
```

## 9. Common failures

- **No log file at all:** `logs/` doesn't exist on Kaya — SLURM opens output
  files before running the script and fails silently. `sync_kaya.sh push`
  creates it; if you bypassed that, `ssh kaya "mkdir -p $KAYA_REMOTE_DIR/logs"`.
- **`CUDA available: False`:** loaded CUDA module, NVIDIA driver, and the
  torch wheel index don't match. Try a different `KAYA_TORCH_INDEX_URL` /
  `KAYA_CUDA` combination.
- **HF download attempted on a compute node:** model wasn't staged (§5), or
  `set_offline` wasn't called, or `HF_HOME` differs between login and compute
  (it shouldn't — it's the same `/group` path either way; double-check
  `env.sh` wasn't edited+re-sourced inconsistently, see the gotcha above).
- **Job stuck in queue:** `ssh kaya 'sinfo -p gpu; squeue -p gpu'` — the GPU
  partition may be full, or `--gres=gpu:1` may not match the partition's GRES
  name (`scontrol show partition gpu`).
- **Quota/permission errors:** `ssh kaya 'quota -s; df -h $MYGROUP'`; make sure
  `MYGROUP`, `HF_HOME`, and `KAYA_ENV` all point at writable `/group` paths,
  not `$HOME`.
- **`sync_kaya.sh push` overwrites remote-only files:** it's `rsync --delete`
  mirroring local -> remote. Use `pull`/`pull-path` first to fetch anything
  generated only on Kaya that you want to keep.
- **conda/setup_env.sh "stuck" for 30+ minutes:** normal, see §4.
- **Edited `env.sh` but values look unchanged:** see "gotcha" in the variable
  reference above — likely a stale `source`d value in your current shell.

## 10. Day-to-day sequence

```text
LOCAL: edit code
LOCAL: bash sync_kaya.sh run gpu_test.sbatch   (after env/CUDA changes)
LOCAL: bash sync_kaya.sh run run_qwen.sbatch ...
LOCAL: inspect logs/ (already pulled), iterate
```

`prestage.sh`/`setup_env.sh`/model downloads still need the login node (§4-5)
since they're one-time/rare environment setup, not per-run jobs — but with the
§0b key in place those are a single non-interactive `ssh kaya bash -lc "'...'"`
command each.
