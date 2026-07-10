# H100 runbook: setup, prestage, and run the H100 generation

This is the whole job for the H100 box: build the envs, download the models and
data, run the generation, and check it came out clean. You do **not** judge or
build tables here. When generate is done and the check is green, hand the
`results/` folder back and you're finished.

This is the `h100_main.yaml` run: Qwen3-VL-8B over the full four-rung ladder
(T/TL/TLV/V) on oracle pages, for the whole answerable question set.

## What you need first

- **conda** on PATH (Miniconda/Anaconda is fine).
- The repo checked out. Run every command below **from the repo root**.
- One **H100** (80 GB) is plenty; the 8B fits with lots of room. If you have
  several and want to pick which, set `CUDA_VISIBLE_DEVICES=0` (or a comma list)
  before the generate step.
- **Disk**: budget ~40 GB under `.cache/` for the 8B reasoner, the retrievers, and
  the parser (see the trimmed prestage below), plus room under `results/`.
- A **Hugging Face token** so downloads don't get rate-limited. Put it in a `.env`
  file at the repo root as `HF_TOKEN=hf_...`, or just `export HF_TOKEN=hf_...`
  before step 2.

## Step 1: build the environments

Run this with your **base** conda active (it creates the project envs for you). It
builds a core reasoning env plus one isolated env per PDF parser, all inside the
checkout at `envs/`, which is where the code looks for them.

```bash
python -m ops.scripts.setup_env --machine H100 --env all --local
```

Notes:
- `--local` is what makes it build into this checkout instead of a cluster path.
- It ends each env with `pip check`. If flash-attn can't build it prints a warning
  and keeps going (the reasoner falls back to a memory-efficient kernel), so that
  is not a failure.
- The TL/TLV rungs need the `parse-paddleocrvl` env, so leave `--env all`.

## Step 2: download models and data

Activate the core env, then prestage with the trimmed config for this run. It lists
only what `h100_main.yaml` needs: the 8B reasoner, the paddleocrvl parser, and the
MMLongBench dataset (G1 uses oracle pages, so no retrievers, and no other model
sizes).

```bash
conda activate "$PWD/envs/core"
python -m ops.scripts.prestage --local --config ops/kaya/h100_main.json
```

It's idempotent, so if it dies partway just run it again and it picks up where it
left off. It prints `HF_TOKEN=set` or `missing` at the top; if it says missing and a
download stalls, set the token and rerun.

## Step 3: smoke test the pipeline first

Before the long run, do a quick end-to-end pass on 5 questions. It's the same
shape as the main run (8B, the full T/TL/TLV/V ladder at med) but capped, so it
exercises every path, including the GPU paddleocr-vl parser on the TL/TLV rungs,
in a few minutes instead of hours. Still in the core env, from the repo root:

```bash
python -m ops.generate --spec ops/specs/h100_smoke.yaml
python -m ops.scripts.check_run --spec ops/specs/h100_smoke.yaml
```

If `check_run` prints `RESULT: OK` you know the envs, the weights, and the parser
are all wired up, so the main run below won't fall over hours in. If it's not
green, see "If the check finds errors" at the bottom, fix it, and rerun this
smoke before moving on. The test writes to its own `g1-8b-test` run_tag, so it
doesn't touch the main run's cells.

## Step 4: run the generation

Still in the core env, from the repo root:

```bash
python -m ops.generate --spec ops/specs/h100_main.yaml
```

What it does: it works through the run(s) in the spec, each written to its own
`run_tag` under `results/cache/`. The parser warms once per run before the reasoner
loads, so the parser and reasoner never share the GPU. Every cell writes exactly one
row whether it succeeds or fails, so a crash mid-run doesn't lose the cells that
already ran.

It finds the weights you prestaged automatically (it points Hugging Face at
`.cache/`). If your H100 node has **no internet**, add `export HF_HUB_OFFLINE=1`
first so it never tries to reach out.

This run is long. Launching it under `tmux`/`screen` (or `nohup ... &`) so it
survives a dropped SSH session is a good idea.

## Step 5: check it ran clean

This is the "did it work" gate. It reads the status row every cell wrote and
reports, per run, how many cells are ok vs oom vs error, and how many are missing
versus expected. It exits nonzero if anything looks broken.

```bash
python -m ops.scripts.check_run --spec ops/specs/h100_main.yaml
```

A healthy run looks like this (every task `OK`, no oom/err, no missing):

```
verdict run_tag              task                       ok  oom  err  miss  note
OK      g1-8b                G1_oracle_ladder         ...    0    0     0  ... cells ok
[check] RESULT: OK - no broken tasks
```

If a task is `FAIL` or `WARN`, the script prints the top failure reasons right under
it. The usual culprit is `ParserCacheMiss` on the TL/TLV cells, which means the
`parse-paddleocrvl` env didn't build or run. On an H100 you should not see `oom`.

## If the check finds errors

1. Read the reason it printed. `ParserCacheMiss` -> the parser env is the problem;
   rebuild it with `python -m ops.scripts.setup_env --machine H100 --env parse-paddleocrvl --local` and confirm it `pip check`s clean.
2. Fix the cause, then retry just the failed cells:

   ```bash
   python -m ops.generate --spec ops/specs/h100_main.yaml --failed-only
   ```

   `--failed-only` reads what each run already wrote, re-runs only the cells whose
   status wasn't `ok`, and upgrades them in place. The cells that already
   succeeded are left untouched, so this is quick. Then rerun `check_run`; repeat
   until it's green.

## When you're done

Once `check_run` says `RESULT: OK`, hand back the whole `results/` folder (the
`results/cache/` tree is what matters, the jsonl rows). That's everything the judging
and table-building steps need; those run elsewhere.
