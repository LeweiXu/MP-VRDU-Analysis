# H100 runbook: setup, prestage, and run the four planned generations

This is the whole job for the H100 box: build the envs, download the models and
data, run the four generation specs, and check they came out clean. You do
**not** judge or build tables here. When generate is done and the checks are
green, hand the `results/` folder back and you're finished.

The four specs, in priority order (run them in this order; each is independently
resumable, so a partial pass is never wasted):

| # | spec | what it is | cells |
|---|------|-----------|-------|
| 1 | `ops/specs/g2_sufficiency.yaml` | LOPO: withhold/isolate one gold page by rank (4 runs) | 11,456 |
| 2 | `ops/specs/g2_robustness.yaml` | all gold + k ranked distractors, blocked by gold count (3 runs) | 12,256 |
| 3 | `ops/specs/g5_faithfulness.yaml` | six prompt modes on both pools (2 runs) | 26,184 |
| 4 | `ops/specs/g0_interleaved.yaml` | TLV vs TLVi ordering comparison (1 run) | 1,694 |
|   | **total** | | **51,590** |

Cell counts are exact (enumerated against the corpus): the robustness blocks
hold 480 / 246 / 40 questions (exactly 1, 2, 3 gold pages), the sufficiency
runs the 358-question hop:multi pool.

## How long it takes (single H100)

Basis: the V100 runs measured ~33 s/cell wall for the 8B (memory-efficient SDPA,
no flash attention, 16 GB). The H100 has flash-attn 2, Hopper tensor cores, and
~3.7x the memory bandwidth, so budget **~7-10 s/cell** for a standard 256-token
cell, and **~15-30 s/cell** for the two CoT modes in g5_faithfulness (decode
budget 2048; most generations stop earlier on EOS, so the top of that range is
pessimistic).

| spec | standard cells | CoT cells | estimate |
|------|---------------|-----------|----------|
| g2_sufficiency | 11,456 | 0 | ~22-32 h |
| g2_robustness | 12,256 | 0 | ~24-34 h |
| g5_faithfulness | 17,456 | 8,728 | ~70-120 h |
| g0_interleaved | 1,694 | 0 | ~4-6 h |
| **total** | | | **~5-8.5 days** |

One-time overheads on a fresh box, before the steady-state rate: model + data
downloads (~25 GB), the PaddleOCR-VL parser warming its markdown cache over the
fed pages (a few hours, cached after the first pass), and ColQwen3 ranking every
document once for the page_set rules (an hour or two, persisted to the retrieval
memo). Run everything under `tmux`/`screen` so a dropped SSH session doesn't
kill it; a killed run resumes from its cache.

## What you need first

- **conda** on PATH (Miniconda/Anaconda is fine).
- The repo checked out. Run every command below **from the repo root**.
- One **H100** (80 GB) is plenty; the 8B fits with lots of room. If you have
  several and want to pick which, set `CUDA_VISIBLE_DEVICES=0` (or a comma list)
  before the generate step.
- **Disk**: budget ~25 GB under `.cache/` (8B reasoner, ColQwen3, the parser)
  plus room under `results/`.
- A **Hugging Face token** so downloads don't get rate-limited. Put it in a
  `.env` file at the repo root as `HF_TOKEN=hf_...`, or `export HF_TOKEN=hf_...`
  before step 2.

## Step 1: build the environments

Run this with your **base** conda active (it creates the project envs for you).
It builds the core reasoning env plus the isolated parser envs, all inside the
checkout at `envs/`, which is where the code looks for them.

```bash
python -m ops.scripts.setup_env --machine H100 --env all --local
```

Notes:
- `--local` is what makes it build into this checkout instead of a cluster path.
- It ends each env with `pip check`. If flash-attn can't build it prints a
  warning and keeps going (the reasoner falls back to a memory-efficient
  kernel); on an H100 you want flash-attn, so if it warns, retry that env once.
- The TL/TLV rungs need the `parse-paddleocrvl` env. Strictly only `core` and
  `parse-paddleocrvl` are needed for these four specs, but `--env all` is
  harmless.

## Step 2: download models and data

Activate the core env, then prestage with the trimmed config for these runs. It
lists exactly what the four specs need: the 8B reasoner, ColQwen3 (the page_set
ranker; bm25 needs no weights), the paddleocrvl parser, and MMLongBench.

```bash
conda activate "$PWD/envs/core"
python -m ops.scripts.prestage --local --config ops/kaya/h100_main.json
```

It's idempotent, so if it dies partway just run it again and it picks up where
it left off. It prints `HF_TOKEN=set` or `missing` at the top; if it says
missing and a download stalls, set the token and rerun.

## Step 3: smoke test the pipeline first

Before the long runs, do a quick end-to-end pass on a handful of questions. It's
the same cells as the real runs but capped, so it exercises every path
(the ColQwen3 ranking, the page_set rules, the GPU parser on TL/TLV, the six
prompt modes and their decode budgets) in minutes instead of days. Because the
capped cells are the same cells the full run needs, nothing is wasted: the full
run resumes over them as cache hits.

```bash
python -m ops.generate --spec ops/specs/g2_sufficiency.yaml --limit 4
python -m ops.generate --spec ops/specs/g5_faithfulness.yaml --limit 4
python -m ops.scripts.check_run --spec ops/specs/g2_sufficiency.yaml --no-expected
python -m ops.scripts.check_run --spec ops/specs/g5_faithfulness.yaml --no-expected
```

If both checks report no err cells, the envs, the weights, the ranker, and the
parser are all wired up, so the main runs below won't fall over hours in.
(`--no-expected` because a capped run is deliberately incomplete.) If not, see
"If the check finds errors" at the bottom, fix it, and rerun this smoke before
moving on.

## Step 4: run the generations, in priority order

Still in the core env, from the repo root:

```bash
python -m ops.generate --spec ops/specs/g2_sufficiency.yaml
python -m ops.generate --spec ops/specs/g2_robustness.yaml
python -m ops.generate --spec ops/specs/g5_faithfulness.yaml
python -m ops.generate --spec ops/specs/g0_interleaved.yaml
```

What it does: each spec works through its run(s), each written to its own
`run_tag` under `results/cache/`. The ranking and parser warm once per run
before the reasoner loads, so they never share the GPU with it. Every cell
writes exactly one row whether it succeeds or fails, so a crash mid-run doesn't
lose the cells that already ran; rerunning the same command resumes.

Two things specific to these specs:
- `g5_faithfulness` writes a `run_settings.json` sidecar next to each run's
  predictions (the decode budgets + the `Answer:` delimiter). Don't edit the
  spec's budgets between a partial run and its resume; the sidecar will refuse
  the mismatch by design.
- If a page_set run logs a `page_set: excluded {...}` line at start, that is
  the documented degenerate-case exclusion policy, not an error (these four
  specs pre-filter by hop, so the counts above already reflect it).

It finds the weights you prestaged automatically (it points Hugging Face at
`.cache/`). If the node has **no internet**, add `export HF_HUB_OFFLINE=1` first
so it never tries to reach out.

## Step 5: check each run came out clean

The "did it work" gate, per spec. It reads the status row every cell wrote and
reports ok / oom / error / missing per run, and exits nonzero if anything looks
broken.

```bash
python -m ops.scripts.check_run --spec ops/specs/g2_sufficiency.yaml
python -m ops.scripts.check_run --spec ops/specs/g2_robustness.yaml
python -m ops.scripts.check_run --spec ops/specs/g5_faithfulness.yaml
python -m ops.scripts.check_run --spec ops/specs/g0_interleaved.yaml
```

A healthy run shows every task `OK` with no oom/err and no missing. On an H100
you should not see `oom`.

## If the check finds errors

1. Read the reason it printed. `ParserCacheMiss` on TL/TLV cells means the
   `parse-paddleocrvl` env is the problem; rebuild it with
   `python -m ops.scripts.setup_env --machine H100 --env parse-paddleocrvl --local`
   and confirm it `pip check`s clean. `RetrievalMemoMiss` on page_set cells
   means the ColQwen3 ranking pass didn't complete; rerun the spec without
   `--skip-retrieval` and it re-ranks.
2. Fix the cause, then retry just the failed cells:

   ```bash
   python -m ops.generate --spec ops/specs/<the-spec>.yaml --failed-only
   ```

   `--failed-only` re-runs only the cells whose status wasn't `ok`, upgrading
   them in place. Then rerun `check_run`; repeat until green.

## When you're done

Once every `check_run` says OK, hand back the whole `results/` folder (the
`results/cache/` tree is what matters, the jsonl rows). That's everything the
judging and table-building steps need; those run elsewhere.
