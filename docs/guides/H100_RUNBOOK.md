# H100 runbook

Build the envs, download models and data, run the five generation specs, check
they came out clean, hand back `results/`. No judging or table-building here.

## What you need first

- **conda** on PATH. Repo checked out; run everything **from the repo root**.
- One **H100** (80 GB). Several? Pick with `CUDA_VISIBLE_DEVICES=0`.
- **Disk**: ~95 GB under `.cache/` (the 32B is ~67 GB of it), plus room under
  `results/`.
- **HF token**: `HF_TOKEN=hf_...` in a repo-root `.env` or exported.

## Step 1: build the environments

With base conda active:

```bash
python -m ops.scripts.setup_env --machine H100 --env all --local
```

If flash-attn warns during the core env build, retry that env once (you want it
on an H100).

## Step 2: download models and data

```bash
conda activate "$PWD/envs/core"
python -m ops.scripts.prestage --local --config ops/kaya/h100_main.json
```

Idempotent; if it dies partway, rerun it. It prints `HF_TOKEN=set|missing` at
the top.

## Step 3: smoke test

A capped pass that exercises every path in minutes; the full run later reuses
these cells as cache hits.

```bash
python -m ops.generate --spec ops/specs/g2_sufficiency.yaml --limit 4
python -m ops.generate --spec ops/specs/g5_faithfulness.yaml --limit 4
python -m ops.scripts.check_run --spec ops/specs/g2_sufficiency.yaml --no-expected
python -m ops.scripts.check_run --spec ops/specs/g5_faithfulness.yaml --no-expected
```

No err cells in either check = good to go. Otherwise see "Debugging" below.

## Step 4: run the generations, in priority order

```bash
python -m ops.generate --spec ops/specs/g2_sufficiency.yaml
python -m ops.generate --spec ops/specs/g2_robustness.yaml
python -m ops.generate --spec ops/specs/g5_faithfulness.yaml
python -m ops.generate --spec ops/specs/g0_interleaved.yaml
python -m ops.generate --spec ops/specs/g0_reasoner.yaml
```

g0_reasoner is the 32B matched-memory pair (bf16 + 4-bit under one tag; the
driver loops the two variants). Everything is cached and resumable: rerunning the same command continues where
it left off. Run under `tmux`/`screen`. If the node has no internet, `export
HF_HUB_OFFLINE=1` first.

## Step 5: check each run came out clean

```bash
python -m ops.scripts.check_run --spec ops/specs/g2_sufficiency.yaml
python -m ops.scripts.check_run --spec ops/specs/g2_robustness.yaml
python -m ops.scripts.check_run --spec ops/specs/g5_faithfulness.yaml
python -m ops.scripts.check_run --spec ops/specs/g0_interleaved.yaml
python -m ops.scripts.check_run --spec ops/specs/g0_reasoner.yaml
```

Healthy = every task `OK`, no oom/err, no missing. You should not see `oom` on
an H100.

## Debugging

- `ParserCacheMiss` on TL/TLV cells: the `parse-paddleocrvl` env is broken.
  Rebuild it (`python -m ops.scripts.setup_env --machine H100 --env
  parse-paddleocrvl --local`) and confirm `pip check` is clean.
- `RetrievalMemoMiss` on page_set cells: the ColQwen3 ranking pass didn't
  finish; rerun the spec (without `--skip-retrieval`) and it re-ranks.
- "run settings mismatch" on g5_faithfulness: the spec's decode budgets or
  delimiter were edited between a partial run and its resume. Revert the edit
  (or use a new run_tag); mixing them within one tag is refused by design.
- A `page_set: excluded {...}` log line at start is documented policy, not an
  error.
- Then retry just the failed cells and re-check, repeating until green:

  ```bash
  python -m ops.generate --spec ops/specs/<the-spec>.yaml --failed-only
  ```

## When you're done

Every `check_run` OK: hand back the whole `results/` folder (the
`results/cache/` jsonl tree is what matters). Judging and tables run elsewhere.

The four specs, in priority order (each independently resumable):

| # | spec | what it is | cells |
|---|------|-----------|-------|
| 1 | `ops/specs/g2_sufficiency.yaml` | LOPO: withhold/isolate one gold page by rank (4 runs) | 11,456 |
| 2 | `ops/specs/g2_robustness.yaml` | all gold + k ranked distractors, blocked by gold count (3 runs) | 12,256 |
| 3 | `ops/specs/g5_faithfulness.yaml` | six prompt modes on both pools (2 runs) | 26,184 |
| 4 | `ops/specs/g0_interleaved.yaml` | TLV vs TLVi ordering comparison (1 run) | 1,694 |
| 5 | `ops/specs/g0_reasoner.yaml` | 32B matched-memory pair, bf16 + 4-bit (1 run, 2 specs) | 6,776 |
|   | **total** | | **58,366** |

Cell counts are exact (enumerated against the corpus): the robustness blocks
hold 480 / 246 / 40 questions (exactly 1, 2, 3 gold pages), the sufficiency
runs the 358-question hop:multi pool, and the 32B pair is 847 questions x 4
rungs x 2 variants.

## How long it takes (single H100)

Basis: ~33 s/cell measured on the V100 (no flash attention); with flash-attn 2
and Hopper, budget ~7-10 s/cell standard, ~15-30 s/cell for the two CoT modes
in g5_faithfulness (budget 2048; most stop earlier on EOS).

| spec | standard cells | CoT cells | estimate |
|------|---------------|-----------|----------|
| g2_sufficiency | 11,456 | 0 | ~22-32 h |
| g2_robustness | 12,256 | 0 | ~24-34 h |
| g5_faithfulness | 17,456 | 8,728 | ~48-65 h |
| g0_interleaved | 1,694 | 0 | ~4-6 h |
| g0_reasoner (32B pair) | 6,776 | 0 | ~45-70 h |

The 32B row is slower per cell than the 8B rate above: ~4x the weights makes
bf16 decode roughly 3x slower (~20-30 s/cell), and bitsandbytes 4-bit is no
faster than bf16 (NF4 dequant overhead, ~25-45 s/cell) even though it fits in
a quarter of the memory.

One-time overheads on a fresh box: downloads (~25 GB), the parser warming its
markdown cache (a few hours, cached after), ColQwen3 ranking every document
once (an hour or two, persisted).
