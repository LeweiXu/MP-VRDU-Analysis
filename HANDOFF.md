# Session handoff: verify GPU memory management, then run T1 for F1

Written 2026-07-04 (end of a debugging session). This is a task script for the
**next session** to execute. Read it top to bottom, then start at "The task".

---

## Background: what this session did

We were fixing CUDA OOM errors on Kaya (V100 16GB) for the Section-2 experiment
grid. Two root causes were found and fixed. Full write-ups are in
`docs/DECISIONS.md` under **"Vision-token cap (CUDA OOM fix)"** and **"GPU memory
management (parser/reasoner co-residence)"** — read those two sections first.

Short version of the fixes (all implemented, all 76 tests pass, all **uncommitted**
on `main`):

1. **Vision-token cap** — `config.max_pixels` + `config.max_pixels_for_spec`
   (2B/4B → 1.0M px ≈ 1300 tok/page; **8B → 602,112 px ≈ 800 tok/page**; 32B →
   401k). Threaded through `get_reasoner(spec, max_new_tokens=, max_pixels=)`.
   Also fixed `_reasoner_for` ignoring `config.max_tokens`.
2. **GPU memory management** (the thing to verify now):
   - **Marker disk cache** (`tools/layout.py`) under `results/cache/marker/` — the
     Surya parser runs once per page, cached to disk; the reasoner phase reads the
     cache and never loads Surya.
   - **Parse pre-pass** (`experiments/driver.py::generate` + `Orchestrator.prewarm_cell`)
     — warms Marker/retrieval caches with the reasoner NOT loaded, then unloads
     retrievers and calls `free_gpu()`. The reason pass then has the whole GPU.
   - **`free_gpu()`** (gc + `torch.cuda.empty_cache` + `synchronize`) after the
     pre-pass, after each spec's reason pass (`LocalVLMBackend.free()`), and after
     `run_side`. Retrievers gained `unload()`.
3. **Kaya CLI fixes** (`kaya/kaya.py`): runner options (`--gres`/`--time`/`--no-wait`)
   now work **before or after** the program path (was a `REMAINDER` bug that
   silently swallowed them → jobs got 30 min / 1 GPU / no `--no-wait`). Added a
   **`cancel`** subcommand. Both verified live.
4. **T4 `--questions` cap** — T4's LongDocURL corpus now honors the run's question
   cap via `config.sample` (was going to run all 2,325 questions). Not used in
   this run, just a fixed footgun.
5. **LongDocURL staging** added to `kaya/prestage.py` (`--skip-longdocurl`) — not
   needed for this run (T4 is deferred).

### Hardware facts (verified via `sinfo`, don't re-derive)
- Kaya has **only V100 16GB** GPUs (`--gres gpu:v100:2` = 2 per node = 32GB
  combined). **No A100** anywhere. AMD MI210 exists but is ROCm (unusable).
- **Qwen3-VL-8B bf16 weights are ~16GB** — do not fit one V100 with attention
  room. On **2×V100** `device_map="auto"` shards it (fits cleanly). On 1×V100 it
  only runs via CPU-offload (slow). **So this run uses `--gres gpu:v100:2`.**
- 32B is out of scope (supervisor's A100). See `docs/DECISIONS.md` "Hardware scope".
- No `bitsandbytes` in the env (no 4-bit quant).

---

## The task

**Goal:** confirm the memory-management fixes let the 8B run without OOM, then
run the full T1 (headline) generation + the cheap dependent tables, to verify the
**F1 Frontier Divergence** gate. Work through the steps; the branch at Step 4
decides whether to scale up or fix-and-retry.

The user approved this plan and is asleep — proceed autonomously through the
steps, using `cancel` to clean up test jobs. Do not run the 32B or T4. Do not
commit anything unless asked.

### Step 1 — submit the pre-flight test (small 8B run)

This is 4 questions × 4 rungs on 2×V100, ~1h walltime. It exercises the exact
GPU-heavy path (8B reasoner, parse pre-pass, free_gpu) at small scale.

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --gres gpu:v100:2 --time 01:00:00 --no-wait \
  --job-name t1-memtest \
  kaya/generate.py -- --experiment T1_headline --full --questions 4 --continue-on-error
```

`submit` auto-pushes the working tree (all the fixes go to the remote). It prints
`Submitted job <id>` and returns immediately (`--no-wait`).

### Step 2 — wait for it, then pull

The GPU queue is busy (expect a wait). Poll without blocking the whole time:

```bash
envs/mpvrdu/bin/python -m kaya.kaya cancel --help   # (reference only)
ssh kaya 'squeue -u "$(whoami)" -o "%.10i %.14j %.8T %.10M %R"'   # check state
```

When it leaves the queue (State disappears / shows COMPLETED), pull results+logs:

```bash
envs/mpvrdu/bin/python -m kaya.kaya pull
```

(Or `envs/mpvrdu/bin/python -m kaya.kaya watch` to block until it finishes, then
it auto-pulls and tails the logs.)

### Step 3 — diagnose the result

Look at:
- `logs/t1-memtest_<jobid>.out` and `.err` (verbose per-cell / per-stage logs).
- `results/cache/full/T1_headline/generate_status.json` (`status`: success/failed
  + traceback).
- `results/cache/full/T1_headline/predictions.jsonl` (one line per completed cell).

**Green looks like:** every T1 cell logs `done … | in_txt=… in_vis≈800 out=… |
ans='…'` with real answers, no `CUDA out of memory`, `generate_status.json`
status=`success`. In the log you should see, per spec: `parse pre-pass (warming
caches, reasoner not loaded)` → `pre-pass done (…s); GPU freed for reasoner` →
per-cell reason lines. `in_vis` around 800 confirms the 8B pixel cap is active.

**If it OOM'd or errored**, diagnose from the traceback + the memory numbers in
the error ("X GiB in use" before the failed alloc):
- OOM inside `model.generate` / attention on the **first** 8B cell → the 8B
  simply needs more than one V100's share even sharded, or `device_map` offload
  thrashing. Options: lower `config.max_pixels` for 8B further (e.g. 8b → 401_408
  in `config.MAX_PIXELS_BY_SIZE`), and/or confirm both GPUs are visible in the
  job (log should show accelerate placing layers on 2 devices). Re-run Step 1.
- OOM only after several cells / in `run_side` → `free_gpu()` isn't reclaiming
  (a lingering model ref). Check that Surya isn't loading during the reason pass
  (it should be a Marker cache hit — grep the log for "Recognizing Layout"
  appearing AFTER "GPU freed for reasoner"; it should not).
- A non-OOM crash → fix the specific bug, re-run tests
  (`envs/mpvrdu-local-gpu/bin/python -m pytest tests/ -q`), re-run Step 1.

Iterate Step 1–3 until green. Use `kaya cancel --job-name t1-memtest` (or
`--all`) to kill stale jobs between attempts. **If you change `max_pixels` or
anything affecting model inputs, clear the stale cache first** so cells re-run:
`ssh kaya 'rm -rf /group/ems036/lxu/mpvrdu/results/cache/full/T1_headline'` and
locally `rm -rf results/cache/full/T1_headline` (the prediction cache key does
NOT include max_pixels, so old successful cells would otherwise be reused).

### Step 4 — when the pre-flight is green: run full T1 + cheap tables + F1

Only T1 needs GPU generation; T2 (analytical) and T5 (composition) are
aggregation-only and build from T1's judged rows (no extra generation). This is
the big run (~4,300 cells on 8B → give it up to 2 days walltime).

```bash
# 1) full T1 generation on 2x V100 (long walltime; --no-wait so you can detach)
envs/mpvrdu/bin/python -m kaya.kaya submit --gres gpu:v100:2 --time 2-00:00:00 --no-wait \
  --job-name t1-full \
  kaya/generate.py -- --experiment T1_headline --full --continue-on-error

# 2) when done: pull
envs/mpvrdu/bin/python -m kaya.kaya pull

# 3) judge + build T1, T2, T5 locally (needs OPENAI_API_KEY in .env; loads no models)
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T1_headline   --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T2_analytical --full --judge gpt-4o-mini
envs/mpvrdu/bin/python -m cli.experiments --phase judge --experiment T5_composition --full --judge gpt-4o-mini

# 4) evaluate the F1 gate on the full Table 1
envs/mpvrdu/bin/python -m cli.gates frontier \
  --table results/tables/full/table1_headline.csv \
  --json-output results/gates/F1_frontier_divergence.json
```

**F1 is Go** if at least two of `text_heavy` / `in_between` / `visual_heavy` have
different `frontier` values in `results/tables/full/table1_headline.csv`. Record
the verdict + numbers in `docs/DECISIONS.md` (F1 is currently "pending" there).

Since the user is asleep: it's fine to submit the full T1 job (Step 4.1) once the
pre-flight is green — it just queues. Do the judge/build/gate (Steps 4.2–4.4)
only after it finishes and is pulled; leave a summary of the F1 verdict.

---

## Reference

**Kaya CLI (fixed this session — options work before or after the program):**
- Submit: `kaya.kaya submit [--gres … --time … --no-wait --job-name …] <prog> -- <prog args>`
- Reconnect/pull: `kaya.kaya watch` (blocks, pulls, tails) or `kaya.kaya pull`
- Cancel: `kaya.kaya cancel --all` | `kaya.kaya cancel <id> …` | `kaya.kaya cancel --job-name NAME` (add `--state PENDING` to restrict)
- Job ids are saved to `.kaya_last_job`.

**Caches / results format** (per experiment under `results/cache/full/<name>/`):
- `predictions.jsonl` — durable reasoner outputs (no judge). The GPU-phase artifact.
- `generate_results.jsonl` — rows scored by a throwaway stub judge (ignore).
- `results.jsonl` — rows scored by the REAL judge (written by the local judge phase).
- `generate_status.json` — success/failure + traceback.
- Final tables → `results/tables/full/tableN_*.csv`.
- Caches are append-only + resumable: re-running skips cached cells, retries
  missing/failed ones. Delete the experiment dir (remote + local) to force a
  clean run.

**Verify the fixes are present (cold-start sanity):**
- `grep -n "prewarm_cell\|free_gpu\|parse pre-pass" experiments/driver.py pipeline/orchestrator.py`
- `grep -n "_marker_cache\|_read_marker_cache" tools/layout.py`
- `grep -n "MAX_PIXELS_BY_SIZE\|max_pixels_for_spec" config.py`
- `grep -n "def handle_cancel\|split_forwarded_args" kaya/kaya.py`
- Tests: `envs/mpvrdu-local-gpu/bin/python -m pytest tests/ -q` → expect 76 passed.

**Gotchas:**
- Judge phase needs `OPENAI_API_KEY` (or `GEMINI_API_KEY`) in the local `.env`;
  it runs locally and loads no models (prediction-cache hits only).
- `--questions N` now caps every experiment including T4 (via `config.sample`).
- Everything from this session is **uncommitted on `main`**. `submit` pushes the
  working tree with `rsync --delete` (data/cache/results/logs/envs are excluded),
  so the remote gets the code but staged data is safe.
- Deferred (not this run): T3 (InternVL-8B), T4 (LongDocURL — stage first with
  `kaya.kaya run kaya/prestage.py -- --skip-models --skip-retrieval-models --skip-tool-caches --skip-dataset`),
  T6 (retrieval), T7 (classifier), T8 (2B ok / 32B out of scope).
