# Handoff (2026-07-12, evening): tooling hardened, Kaya generation in flight

## Code state

Two commits since the last handoff carry the generate/judge split (`6064117`) and a
batch of tooling (`46c92df` "actually all working"). The generation pipeline itself is
unchanged; everything below is entry-point / ops tooling.

**Committed (`6064117` + `46c92df`):**
- **Generate/judge split.** Generate writes only `predictions.jsonl` (one `PredictionRow`
  per cell incl. failures, no judging). `ops.judge --spec <yaml> --judge-spec <spec>` writes
  `results.jsonl` (a strict superset: prediction + verdict). `judge_spec` removed from specs;
  `ResultRow` shape + cache keys unchanged. See `docs/DECISIONS.md` top entry.
- **Kaya preflight.** `ops/scripts/preflight.py`, auto-run by `kaya submit` on the login node
  for any spec-driven submit (`--no-preflight` skips): checks spec parse, imports, corpus
  resolve (+ cell counts), staged weights, parser env, `--gres` sizing.
- **Kaya job-name fix.** `kaya.py::spec_job_name` derives the SLURM job name from the spec's
  `run_tag` (jobs were all named `generate`). Rename running jobs with `scontrol update
  JobId=N JobName=...`.
- **Judge `.env` loading.** `ops/judge.py` now loads the repo `.env` (`GEMINI_API_KEY` /
  `_SECONDARY` / `OPENAI_API_KEY`) into the environment before scoring; `gemini-judge` added
  as a `get_judge` alias. `gemini-flash` needs no OpenAI key.
- **check_run resolution fix.** `_expected_rows` now multiplies by `visual_resolutions` (the
  resolution ladder was under-reported, e.g. 1314 instead of 3942).
- **Spec renames.** `kaya_g1_quantization_per_doc_type_80.yaml` -> `kaya_g1_quantization_full.yaml`
  (8bit+4bit, full corpus); `kaya_g1_reasoner_per_doc_type_80.yaml` -> `kaya_g1_reasoner_full.yaml`
  (2B/4B/InternVL3-8B).

**Uncommitted (working tree) — commit these when convenient:**
- **`--skip-retrieval`** (`ops/generate.py`, `experiments/engine/driver.py` + test). Skips the
  stage-1 retrieval benchmark on a *normal* run and reuses the existing retrieval memo, so a
  resumed / supervisor inference pass never re-ranks or rewrites `retrieval.jsonl`. Needs the
  memo present (else it warns and re-ranks). See the G2 workflow below.
- **check_run `--check-all`** (`ops/scripts/check_run.py`): reads every `ops/specs/*.yaml`,
  prints each in detail, and ends with a cross-run summary table (done% / oom% per run_tag).
- **qwen3-embedding OOM fix** (`retrievers/text.py`): the 4B embedder's OOM was activation,
  not weight, memory, so the encode batch is 8 -> 2 and `max_seq_length` is capped at 4096
  (it fits one 16 GB V100 now; the weights already fit at 8 GB fp16).
- **`ops/scripts/complete_retrieval.py`**: rebuild the retrieval **memo** for named methods,
  writing the benchmark rows to a separate `--filename` (it refuses `retrieval.jsonl`).
- **`ops/scripts/g2_rerun.py`**: one GPU job = stage 1 completes the qwen3-embedding memo,
  stage 2 runs `ops.generate --skip-retrieval`; the retriever frees the GPU before the reasoner.

Full suite: **216 green.**

## Kaya jobs (status as of the last pull; re-check with `check_run --check-all`)

Run `envs/mpvrdu/bin/python -m ops.kaya.kaya run --no-push ops/scripts/check_run.py -- --check-all`
for live state. Last observed:

| run_tag | wall | cells | done | note |
|---|---|---|---|---|
| `g1-representation-full` | 24h | 2,628 | **100%** (2409 ok / 219 oom) | FINISHED |
| `g3-hallucination-full` | 24h | 2,928 | ~69% (1848 ok / 176 oom) | will time out, resubmit to resume |
| `g2-retrieval-full` | 72h | 25,410 inf | 0.3% (69 ok / 46 oom) | inference started, see G2 below |
| `g1-quantization-full` | 60h | 5,256 | ~6% | running (8bit first, slow) |
| `g1-reasoner-full` | 60h | 7,884 | ~6% | running (2B first) |
| `g1-resolution-full` | 54h | 3,942 | ~6% | running |

OOM failures are the expected V100 pressure on big-context TLV/V (and high-k) cells: all CUDA
OOM, zero real errors, marked `oom`, job continues, completed later on the supervisor via
`--failed-only`.

## Deadline: Kaya down Fri 2026-07-17 17:00 for migration

Per task:
- **G1 representation:** done. Supervisor `--failed-only` for the 219 OOM cells, then judge + build.
- **G3:** let it hit the 24h wall (do NOT cancel, requeue risk), then resubmit the same spec to
  resume from cache + run the classifier side artifact. Then supervisor `--failed-only`.
- **G2:** reduce k and hand inference to the supervisor (workflow below).
- **G1 quant/reasoner/resolution:** all finish inside their walls; supervisor `--failed-only` for OOMs.

## G2: reduce k, then supervisor inference-only (uses `--skip-retrieval`)

**OOM scales hard with k** (8B, 2xV100): k=1 0%, k=3 8%, k=5 38%, k=7 58%, k=10 **100%**. So the
25,410-cell inference is dominated by high-k OOMs. Plan:
1. `scancel` the G2 job.
2. Edit `kaya_g2_full.yaml`: `k_values: [1, 3, 5]` (or `[1, 3]` for ~zero OOM); trim `joint_k_values`.
3. Push (cancel G2 first so the push does not race it; the change is additive and does not touch
   the other running jobs' in-memory modules).
4. Kaya resubmit **with `--skip-retrieval`**: reduced-k inference, memo reused, `retrieval.jsonl`
   left untouched (full k preserved, no backup needed). Cells k=1,3 rarely OOM; k=5 ~38% become
   failed cells.
5. Supervisor: sync the run_tag cache (`predictions.jsonl` + the `retrieval/` **memo** +
   renders/parser caches; `retrieval.jsonl` itself is not needed for inference), then
   `ops.generate --spec kaya_g2_full.yaml --skip-retrieval` for inference-only on the bigger GPU,
   no retrieval rerun. (`--failed-only` also works if Kaya attempted every cell; `--skip-retrieval`
   also covers cells Kaya never reached.)

**Combined job (recommended):** `ops/scripts/g2_rerun.py` does stage 1 (complete the
qwen3-embedding memo) + stage 2 (`generate --skip-retrieval`) in one allocation:
```bash
kaya submit --gres gpu:v100:2 --time 48:00:00 ops/scripts/g2_rerun.py -- --spec ops/specs/kaya_g2_full.yaml
```
Pass `--skip-complete` to skip stage 1 if the retrieval rung is already done. Job is named after
the run_tag (`g2-retrieval-full`), not `generate`.

Orphan note: reducing k leaves the old k=7,10 rows in `predictions.jsonl` (the key-cache never
deletes). Filter conditions containing `_k7`/`_k10` at build, or clear-cache + rerun.

## qwen3-embedding retrieval rung: cause found + fixed, still to complete

Two separate files were involved (this caused confusion once):
- the **memo** `results/cache/g2-retrieval-full/retrieval/qwen3-embedding__dpi200.jsonl` had
  **65 of 847** rankings;
- the **benchmark** `.../full/G2_retrieval/retrieval.jsonl` had **zero** qwen3-embedding rows
  (and no qwen3-embedding|colqwen3 joint).

Cause: **Qwen3-Embedding-4B OOM'd during stage-1 ranking** after ~65 questions. It is NOT a
fundamental limit and NOT a leak: the 8 GB fp16 weights fit a 16 GB V100 (loaded and ranked 65
docs fine); the OOM was **activation memory** on a document with long page text, encoded at
`batch_size=8`. The `device="cuda"` load also strands the whole model on one card even with
`gpu:v100:2` (device_map wouldn't shard a model that fits on one device). **Fixed** in
`retrievers/text.py`: encode batch 2 + `max_seq_length` 4096.

Completing the rung (the memo regenerates **in place**; `--filename` only steers the *benchmark*
output away from `retrieval.jsonl`):
```bash
# standalone, or folded into the G2 rerun via g2_rerun.py
python -m ops.scripts.complete_retrieval --spec ops/specs/kaya_g2_full.yaml \
    --text-methods qwen3-embedding --vision-methods colqwen3 --joints matched \
    --filename retrieval_qwen3.jsonl
```
This fills `retrieval/qwen3-embedding__dpi200.jsonl` (resumes 65 -> 847), reuses the complete
colqwen3 memo for the joint (no model load), and writes the qwen3-embedding + joint benchmark rows
to `retrieval_qwen3.jsonl`. `cat retrieval_qwen3.jsonl >> retrieval.jsonl` to fold them into the
table. It does NOT affect inference (which uses bge-m3 + colqwen2.5, both already complete).

## Known behaviour: answers ~35x longer than the v3 (`old/`) runs

New answers average ~94 words / ~140 output tokens (hitting the 256 cap) vs old ~2.7 words / ~6
tokens. Two compounding causes: (a) `DEFAULT_MAX_TOKENS` 64 (v3) -> 256 (v4); (b) the G1 specs use
`prompt_modes: [none]` = empty instruction, while v3 always appended "Keep the answer concise"
(only v4's `generic`/`targeted` modes carry that line). Accuracy is mostly unaffected (the gold
answer is embedded in the verbose response, so the judge still marks it correct), but decode
cost/latency is inflated ~20x and exact-match/abstention metrics would differ. To restore terse
answers set `prompt_modes: [targeted]` (or `[generic]`) and/or lower `max_tokens` — note the prompt
mode rides the condition name, so it is a new set of cache cells, not a re-judge.

## Cell-count reference

questions x representations x prompt_modes x k-values x retrieval-methods (oracle collapses k +
methods to 1). mmlongbench = 1091 = 847 answerable + 244 unanswerable; 657 answerable are digital.
- G1 (oracle): representation 657x4=2,628; quantization 657x4x2=5,256; reasoner 657x4x3=7,884;
  resolution 657x2x3=3,942.
- G3 (bm25 text arm, 3 prompts): 244x4x3=2,928.
- G2 (TLV,V x k{1,3,5,7,10} x text/vision/joint): 847x2x5x3=25,410 inference cells.

## After generation: judge + build

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya pull                                   # bring predictions local
envs/mpvrdu/bin/python -m ops.judge --spec ops/specs/<spec>.yaml --judge-spec gemini-flash
envs/mpvrdu/bin/python -m ops.build --task all
```

Judge runs locally (loads no models, reads the `.env` key). Judge each task as it finishes: the
Gemini Tier-1 key is ~10k/day and the total prediction count is ~30k+ if G2 runs full, so lean on
the two-key fallback (`GEMINI_API_KEY_SECONDARY`) and/or stub-judge first for immediate tables.
**Gap:** `ops.build` reads the un-tagged cache (`ExperimentConfig()` with no run_tag), so building a
run-tagged run needs its tag wired through `ops.build` first (not yet done).

## Watch-outs

- **Don't push while jobs run** unless the diff is additive/orthogonal. `push`/`submit` do
  `rsync --delete` on the remote root; `results/ .cache/ .data/ envs/ logs/` are excluded, so
  caches/weights/data are safe, but source files are replaced.
- **`retrieval.jsonl` vs `predictions.jsonl` on rerun:** predictions is key-cached (edits survive,
  reduced-spec reruns skip done cells and leave orphans); `retrieval.jsonl` is rewritten `"w"` each
  stage-1 run (edits are discarded) unless you use `--skip-retrieval`.
- **check_run:** `--check-all` sweeps every spec; `--no-push` when running it on the login node.
  Multi-resolution expected counts are now correct.
