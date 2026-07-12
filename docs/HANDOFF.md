# Handoff (2026-07-12): tooling + tables ready, scanned/G2 jobs queued for the migration

## Deadline

Kaya goes down **Fri 2026-07-17 17:00** for migration. Everything below is scoped to
land before then.

## Code state (working tree, uncommitted — commit when convenient)

On top of `46c92df` ("actually all working"), this session added:

- **`--skip-oom`** (`ops/generate.py` -> `driver.generate`, passed through by
  `g2_rerun.py`). Drops cells already recorded `oom` from a resume, prewarm/parser-warm
  included, so a V100 resume does not re-parse cells that only OOM again. Mirror of
  `--failed-only`; do not combine them.
- **qwen3-embedding encodes at `batch_size=1` + `max_seq_length=4096`** (`retrievers/text.py`).
  batch=1 alone still OOM'd on a dense page (attention is O(seq^2)), so the 4096 cap is
  back. Only affects the retrieval memo build, not inference.
- **`ops/build.py --run-tag`** builds from `results/cache/<run_tag>/…` (closes the old
  "build reads the un-tagged cache" gap).
- **Report tables group by the 7 native mmlongbench doc_type classes** (was the modality
  bin); the `bin` column is renamed `doc_type`.
- **`kaya.py` split into the `ops/kaya/runner/` subpackage** (config/remote/sync/sources/
  slurm/jobs/status/commands); `kaya.py` is now just parse + dispatch.
  `ops/scripts/kaya_status.py` folded in and reachable as `ops.kaya.kaya status`.
- New specs `ops/specs/kaya_jobB_scanned_resume.yaml` + `kaya_jobC_scanned_resume.yaml`;
  `kaya_g2_full.yaml` reduced to `k_values: [1,3,5]`, `joint_k_values: [1,3]`.

Full suite: **217 green.**

## Kaya jobs (re-check live with `ops.kaya.kaya status` or check_run)

When last checked: `g1-quantization-full`, `g1-reasoner-full`, `g1-resolution-full` all
running (scan:digital, ~10%); `g2-retrieval-full` running at full k (slow, to be
cancelled); `g1-representation-full` FINISHED (2409 ok / 219 oom, being judged);
`g3-hallucination-full` walled at 24 h (~72%, needs resume + its classifier artifact).

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run --no-push ops/scripts/check_run.py -- --spec ops/specs/<spec>.yaml
```

## Plan for the six tasks

**G2 — cancel, memo on Kaya, inference on the H100.** V100 has no FlashAttention, so
~15k reduced-k image cells is ~130 h on V100 (no wall covers it). qwen3-embedding is a
*benchmark-only* method (not an inference arm), so inference (bge-m3 / colqwen2.5) is
independent of the memo. Steps:
1. `scancel` the g2 job.
2. Memo regen on one V100 (also smoke-tests the batch=1 fix; fills `retrieval.jsonl`):
   ```bash
   envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:1 --time 12:00:00 \
     ops/scripts/complete_retrieval.py -- --spec ops/specs/kaya_g2_full.yaml \
     --text-methods qwen3-embedding --vision-methods colqwen3 --joints matched --filename retrieval_qwen3.jsonl
   ```
   Fold in with `cat retrieval_qwen3.jsonl >> …/retrieval.jsonl`.
3. Reduced-k G2 inference on the supervisor H100 (FlashAttention, several× faster):
   `ops.generate --spec kaya_g2_full.yaml --skip-retrieval`, then `… --failed-only` to
   upgrade the V100 `oom` rows. Note: joint inference keys off `k_values` (1,3,5), not
   `joint_k_values` — dropping joint-k5 only trimmed the benchmark.

**Scanned + resumes — Jobs B and C (2×V100 each, ~60 h wall):**
```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --time 60:00:00 \
  ops/generate.py -- --spec ops/specs/kaya_jobB_scanned_resume.yaml --skip-oom
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --time 60:00:00 \
  ops/generate.py -- --spec ops/specs/kaya_jobC_scanned_resume.yaml --skip-oom
```
- Job B: g3 resume (+ its `classifier.jsonl`), then `g1-reasoner-scanned`.
- Job C: `g1-representation` scan:any (adds the ~190 scanned docs on top of the 2628
  cached digital cells), `g1-quantization-scanned`, `g1-resolution-scanned`.
- The scanned runs use NEW run_tags so they do not race the still-running digital jobs on
  `predictions.jsonl`; scan:digital ∪ scan:scanned = scan:any, merged at build.

**Running G1 digital jobs (quant/reasoner/resolution):** let them finish inside their
walls; supervisor `--failed-only` for the OOM cells.

## After generation: judge + build

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya pull
envs/mpvrdu/bin/python -m ops.judge --spec ops/specs/<spec>.yaml --judge-spec gemini-flash   # local, needs .env key
envs/mpvrdu/bin/python -m ops.build --task all --run-tag <run_tag>                            # --run-tag now wired
```
Judge one run at a time (Gemini Tier-1 ~10k/day; two-key fallback via
`GEMINI_API_KEY_SECONDARY`). For the scan:any tables, judge + build both the digital and
the `*-scanned` run_tags and merge.

## Known behaviour: answers ~35× longer than the v3 (`old/`) runs

New answers average ~94 words / ~140 tokens (hitting the 256 cap) vs old ~2.7 words / ~6
tokens. Two causes: `DEFAULT_MAX_TOKENS` 64 -> 256, and the G1 specs use
`prompt_modes: [none]` (empty instruction) while v3 always appended "Keep the answer
concise". Accuracy is mostly unaffected (the gold answer is embedded in the verbose
response, so the judge still marks it correct), but decode cost is inflated ~20× and
exact-match/abstention would differ. Set `prompt_modes: [targeted]` and/or lower
`max_tokens` to restore terse answers (the prompt mode rides the condition name, so it is
a new set of cache cells, not a re-judge).

## Cell-count reference

mmlongbench = 1091 = 847 answerable (657 digital) + 244 unanswerable.
- G1 oracle per run: representation 847×4; quantization 657×4×2; reasoner 657×4×3;
  resolution 657×2×3 (digital) — each scanned companion adds the 190 scanned questions.
- G3 (bm25 text arm, 3 prompts): 244×4×3 = 2,928.
- G2 reduced inference: 847 × 2 reps × 3 k × 3 arms (text/vision/joint) ≈ 15.2k.

## Watch-outs

- **Don't push while jobs run** unless the diff is additive/orthogonal: `push`/`submit`
  do `rsync --delete` on the remote root; `results/ .cache/ .data/ envs/ logs/` are
  excluded (caches/weights/data safe), but source files are replaced.
- **`retrieval.jsonl` vs `predictions.jsonl`:** predictions is key-cached (edits survive,
  reduced-spec reruns skip done cells and leave orphans); `retrieval.jsonl` is rewritten
  each stage-1 run unless you pass `--skip-retrieval`.
- **check_run** `--check-all` sweeps every spec (uncommitted flag; use `--no-push` on the
  login node). Multi-resolution expected counts are correct.
