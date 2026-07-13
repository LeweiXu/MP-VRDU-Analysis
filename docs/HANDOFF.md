# Handoff (2026-07-12): tooling + tables ready, scanned/G2 jobs queued for the migration

## Deadline

Kaya goes down **Fri 2026-07-17 17:00** for migration. Everything below lands before then.

## Code state (working tree on top of `46c92df`, uncommitted — commit when convenient)

This session's changes; full suite **221 green**.

- **`--skip-oom`** (`ops/generate.py` -> `driver.generate`, passthrough in `g2_rerun.py`):
  drops cells already recorded `oom` from a resume (prewarm/parser-warm included). Mirror
  of `--failed-only`; do not combine.
- **qwen3-embedding memo build: `batch_size=1` + `max_seq_length=4096`** (attention is
  O(seq^2) with no FlashAttention on a V100, so a long page OOMs without the cap). Both
  knobs now live in `config.py` (`QWEN3_EMBEDDING_MAX_SEQ_LEN` / `_ENCODE_BATCH`).
- **Retrieval benchmark rides through failures.** A method that fails to *load* skips; once
  loaded, each question ranks independently. An OOM records a memo status row
  (`status`+`skipped_reason`, no ranking) and continues; failed questions are left out of
  the scored rows. `--fresh` (`complete_retrieval --fresh`, `g2_rerun --fresh-complete`)
  wipes each method's memo so the rung re-ranks uniformly (no mixing capped/uncapped).
- **Inference reuse is strict.** On `--skip-retrieval` the arms are `reuse_only`: a memo
  miss records that inference cell as a failure (with reason) and rides on, instead of
  silently re-ranking. Failures stay self-contained + rerunnable (no run-level guardrail).
- **Dense memo rows carry truncation telemetry** (`seq_len_cap`, `page_token_lens`,
  `truncated_pages`); additive, qwen3-embedding only (bge-m3's wrapper has no tokenizer).
- **`ops/build.py --run-tag`** builds from `results/cache/<run_tag>/…` (old un-tagged-cache
  gap closed). Report tables now group by the **7 native mmlongbench doc_type classes**
  (the `bin` column is renamed `doc_type`).
- **`kaya.py` split into `ops/kaya/runner/`** (config/remote/sync/sources/slurm/jobs/status/
  commands); `kaya.py` is parse+dispatch only. `kaya_status.py` folded in as
  `ops.kaya.kaya status`.
- New specs `kaya_jobB_scanned_resume.yaml` + `kaya_jobC_scanned_resume.yaml`;
  `kaya_g2_full.yaml` reduced to `k_values: [1,3,5]`, `joint_k_values: [1,3]`.

## Kaya jobs (re-check live with `ops.kaya.kaya status`, or check_run per spec)

Last seen: `g1-quantization-full`, `g1-reasoner-full`, `g1-resolution-full` running
(scan:digital, ~10%); `g2-retrieval-full` running at full k (slow — cancel it);
`g1-representation-full` FINISHED (2409 ok / 219 oom, judging); `g3-hallucination-full`
walled at 24 h (~72%, needs resume + its classifier).

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya run --no-push ops/scripts/check_run.py -- --spec ops/specs/<spec>.yaml
```

## Plan for the six tasks

**G2 — cancel, regen memo on Kaya (`--fresh`), run inference on the H100.** V100 has no
FlashAttention, so ~15k reduced-k image cells is ~130 h (no wall covers it). qwen3-embedding
is benchmark-only, so inference (bge-m3 / colqwen2.5) is independent of its memo. Steps:
1. `scancel` the g2 job.
2. Regen the qwen3-embedding memo, 1 V100, from scratch (measured ~1.5 s/question ->
   ~25-30 min for all 847; 1 h wall is plenty):
   ```bash
   envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --no-preflight \
     --gres gpu:v100:1 --time 01:00:00 ops/scripts/complete_retrieval.py -- \
     --spec ops/specs/kaya_g2_full.yaml --text-methods qwen3-embedding \
     --vision-methods colqwen3 --joints matched --filename retrieval_qwen3.jsonl --fresh
   ```
   Fold in with `cat retrieval_qwen3.jsonl >> …/retrieval.jsonl`.
3. Reduced-k G2 inference. On the H100 (fast): `ops.generate --spec kaya_g2_full.yaml
   --skip-retrieval` then `… --failed-only`. To bank progress on Kaya first (won't finish):
   ```bash
   envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 \
     --time 72:00:00 ops/generate.py -- --spec ops/specs/kaya_g2_full.yaml --skip-retrieval --skip-oom
   ```
   Note: joint inference keys off `k_values` (1,3,5), not `joint_k_values` — dropping joint-k5
   only trimmed the benchmark.

**Scanned + resumes — Jobs B and C (2×V100 each, ~60 h wall):**
```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --time 60:00:00 \
  ops/generate.py -- --spec ops/specs/kaya_jobB_scanned_resume.yaml --skip-oom
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --time 60:00:00 \
  ops/generate.py -- --spec ops/specs/kaya_jobC_scanned_resume.yaml --skip-oom
```
- Job B: g3 resume (+ its `classifier.jsonl`), then `g1-reasoner-scanned`.
- Job C: `g1-representation` scan:any (adds ~190 scanned docs on top of the 2628 cached
  digital cells), `g1-quantization-scanned`, `g1-resolution-scanned`.
- Scanned runs use NEW run_tags so they don't race the running digital jobs;
  scan:digital ∪ scan:scanned = scan:any, merged at build.

**Running G1 digital jobs (quant/reasoner/resolution):** let them finish inside their walls;
supervisor `--failed-only` for the OOM cells.

## After generation: judge + build

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya pull
envs/mpvrdu/bin/python -m ops.judge --spec ops/specs/<spec>.yaml --judge-spec gemini-flash   # local, needs .env key
envs/mpvrdu/bin/python -m ops.build --task all --run-tag <run_tag>                            # --run-tag now wired
```
Judge one run at a time (Gemini Tier-1 ~10k/day; two-key fallback via
`GEMINI_API_KEY_SECONDARY`). For the scan:any tables, judge + build both the digital and the
`*-scanned` run_tags and merge.

## Deferred: finish centralising experiment knobs into config.py

Moved this session: the qwen3 seq cap + encode batch. Still hard-coded in modules and worth
pulling into `config.py` (asked the user which subset to do next):
- **Direct analog:** vision embedder knobs `PAGE_EMBED_BATCH_SIZE` / `PAGE_EMBED_CACHE_DOCS`
  (`retrievers/vision.py`).
- **Core science params, buried:** `JUDGE_SYSTEM_PROMPT` + judge model strings
  (`gpt-4o-mini` / `gemini-2.5-flash`, `pipeline/judge.py`); bootstrap CI (`n_bootstrap=1000`,
  `seed`, 95%, `scoring/accuracy.py`); `ABSTENTION_FORMS` (`scoring/abstention.py`);
  `SCANNED_MIN_CHARS_PER_PAGE=20`, the digital/scanned threshold (`data/render.py`).
- **Three-way duplications (single-source-of-truth):** the T/TL/TLV/V ladder
  (`config.representations` vs `scoring/frontier.RUNG_ORDER` vs `pipeline/representation.RUNGS`);
  bin labels (`config.DEFAULT_BINS` vs `annotations.BIN_LABELS` vs `binning.BINS`);
  `("4bit","8bit")` and the parser/scan label tuples re-declared in modules + config.
- **Bigger call:** model IDs (retriever `COL*`/`BGE` + `models/*.MODEL_IDS`) — centralise into
  one registry, not scatter; and `PROMPT_HEADER`/`PROMPT_BODY` duplicated in qwen3vl + internvl.

## Known behaviour: answers ~35× longer than the v3 (`old/`) runs

~94 words / ~140 tokens (hits the 256 cap) vs old ~2.7 words. Causes: `DEFAULT_MAX_TOKENS`
64 -> 256, and G1 specs use `prompt_modes: [none]` (empty instruction) vs v3's "keep concise".
Accuracy mostly unaffected (gold is embedded in the verbose answer), but decode cost ~20× and
exact-match/abstention differ. Set `prompt_modes: [targeted]` and/or lower `max_tokens` to
restore terse answers (new cache cells, not a re-judge).

## Cell-count reference

mmlongbench = 1091 = 847 answerable (657 digital) + 244 unanswerable.
- G1 oracle per run: representation 847×4; quantization 657×4×2; reasoner 657×4×3; resolution
  657×2×3 (digital) — each scanned companion adds the 190 scanned questions.
- G3 (bm25 text arm, 3 prompts): 244×4×3 = 2,928.
- G2 reduced inference: 847 × 2 reps × 3 k × 3 arms (text/vision/joint) ≈ 15.2k.

## Watch-outs

- **Don't push while jobs run** unless the diff is additive/orthogonal: `push`/`submit` do
  `rsync --delete` on the remote root (`results/ .cache/ .data/ envs/ logs/` excluded, so
  caches/weights/data are safe, but source is replaced).
- **`retrieval.jsonl` vs `predictions.jsonl`:** predictions is key-cached (edits survive,
  reduced-spec reruns skip done cells, leave orphans); `retrieval.jsonl` is rewritten each
  stage-1 run unless `--skip-retrieval`.
- **check_run** `--all` sweeps every spec except `*template*`/`*smoke*` (uncommitted flag; `--no-push` on the login
  node).
