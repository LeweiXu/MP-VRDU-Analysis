# Handoff (2026-07-14): generation runs (nearly) done, verification found 3 fixables

## Deadline

Kaya goes down **Fri 2026-07-17 17:00** for migration. Everything below lands before then.

## Current status (2026-07-14) — read this first

Generation is essentially complete. All G1 digital + scanned runs, G3, and the G2 retrieval
benchmark finished; **jobC** is on its last leg (~61 high-res cells, finishes well before wall)
and **g2 inference** is banking partial progress (won't finish, by design → H100). A
`check_run --all` verification sweep found **three real component failures** (not just OOM),
see "Post-run verification findings" below — the biggest is that **InternVL3-8B produced zero
valid cells** (einops missing on Kaya). Fix those + do the OOM `--failed-only` sweep, then
judge + build.

## Code state (committed: `d50c6bd` "config fix + dpi experiment", on `934e53`)

Full suite **221 green**. The doc_type/overall-table fixes landed in `934e53`; the config
centralisation, DPI study, and check_run change landed in `d50c6bd`. (A local `.git`
corruption on 2026-07-13 was repaired from origin; nothing lost.) Recent additions:

- **config Tier-1/2 centralised** into `config.py`: `REPRESENTATION_LADDER` + `DEFAULT_BINS`
  single-sourced (frontier/representation/annotations import them); bootstrap CI, abstention
  forms, scanned-char threshold, and the judge prompt+model ids moved out of the modules.
- **DPI visual-retrieval study** wired in (see its section below).
- **`check_run.py`: `--check-all` → `--all`**, and the sweep skips `*template*`/`*smoke*` specs.
- **Retrieval side-artifact carries `doc_type` + `dpi`**; `ops.build` backfills both on older
  `retrieval.jsonl` rows. New tables `retrieval_accuracy_overall` (no doc_type split) and
  `retrieval_dpi`.

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

## Kaya jobs (re-check live with `ops.kaya.kaya status`, or `check_run --all`)

As of 2026-07-14, all cells written per run (ok+oom+err = expected) unless noted:

| run_tag | ok | oom | err | state |
| --- | --- | --- | --- | --- |
| g1-quantization-full | 4961 | 295 | 0 | done (OOM sweep pending) |
| g1-reasoner-full | 5080 | 176 | **2628** | done, InternVL arm failed (einops) |
| g1-representation-full | 3143 | 245 | 0 | done |
| g1-resolution-full | 3133 | 321 | **488** | done, CUDA launch faults |
| g3-hallucination-full | 2663 | 247 | **18** | done, 2 bad docs |
| g1-reasoner-scanned | 1496 | 24 | **760** | done, InternVL arm failed (einops) |
| g1-quantization-scanned | 1488 | 32 | 0 | done |
| g1-resolution-scanned | ~1079/1140 | — | 0 | **jobC, ~61 high-res cells left** |
| g2-retrieval-full (inference) | 1713 | 489 | 0 | 14% — won't finish (banking → H100) |

**jobC (1033386)** at 39 h of its 60 h wall: representation + quant-scanned complete,
resolution-scanned in its high-res tail (~61 cells). **Finishes with ~15 h+ margin.**
**g2 inference (1033908)** is expected not to finish (design); the retrieval benchmark
side-artifact (`retrieval.jsonl`) is complete.

```bash
envs/mpvrdu/bin/python -m ops.scripts.check_run --all        # local, over pulled results
```

## Post-run verification findings (2026-07-14) — fix before judge+build

Three real failures beyond OOM (all `--failed-only`-rerunnable once the cause is fixed):

1. **InternVL3-8B produced zero valid cells — `einops` missing on Kaya. FIXED (2026-07-14).**
   All 3388 internvl cells (2628 `g1-reasoner-full` + 760 `g1-reasoner-scanned`) errored with
   `ImportError: ... requires ... einops`. Fix applied: `einops==0.8.2` added to
   `docs/requirements/core.txt` and installed into `envs/core` on Kaya; verified with
   transformers `check_imports` that every InternVL modeling file now imports cleanly (no
   submit needed). Still to do: the `--failed-only` rerun (see `kaya_failed_rerun.yaml`).
2. **`g1-resolution-full`: 488 `CUDA error: unspecified launch failure`** (qwen3vl-8b, not
   OOM) — a GPU/node fault mid-run, cascaded to the rest of that process. Rerun the 488 via
   `--failed-only` (ideally on a fresh node / the supervisor).
3. **`g3-hallucination-full`: 18 errors = 2 documents. FIXED (2026-07-14).** Both were
   `2306.05425v1.pdf` (a VLM paper) whose text literally contains `<image>`, colliding with
   the `IMAGE_PLACEHOLDER` sentinel so the placeholder-vs-image count rejected the prompt.
   `ModelInput.to_local_prompt` now neutralises a literal sentinel in document text to
   `[image]` (regression test `tests/test_image_placeholder_collision.py`; cache keys
   unaffected — prompt text is not in the key). The `--failed-only` rerun picks it up.

Plus the expected OOM tail on every run (`--failed-only` sweep on a 16 GB+ GPU) and the g2
inference remainder (H100). None of the above blocks judging the cells that did complete.

**Rerun specs (2026-07-14).** `ops/specs/kaya_failed_rerun.yaml` reruns the four failed
run_tags (verbatim configs so `--failed-only` matches the cached failures); submit it with
`--failed-only`. `ops/specs/kaya_failed_rerun_smoke.yaml` is the same four runs capped to one
question each on isolated `*-smoke` run_tags — run it fresh (no `--failed-only`) first to
confirm the InternVL load + generate works end to end before the real rerun.

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

## DPI visual-retrieval study (cheap; runnable on the local RTX 5070)

A retrieval-only ablation: re-embed + re-rank pages at several render DPIs, compare page
P/R/F1. Cheap because it's retrieval-only (no reasoner, no judge), vision-only (text spans
are dpi-independent, so only `colqwen*` move), and lower DPI is *faster* than the 200
baseline. Wired in: `RetrievalEvalRow.dpi`, the `retrieval_dpi` table (grouped by
retriever/k/dpi), `complete_retrieval.py --parser-dpi N`, and spec `ops/specs/kaya_g2_dpi.yaml`
(vision-only; methods come from `--vision-methods`). Run one pass per DPI, then merge into one
`retrieval.jsonl` under the `g2-dpi` run_tag and `ops.build --task G2_retrieval --run-tag g2-dpi`:

```bash
for d in 72 110 150; do
  python -m ops.scripts.complete_retrieval --spec ops/specs/kaya_g2_dpi.yaml \
    --vision-methods colqwen2.5,colqwen3 --parser-dpi $d --filename retrieval_dpi$d.jsonl
done
```

Caveat: the colqwen processors resize internally, so the informative range is low→moderate DPI.

**On the local RTX 5070 (12 GB): yes for retrieval, not the reasoner.** The stock env's
`torch 2.7.0+cu126` has no sm_120 (Blackwell) kernels — every CUDA op dies with "no kernel
image available", so first build a **separate** retrieval-only venv on **torch cu128+**
(don't upgrade `envs/mpvrdu`: `vllm 0.9.2` / `xformers 0.0.30` are ABI-pinned to 2.7/cu126, and
neither is needed for retrieval; paddle isn't either — the parser only feeds the reasoner).
12 GB < the V100's 16 GB, but Blackwell has flash/efficient SDPA (the exact thing the V100
lacked that caused the O(seq²) OOMs), and the benchmark loads one ≤4B model at a time
(peak ~8–9 GB), so retrieval fits and likely beats the V100. The 8B reasoner (16 GB weights)
does **not** fit — 5070 is a retrieval box, not an inference box.

## Deferred: finish centralising experiment knobs into config.py

Done (Tier 1/2): the representation ladder + modality bins are single-sourced in `config.py`
(`REPRESENTATION_LADDER` / `DEFAULT_BINS`), and the judge prompt+models, bootstrap CI,
`ABSTENTION_FORMS`, and `SCANNED_MIN_CHARS_PER_PAGE` moved out of the modules. Still deferred
(Tier 3):
- **Direct analog:** vision embedder knobs `PAGE_EMBED_BATCH_SIZE` / `PAGE_EMBED_CACHE_DOCS`
  (`retrievers/vision.py`).
- **Remaining duplications:** `("4bit","8bit")` and the parser/scan label tuples re-declared
  in modules + config.
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
