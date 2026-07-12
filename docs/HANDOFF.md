# Handoff (2026-07-12): generate/judge split shipped, Kaya generation in flight

## Code state

- **Committed (`6064117` "results generation unification" + `8d0bb6a`): the generate/judge
  split.** Generate now writes only `predictions.jsonl` (one `PredictionRow` per cell,
  including failures, no judging). `ops.judge --spec <yaml> --judge-spec <stub|gemini-flash|gpt-4o-mini>`
  reads it and writes `results.jsonl` (= every prediction row plus the verdict; a strict
  superset). `judge_spec` was removed from all specs; `ResultRow` shape and the cache keys
  are unchanged. Top entry in `docs/DECISIONS.md` has the details.
- **Uncommitted (working tree): the Kaya pre-submission preflight.**
  `ops/scripts/preflight.py` plus `kaya submit` wiring that auto-runs it on the login node
  for any spec-driven submit (`--no-preflight` skips), plus `tests/test_preflight.py` and
  KAYA guide updates. It checks: spec parses, pipeline imports, dataset loads + corpus
  resolves (reports question/cell counts), every reasoner/retriever/classifier weight is
  staged offline, parser env exists, and a soft `--gres` sizing hint. Full suite is 215 green.

## Kaya jobs in flight (as of Sun 2026-07-12 ~10:00 node time)

All are 8B qwen3vl, 2xV100, med resolution. The OOM failures are the expected V100 pressure
on the big-context TLV/V cells (all CUDA OOM, zero real errors); they are marked `oom`, the
job keeps going, and they get completed later on the supervisor via `--failed-only`.

| Job | spec | wall | cells | progress |
|---|---|---|---|---|
| 1028727 | `kaya_g1_representation_full` | 24h | 2,628 | ~60% (1481 ok / 106 oom), finishes ~Sun evening |
| 1028725 | `kaya_g3_full` | 24h | 2,928 | ~42% (1122 ok / 99 oom), will not clear the wall |
| 1028729 | `kaya_g2_full` | 72h | 25,410 inf | stage-1 done, inference not started (see below) |

**G2 detail.** The stage-1 retrieval benchmark is effectively complete: `retrieval.jsonl`
has 26,257 rows (bm25, bge-m3, colmodernvbert, colqwen2.5, colqwen3, plus 2 joints).
qwen3-embedding-4B OOM'd, so its rows and the qwen3-embedding|colqwen3 joint are absent
(known casualty of the 4B embedder on a 16 GB V100). The job is now in the inference
parser/render pre-pass (confirmed alive: parser markdown being written). Inference is 25,410
cells and will not finish in 72h. **User decision: leave it running; the supervisor finishes
it in full if the wall is not enough.**

## Deadline: Kaya down Fri 2026-07-17 17:00 for migration

Everything must be generated (ideally judged + built) by then. Runway from Sun ~10:00 is
about 127h. Per task:

- **G1 representation:** finishes on its own ~Sun evening, then `--failed-only` on the
  supervisor for the 106 OOM cells.
- **G3:** let it hit the 24h wall (do NOT cancel now, requeue risk), then resubmit the same
  spec. It resumes from the ~1,200 cached cells and runs the classifier side artifact.
  Then `--failed-only` on the supervisor for the OOMs.
- **G2:** as above, the supervisor completes it if 72h is short.

## Additional G1 sub-experiments to submit (Kaya is relatively free)

Same 657 digital-and-answerable corpus, oracle ladder. The parser markdown cache is shared
(`results/cache/parser/...`, already warm from g1-representation), so these skip the slow
parser pass; only renders (per run_tag) repeat, roughly 1-2h per job.

| spec | cells | breakdown | suggested |
|---|---|---|---|
| `kaya_g1_quantization_full` | 5,256 | 657 x 4 rungs x 2 quants (8bit, 4bit) | `--gres gpu:v100:2 --time 72:00:00` |
| `kaya_g1_reasoner_full` | 7,884 | 657 x 4 rungs x 3 models (2B, 4B, InternVL3-8B) | `--gres gpu:v100:2 --time 60:00:00` |
| `kaya_g1_resolution_full` | 3,942 | 657 x 2 rungs (TLV, V) x 3 res (low, med, high) | `--gres gpu:v100:2 --time 54:00:00` |

Walls anchored to the observed ~120 cells/h for 8B bf16 on 2xV100 (pre-pass included) and
padded: 8-bit is ~1.5-2x slower (drives the 72h quant wall); InternVL3-8B needs the 2 V100s
and is the slow leg of the reasoner job; high-res TLV/V is slow and OOM-heavy. Jobs exit when
done and resume from cache on a wall hit, so the walls are safe upper bounds.

```bash
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --mem 64G --time 72:00:00 ops/generate.py -- --spec ops/specs/kaya_g1_quantization_full.yaml
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --mem 64G --time 60:00:00 ops/generate.py -- --spec ops/specs/kaya_g1_reasoner_full.yaml
envs/mpvrdu/bin/python -m ops.kaya.kaya submit --no-wait --gres gpu:v100:2 --mem 64G --time 54:00:00 ops/generate.py -- --spec ops/specs/kaya_g1_resolution_full.yaml
```

`submit` pushes first (safe here: the diff since the last push is additive, being the
preflight tooling plus these specs plus docs, and touches none of the running jobs'
executing modules or the rsync-excluded `results/` `.cache/` `.data/`), then auto-runs the
preflight on the login node and only queues if it passes. Quantized 8B fits one V100, so the
quant job can use `--gres gpu:v100:1` to ease contention at the cost of more OOM cells to mop
up later.

## Cell-count reference

Formula: questions x representations x prompt_modes x k-values x retrieval-methods (oracle
runs collapse k and methods to 1). mmlongbench = 1091 questions = 847 answerable + 244
unanswerable; 657 of the answerable are digital-scan.

- **G1 (oracle):** representation 657x4 = 2,628; quantization 657x4x2 = 5,256;
  reasoner 657x4x3 = 7,884; resolution 657x2x3 = 3,942.
- **G3 (bm25 text arm only, 3 prompt sweep):** 244x4x3 = 2,928.
- **G2 (TLV,V x k{1,3,5,7,10} x text/vision/joint):** 847x2x5x3 = 25,410 inference cells.

## After generation

Per task, on the supervisor if OOMs remain: `ops.generate --spec <spec> --failed-only`
(the bigger GPU fits the cells the V100 OOM'd on). Then judge and build:

```bash
envs/mpvrdu/bin/python -m ops.judge --spec ops/specs/<spec>.yaml --judge-spec gemini-flash
envs/mpvrdu/bin/python -m ops.build --task all
```

Judge each task as it finishes rather than batching: the Gemini Tier-1 key is ~10k
requests/day and the total prediction count is ~30k+ if G2 runs full, so use the two-key
fallback (secondary `GEMINI_API_KEY_SECONDARY` in `.env`) and/or stub-judge first for
immediate tables.

## Watch-outs

- **Don't push while jobs run** unless the diff is additive and orthogonal (as it is now).
  `push`/`submit` do `rsync --delete` on the remote root; `results/ .cache/ .data/ envs/
  logs/` are excluded, so caches/weights/data are safe, but source files are replaced.
- **G2's 25,410 inference cells** (k-sweep x 3 methods) is the one thing that fits no single
  wall and blows the Gemini judge quota. Trimming inference k is possible (retrieval.jsonl
  already covers k-accuracy) but `write_retrieval_eval` opens `retrieval.jsonl` in `"w"`
  mode, so back that file up first, or add an additive `inference_k_values` spec field to
  decouple inference k from the benchmark k.
- **check_run** (`ops.scripts.check_run --spec <spec>`) reads `predictions.jsonl` and reports
  ok/oom/error vs expected per run_tag; run it (login node, `--no-push`) to gauge progress.
