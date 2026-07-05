# Session handoff — 2026-07-05

Written at the end of a long working session. Read top to bottom. The durable
record is in `docs/AGENT_GUIDE.md`; this is the "what changed today and what's
next" summary.

## Live job to watch

- **`t1-full` (job 1006495)**: bf16 Qwen3-VL-8B on `--gres gpu:v100:2`, the
  ~309-question (100/domain) T1 headline. Running healthy on k030, no OOM,
  resumable cache. When it finishes:
  1. `kaya.kaya pull`
  2. `cli.experiments --phase judge --experiment T1_headline --full` (gemini judge; needs `GEMINI_API_KEY` exported from `.env`)
  3. Also judge T2 and T5 (aggregation-only from T1): same command with `T2_analytical` / `T5_composition`.
  4. F1 gate: `cli.gates frontier --table results/tables/full/table1_headline.csv --json-output results/gates/F1_frontier_divergence.json`. Go if >=2 of the 3 bins have different frontiers. Record the verdict in `docs/AGENT_GUIDE.md`.
- A background monitor is polling it and will pull + report on completion or OOM.

## What changed today

**Three real OOM/critical-path bugs found (via the 4-bit smoke) and fixed:**
1. **Attention math kernel OOM** (~105 GiB). Probe `1004834` proved the V100
   (sm_70) has *no* memory-efficient or flash attention for Qwen3-VL, only the
   O(seq^2) math kernel. A dense-page `TL` layout-JSON cell (~30k tokens) blows
   up. Fix: size-aware **input-token cap** (`config.max_input_tokens`, 8B->4096)
   that truncates long context, keeping image placeholders.
2. **2xV100 shard-headroom OOM** (missed by 0.14 GiB). `device_map="auto"` left
   no activation room. Fix: `LocalVLMBackend._max_memory_map` reserves ~5 GiB/GPU.
   Both fixes verified by a small 2xV100 memtest before the full resubmit.
3. These would have broken the real bf16 runs too, not just 4-bit.

**Quantization** (`--quantization {4bit,8bit}` on `cli.experiments` and
`kaya.generate`): a model-spec suffix (`qwen3vl-8b-local-4bit`) with its own
cache rows. 4-bit fits one 16GB V100 (~7 GiB) and is verified end-to-end
(generate -> Gemini judge -> tables). `bitsandbytes==0.49.2` added to both
requirements files and installed on Kaya. Feasibility writeup:
`SINGLE_GPU_8B_FEASIBILITY.md` (repo root). Main tables stay bf16 for fidelity.

**Per-bin subset**: a full mmlongbench run defaults to ~100 questions/domain
(document-level, ~309 Q); `--per-bin-questions 0` for the whole corpus,
`--sample-seed N` for a different draw.

**Table 4 reworked**: it was silently *not* binning LongDocURL into the 3
domains. Now it is a **held-out MMLongBench subset** — text_heavy/in_between draw
~100 Q from documents disjoint from T1 (verified 0 doc overlap), visual_heavy
reuses T1's questions (too thin to hold out; SlideVQA is the planned visual
replication, out of scope). The LongDocURL loader still exists but is unused.

**`clear-cache`**: `kaya.kaya clear-cache [--mode --experiment --renders --logs --all --local --dry-run --yes]`
removes cached generation results on Kaya (and locally with `--local`),
path-guarded to `results/`/`logs/`.

**Docs consolidated**: `docs/` is now just `implementation_plan.md`,
`dataset_stats.md`, `dataset_label_distributions.csv`, plus two merged guides:
`USER_GUIDE.md` (what/why + Runbook) and `AGENT_GUIDE.md` (decisions + findings +
models/data/tools/eval reference). The former DECISIONS/MODELS/DATA/TOOLS/
EVALUATION/RUNBOOK/PROJECT_SPEC/context files were merged or dropped; references
updated in `CLAUDE.md`, `implementation_plan.md`, code, and the kaya guides.

All 82 tests pass (1 skipped: the bnb-config test, bitsandbytes is Kaya-only).

## Next steps / open items

- **Rerun T4** after the current job: its cache is stale from the rework, so
  `kaya.kaya submit --gres gpu:v100:2 ... kaya/generate.py -- --experiment T4_dataset --full`,
  then judge + build. (Cheap: one table.)
- **Held 4-bit full-7 job** (user decided to wait for bf16 t1-full first): once
  it's done and the cache is clear, `kaya.kaya submit --gres gpu:v100:1 ... -- --experiment section2 --full --quantization 4bit --continue-on-error`.
  Caveat: `section2` includes T3 (InternVL, a separate bf16 backend that
  `--quantization` doesn't touch, won't fit 1 GPU) and T4 is now MMLongBench, so
  the 1-GPU-4bit runnable set is T1,T2,T4,T5,T6,T7 (T3 needs InternVL quant support added).
- **F2/F3 gates** (judge-human kappa, classifier pilot) still pending, per the plan.

## Gotchas

- Judge phase flags (`--full`, `--per-bin-questions`, `--quantization`) MUST
  match the generate phase, or it resolves a different corpus/spec and the
  prediction-cache guard fires.
- Two jobs writing the same experiment dir on `/group` (Lustre) can corrupt
  `predictions.jsonl` via concurrent appends. Don't run overlapping generates for
  the same experiment.
- V100 per-image-cell latency is ~40-50s (math attention), so a 309-Q T1 is
  ~8-12h. 4-bit on 1 GPU is the fast/schedulable alternative (backfills in minutes).
- Nothing is committed (per standing instruction). Everything is uncommitted on `main`.
