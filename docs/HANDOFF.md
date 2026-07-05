# Session handoff — 2026-07-05

End of a long session. Read top to bottom. Durable details are in
`docs/AGENT_GUIDE.md`; how to run is in `docs/USER_GUIDE.md` (local) and
`kaya/KAYA_USER_GUIDE.md` (cluster). Nothing is committed — everything is
uncommitted on `main`.

## The full T1 run: OOM'd at ~56% (needs a decision)

The bf16 Qwen3-VL-8B T1 headline on 2×V100 (job **1006495**) ran 2h34m and the
SLURM job exited 0, but with `--continue-on-error` the **T1_headline experiment
itself failed with a CUDA OOM at ~56% (698 of 1236 bf16 cells cached)**. The
predictions already generated are safe in
`results/cache/full/T1_headline/predictions.jsonl` (append-only, resumable).

**Why it OOM'd (diagnosed):** the `max_input_tokens` cap truncates *text* but not
*vision* tokens. A full-corpus question with 7-8 gold pages contributes ~6000
vision tokens, past the 4096 cap, and the attention score (~4.3 GiB, V100 has
only the O(seq²) math kernel) tips a bf16 GPU over. The earlier memtest passed
because its documents topped out at 4 evidence pages. Full write-up:
`docs/AGENT_GUIDE.md` → "Third OOM: vision tokens are not capped (open)".

### Your options (pick one)

1. **Judge what's cached now (partial Table 1).** Fastest way to see numbers:
   ```bash
   set -a; . ./.env; set +a                 # export GEMINI_API_KEY
   python -m cli.experiments --phase judge --experiment T1_headline --full --continue-on-error
   ```
   You'll get a Table 1 built from the 698 cached cells (not all bins fully
   covered). Good enough to eyeball the frontier; not the final gate number.

2. **Switch the full run to 4-bit on ONE V100 (recommended).** 4-bit weights are
   ~7GB vs bf16's ~13GB, so the many-page cell that OOM'd bf16 *fits* in 4-bit
   (7 + 5 < 16), and 1-GPU jobs backfill in minutes instead of waiting for a
   scarce 2-GPU node. The 4-bit path is already verified end-to-end. This is a
   *different reasoner* (quantized ≠ pre-registered bf16), so treat it as the
   working/preview run or an appendix, per your call:
   ```bash
   # on the cluster (see kaya/KAYA_USER_GUIDE.md for push/submit/pull):
   kaya.kaya clear-cache --mode full --experiment T1_headline --local --yes   # optional clean start
   kaya.kaya submit --gres gpu:v100:1 --time 06:00:00 --job-name t1-4bit \
     cli/experiments.py -- --phase generate --experiment T1_headline --full --quantization 4bit --continue-on-error
   ```

3. **Keep bf16 on 2×V100 but cap vision tokens first.** Implement the missing fix
   (cap total vision tokens per cell: limit page count or shrink `max_pixels`
   when a cell has many pages), smoke-test it, then resubmit — it resumes from the
   698 cached cells. This is the "correct pre-registered" path but needs the code
   fix + another smoke cycle.

## How to monitor a (re)submitted job

```bash
python scripts/kaya_status.py                                   # queue + node view
ssh kaya 'squeue -u lxu -o "%.10i %.10j %.8T %.10M"'            # your jobs
ssh kaya 'sacct -j <jobid> -o JobID,State,Elapsed,ExitCode'    # REAL end state (not squeue)
```
Gotcha learned this session: a transient `squeue` blip returns empty and looks
like "job gone". Always confirm a finished job's state with `sacct` (COMPLETED /
FAILED / TIMEOUT) before trusting it. After it ends: `kaya.kaya pull`, then check
`results/cache/full/<name>/generate_status.json`.

## How to run the judge phase + gates

Full detail (every arg, the cache format, judge output format) is in
`docs/USER_GUIDE.md` → Runbook. Short version:
```bash
set -a; . ./.env; set +a
python -m cli.experiments --phase judge --experiment T1_headline --full   # + T2_analytical, T5_composition (aggregation-only from T1)
python -m cli.gates frontier --table results/tables/full/table1_headline.csv \
    --json-output results/gates/F1_frontier_divergence.json               # F1: Go if >=2 bins differ
```
The judge phase MUST use the same corpus/model flags as the generate phase
(`--full`, `--per-bin-questions`, `--quantization`) or it looks for predictions
that were never generated. If you ran generate in 4-bit, add `--quantization 4bit`
to the judge command too.

## What this session changed (all uncommitted)

- **Repo restructure:** `kaya/` now holds only the SLURM dispatcher (`kaya.py`,
  `config.json`, the two KAYA guides). Moved to `scripts/`: `download_hf`,
  `prestage`, `setup_env`, and the probes (`gpu_test`, `single_gpu_probe`,
  `attn_probe`); `scripts/` is now an importable package. Moved to `cli/`:
  `generate.py` (was `kaya/generate.py`); `kaya/run_probe.py` deleted (use
  `cli/run_probe.py`). All imports/refs updated; 82 tests pass.
- **Docs consolidated + split by audience.** `docs/` = `implementation_plan.md`,
  `dataset_stats.md`, `dataset_label_distributions.csv`, plus `USER_GUIDE.md`
  (what/why + a comprehensive local Runbook: args, cache format, judge format) and
  `AGENT_GUIDE.md` (decisions + findings + models/data/tools/eval reference).
  Kaya operational how-to lives only in `kaya/KAYA_*_GUIDE.md`.
- **Table 4 reworked** to a held-out MMLongBench subset (disjoint docs for
  text_heavy/in_between, reused visual_heavy), not LongDocURL. Its cached rows are
  stale; rerun T4 when convenient.
- **Quantization** (`--quantization {4bit,8bit}`) wired end-to-end;
  `bitsandbytes==0.49.2` in both requirements files. Feasibility:
  `docs/SINGLE_GPU_8B_FEASIBILITY.md`.
- Earlier fixes still in place: input-token cap, `max_memory` shard headroom,
  per-bin subset default, `clear-cache` command.

## Deferred / next

- Decide the T1 path above (4-bit recommended). Then judge + F1 gate.
- Rerun **T4** (held-out subset) once a reasoner run is chosen.
- **T3** (InternVL) and **T8-32B** don't fit one V100; T3 needs InternVL quant
  support added for a 1-GPU 4-bit `section2` run; 32B is supervisor's-A100 scope.
- Gates **F2** (judge-human κ) and **F3** (classifier pilot) still pending.
- Nothing committed; `git status` shows the full set of changes.
