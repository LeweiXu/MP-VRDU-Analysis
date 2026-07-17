# Status @ 12:15 Fri 2026-07-17 (Kaya shuts down 17:00)

## Headline
Parser non-OOM failures fully fixed. g3 done. g2 timed out at 36%, resubmitted to
grab more before 17:00. pytest green (241 passed). Nothing committed.

## Parser (unlimited) — DONE, error=0
- Was: 1447 ok / 144 error (ParserCacheMiss) / 103 oom.
- Root cause: Unlimited-OCR gundam crop-mode CUDA OOM on 12 landscape/dense docs
  (101 pages); worker swallowed it, surfaced as ParserCacheMiss.
- Fix (tools/parser_worker.py): gundam -> base-1024 -> base-640 infer fallback +
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments. Smoke job 1063873 recovered 4/4.
- Recovery: dropped the 144 error rows from remote predictions (backed up), normal
  resume job 1063881 (--skip-oom, interrupt-safe) re-warmed the 101 pages + reasoned
  the 144 cells.
- Verdict-cache poisoning (HANDOFF §42) hit: the earlier judge chain had written 144
  stale `error` verdicts before the recovery pull, so re-judge scored 0. Dropped the
  stale verdicts (backed up), re-judged -> 144 scored.
- **Now: 1585 ok / 109 oom / 0 error. parser.csv n = 1585.** Of the 144 recovered:
  138 judged-ok, 6 turned out to be genuine reasoner OOM.

## g3 (hallucination) — DONE
- 2698 ok / 230 oom / 0 error. All non-ok are reasoner OOM (not fixable on V100).
  Judged + built by the overnight/g3+parser chains.

## g2 (retrieval) — PARTIAL, resume running
- Timed out at its 24h wall ~11:41. Pulled: **5464/15246 = 35.8%** (4218 ok, 1246
  oom, 0 error). Retrieval inference is slow; 22.8% OOM.
- Resubmitted normal resume job **1065168** (--skip-retrieval --skip-oom, 2x V100,
  running on k040) to grab more non-OOM cells before 17:00.
- **A FINAL PULL IS NEEDED BEFORE 17:00** to capture the resume's extra cells.
  Detached watcher logs/g2_final_watch.sh does this automatically (pulls + judges +
  rebuilds g2 when 1065168 ends/gets killed). g2 answer-judging of the current 5464
  is running now (logs/g2_judge_now.log).

## Genuine OOM cells (NOT fixable on V100 — need A100/H100)
- parser 109, g3 230, g2 1246. These are reasoner OOMs on big contexts.

## Tests / tree
- pytest: **241 passed**.
- Uncommitted (left for review, NOT committed): tools/parser_worker.py (the fix),
  regenerated docs/generated/{build_status,mined_tables}.md. New scratchpad/ helpers.

## For the user (back ~14:00)
- Parser + g3 are complete and clean; enjoy.
- Decide whether the g2 resume is worth keeping past 14:00; either way pull once more
  before 17:00 (the watcher already does this) so no g2 cells are stranded.
- Optional later on a bigger GPU: the ~1585 total OOM cells across arms.
