# Handoff (2026-07-16): judge/build/mine done; unlimited-parser + g2 inference running on Kaya

## Deadline

Kaya goes down **Fri 2026-07-17 17:00** for migration. The running jobs and any
resume must land (and be pulled) before then.

## One-line status

Core judge+build+mine landed and is committed (`f4c77d6 "build mine"`). The stable
G1/G3 arms are judged and their tables built. Three Kaya jobs are finishing: g3
recovery is **done and complete**, the Unlimited-OCR parser rerun and the G2
inference are still running. A few late fixes are uncommitted (see below).

## Kaya jobs (as of ~16:00 AWST 2026-07-16)

| job | id | state | notes |
| --- | --- | --- | --- |
| g3-hallucination-full (recovery) | 1061557 | **COMPLETED** | distinct=2928 (full), ok=2698 oom=230 err=0. No missing cells. The 243 old CUDA-faults resolved (34→ok, ~209 are genuine big-context V100 OOMs). Needs pulling + judging. |
| g1-parser-full-unlimited | 1061530 | RUNNING | still in the **parser-warm pre-pass** (~1041/~1224 pages), reasoner not started (14 cells). 24h wall, ~17h left. |
| g2-retrieval-full (inference) | 1061344 | RUNNING | banking reduced-k inference (`--skip-retrieval --skip-oom`), ~17h left. Incomplete (was 4137/15246). |

**Will unlimited finish in its 24h wall? Borderline — likely not quite.** The
Unlimited-OCR warm pass is slow (~8h for ~1224 pages), leaving ~16h for the 1680
reasoner cells (~14–15h needed at ~33s/cell, more with TLV + OOM retries). It will
either just finish or time out with a modest reasoner remainder. **If it times out,
recover with a NORMAL resume (NOT --failed-only)** — see the interrupt-safety note
below: `ops.kaya.kaya submit --gres gpu:v100:2 --time 12:00:00 ops/generate.py --
--spec ops/specs/kaya_g1_parser_unlimited_full.yaml`. **Do NOT cancel it** (that
loses cells). With the Fri 17:00 shutdown the resume window is tight.

## Judging status

`ops.judge` (gemini-flash, Tier-1 ~10k/day). Fully judged and stable: representation,
quantization, resolution, reasoner (incl. all InternVL), reasoner-scanned, mineru.
A background chain (`logs/judge_resume.log`) is finishing the last scanned companions
(quant-scanned ~1341/1488, resolution-scanned pending) then rebuilds + mine + pytest.

**Not yet judged (deferred, needs the Kaya jobs done + a pull):**
- g1-parser-full-unlimited — its local cache is still the stale 1680 error rows;
  judging it now would poison the result cache (error verdict keyed same as the
  future ok cell). Judge only AFTER its job completes and a fresh pull.
- g3 recovery's new ok cells (pull g3 first, then re-judge).
- g2 inference — leave until inference is complete (else partial matched_cross/kdepth).

A one-shot catch-all judge cron is scheduled: **d2e49a7b, 2026-07-17 15:20 AWST** —
pulls, judges everything still unjudged (unlimited, g3, g2-if-complete), rebuilds,
pytest. Delete it (`CronDelete d2e49a7b`) if you'd rather do it by hand.

## Tables (docs/generated/build_status.md)

Built: headline, composition, parser (paddleocrvl + mineru once its build runs),
scale (model-size + quantization), routing, **InternVL family**, hallucination,
retrieval accuracy (overall + doc_type), + 6 mined_* tables. Partial/blocked:
resolution (OOM tail), matched_cross/kdepth (g2 inference incomplete), retrieval_dpi
(DPI sweep never run), parser comparison as a *merged* paddle-vs-mineru-vs-unlimited
table needs a small cross-run_tag builder (each parser currently builds its own).

## Cell-count audit (no silent missing data)

`scratchpad/audit_cells.py` verified theoretical (from yaml) == produced for every
G1/G3/parser/scanned run. Only g2 inference is short (known, still running). OOM
rates: ~5% G1/G3, 13.5% mineru, 21.5% g2; the OOM cells need a bigger GPU than the
16 GB V100 (supervisor A100/H100 --failed-only sweep).

## Uncommitted changes (this session, for your review/commit)

- **`tools/parser_worker.py`** — Unlimited-OCR fix: it's a DeepSeek-OCR-style model
  (custom `UnlimitedOCRConfig` → `AutoModel`, custom `model.infer(...)`, not the
  vision auto-classes). New `_load_unlimited` / `_unlimited_markdown` path
  (prompt `"<image>document parsing."`, gundam args, reads back `result.md`).
  Smoke-tested green on Kaya (job 1061368). setup_env/prestage were already correct
  (parse-unlimited env + baidu/Unlimited-OCR staging); no change needed there.
- **`pipeline/orchestrator.py`** — Result/Prediction caches now skip a truncated
  last line (from a host reboot killing a writer) instead of crashing the whole run.
- **`ops/scripts/build_status.py`** + regenerated `docs/generated/{build_status,mined_tables}.md`
  — dynamic InternVL row, refreshed resolution/parser/g2 reasons.
- New specs: `ops/specs/kaya_g1_parser_unlimited_full.yaml` (unlimited-only, explicit
  run_tag, --failed-only), `kaya_g1_parser_mineru_full.yaml` (mineru-only judge
  target), `kaya_g1_parser_unlimited_smoke.yaml` (1-GPU smoke).
- (Not mine: `ops/kaya/kaya.py`, `runner/status.py`, `KAYA_AGENT_GUIDE.md`,
  `tests/test_kaya_status.py` were already-uncommitted kaya WIP at session start.)

## Known issues / recommendations

1. **Interrupt-unsafe `--failed-only`** (`experiments/engine/driver.py::_prepare_failed_only`):
   it rewrites predictions.jsonl dropping all failed rows up front, then recomputes.
   A cancel/kill/timeout mid-reprocess **loses** the dropped-not-yet-recomputed cells
   (this bit us on g3 — 261 cells lost after two cancellations, since recovered via a
   normal resume). Rules: **never cancel a running --failed-only job; recover with a
   normal resume** (runs missing cells non-destructively). Recommended fix: make
   --failed-only force-recompute the failed keys via an in-memory skip set + merge at
   the end, leaving on-disk rows intact until completion.
2. **Gemini daily cap ~10k/day** — full judging spans ~2 days on these keys; a paid
   key removes the cap. Judging is idempotent (re-judge = 0-cost on scored cells).
3. **OOM cells** (~1.7k across G1/G3 + g2's) need a bigger GPU; Kaya is all V100.

## Next steps

1. Let unlimited + g2 run; do NOT cancel. Pull when they finish (before Fri 17:00).
2. After pull: judge g3 (recovery cells), unlimited, and g2 (if inference complete);
   rebuild tables + mine + build_status; run pytest. (The d2e49a7b cron does this.)
3. Review + commit the uncommitted fixes above.
4. Optional: fix interrupt-unsafe --failed-only; add a merged parser-comparison builder.
