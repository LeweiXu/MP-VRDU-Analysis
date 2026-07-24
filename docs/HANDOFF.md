# Handoff (2026-07-24): pipeline extension built + smoke-verified, specs reorganised, H100 run staged

## One-line status

The eight-stage pipeline extension (`docs/PIPELINE_EXTENSION_PLAN.md`, now
deleted) is fully implemented, unit-tested, and smoke-verified on the local
RTX 5070 with real 2B generation. The `ops/specs/` tree is reorganised to the
E1-E5 taxonomy, and the H100 job is staged: five run specs, a prestage config,
a pre-flight smoke, and a runbook. Everything is committed on `main`, tree
clean. Tests: 299 passed + the 2 known gemini env failures (green). `python -m
ops.build` clean at 39 tables.

## What this session did (newest first; see `docs/DECISIONS.md` for the detail)

1. **H100 smoke + runbook** (`04fecd8`, `94580c0`, `7375941`). New
   `ops/specs/h100_smoke.yaml`: one tiny capped run per mechanism the five H100
   specs use (parser, page_set both rule types, both rankers, six prompt modes,
   both pools, TLVi, 32B under bf16 + 4-bit), on isolated `smoke-*` tags, ~35
   cells. `docs/guides/H100_RUNBOOK.md` shortened and retargeted to the five
   specs; `ops/kaya/h100_main.json` trimmed to exactly what they need (8B + 32B
   + ColQwen3 + PaddleOCR-VL + MMLongBench).
2. **Robustness +k design fix** (`b3b9b4f`). The distractor arm was wrong: it
   now blocks questions by EXACT gold count (`corpus.hop: 1/2/3`) and feeds all
   their gold + k distractors, so d=0 is literally the oracle condition already
   in `g1-representation-full`. `filter_by_hop` gained exact-count support; the
   `selection` builder blocks by corpus gold count with a per-block oracle
   baseline row.
3. **CoT budgets doubled to 2048** (`99d78eb`) so a reasoning-bearing cell is
   very unlikely to truncate before its `Answer:` line. Truncation is still not
   a hard failure: `output_truncated` is on every row and the judge falls back
   to whole-text; the scoring policy for flagged cells is deliberately left
   open (we are only worrying about generation now).
4. **Specs reorganised to E-taxonomy** (`9cb7871`, `9790b75`). 27 files -> 13,
   named `g<E>_<name>.yaml` after the failure mode, `g0_*` for interventions.
   Run_tags and task_names are unchanged (they are the cache identifiers), so
   every cached tag stays judge-reachable. Prestage config gained the new
   reasoner ids.
5. **Extension phases A-D** (`bc6825f`, `b89a7f0`, `bd5eff7`, `0a94f7c`,
   `13e3cb4`). Six-mode prompt set + per-mode decode budgets + judge-time
   delimiter extraction + output-truncation canary; the page_set condition
   grammar + `PageSetConditioner` + hop filter; per-backend prompt templates +
   the Thinking and Llama-Vision backends + no-stub-fallthrough; the four new
   builders (selection, faithfulness_pools, reasoner_unified, levers) + the
   build-time reconciliation gate. Smoke-verified end to end on the 5070.
6. **Reporting fixes** earlier in the session (`232a8ac`, `f67f2d4`,
   `553d6dc`, `9df102a`): fidelity-transition by doc_type, the qwen3-embedding
   retrieval memo fold + audit, the hop_doctype cross-tab.

## Current state

- **Specs** (`ops/specs/`): the runnable set is `g<E>_<name>.yaml` + `g0_*` +
  `template.yaml` (reference menu) + `h100_smoke.yaml`. In `g0_reasoner.yaml`
  everything except the 32B matched-memory run is commented out.
- **Environments**: no new conda envs from the extension. `setup_env.py`'s four
  envs (core + three parser) are unchanged; `core.txt` already pins
  `bitsandbytes==0.49.2`, so the 32B 4-bit path needs no extra setup. A local
  GPU env `~/venvs/mpvrdu-gpu` (torch 2.11 cu128 for the 5070's Blackwell) was
  built for the smoke; it is NOT the repo's supported env, just this machine.
  See the [[mpvrdu-python-envs]] memory.
- **Data**: unchanged from before this session. The completed runs
  (`g1-representation-full`, the g1 sweeps, `g2-retrieval-full` partial,
  `g3-hallucination-full`) are all still cached and judged; this session added
  no new experiment data beyond the throwaway `smoke-*` caches on the 5070.

## What is ready to run, and how

The H100 job, in `docs/guides/H100_RUNBOOK.md`. Order: setup_env -> prestage
(`ops/kaya/h100_main.json`) -> `h100_smoke.yaml` (pre-flight) -> the five specs
in priority order (g2_sufficiency, g2_robustness, g5_faithfulness,
g0_interleaved, g0_reasoner) -> check_run each. ~58k cells, ~9-13 days on one
H100. Generation only; judge + build run elsewhere afterward.

## Known gaps / open items

- **Scoring policy for truncated CoT cells** is unimplemented by choice. When
  judging lands, decide whether a delimiter-configured cell with
  `output_truncated` and no delimiter in its output should be forced incorrect
  or excluded. Right now it is judged on whole text.
- **Robustness gold-2/gold-3 blocks have no same-evidence clean baseline** now
  that d=0 is reused as oracle: their dilution slope reads d=1 -> d=2 only. This
  is stated in the spec header and the selection builder's note.
- **Llama-3.2-11B-Vision is a gated repo**: prestage needs the HF account to
  have accepted Meta's license, not just a token. Its run is commented out in
  `g0_reasoner.yaml` anyway.
- **`annotations/model_weights.csv` lacks the thinking/llama/32B rows** (the
  `reasoner_unified` weight column shows `-` for them). Regenerate with
  `ops/scripts/model_weight_sizes.py` (needs network + token) when convenient.
- **The reconciliation gate is live but mostly SKIP** until the G4/G5 runs land
  (8 PASS / 7 SKIP today); the anchors are the headline ladder and the G3
  abstention rates.

## Next steps

1. Run the H100 job per the runbook (smoke first).
2. When `results/` comes back: judge (`ops.judge`) then `ops.build`; the four
   pending builders (selection, faithfulness_pools, reasoner_unified, levers)
   fill in and the reconciliation checks flip from SKIP to PASS/FAIL.
3. Decide the truncated-CoT scoring policy before trusting the CoT lever rows.
