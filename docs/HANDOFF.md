# Handoff: make `setup_env` + `prestage` guarantee `generate` works offline

Date: 2026-07-10. This handoff is about the environment/staging contract, not the
science. It carries two diagnosis tasks and the end goal, plus the concrete findings
from the smoke run (Kaya job 1026182) that motivated it.

## End goal (the contract we want)

Two scripts, one promise: after you run them, any generation task runs on the target
machine with no network.

1. **`ops/scripts/setup_env.py --machine {V100|H100} [--env all]`** installs the FULL
   environment for that machine (reasoners, retrievers incl. colpali, all parsers,
   judges, data libs). One command per machine, and it either finishes green or tells
   you exactly which env failed and why.
2. **`ops/scripts/prestage.py --config <json>`** stages EVERY weight and dataset that
   `generate.py` will touch, so the offline compute node never reaches for HF. Scope:
   everything except the 32B reasoner for now.
3. **`ops/generate.py`** then works for any task. If it doesn't after (1) and (2),
   the bug is in `setup_env`, `prestage`, or their JSON config, and that is what to
   fix. `generate.py` should not need manual weight fetches or env tweaks.

Machine naming: `setup_env` accepts `V100` (sm_70, no flash-attn) or `H100`
(sm_90, flash-attn). `--local` controls whether the environments are built in the
current checkout rather than the configured Kaya root.

**Scope narrowed (2026-07-10):** the only two targets that matter are **Kaya V100s**
and the **supervisor H100 (run locally with `--local`)**. Non-Kaya / arbitrary
machines are out of scope.

---

## Setup verdict for V100 + H100 (confirmed 2026-07-10)

For the core science (G1/G2/G3, paddleocrvl parser, reasoners ≤ 8B, retrievers
bm25 / bge-m3 / colqwen2.5 / colmodernvbert), **setup + prestage work on both boxes —
no setup blocker.** Verified against the code:

- Both machines exist in `setup_env` (`setup_env.py:40-42`): V100 = cu126 + no
  flash-attn, H100 = cu126 + flash-attn. `core` = torch 2.7.0, `parse-paddleocrvl` =
  paddle 3.3.1 (`setup_env.py:30-35`); torch 2.7 cu126 covers sm_70 and sm_90.
- flash-attn on H100 is best-effort: the build is try/excepted and only warns
  (`setup_env.py:135-142`); the reasoner uses SDPA regardless. So H100 setup does not
  hard-fail if flash-attn does not build. On the supervisor, run both scripts with
  `--local` (else they target the Kaya remote root).
- `prestage` (`ops/kaya/config.json`) stages 2B/4B/8B + InternVL3-8B (not 32B), the
  retriever adapters, and **the colpali bases**: colqwen2.5-base auto-discovered from
  `adapter_config.json` (`prestage.py:122`), colmodernvbert-base + siglip2 via
  `retrieval_model_dependencies` (`config.json:68-72`); plus paddleocrvl
  (+ PP-DocLayoutV2) and MMLongBench. bm25 needs no model.

What actually works now (verified against logs job 1025361 on a V100 + the newest
retriever probe job 1027198, `{"passed": 6, "failed": []}`):

- **All six retrievers load and rank offline**, including `colqwen3` — it does NOT use
  colpali_engine (which lacks a ColQwen3 class); `ColQwen3Retriever._load` loads the
  repo's own transformers model via `AutoModel.from_pretrained(trust_remote_code=True,
  local_files_only=True)` (`retrievers/vision.py:271-289`). So colqwen3 is not a blocker.
- `qwen3-embedding-4B` is the only retriever that fails on a 16 GB **V100** (CUDA OOM,
  final_probe:16); it loads on the H100. Hardware only, per-method skip.
- `paddleocrvl` works after the paddle 3.0.0 → 3.3.1 fix (commit `9488b03`). The
  `[FAIL] parser.paddleocrvl` in the old job 1025361 (line 21, PIR strides error) is
  pre-fix and stale.

Core G1/G2/G3 need only `core` + `parse-paddleocrvl`, so the practical build is:
`setup_env --machine {V100|H100} --env core,parse-paddleocrvl` (+ `--local` on the H100),
`prestage --config ops/kaya/config.json` (+ `--local`), then `generate` on a G1/G2/G3
spec using paddleocrvl and ≤ 8B. **No setup blocker for that scope.**

### Blockers for running `ops/specs/target_architecture.yaml` IN FULL (H100 end goal)

The full reference spec sweeps parsers, 32B, and LongDocURL, which the core build does
not cover. Status of the three gaps:

1. **`mineru` + `unlimited` parser loader — FIX LANDED (needs Kaya re-probe).**
   `parser_worker` was loading VL models via `AutoModelForCausalLM` — wrong class:
   MinerU2.5 is a Qwen2-VL (`Qwen2VLConfig`) and Unlimited-OCR a custom
   image-text-to-text model (`UnlimitedOCRConfig`), so both crashed with "Unrecognized
   configuration class" (final_probe:38-59). New `_load_vlm` (`tools/parser_worker.py`)
   loads through `AutoModelForImageTextToText` → `AutoModelForVision2Seq` (with
   `trust_remote_code`). Compiles locally but **cannot be runtime-verified here** (parser
   envs + weights are Kaya-only) — re-probe on Kaya to confirm both load and generate.
   Caveat: this keeps the raw-transformers `generate()` path; if MinerU output is poor,
   the next step is its own `mineru[vlm]` pipeline (which uses the staged layout/formula
   aux models). NB the `9488b03` commit fixed **paddleocrvl/paddle**, a different parser.
2. **32B reasoner — STAGED for the H100.** Added `Qwen/Qwen3-VL-32B-Instruct` to
   `ops/kaya/h100_main.json`, which now stages the full G1 + G2 set (all reasoners incl.
   32B, all six retrievers + colpali bases, all three parsers, MMLongBench). 32B runs on
   the H100, not the V100.
3. **LongDocURL — abandoned.** Kept in `target_architecture.yaml` for demo/completeness
   only; intentionally NOT staged (its `dataset` sweep is not run).

(The per-parser try/except means a broken parser yields skip rows rather than crashing —
the job "completes" but those cells are empty; "no errors" needs #1 confirmed on Kaya.)

---

## Diagnosis task 1: why does `setup_env --env all` fail?

**Known symptom.** Building all four envs in one `--env all` invocation has failed and
left broken partial dirs for `parse-mineru` and `parse-unlimited` (corrupted conda
package cache + a half-created env prefix). Building `core` and `parse-paddleocrvl`
individually works fine (both were rebuilt green on 2026-07-10).

**Confirm the root cause** (don't just trust the symptom):
- `main()` loops `ENVS` and calls `build_env` with `subprocess.run(check=True)`. There
  is **no per-env isolation and no cleanup**: the first env that fails aborts the whole
  run and leaves its half-built prefix on disk. A later `--env all` then trips over the
  stale prefix (`build_env` skips `conda create` when `conda-meta` exists, then runs pip
  into a broken env).
- The shared conda package cache (`.cache/conda-pkgs`, set via `CONDA_PKGS_DIRS` by the
  containment fix) gets corrupted if a download is interrupted; the next env reusing it
  hard-fails. Verify `CONDA_PKGS_DIRS`/`PIP_CACHE_DIR` are actually pointing in-project
  and not leaking to `~/.conda` / `~/.cache/pip`.
- `parse-mineru` (torch 2.7) and `parse-unlimited` (torch 2.10) pull heavier, differently
  pinned stacks than `core`; check for a genuine dependency/wheel-build failure on the
  GPU-less login node, separate from the cache corruption.

**Likely fixes:** build each env in isolation and continue-on-error (report which envs
passed/failed at the end instead of aborting on the first), remove a half-created prefix
before retrying it, and `rm -rf .cache/conda-pkgs` before a full rebuild. Then actually
run `--env all` (or build mineru + unlimited individually) and read the real error.

**Scope note:** G1-G3 only need `core` + `parse-paddleocrvl`. `parse-mineru` /
`parse-unlimited` are used **only** by the G1 parser-comparison sweep. So `--env all`
only has to work if you want that sweep; core + paddleocrvl is enough for everything
else.

---

## Diagnosis task 2: prestage → generate offline contract (CONFIRMED BROKEN)

Running `prestage` today does **not** make retrieval work offline. Smoke job 1026182
(2B, `limit:2`, all three tasks, COMPLETED) proved it. The health gate was green
(G1 18 ok, G2 36 ok, G3 6 ok, 0 failures) but that green is misleading, because two of
the "ok" paths silently degraded. Findings, most important first:

1. **colpali vision retrievers are ADAPTERS whose base model is not staged.**
   `vidore/colqwen2.5-v0.2` is a LoRA adapter with
   `base_model_name_or_path: vidore/colqwen2.5-base`; `ModernVBERT/colmodernvbert` →
   base `ModernVBERT/colmodernvbert-base`. `prestage` stages the adapters but not the
   bases, so on the offline node `from_pretrained` reads the adapter config, tries to
   fetch the base from HF, and fails: *"couldn't connect to huggingface.co ... couldn't
   find them in the cached files."* **Both `colqwen2.5` and `colmodernvbert` fail to load.**
   - **Fix:** `prestage` must also stage each colpali adapter's base. Read
     `base_model_name_or_path` from the staged `adapter_config.json` and stage that repo
     too (or add an explicit base list to the JSON config). Bases needed:
     `vidore/colqwen2.5-base`, `ModernVBERT/colmodernvbert-base` (and colqwen3's base if
     colqwen3 gets supported). Verify each base's own config doesn't reference yet another
     uncached repo.
2. **This also corrupts the G2 inference vision arm, not just the benchmark.**
   `colqwen2.5` is the vision retriever `driver.build_retrievers` feeds the reasoner, and
   it builds with `allow_text_fallback=True`. So when colqwen2.5 failed to load, the G2
   inference vision cells silently fell back to a text/order heuristic ranking — they
   counted as "ok" but are scientifically wrong. So this offline-load bug hits **both**
   the retrieval side-artifact and the G2 inference. High priority; it must be fixed
   before the real run is meaningful. (The six-method side-artifact uses
   `allow_text_fallback=False`, so there it fails loudly and is skipped, which is why the
   log showed the errors.)
3. **`colqwen3` (`OpenSearch-AI/Ops-Colqwen3-4B`) is unsupported by the installed
   colpali_engine.** The log shows `ColQwen3` / `ColQwen3Processor` are *not in
   colpali_engine* (this model is not an adapter, so there's no base to stage). This is an
   **env** problem: either install a colpali_engine version that supports Qwen3-based
   ColPali, or drop `colqwen3` from the vision ladder. Track under diagnosis task 1.
4. **`qwen3-embedding-4B` OOMs on a 16GB V100.** Not an env/prestage bug, a hardware
   limit: the 4B embedder doesn't fit on Volta. It loads and runs on the H100 supervisor.
   Decide: run the retrieval side-artifact on the H100, or accept the expensive text rung
   is skipped on Kaya (the per-method `try/except` in `write_retrieval_eval` already skips
   it cleanly and the run continues).
5. **What already works offline:** the reasoners, PaddleOCR-VL + its paddlex pdx cache,
   mmlongbench, `bm25` (no model), and `bge-m3`. So the staging pattern is right; it's
   just missing the colpali bases.

**Prestage config scope for "any task works, excl. 32B".** `config_minimal.json` today
lists 2B + 8B reasoners, the five retriever adapters (no bases), paddleocrvl, mmlongbench.
To cover the G1 sweeps it also needs: reasoners `4B` and `InternVL3-8B` (still **not**
32B); the colpali **base** models; and, only if the parser sweep is wanted, the
`mineru` + `unlimited` parser models and their envs. LongDocURL is no longer in
scope for prestaging.

---

## Acceptance test (how you know it's fixed)

1. `setup_env --machine V100 --env all` finishes green (all four envs `pip check` clean).
2. `prestage --config <full json>` stages reasoners + retrievers **+ colpali bases** +
   parsers + datasets, idempotently.
3. Submit `ops/specs/kaya_smoke.yaml`, pull, `check_run`. The retrieval side-artifact must
   now write rows for `colqwen2.5` and `colmodernvbert` (not just bm25/bge-m3), and the G2
   inference vision cells must use **real** colqwen2.5 rankings, not the fallback. Confirm
   the latter by spot-checking a `retrieved_vision_k*` cell's pages against the
   `retrieval.jsonl` colqwen2.5 ranking. `qwen3-embedding` and `colqwen3` may still be
   skipped on the V100 (OOM / unsupported) — acceptable, and noted above.
4. Then the real run `ops/specs/kaya.yaml` (`--gres gpu:2 --mem 64G --time 08:00:00`).

## State snapshot (2026-07-10)

- Envs on Kaya: `core` + `parse-paddleocrvl` rebuilt green (paddle 3.3.1). `mineru` /
  `unlimited` not needed for G1-G3, `--all` still suspect (task 1).
- Prestaged at diagnosis time: 2B/8B reasoners, the 5 retriever adapters (no bases),
  PaddleOCR-VL (+pdx cache), and MMLongBench. Missing then: colpali bases, 4B, and
  InternVL3-8B.
- Code: G1-G3 + the six-method retrieval side-artifact + retrieval timing are landed and
  pytest is green; robustness (per-method skip on load failure) is proven by the smoke.
- Render cost observed: first-time page rasterization is ~21s/page on the `/group` network
  FS (one-time, cached to `.cache/renders/`), so the real run needs generous walltime.

## Coordinate / not in scope here

Both of the items that were in flight when this handoff was written have since **landed**
(2026-07-10); details in `docs/DECISIONS.md`. What they change for the offline/retrieval
contract:

- **G4 → G3 merge — LANDED** (committed). Tasks are now G1/G2/G3; the classifier is G3's
  optional side artifact (`classifier.jsonl`, gated on `config.classifier_spec`). The
  driver `run_side` call site, `G3_hallucination`, and the registry all moved; the G4
  module is gone. `docs/WIP_g4_g3_merge.md` is deleted.
- **Per-sweep YAML expander — LANDED** (uncommitted; `docs/WIP_yaml_expander.md`). It
  reworks the retrieval wiring this handoff cares about, but **behaviour-preserving on the
  defaults**, so the findings above still hold at the same spots:
  - `driver.build_retrievers` now builds the inference arms from
    `config.inference_text_retriever` / `_vision_retriever` via `get_text_retriever` /
    `get_vision_retriever` (defaults still bm25 / colqwen2.5, same fallback default). So
    finding #2 (silent `allow_text_fallback` degradation) is unchanged — fix it at the
    retriever construction, now config-driven.
  - `side_artifacts.write_retrieval_eval` takes its method sets as params
    (`text_methods` / `vision_methods` / `joint_pairs`, defaults = the old constants), fed
    from the G2 `retrieval` block. Per-method try/except + skip is unchanged, so findings
    #1/#3/#4 (colpali bases, colqwen3 unsupported, qwen3-embedding OOM) are unaffected.
  - **Prestage-scope shift:** `parser` and `dataset` are now real spec axes. A `parser`
    sweep needs the `mineru` / `unlimited` envs+weights (task 1 / the scope note above); a
    `dataset` sweep selects the loader in `ops/generate.py` (`DATASET_LOADERS`:
    mmlongbench / longdocurl), so **LongDocURL is back in prestage scope IF the dataset
    sweep is run** (it is skipped otherwise, so the "no longer in scope" line stays true
    for the default kaya run).

---

## Next work (deferred — NOT this turn)

Separate from the env/prestage contract. Both confirmed by the 2026-07-10 audit against
the code; captured here so the next session fixes them.

1. **Pivot-v4 G2 stage reuse + continuous write.** Today Stage 2 (inference) does not
   consume Stage 1 (retrieval) — they compute rankings independently and in the wrong
   order:
   - The inference retrievers (bm25 / colqwen2.5) run first, memoized under
     `<run_tag>/retrieval/` (`driver.py:129-135`); the benchmark runs **after** all
     reasoner cells (`driver.py:357`) and rebuilds every method from scratch with **raw**
     (non-memoized) retrievers (`side_artifacts.py:50,52`). bm25/colqwen2.5 are ranked
     twice, and — because the two paths can set different `allow_*` fallback flags — the
     pages fed to the reasoner can silently diverge from the ranking scored in
     `retrieval.jsonl` (the env finding-#2 hazard).
   - `write_retrieval_eval` ranks every method into in-memory dicts, then opens
     `retrieval.jsonl` `"w"` and dumps all rows at the end (`side_artifacts.py:90-113,
     122-123`) — it is **not** written incrementally.
   - Fix direction: run retrieval Stage 1 FIRST, write `retrieval.jsonl` rows as each
     method finishes (incremental append), and have inference READ the persisted stage-1
     ranking for bm25/colqwen2.5 instead of recomputing. Add a test asserting the
     inference pages match the `retrieval.jsonl` ranking for the same (question, method).

2. **YAML reader must enforce bm25 + colqwen2.5 are present.** Every G2 `retrieval` block
   must contain `bm25` in `text_retrievers` and `colqwen2.5` in `vision_retrievers` — they
   are the fixed matched/cross arms the inference stage feeds and that G3's similarity
   retrieval uses. Today `_g2_spec` (`experiments/corpus/yaml_spec.py`) only validates that
   the inference PICKS are a subset of the benchmark lists; it does not require
   bm25/colqwen2.5 to exist. Add the presence check (raise `SpecError` if either is
   missing) alongside the existing subset validation.
