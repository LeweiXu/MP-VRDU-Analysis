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

- **G4 → G3 merge is in flight by another agent** (`docs/WIP_g4_g3_merge.md`): don't touch
  the driver `run_side` call site, `G3_hallucination`, the registry, or the deleted G4.
- **Per-sweep YAML expander is deferred**; its target schema is
  `ops/specs/target_architecture.yaml` (user-owned, with the precedence rules). Not part
  of this env/prestage work.
