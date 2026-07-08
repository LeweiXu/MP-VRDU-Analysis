# DECISIONS — pivot-v4 changelog + probe/env verdicts

This is the pivot-v4 decision log called for by `pivot_v4_implementation.md`. It
records the do-over reference point, the Phase-1 probe/env verdicts, and every
real judgement call made while building v4 (one line each: what, why, what it
affected).

> Note on where this lives: the repo's `CLAUDE.md` says the DECISIONS content was
> folded into `AGENT_GUIDE.md`. The v4 implementation plan reintroduces a standalone
> `docs/DECISIONS.md` as the pivot changelog. This file is that changelog. Final
> reconciliation of `AGENT_GUIDE.md` / `PROJECT_SPEC.md` / `README.md` to v4 (and
> folding `pivot_v4.md` in here) happens in the Phase-4 cleanup, not now.

---

## Phase 0 — capture (2026-07-08)

- **Reference commit (v3 snapshot):** `e73ee892b9e627f313ab780ec2c199470175bb5c`
  ("pivot v4", branch `main`). This is the v3 state the do-over forks from.
- **v3 structural fixtures preserved** at `tests/fixtures/v3_results/`, copied from
  `results/cache/{bf16-lowres,yaml-g1-g2-g5-rerun}/full/`. jsonl only (no
  render/marker/ocr blobs). Covers task shapes G1/G2/G3/G5/G6 incl. judged
  `results.jsonl`, retrieval and classifier side-artifacts. Labelled **v3-shaped,
  values NOT comparable to v4** (see the fixtures README).
- **Gitignore:** added `!tests/fixtures/v3_results/**/*.jsonl` to override the
  blanket `*.jsonl` ignore so the fixtures are tracked.

---

## Phase 1 — probes & decisions

### 1a. Resolution probe

Script written at `ops/scripts/resolution_probe.py`. It sweeps the five presets
(`min`/`low`/`med`/`high`/`full`) on the V rung (image-only, parser-independent),
worst case ~10 pages, with the 8B primary reasoner loaded exactly like production
(HF, `device_map="auto"`, 5GiB/GPU reserve, memory-efficient SDPA). Per preset it
records per-GPU peak VRAM and OOM; the highest preset that stays under 16GiB (else
the highest that does not OOM) is the deployment resolution.

**Submitted to Kaya (2026-07-08): job `1017226`**, partition `gpu`, `gpu:v100:2`,
30 min. The `gpu` partition was saturated (the long `g1g2g5-full` run holds a
node), so the scheduler estimated a start up to ~2026-07-10; backfill may run it
sooner. Pull results with `python3 -m kaya.kaya watch 1017226` (or `pull`); the
verdict lands in `results/probes/resolution_probe.json`.

_Verdict (chosen deployment resolution preset): PENDING job 1017226._

### 1b. Environment / dependency decision

**vLLM verdict: DROP.** Evidence (2026-07-08):

- The live inference path already uses plain HF `transformers`
  (`Qwen3VLForConditionalGeneration.from_pretrained(device_map="auto")` +
  `model.generate(do_sample=False)` under the memory-efficient SDPA kernel, in v3
  `models/local_vlm.py`). It never imports vLLM.
- The only real `import vllm` in the whole tree is the retired feasibility probe
  `scripts/run_probe.py:578`, which v4 replaces. The other two hits are docstring
  mentions.
- v4 reasoning is batch-1 and latency-measured, which is exactly the regime where
  vLLM's serving/throughput machinery buys nothing.

**What dropping vLLM frees** (these pins existed only for it):

- `openai` — `requirements.txt` documents `openai<=1.90.0` as a vLLM 0.9.2 cap.
  Freed; judge SDKs can use a current `openai`.
- `torch==2.7.0` exact — relax to a cu126-compatible range (Kaya is `cuda/12.6.3`).
- `transformers==4.57.6` exact — keep the **floor** `>=4.57` (Qwen3-VL's
  `Qwen3VLForConditionalGeneration` landed there); relax the ceiling.
- `pillow==10.4.0` / `pillow<11` — the `<11` pressure came from marker/docling,
  which v4 drops; can move to pillow 11.

**Env partition (target):**

- **Core reasoning env** — the reasoners (Qwen3-VL + InternVL via timm), quant
  (bitsandbytes), the T-rung cheap text (PyMuPDF), all retrievers (rank-bm25,
  FlagEmbedding/BGE-M3, colpali-engine for ColQwen/ColModernVBERT, Qwen3-Embedding),
  the judge SDKs (openai, google-genai), and data/util libs. No vLLM.
- **One isolated env per parser that will not co-exist** — PaddleOCR-VL, MinerU 2.5,
  Unlimited OCR. They are heavy, separately-pinned VLM stacks; they cross to the
  reasoner only via the disk cache (the pre-pass warms the parser cache), so they
  never share the reasoner's env or its VRAM. Parsers that happen to `pip check`
  clean together may share one `parse` env.
- **Dropped stacks** (and any pins that existed only for them): vLLM, marker-pdf,
  docling, and the old paddleocr/paddlex/paddlepaddle parser stack (superseded by the
  isolated PaddleOCR-VL env). Marker caches may be kept as appendix continuity.
- Keep the local-Blackwell (RTX 5070) env as-is.

**Still to verify empirically (Phase 4, per the plan):** the exact package
names/pins for PaddleOCR-VL / MinerU 2.5 / Unlimited OCR, and a clean `pip check`
per env. That is decided by attempting installs, not statically, so it is
deliberately deferred to Phase 4 finalization. This Phase-1 deliverable is the
verdict + partition, which are settled above.

---

## Phase 2 — park & scaffold (2026-07-08)

- **v3 snapshot in `old/`** (untouched, 100 modules): `config.py`, `schema.py`,
  root `__init__.py`, and the packages `cli covariates data experiments gates
  metrics models pipeline reporting tools scripts specs kaya`, plus the v3
  `test_*.py` under `old/tests/`. Deleted in the final commit once v4 is green.
- **Phase-0 fixtures kept at root** `tests/fixtures/v3_results/` (not moved into
  `old/`), so Phase-3 I/O tests can read them.

**Direct-copy set (to v4 homes):**

| File(s) | v4 home | Mark |
|---|---|---|
| `kaya/{kaya.py,__init__.py,config.json,KAYA_*_GUIDE.md}` | `ops/kaya/` | copied-pending-rework (see below) |
| `download_hf,gpu_test,kaya_status,setup_env,dataset_stats,profile_datasets,split_docs_by_type` | `ops/scripts/` | clean copy |
| `dump_docstrings.py` | `ops/scripts/` | copied-pending-rework (stale SUMMARY_OVERRIDES) |
| `annotate_docs,prestage,inspect_results,run_probe` | `ops/scripts/` | copied-pending-rework (per plan §Phase 2) |
| `ANNOTATION_GUIDE.md` | `ops/scripts/` | clean copy |
| `specs/*.yaml` | `ops/specs/` | clean copy |

**Deviations recorded:**

- `ops/kaya/kaya.py`: `LOCAL_ROOT` was `Path(__file__).resolve().parents[1]`
  (repo root under `kaya/`); at `ops/kaya/` that resolved to `ops/`, breaking the
  rsync source and program-path anchoring. Changed to `parents[2]`. Verified:
  `python3 -m ops.kaya.kaya show-config` resolves the repo root and config. The
  Kaya driver is now invoked as **`python3 -m ops.kaya.kaya`** (was `kaya.kaya`);
  the live resolution-probe job `1017226` is pulled with
  `python3 -m ops.kaya.kaya watch 1017226`.
- `ops/scripts/dump_docstrings.py`: its `SUMMARY_OVERRIDES` still key off v3 paths
  and contain "v4 should" plan-talk (which the new docstring rule forbids). It is
  copied for reference but must have the overrides cleared and be regenerated in
  Phase 4. Reclassified clean-copy -> copied-pending-rework.

**Scaffold:** empty v4 tree created with 1-3 sentence module docstrings; all 65
spine modules + the `ops` entry points import cleanly. `docs/generated/` created
and the generated outputs (`dataset_stats.md`, `dataset_label_distributions.csv`)
moved there. `docs/REPO_STRUCTURE.md` written (tree + auto-gen marker; per-file map
regenerated in Phase 4). The `CLAUDE.md` module-docstring rule was added.

---

## Phase 3 — tests first (2026-07-08)

- v3 tests are parked in `old/tests/` (deleted from the active tree). `pytest.ini`
  scopes collection to `tests/` and excludes `old/` and the local `.cache/` conda
  package tree.
- v4 suite written as executable specs of the `pivot_v4.md` invariants. Result:
  **150 tests, 0 collection errors, 128 green, 22 red** against the stubs — the red
  ones are the invariants Phase 4 must satisfy.
- Design: `tests/conftest.py::require(module, attr)` fetches an intended v4 symbol
  and fails cleanly if the stub lacks it, so unfinished work reads as a red test
  rather than a collection-time ImportError. Fixtures loaded from
  `tests/fixtures/v3_results/`.
- Green now (real guards, not stubs): every spine module imports; the module-
  docstring rule holds on all v4-authored modules; `config` has no input-token cap
  symbols; `pipeline/representation.py` references no bbox and imports no model
  backend; the four fixture shapes parse. Red now: registry task discovery, schema
  telemetry + truncation canary, cell robustness + `--failed-only`, machine-
  independent keying, corpus sampling modes, YAML `parse_spec`, and the v4
  jsonl reader / build grouping.
- Test files: `test_imports_registry`, `test_schema_telemetry`,
  `test_config_cap_removed`, `test_engine_robustness`, `test_keying`,
  `test_representation`, `test_corpus_scope`, `test_yaml_spec`, `test_io_fixtures`,
  `test_docstrings`.

---

## Deviation & decision log (Phase 4+)

_One line per real judgement call: what, why, what it affected._
