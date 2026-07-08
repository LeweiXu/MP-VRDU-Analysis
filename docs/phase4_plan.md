# Phase 4 build plan (concrete, staged)

Companion to `pivot_v4_implementation.md` (the paradigm and rules) and
`pivot_v4.md` (the science). That doc says *build bottom-up, one green layer at a
time*; this doc pins down **which stages, in what order, which of the 22 red
tests each turns green, and exactly what is lifted from `old/` vs written from
scratch.**

Ground truth captured while writing this:

- Test baseline right now: **128 green / 22 red** (`envs/mpvrdu/bin/python -m pytest -q`).
- Phase-0 fixtures are in place under `tests/fixtures/v3_results/` (G1/G3/G5 v3
  rows). Phase-1 env build is done on Kaya (4 envs, `pip check` clean).
- Prestage is done on Kaya: 4 reasoners, the **v3** retrieval set, 3 parsers, and
  both datasets are all staged. The v4 retrieval set is not staged yet (see
  Stage 0).
- The v4 spine is scaffolded as one-line-docstring stubs. `old/` holds the full
  untouched v3 tree.

## How to read the reuse column

- **lift** = copy the v3 file/function into its v4 home and adapt names/imports;
  the logic is basically right.
- **trim** = lift but delete a large chunk (a dropped mechanism).
- **rewrite** = the v3 file exists but the mechanism changed enough that only the
  scaffolding/patterns carry over.
- **new** = no v3 equivalent; write from scratch.

A recurring detail: in v3 the two cache-key functions and `ResultRow` live in
`pipeline/orchestrator.py`. v4 moves the **keys** to `experiments/engine/paths.py`
and **`ResultRow`** to `schema.py` (the telemetry test imports it from `schema`).

---

## The 22 red tests, grouped by the stage that greens them

| Stage | Red tests greened | Count |
|---|---|---|
| 1 schema + config | `test_schema_telemetry` (3), `test_config_cap_removed::test_resolution_presets_present` (1) | 4 |
| 2 engine plumbing | `test_keying` (3), `test_engine_robustness` (3) | 6 |
| 3 data + corpus | `test_corpus_scope` (5) | 5 |
| 4 tools + retrievers + representation | `test_representation::test_four_rungs_are_addressable` (1) | 1 |
| 5 models + pipeline + orchestrator | none direct (enables end-to-end) | 0 |
| 6 tasks + registry + yaml | `test_imports_registry` (2), `test_yaml_spec` (2) | 4 |
| 7 scoring + reporting | `test_io_fixtures` (2) | 2 |
| 8 ops + cleanup | none (docstrings test already green; keep it green) | 0 |

`test_config_cap_removed::test_no_input_token_cap_symbols` and the whole
`test_representation` file's *no-bbox / no-model-import* asserts are already green
against the empty stubs and must **stay** green as each stage fills code in.

Stages 1 and 2 are pure plumbing (no GPU, no models), so they can land first and
fast even though the dependency order in `pivot_v4_implementation.md` lists the
engine later. Everything Stage 5 and up wants a real model for is deferred behind
the disk-cache boundary, so the suite goes fully green before any GPU smoke.

---

## Stage 0 — one-time prep (do before Stage 1)

- **Bump the cache namespace/version** so v3 and v4 cells never co-mingle
  (`pivot_v4_implementation.md` §Phase 4). This is a constant in `config.py` /
  the cache path builder; set it once here.
- **Swap the retriever catalog** in `ops/kaya/config.json::retrieval_models` from
  the v3 ids to the v4 set: BM25 (no weights), **BGE-M3**, **Qwen3-Embedding-4B**
  (text); **ColModernVBERT**, **ColQwen2.5-v0.2** (kept), **ColQwen3-4B** (vision).
  Then re-run `prestage.py` to stage the three new ids. colqwen2.5 is already
  staged; colpali/colqwen2/bge-small can stay staged as dead weight or be pruned
  later.
- No test depends on this; it just stops us from threading v3 ids through the new
  retriever code.

---

## Stage 1 — schema + config (the foundation)

**Goal:** the frozen contracts carry v4 telemetry and the cap is gone.
**Greens:** `test_schema_telemetry` (3), resolution-presets (1).

| v4 target | from | how |
|---|---|---|
| `schema.py` (Question, Page, PageSet, TextSpan, Text/ImagePart, Payload, Prediction, Score, `derive_hop`, `is_not_answerable`) | `old/schema.py` | **lift** almost verbatim. |
| `schema.py::ResultRow` + `STATUS_VALUES` | `old/pipeline/orchestrator.py::ResultRow` | **rewrite**: move into schema; add the telemetry fields the test names (`status`, `skipped_reason`, `bin_label`, `scan_label`, `machine`, `total_text_tokens`, `total_visual_tokens`, `text_tokens_fed`, `output_tokens`, `latency_s`, `prefill_latency_s`, `decode_latency_s`, `peak_vram_bytes`, `oom_occurred`, `tokens_dropped`); add `STATUS_VALUES = ("ok","oom","error")`. |
| `schema.py::PageSetProvenance` | `old/schema.py` | add `"similarity"` (hallucination page selection); drop `"buried"`. |
| `config.py` | `old/config.py` | **trim**: delete `max_input_tokens`, `MAX_INPUT_TOKENS_BY_SIZE`, `max_input_tokens_for_spec`; keep paths + `ExperimentConfig`; **add** `VISUAL_RESOLUTION_PRESETS` (keys `min/low/med/high/full`, per-page pixel caps) and the fixed deployment preset chosen by the resolution probe. |

**From scratch:** the telemetry-canary invariant (`text_tokens_fed == total_text_tokens`, `tokens_dropped == 0`) as a helper/validator the orchestrator will call.

**Note:** the v3 `Prediction` splits `input_text_tokens` / `input_visual_tokens`
already; v4 renames to the `total_*` / `*_fed` scheme in `ResultRow`. Keep
`Prediction` mostly as is and map into `ResultRow` at telemetry-capture time.

---

## Stage 2 — engine plumbing: keying + robustness (no models)

**Goal:** machine-independent keys and one-row-per-cell + `--failed-only`, as pure
functions testable without a GPU.
**Greens:** `test_keying` (3), `test_engine_robustness` (3).

| v4 target | from | how |
|---|---|---|
| `experiments/engine/paths.py::prediction_key(question_id, doc_id, condition, representation, model_spec, page_indices)` | `old/pipeline/orchestrator.py::make_prediction_key` | **rewrite**: SHA-256 over a sorted JSON of the identity dict; signature now takes `page_indices` and **no dpi** (resolution preset lives in the run manifest, not the key); nothing machine-dependent may enter it (the test monkeypatches hostname + `torch.cuda.device_count`). |
| `experiments/engine/paths.py::result_key(...)` (adds judge_spec) | `old/...::make_cache_key` | **lift** the two-layer split (prediction key without judge, result key with it). |
| `experiments/engine/paths.py` (ExperimentPaths, logging, `free_gpu`, `mode`) | `old/experiments/paths.py` | **lift**. |
| `experiments/engine/driver.py::run_cells(cells, run_one)` | new shape over `old/experiments/driver.py::generate`'s loop | **rewrite**: wrap each `run_one(cell)` in try/except; success → its row; failure → a row with `status ∈ {oom,error}` (classify on "out of memory" in the message) + `skipped_reason`; **always N rows for N cells**. |
| `experiments/engine/driver.py::select_failed(rows)` | new | **new**: return rows with `status != "ok"`. |
| `experiments/engine/driver.py` failed-only upgrade | new | **new**: read a jsonl, re-run only `select_failed`, upgrade those rows **in place** (same file), leave `ok` rows byte-identical. |

The rest of `old/driver.py` (the parse pre-pass, reasoner load/free, side work)
is **not** in this stage; it lands in Stage 5 where it has real models to
sequence. Stage 2 builds only the pure functions the tests call.

---

## Stage 3 — data + corpus + annotations

**Goal:** questions load, carry `bin_label` from manual annotation, split
answerable/unanswerable, and sample document-coherently.
**Greens:** `test_corpus_scope` (5).

| v4 target | from | how |
|---|---|---|
| `data/loader.py` | `old/data/loader.py` | **lift** (MMLongBench + LongDocURL); add a `split_answerable(questions)` / task-pool helper. `is_unanswerable` is already derived in `schema`. |
| `data/render.py` | `old/data/render.py` | **lift** (`render_pdf`, `pdf_page_count`, `extract_text_spans`, `classify_scanned`, `render_question_pages`). |
| `data/annotations.py` | scan-label reader in `old/tools/text.py` + `old/scripts/annotate_docs.py` | **new** module: read/validate the 3-label table (`bin_label` in {text-dominant, mixed-modality, visual-dominant}, `scan_label`, `dominant_visual`). |
| `data/binning.py` | `old/data/binning.py` | **rewrite**: was `doc_type → bin`; now `doc_id → bin_label` lookup from `annotations`, new bin names, and a helper that stamps `bin_label` onto each `Question` (the corpus test reads `q.bin_label`). Keep the bin-order / count-structure idea. |
| `experiments/corpus/resolve.py::sample_per_bin(corpus, per_bin, seed)` | `old/experiments/corpus.py::sample_questions_per_bin` + `_draw_documents` | **lift + rename**: keep the whole-document draw (doc-coherent); add `full` and `limit` modes; **bind the answerable pool by task** (G1/G2 answerable, G3 unanswerable) so a spec can't cross-contaminate. |
| `experiments/corpus/smoke.py` | `old/experiments/smoke.py` | **lift**; retarget bins to v4 names. |

**Blocker to flag:** bins can't be assigned until the manual annotation CSV
exists (the `annotations/` dir is currently empty). The reworked `annotate_docs`
(Stage 8) writes it, then a <1 hr human pass fills it. Until then, `binning`/
`corpus` can be unit-tested against a fixture annotation table (the corpus test
already fabricates its own), but a real run is blocked on the annotation pass.

---

## Stage 4 — tools + retrievers + representation

**Goal:** the four channels build; the cost-ordered ladder is addressable; no bbox,
no model import in the representation layer.
**Greens:** `test_representation::test_four_rungs_are_addressable` (1). Keep the
already-green no-bbox / no-model-import asserts green.

| v4 target | from | how |
|---|---|---|
| `tools/text.py` | `old/tools/text.py` | **trim hard**: keep `embedded()` (PyMuPDF per-page text = the T channel); delete the Marker/OCR routing, the PaddleOCR engine, `text_channel`, scan-label lookup. |
| `tools/visual.py` | `old/tools/visual.py` | **lift**: `VisualArtifact`, `estimate_visual_tokens`, `resolution(scale)`, `full_page`, `visual_channel`; wire scaling to `VISUAL_RESOLUTION_PRESETS`. Drop `region_crop`. |
| `tools/parser.py` | caching pattern from `old/tools/layout.py`; backends are new | **rewrite/new**: PDF-parser markdown for TL/TLV via PaddleOCR-VL / MinerU 2.5 / Unlimited OCR, each in its **isolated env**, output warmed to a **disk cache** (reuse the `_safe_stem` / cache-root / read-write-cache pattern from `layout.py`). **No bbox JSON.** This is the heaviest new tools work; the parser↔reasoner boundary is the disk cache (subprocess into the parser env). |
| `retrievers/text.py` | `old/covariates/retriever.py::BM25BGERetriever` | **rewrite**: BM25 + BGE-M3 + Qwen3-Embedding-4B as three cost rungs; reuse BM25/embedding scoring + `MemoizedRetriever`. |
| `retrievers/vision.py` | `old/covariates/retriever.py::ColQwenRetriever` | **rewrite**: ColModernVBERT + ColQwen2.5 (kept) + ColQwen3-4B; reuse the ColQwen scaffolding. |
| `retrievers/joint.py` | new | **new**: deduplicated union of one text + one vision page set (matched-tier pairs, k ∈ {1,3,5}); no score fusion (§4.1). |
| `retrievers/__init__.py` (Retriever ABC, registry) | `old/covariates/retriever.py::Retriever` + `_GuardRetriever` idea | **lift** the ABC + memoization + guard-retriever pattern. |
| `pipeline/representation.py` | `old/pipeline/representation.py` | **rewrite**: T = embedded text; TL/TLV = **parser markdown** from `tools/parser`; V = image. Remove `layout_channel` / bbox entirely. Keep the `Representation` ABC + `get_representation` registry + `ChannelRepresentation`. Must not import any model backend (test asserts this). |

**Note on cost-ordering:** the ladder is non-cumulative now (`T ⊄ TL`); the
composer builds each rung independently, it does not stack T under TL.

---

## Stage 5 — models + pipeline + orchestrator + engine lifecycle

**Goal:** real reasoners load/free correctly, telemetry is captured per cell, and
the parse pre-pass keeps parser/retriever/reasoner off the GPU together. No new
red tests, but this is what makes an end-to-end cell run.

| v4 target | from | how |
|---|---|---|
| `models/qwen3vl.py` | `old/models/local_vlm.py` | **trim + extend**: rename; **delete `_truncate_context`** (cap gone); keep load/free, quantization config, token counting; **add** prefill/decode latency split + `peak_vram_bytes` via `max_memory_allocated()` reset per cell. |
| `models/internvl.py` | `old/models/internvl.py` | same trim (drop `_truncate_context`) + latency/vram capture. |
| `models/classifier.py` | `old/covariates/classifier.py` | **rewrite** the binning dependency (`doc_type → bin_label`); keep the first-two-pages small-model classify path. |
| `models/payload.py` | `old/models/payload.py` | **lift** (`ModelInput` is a frozen interface). |
| `models/__init__.py::get_reasoner` | `old/models/__init__.py` | **lift**; register `qwen3vl` (was `local_vlm`) + `internvl`. |
| `pipeline/reasoner.py` | `old/pipeline/reasoner.py` | **lift** (ABC + StubReasoner). |
| `pipeline/conditioner.py` | `old/pipeline/conditioner.py` | **trim + add**: keep Oracle / RetrievedTopK / FullDoc; **add `SimilarityTopK`** for the hallucination study; drop `BuriedOracle`. |
| `pipeline/judge.py` | `old/pipeline/judge.py` | **lift** (Gemini + GPT judges, retry/quota/2-key fallback); repoint the abstention import from `metrics` to `scoring`. |
| `pipeline/orchestrator.py` | `old/pipeline/orchestrator.py` | **rewrite**: keep the two-cache design + `prewarm_cell` non-co-residence; `ResultRow` now imported from `schema`; keys now come from `engine/paths`; **capture the full telemetry** (status, prefill/decode, vram, canary) into `ResultRow`. |
| `experiments/engine/driver.py` (lifecycle half) | `old/experiments/driver.py::generate` | **lift + adapt**: the parse pre-pass (warm caches with a spec-only reasoner, unload retrievers/parser, `free_gpu`, then load real reasoner), lazy reasoner free between specs; add the **systemic-failure abort** (configurable threshold, ~15-20 consecutive or >50% early). Wire it to `run_cells` from Stage 2. |
| `experiments/engine/side_artifacts.py` | `old/experiments/side_artifacts.py` | **lift** (retrieval + classifier writers). |
| `experiments/engine/artifacts.py` | `old/experiments/artifacts.py` | **lift** (manifest-driven judge/build). |

Model-lifecycle invariants (`pivot_v4_implementation.md` §7) are the acceptance
bar here: parser output crosses to the reasoner **only via disk**; one engine
load per run, unloaded before any reasoner; parser/retriever/reasoner never share
VRAM. The v3 `generate()` already does exactly this; keep that discipline.

---

## Stage 6 — tasks + registry + yaml scope

**Goal:** four `G[num]_[name]` tasks discoverable; sweeps are YAML variants; no
`machine:` concept.
**Greens:** `test_imports_registry` (2), `test_yaml_spec` (2).

| v4 target | from | how |
|---|---|---|
| `experiments/tasks/base.py` | `old/experiments/base.py` | **lift** (`Cell`, `Retrievers`, `GenerationTask` ABC, `oracle_ladder_cells`, `matched_cross_cells`). |
| `experiments/tasks/G1_oracle_ladder.py` | `old/experiments/G1_sufficiency.py` | **rewrite**: oracle × {T,TL,TLV,V}, answerable-only; base grid the parser/resolution/family/dataset/quant/size sweeps reuse as YAML variants. |
| `experiments/tasks/G2_retrieval.py` | `old/experiments/G5_retrieval.py` | **rewrite**: v4 method set + joint; single-method k ∈ {1,3,5,7,10}, joint k ∈ {1,3,5}; inference scorer at TLV/V + retrieval side-artifact scorer. |
| `experiments/tasks/G3_hallucination.py` | new (reuses `SimilarityTopK` + G2 cache) | **new**: unanswerable × similarity 2-3 pages × TLV × {no/generic/targeted prompt}; correct = abstention; no oracle arm. |
| `experiments/tasks/G4_classifier_pricing.py` | `old/experiments/G6_classifier.py` | **lift + adapt**: side-only classifier pricing (latency/VRAM), no reasoner cells. |
| `experiments/registry.py` | `old/experiments/registry.py` | **rewrite**: list exactly the four v4 tasks; no legacy names (test asserts both). |
| `experiments/corpus/yaml_spec.py::parse_spec` | `old/experiments/yaml_spec.py` | **rewrite**: return a spec object exposing `task`, `representations`, `corpus` (a `sampling` block: `full` / `{per_bin,seed}` / `{limit}`); **reject any `machine` field**; drop the scan-label reading. Keep `YamlGenerationTask` / manifest-writing scaffolding. |
| `ops/specs/*.yaml` | `old/specs/` | port saved specs; strip any `machine:` line (test scans for it). |

---

## Stage 7 — scoring + reporting

**Goal:** cached rows become numbers and tables; the v4 reader parses fixtures and
the build groups at cell cardinality.
**Greens:** `test_io_fixtures` (2).

| v4 target | from | how |
|---|---|---|
| `scoring/accuracy.py` | `old/metrics/accuracy.py` | **lift** (doc-level accuracy + bootstrap CIs). |
| `scoring/cost.py` | `old/metrics/cost.py` | **lift + extend**: add prefill/decode + `peak_vram` aggregation. |
| `scoring/frontier.py` | `old/metrics/frontier.py` + `old/gates/core.py::frontier_divergence_gate` | **lift**: the sufficiency-frontier rule + 3-point margin. |
| `scoring/retrieval.py` | `old/metrics/retrieval.py` | **lift** (page P/R/F1, per-bin/modality summaries). |
| `scoring/abstention.py` | `old/metrics/abstention.py` | **lift**. |
| `scoring/agreement.py` | `old/gates/core.py` (kappa bits) | **trim**: keep `cohen_kappa` + agreement-sheet math; **drop** the F1/F2/F3 gate CLI + threshold scaffolding. |
| `scoring` skips `status != ok` | new rule | **new**: scoring ignores oom/error rows (no exclusion list needed). |
| `reporting/build.py` | `old/reporting/build.py` | **rewrite** the task→table routing to v4 table names + **build-time routing assembly** (routing table = G1 rows + G4 classifier price); keep `TableSpec` / blocker / bootstrap machinery. Provide the v4 jsonl reader + cell grouping the io_fixtures test exercises. |
| `reporting/tables/_common.py`, `_markdown.py` | `old/reporting/tables/_common.py`, `_markdown.py` | **lift**; update bin/rung helpers to v4 bin names. |
| `reporting/tables/headline.py` | `old/reporting/tables/T1_headline.py` | **rewrite**: cost-ordered ladder × bin. |
| `reporting/tables/routing.py` | `old/reporting/tables/T7_routing.py` | **lift + adapt** to build-time assembly. |
| `reporting/tables/scale.py` | `old/reporting/tables/T8_scale.py` | **lift** (size/quant frontier). |
| `reporting/tables/composition.py` | `old/reporting/tables/T5_composition.py` | **lift** (appendix). |
| `reporting/tables/matched_cross.py` | `old/reporting/tables/T6_retrieval.py` (partial) | **rewrite** (matched-vs-cross per k). |
| `reporting/tables/parser.py`, `resolution.py`, `kdepth.py`, `retrieval_accuracy.py`, `hallucination.py` | new (headline/T6 patterns) | **new** builders for the parser sweep, resolution sweep, k-depth, page-F1 benchmark, and abstention×prompt. |

---

## Stage 8 — ops entry points + scripts + final cleanup

**Goal:** the three role entry points and the reworked scripts; then the do-over's
final cleanup. Docstrings test must stay green throughout.

| v4 target | from | how |
|---|---|---|
| `ops/generate.py`, `ops/judge.py`, `ops/build.py` | `old/cli/generate.py`, `judge.py`, `build.py` | **lift** the entry points to `ops/` root. |
| `ops/scripts/annotate_docs.py` | `old/scripts/annotate_docs.py` | **rework**: write the 3-label schema (`bin_label`/`scan_label`/`dominant_visual`). Unblocks the annotation pass Stage 3 needs. |
| `ops/scripts/inspect_results.py` | `old/scripts/inspect_results.py` + `old/gates/viewer.py` | **rework**: absorb the retired viewer. |
| `ops/scripts/run_probe.py` | `old/scripts/run_probe.py` | **rework** to the v4 feasibility set. |
| `ops/scripts/resolution_probe.py` | already present | confirm it matches the Phase-1 V-rung probe; re-run if the parser path shifts the sequence profile. |

**Final cleanup (only when all 22 are green + a GPU smoke passes):** delete `old/`;
remove dead code; regenerate `REPO_STRUCTURE.md`'s file map via
`dump_docstrings.py`; reconcile `PROJECT_SPEC.md` / `README.md` / `AGENT_GUIDE.md`
to v4; fold `pivot_v4.md` into `DECISIONS.md`; reconcile the CLAUDE.md pointer
(decisions live in `DECISIONS.md`, not `AGENT_GUIDE.md`) and the Kaya-driver
command form (`python -m ops.kaya.kaya`).

---

## Cross-cutting things not owned by one stage

- **Cache namespace bump** (Stage 0) is the single guard against v3/v4 co-mingle.
- **Machine independence** (Stage 2 keys) is what makes `--failed-only` on the
  supervisor complete the *same* file. Nothing device/hostname/cuda-derived may
  enter a key or the resolved cell list.
- **Telemetry uniformity**: every task writes the full `schema` telemetry every
  run; the truncation fields are a zero-canary, never analysis fields, never
  deleted.
- **Isolated parser envs** are already built on Kaya; Stage 4's `tools/parser.py`
  is where the subprocess/disk boundary to them gets written and GPU-smoked.
- **Deployment resolution** is still pending the resolution probe verdict (job
  `1017226`); Stage 1's `config.VISUAL_RESOLUTION_PRESETS` can be filled with the
  preset names now and the chosen deployment preset set once the probe returns.
</content>
</invoke>
