# Handoff: per-sweep YAML config architecture

Date: 2026-07-10. Goal: make every sub-generation sweep in G1/G2/G3 individually
controllable from the spec YAML. Right now some sweeps are list fields, some are
single-value-per-run (so you need a separate run block to sweep them), and the G2
retriever set is hardcoded and not spec-controllable at all. The target is that a
spec spells out every axis explicitly, and you dial each one up or down (or off) on
its own.

The mock-up of the target schema is checked in at **`ops/specs/target_architecture.yaml`**
(heavily commented, not wired to anything yet). Read that first; it's the concrete
shape we're building toward. This file is the "how to build it".

Note: the previous HANDOFF (the full G1-G3 doc_type-sampled Kaya build) is done. Its
decisions live in `docs/DECISIONS.md` under "Full G1-G3 doc_type-sampled Kaya
submission (2026-07-10)". The G1/G2/G3 code, the six-method retrieval side-artifact,
retrieval timing, and `kaya.yaml`/`kaya_smoke.yaml` are all landed and pytest is
green; this handoff is only about the config/YAML rework on top of that.

## The idea in one paragraph

Each main task (G1/G2/G3) gets a `base:` block of baseline scalars plus named
sub-sweep blocks. A sub-sweep varies exactly ONE axis and holds everything else at
`base` (this is the pivot's "one field changed each"; it deliberately avoids a full
cross-product of coupled axes). You control a sweep by editing its list: multi-value
runs it, one value collapses it to the baseline, deleting the block (or
`enabled: false`) turns it off. Each sub-sweep is its own cache namespace,
`run_tag = "<base run_tag>-<sweep name>"`.

## Design decisions already made (confirm before deviating)

1. **Independent per-axis sub-sweeps, NOT one big cross-product.** A flat "list per
   axis, multiply everything" both explodes (32B x 4bit x high-res x mineru x
   longdocurl) and wastes compute (T/TL are identical across resolutions). If the
   user ever wants true cross-products (e.g. size x resolution together) that's a
   different engine; ask.
2. **Coupled sweeps carry their own `representations` override.** `resolution` runs
   at `[TLV, V]` (resolution is meaningless without a vision channel); `parser` runs
   at `[TL, TLV]` (parser only feeds the parsed rungs). See the mock-up.
3. **G2 is two explicit stages.** `retrieval:` is the accuracy benchmark (no
   reasoner) that sweeps every listed method x k into the side-artifact.
   `inference:` is the reasoner k-sweep that picks WHICH retrievers (from the
   retrieval stage) to actually feed the model, at TLV and V. The inference
   retrievers must be a subset of the retrieval-stage lists, because inference reuses
   the stage-1 cached rankings.
4. **`joints: matched`** auto-pairs by cost tier: cheap (bm25 | colmodernvbert), mid
   (bge-m3 | colqwen2.5), expensive (qwen3-embedding | colqwen3). Or list explicit
   `[[text, vision], ...]` pairs, or `[]` to skip joints.

## What exists today (starting point)

- **`experiments/corpus/yaml_spec.py`** — `ALLOWED_KEYS`, the `Spec` dataclass,
  `parse_spec`, `parse_specs` (base+runs merge, run keys win), `config_from_spec`,
  `corpus_limit`, `load_yaml_specs`. Today a spec is flat: one task, list fields
  `representations` / `reasoner_specs` / `visual_resolutions` / `k_values`, and
  single-value `parser` / `quantization` / `visual_resolution`. `dataset` is NOT in
  `ALLOWED_KEYS` (so it can't be swept from a spec yet, even though the config field
  exists).
- **`config.py` `ExperimentConfig`** — already has `dataset` (default "mmlongbench"),
  `reasoner_spec`/`reasoner_specs`, `quantization` (single), `parser_tool`,
  `visual_resolution`/`visual_resolutions`, `k_values`, `representations`. Quant is
  folded into the reasoner spec string in `__post_init__` (so a quantized run caches
  apart). There are no retriever fields at all.
- **`experiments/engine/driver.py` `build_retrievers`** (~line 110) — hardcodes
  `text = MemoizedRetriever(Bm25Retriever(...))`, `vision =
  MemoizedRetriever(ColQwen25Retriever(...))`. This is what the G2 inference stage
  uses. Registries `get_text_retriever` / `get_vision_retriever` already exist in
  `retrievers/text.py` / `retrievers/vision.py`.
- **`experiments/engine/side_artifacts.py` `write_retrieval_eval`** — the six-method
  benchmark. The method set is hardcoded module constants `_TEXT_METHODS`,
  `_VISION_METHODS`, `_JOINT_PAIRS`, and the k lists come in as `single_ks`/`joint_ks`
  kwargs from the task. Already robust per-method (each try/excepted, big-model OOM
  skips just that method).
- **`experiments/tasks/G2_retrieval.py`** — `INFERENCE_REPRESENTATIONS = ("TLV","V")`,
  `JOINT_K_VALUES = (1,3,5)` as module constants; `generation_cells` builds the
  inference cells via `matched_cross_sweep_cells` (bm25 + colqwen2.5 + joint);
  `run_side` calls `write_retrieval_eval`.
- **`experiments/tasks/base.py`** — `Retrievers` is a 2-slot `(text, vision)`
  dataclass; `matched_cross_sweep_cells(questions, *, retrievers, ks, joint_ks,
  representations)`.
- **`ops/generate.py`** — loads the dataset once for the first config
  (`load_mmlongbench(first_config.paths.data_dir)`), then runs each spec.

## What to build

### 1. `yaml_spec.py` — the parser is where most of the work is

Expand a `base` + `sweeps` (and G2's `retrieval`/`inference`) block into a list of
plain flat `Spec`s AT PARSE TIME, one per sweep value-set, each with a derived
`run_tag`. If you do this, the driver needs no changes: it already runs one pass per
spec. Concretely:

- Add the new keys to `ALLOWED_KEYS`: `base`, `sweeps`, `retrieval`, `inference`,
  `dataset`, `datasets`, `quantizations`, `parsers`, `text_retrievers`,
  `vision_retrievers`, `joints`, `joint_k_values`, `prompt_modes`, `similarity_k`,
  `retriever`, `enabled`.
- **Backward compatibility:** if a run block has none of `base`/`sweeps`/`retrieval`/
  `inference`, treat it as a flat spec exactly as today, so the current `kaya.yaml`
  and `kaya_smoke.yaml` keep working unchanged. Only the new nested shape triggers the
  expander. (Decide with the user whether to also migrate kaya.yaml to the new shape,
  or leave it flat.)
- For a G1-style block: emit the `base` as one Spec (`run_tag = base tag`), then for
  each enabled sweep emit one Spec that overlays that sweep's axis list (and any
  `representations` override) onto the base, with `run_tag = "<base>-<sweep>"`. A
  sweep whose list has a single value that equals base can be skipped (it would be a
  duplicate of base).
- For a G2 block: emit one Spec carrying both the `retrieval` method lists (for the
  side-artifact) and the `inference` retriever picks + reps. These two live on the
  same task/run so they share the retrieval cache; don't split them into separate
  run_tags or the inference stage won't find the stage-1 rankings.
- `config_from_spec` maps all the new fields onto `ExperimentConfig`.

### 2. `config.py` — new fields

Add: `quantizations: tuple[str,...]`, `parsers: tuple[str,...]`, `datasets:
tuple[str,...]`, and the G2 retriever fields — `text_retrievers`,
`vision_retrievers`, `joints` (either the literal "matched" or a tuple of pairs),
`joint_k_values`, plus the inference picks `inference_text_retriever`,
`inference_vision_retriever`, `inference_joint: bool`, and `inference_representations`.
Keep the singular `quantization`/`parser_tool`/`dataset`/`visual_resolution` as the
baseline values the sweeps vary. Quant-in-spec-string folding already exists; extend
it if a `quantizations` sweep is expanded (each expanded Spec still sets a single
`quantization`).

### 3. `driver.build_retrievers` — build from the spec, not hardcoded

Read `config.inference_text_retriever` / `config.inference_vision_retriever` (default
bm25 / colqwen2.5) via the `get_*_retriever` factories, wrap in `MemoizedRetriever`
sharing `cache/retrieval`. The joint arm is still `JointTopK(text, vision)`.

### 4. `side_artifacts.write_retrieval_eval` — method lists from config

Replace the `_TEXT_METHODS` / `_VISION_METHODS` / `_JOINT_PAIRS` module constants with
values from the config: `config.text_retrievers`, `config.vision_retrievers`, and the
joints (expand "matched" to the tier pairs, or use the explicit list). Keep the
per-method try/except + `free_gpu()`. Everything else (timing, scoring, robustness)
stays.

### 5. `G2_retrieval.py` — read inference reps/retrievers from config

`generation_cells` should pass `config.inference_representations` (default
`("TLV","V")`) and the joint on/off from config into `matched_cross_sweep_cells`, and
`run_side` should pass `config.text_retrievers` / `config.vision_retrievers` /
`config.joints` (or new single_ks/joint_ks) into `write_retrieval_eval`. The two
module constants become config-driven.

### 6. `generate.py` — per-dataset corpus load (the dataset sweep)

Today the corpus is loaded once from mmlongbench. When a `dataset` axis is present,
each expanded Spec names its dataset; load the corpus for that Spec's dataset (there's
a loader for LongDocURL alongside `load_mmlongbench`; wire a small dataset->loader
map). Cache the load per dataset so two runs on the same dataset don't reload.

## Semantic rules to enforce (and test)

- Collapse-to-disable: a sweep list with one element (or absent) contributes only the
  baseline.
- Coupled reps: `resolution` sweep uses `[TLV, V]`, `parser` sweep uses `[TL, TLV]`,
  unless the block overrides `representations` explicitly.
- Inference retrievers must be a subset of the retrieval-stage lists (validate and
  raise a clear `SpecError` if not; otherwise the inference stage points at rankings
  that were never computed).
- Frozen cache keys still hold (CLAUDE.md): new conditioner names and additive config
  fields are fine; don't reshape the key. Distinct `run_tag` per expanded sweep keeps
  caches from colliding.

## Testing

- `tests/test_yaml_spec.py` — add cases for the expander: a G1 block with two sweeps
  expands to base + one Spec per sweep with the right run_tags and overlaid axes; a
  collapsed sweep (single value) does not duplicate base; a flat legacy spec still
  parses unchanged; the inference-subset validation raises on a bad pick.
- Keep the existing specs (`kaya.yaml`, `kaya_smoke.yaml`) parsing green (backward
  compat).
- Whole suite green before submitting anything.

## Not in scope for this handoff

Getting the current runs green on Kaya is a separate, in-progress thread (the smoke
`kaya_smoke.yaml` shakeout and then the real `kaya.yaml`). That uses the existing flat
specs and does not depend on this rework. Don't block the config rework on it or vice
versa.
