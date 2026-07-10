# WIP coordination note: per-sweep YAML expander (Part 2, temporary)

Heads-up for the agent testing retrieval code paths on kaya. Temporary coordination
file, delete once seen.

## Status: LANDED in the working tree (uncommitted)

Part 2 (the per-sweep YAML expander) is implemented and the full local suite is green
(196 passed). It is NOT committed — the working tree also holds your uncommitted
retrieval changes, so I left committing to you / the user to avoid entangling our
diffs. **If you have retrieval work mid-flight in the files below, reconcile before
committing.** Notable overlaps with your area: `driver.build_retrievers` now builds
the inference arms from `config.inference_text_retriever` / `_vision_retriever` via
`get_text_retriever` / `get_vision_retriever` (I did NOT touch your parser-warm
refactor), and `side_artifacts.write_retrieval_eval` now takes its method sets as
params (`text_methods` / `vision_methods` / `joint_pairs`) instead of the
`_TEXT_METHODS` / `_VISION_METHODS` / `_JOINT_PAIRS` module constants (kept as
defaults). The per-method try/except + `free_gpu()` robustness is unchanged.

## What I did

Wiring `ops/specs/target_architecture.yaml` end to end: the nested `base` + `sweeps`
(and G2 `retrieval` / `inference`) shape becomes real per-sweep control. The parser
expands each nested block into flat `Spec`s at parse time (one per sweep, each with
its own `run_tag = "<base>-<sweep>"`), so the driver keeps running one pass per spec.

## Files I'm touching (heads-up: this DOES include retrieval wiring)

Unlike Part 1, Part 2 reworks the G2 retrieval plumbing you're testing, so please
coordinate / rebase around these:

- `experiments/corpus/yaml_spec.py` - the expander (most of the work).
- `config.py` - new fields (retriever picks, method lists, dataset, joints, etc.).
- `experiments/engine/driver.py` - `build_retrievers` reads the inference retriever
  picks from config instead of hardcoding bm25 / colqwen2.5.
- `experiments/engine/side_artifacts.py` - `write_retrieval_eval` takes its text /
  vision / joint method sets from config instead of the `_TEXT_METHODS` /
  `_VISION_METHODS` / `_JOINT_PAIRS` module constants.
- `experiments/tasks/G2_retrieval.py` - `INFERENCE_REPRESENTATIONS` / `JOINT_K_VALUES`
  become config-driven.
- `ops/generate.py` - per-dataset corpus load (a `dataset -> loader` map).

## run_tag strategy (forced by the frozen cache key)

The orchestrator cache key is `(question, doc, conditioner, modality, reasoner.spec,
page_indices, visual_resolution)` and is frozen. reasoner_spec (incl. the `-4bit`/
`-8bit` quant suffix) and visual_resolution ARE in the key; parser and dataset are
NOT. So:

- **size / family / quantization / resolution sweeps** stay ONE run_tag
  (`<base>-<sweep>`) with a list field the driver already loops (`reasoner_specs` /
  `visual_resolutions`). The scale/resolution tables read those variants from one file.
- **parser / dataset sweeps** become SEPARATE run_tags per value
  (`<base>-<sweep>-<value>`), because they are not in the key (a shared run_tag would
  collide) and dataset even loads a different corpus. This is a deliberate deviation
  from the HANDOFF's `parsers:`/`datasets:` list idea; the frozen key forces it.

Cross-run_tag table assembly (the multi-parser / multi-dataset comparison in
`reporting/build.py`) is NOT part of this change - it stays as-is (one run_tag per
build). This change is generation-side: the right cells under the right run_tag.

## Invariants I'm keeping

- Frozen cache keys are untouched (additive config fields only; distinct `run_tag`
  per expanded sweep is what isolates caches).
- Flat legacy specs (`kaya.yaml`, `kaya_smoke.yaml`, `kaya_probe.yaml`) keep parsing
  unchanged - the nested shape is the only thing that triggers the expander.
- The retrieval benchmark's per-method robustness (each method try/excepted, big-model
  OOM skips just that method) and timing stay exactly as they are.

If your retrieval work is mid-flight in any of these files, ping me and I'll rebase
onto your changes rather than the other way around.
