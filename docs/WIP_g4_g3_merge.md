# WIP coordination note: G4 -> G3 merge (temporary)

Heads-up for the agent testing retrieval code paths on kaya. This is a temporary
coordination file, delete it once you've seen it.

## Status

**Part 1 (the G4 -> G3 merge) is LANDED** and the decision is folded into
`docs/DECISIONS.md` ("G4 folded into G3"). Full local suite is green (181 passed).
**Part 2 (the big per-sweep YAML expander) is still deferred** until your retrieval
testing wraps (see "What I'm deliberately NOT doing yet" below).

## What I did

Merged the G4 classifier task into G3 (folded the one-shot document classifier
into `G3_hallucination.run_side`) and deleted `G4_classifier_pricing` as a task.
Routing stays a build-time table over G1's cached rows; the classifier is the only
GPU work it needed. The short version:

- Driver hands `run_side` the **full** corpus + smoke `limit` (each side writer
  re-resolves its own scope).
- G3 gains `side_artifact = "classifier.jsonl"` and a `run_side` that classifies
  G1's answerable doc_type-sampled docs when `config.classifier_spec` is set.
- G4 task module, registry entry, and reporting/probe references removed.

## Files I'm touching (so we don't collide)

- `experiments/engine/driver.py` - the `run_side` call site (line ~337) only.
- `experiments/tasks/base.py` - `run_side` signature gains a `limit` kwarg.
- `experiments/tasks/G3_hallucination.py` - adds the classifier side-artifact.
- `experiments/tasks/G4_classifier_pricing.py` - deleted.
- `experiments/registry.py`, `config.py`, `reporting/build.py`,
  `reporting/tables/routing.py`, `ops/scripts/final_probe.py`, and a few tests/docs.
- `experiments/tasks/G2_retrieval.py` - **one line added to `run_side`** to
  re-filter to G2's answerable pool now that it receives the full corpus. I am NOT
  touching G2's retrieval logic, the retriever registries, `side_artifacts.write_retrieval_eval`,
  or `matched_cross_sweep_cells`. The retrieval benchmark behavior is unchanged.

## What I'm deliberately NOT doing yet

The big per-sweep YAML expander (`ops/specs/target_architecture.yaml` / the HANDOFF
rework) reworks G2's retrieval wiring (method sets from config, `build_retrievers`
from config, `joints: matched`, etc.). That is **Part 2 and deferred until after
your retrieval testing wraps**, specifically so it does not collide with your work.
If you need me to hold off on the G2 `run_side` one-liner too, ping me.
