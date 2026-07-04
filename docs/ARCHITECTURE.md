# Architecture

How the code tree maps to the paper, and the interfaces frozen at Stage 3. If
you know `PROJECT_SPEC.md`, you should recognise the pipeline in the file tree:
the four pipeline stages and the two covariates are first-class objects with the
same names.

> This file is the structural contract. Per the global rule, once it exists,
> changing it (new module, moved files, restructuring) is a confirm-first step,
> not a silent edit. The interfaces listed under "Frozen interfaces" below are
> frozen at the end of Stage 3; changing any of them is a checkpoint conversation
> recorded in `docs/DECISIONS.md`.

## The data flow

A `Question` (from the data layer) is resolved to pages by an `InputConditioner`,
those pages are rendered, encoded by a `Representation` into a `Payload`, mapped
to a backend-agnostic `ModelInput`, answered by a `Reasoner`, and scored by a
`Judge`. A `Retriever` and a `DocTypeClassifier` are covariates that feed and
annotate the flow.

```
Question
   │  InputConditioner.condition(question, page_count)      pipeline/conditioner.py   (Stage A)
   ▼
PageSet ──render──► [Page, …]                               data/render.py
   │  Representation.build(pages)                           pipeline/representation.py (Stage B)
   ▼
Payload  (modality + ordered text/image parts)             schema.py
   │  ModelInput.from_payload(payload)                      models/payload.py
   ▼
ModelInput ── to_chat_messages() / to_local_prompt() ──►   (the swap boundary)
   │  Reasoner.answer(question, model_input)                pipeline/reasoner.py       (Stage C)
   ▼
Prediction  (answer + split token/latency cost)            schema.py
   │  Judge.score(question, prediction)                     pipeline/judge.py          (Stage D)
   ▼
Score ──► ResultRow (cached)                                pipeline/orchestrator.py
```

## Tree ↔ paper mapping

| Path | Paper role |
|---|---|
| `schema.py` | The data contracts: `Question`, `PageSet`, `Page`, `Payload`, `Prediction`, `Score`, and the shared `TextPart`/`ImagePart`. |
| `config.py` | `ExperimentConfig` (dataset, reasoner + scaling specs, conditions, k / burying grids, representation ladder, sufficiency margin) and root-relative `ProjectPaths`. |
| `data/loader.py`, `data/render.py` | The single-dataset loader and the PDF→pages substrate shared by representation and retrieval. |
| `pipeline/conditioner.py` | Stage A — input conditions: `Oracle`, `RetrievedTopK`, `FullDoc`, `BuriedOracle`. |
| `pipeline/representation.py` | Stage B — the `T`/`TL`/`TLV`/`V` modality ladder; enforces the modality boundary. |
| `pipeline/reasoner.py` | Stage C — the backend-agnostic `Reasoner` ABC (the swap point). |
| `pipeline/judge.py` | Stage D — the uniform judge protocol. |
| `pipeline/orchestrator.py` | Composes A→B→C→D for one cell; owns the caching contract. |
| `models/payload.py` | `ModelInput` + the two adapters (`to_chat_messages`, `to_local_prompt`) that make the reasoner family swappable. |
| `models/__init__.py` | `get_reasoner(spec)` registry — maps a spec to a backend. |
| `covariates/retriever.py` | Retrieval covariate (RQ7 decomposition, RQ2 modality divergence). |
| `covariates/classifier.py` | Document-type classifier covariate (RQ3 routing). |
| `tools/{text,layout,visual}.py` | Modular non-VLM channel functions the representation composers call. |
| `metrics/*` | Accuracy, retrieval, abstention, cost, and the sufficiency frontier. |
| `experiments/*`, `cli/*` | Config → cells → cached predictions → result CSVs. |

## The `ModelInput` contract (why the family is swappable)

`models/payload.py::ModelInput` is a backend-agnostic container of ordered text
and image parts. A `Representation` produces a `Payload`; `ModelInput.from_payload`
maps it across. Two adapters render it for either backend:

- `to_chat_messages()` → an OpenAI/Gemini/Anthropic-style `messages` array with
  base64 `image_url` parts (the HTTP API backend consumes this).
- `to_local_prompt()` → a prompt string with one `<image>` placeholder per image
  plus the ordered image parts (the local vLLM/HF backend consumes this).

The pipeline never imports a concrete backend. It asks `get_reasoner(spec)` for a
`Reasoner` and hands it a `ModelInput`. Adding Qwen3-VL sizes, InternVL/Gemma, or
GPT/Gemini is a new registry entry reading one of these two adapters — no pipeline
code changes. Images are carried by reference (`ImagePart.image_path`) or inline
(`ImagePart.data`); `read_bytes()`/`data_uri()` hide the difference so adapters
never care where an image came from.

## The modality-boundary rule (structural)

Only `TLV` and `V` may attach images; `T` and `TL` attach strings only. This is
enforced twice: the composers structurally add image parts only for `TLV`/`V`,
and `Payload.__post_init__` re-checks it so a future bug cannot leak an image
into a text-only condition. The `T`/`TL` channels are produced by modular non-VLM
tools (`tools/text.py`, `tools/layout.py`); the VLM is confined to the visual
channel.

## The caching contract (frozen at Stage 3)

Every cell is keyed by `make_cache_key`, a SHA-256 over
`{question_id, doc_id, condition, representation, model_spec, judge_spec, dpi}`,
and stored as one jsonl line in `results/cache/orchestrator/results.jsonl`
(`pipeline/orchestrator.py::ResultCache`). `run_cell` returns the cached row on a
hit, so re-running is idempotent and a fresh orchestrator resumes from disk with
no recomputation. This is what makes the multi-condition sweep affordable. The
model spec is part of the key, so the scaling sweep and any family swap produce
distinct, mergeable rows.

## Root-relative paths and local/Kaya execution

`config.py` derives `ROOT` by walking up to `docs/implementation_plan.md` and
defaults every artifact path under it (`hf_home=<root>/.cache`,
`data_dir=<root>/.data`, `results_dir=<root>/results`, `env_dir=<root>/envs`). No
absolute machine paths appear in code, so the same config runs locally or on a
Kaya compute node and the caches/results produced in either place are mergeable.

Local is for editing and small/cheap runs; Kaya is for the GPU-heavy grid. The
two-machine flow is driven from local via `envs/mpvrdu/bin/python -m kaya.kaya`
(`push` / `run` / `submit` / `pull`); the mechanics and rules live in `kaya/`
(`KAYA_AGENT_GUIDE.md`, `KAYA_USER_GUIDE.md`) and section 2b of
`docs/implementation_plan.md`. `ProjectPaths` is overridable (e.g. tests point
`data_dir`/`cache_dir` at a fixture) without touching any pipeline code.

## Frozen interfaces (do not change without a checkpoint)

- `schema.py`: `Question`, `PageSet`, `Page`, `TextSpan`, `TextPart`, `ImagePart`,
  `Payload`, `Prediction`, `Score`.
- `models/payload.py`: `ModelInput` and its `to_chat_messages` / `to_local_prompt`
  / `from_payload` contract.
- `pipeline/conditioner.py`: `InputConditioner.condition(question, page_count)`.
- `pipeline/representation.py`: `Representation.build(pages)`.
- `pipeline/reasoner.py`: `Reasoner.answer(question, model_input)`.
- `pipeline/judge.py`: `Judge.score(question, prediction)`.
- `covariates/retriever.py`: `Retriever.retrieve(question, page_count, k)`.
- `covariates/classifier.py`: `DocTypeClassifier.classify(question)`.
- `pipeline/orchestrator.py`: the cache key and `ResultRow` shape.

Everything after Stage 3 fills implementations behind these; any pressure to
change one is a checkpoint conversation recorded in `docs/DECISIONS.md`.
