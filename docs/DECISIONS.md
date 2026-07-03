# Decision Log

This log records decisions that later stages must treat as fixed unless a
checkpoint explicitly changes them.

## Fixed for v1

- Dataset: MMLongBench-Doc only. Other datasets are deferred to optional Stage
  10 because MMLongBench-Doc is the only v1 source with document type,
  evidence-modality labels, gold evidence pages, and the unanswerable signal.
- Evaluated model family: Qwen3-VL at 2B, 4B, 8B, and 32B, with 8B as the
  center configuration. Stage 1 must confirm concrete checkpoint availability
  and 32B feasibility on Kaya before full runs depend on it.
- Model swap point: all reasoners sit behind one `Reasoner` interface, with
  local-weight and HTTP-API backends. Closed models are allowed for comparison
  and judging, not for the deployment recommendation.
- Pipeline stages mirror the paper: input conditioning, representation,
  reasoning, and scoring, with retrieval and document-type classification as
  covariates.
- Representation ladder: `T`, `TL`, `TLV`, and `V`. Text/layout channels must
  be produced by modular non-VLM tools; only `TLV` and `V` may attach images.
- Distractor-burying is in scope as a deployment instrument: gold pages remain
  present and same-corpus distractors test how much irrelevant context the
  model tolerates.
- Paths are root-relative. `.cache/`, `.data/`, `envs/`, `results/`, and
  `logs/` live under the repository root on both local and Kaya.
- Kaya execution uses a two-machine model: local edits and sync; Kaya login for
  environment/model/data staging; Kaya compute for offline GPU jobs.

## Stage 0 implementation notes

- `data/` is reserved for the importable Python package. Downloaded datasets,
  synthetic samples, and rendered pages live under `.data/` so artifact storage
  cannot conflict with importable code.
- `requirements.txt` is a declaration only at Stage 0. Stage 1 must validate
  the pins against Kaya's actual module, CUDA, and GPU partition configuration.
- `kaya/` remains the standalone reference/demo kit. Pipeline operations use
  `scripts/kaya/`.

## Open items Stage 1 confirms

- MMLongBench-Doc fetch/render path and field parsing.
- Whether MMLongBench-Doc has a real scanned-document slice or requires
  synthetic degradation for text-recovery analysis.
- Whether in-page evidence boxes exist in v1; expected outcome is page-level
  crops only.
- Qwen3-VL 2B, 4B, 8B, and 32B availability and Kaya feasibility.
- Local-backend and API-backend instantiation through the same `Reasoner`
  contract.
- ColPali/ColQwen and BM25+BGE feasibility on target hardware.
- Native unanswerable count and the exact abstention definition.
- MMLongBench-Doc `doc_type` distribution and the text/in-between/visual
  spectrum mapping.
- Confirmed Kaya module names, CUDA version, GPU partition, and GPU request
  syntax.
