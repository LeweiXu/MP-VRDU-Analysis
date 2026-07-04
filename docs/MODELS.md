# Models

This file records the reasoner backends and prompt contracts used by the MVP
and Section-2 replications.

## Local VLM Paths

M3 uses the Hugging Face `transformers` load path for the smoke reasoner:

- `transformers==4.57.6`
- `vllm==0.9.2`
- `colpali-engine==0.3.13`
- `marker-pdf==1.10.2`
- `surya-ocr==0.17.1`

`transformers==4.57.6` is the top version currently inside the installed
`colpali-engine` constraint (`transformers>=4.53.1,<4.58.0`) and exposes
`Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`, and
`Qwen3VLProcessor`. That resolves the Stage-1 Qwen3-VL class gap without moving
Marker, Surya, vLLM, or ColPali outside their declared compatibility windows.

The registry dispatches Qwen3-VL local specs to the shared Hugging Face backend:

```text
Qwen/Qwen3-VL-2B-Instruct
Qwen/Qwen3-VL-4B-Instruct
Qwen/Qwen3-VL-8B-Instruct
Qwen/Qwen3-VL-32B-Instruct
```

Section F4 adds the first non-Qwen local backend:

```text
internvl3-8b-local -> OpenGVLab/InternVL3-8B
```

`models.internvl.LocalInternVLBackend` uses the checkpoint's Hugging Face
`chat()` helper behind the same `Reasoner.answer(question, model_input)`
contract as Qwen. Other non-Qwen families remain stubbed until a concrete
backend is added.

## Frozen Prompt

Prompt template version: `m3-qwen3vl-v1`.

```text
You are answering a question about a document.
Use only the provided document evidence. If the evidence does not contain the answer, answer exactly: Not answerable.
Keep the answer concise.

Question:
{question}

Document evidence:
{context}

Answer:
```

`ModelInput.to_local_prompt()` supplies `{context}`. Text parts are inserted as
text. Each `<image>` placeholder is replaced by the corresponding Qwen chat
image block, preserving the order of page images for `TLV` and `V`.

InternVL uses prompt template version `f4-internvl3-v1` with the same document
question/answer instruction and converts image parts into the model's expected
pixel tensor input.

## Accounting

`LocalVLMBackend` records:

- `input_text_tokens`: tokenizer count over the rendered prompt text with image
  placeholders removed.
- `input_visual_tokens`: Qwen image-grid estimate from `image_grid_thw`.
- `output_tokens`: number of generated ids after trimming prompt ids.
- `latency_s`: wall-clock batch=1 generation time around `model.generate()`.

The metadata also records backend, model id, prompt-template version,
`max_new_tokens`, image count, offline/cache mode, and load class.

## Closed Models

Closed or hosted models are comparison and judge backends only. They must remain
behind the same `Reasoner` ABC and consume `ModelInput.to_chat_messages()`; the
pipeline should not import vendor SDKs directly.

## Smoke Commands

Local unit coverage, no weights loaded:

```bash
envs/mpvrdu/bin/python -m pytest tests/test_reasoner.py
```

Kaya GPU smoke, after `setup_env.py` and `prestage.py --smoke` (generate the
headline experiment's predictions on the GPU):

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit kaya/generate.py -- --experiment T1_headline
```
