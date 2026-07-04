# Models

This file records the reasoner backends and prompt contract used by the MVP.

## Stage M3 Qwen3-VL Path

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

The M3 registry dispatches `qwen3vl-2b-local` to:

```text
Qwen/Qwen3-VL-2B-Instruct
```

The remaining local size specs stay stubbed until their scaling stages wire them
deliberately.

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

Kaya GPU smoke, after `setup_env.py` and `prestage.py --smoke`:

```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --time 00:30:00 --mem 64G kaya/reasoner_smoke.py -- --fresh-cache
```
