"""Run local Qwen3-VL checkpoints behind the `Reasoner` interface.

Purpose:
    Implements the Stage-M3 local vision-language backend for the smoke
    reasoner path. The backend resolves repo model specs such as
    `qwen3vl-2b-local` to Hugging Face model ids, formats a single frozen prompt
    template, loads Qwen3-VL through `transformers`, and returns the common
    `schema.Prediction` contract.

Pipeline role:
    `models.get_reasoner()` instantiates `LocalVLMBackend` for the smoke Qwen3-VL
    spec. The orchestrator still talks only to the `Reasoner` ABC and hands over
    a backend-neutral `ModelInput`; this module is the only place that knows
    about Qwen chat templates, processor image binding, or generation calls.

Load path:
    Stage M3 uses `transformers==4.57.6`, where
    `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`, and
    `Qwen3VLProcessor` are present. Compute-node jobs run with the repo-level
    Hugging Face cache and respect `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`.

Arguments:
    None at the command line. Import callers instantiate `LocalVLMBackend` with
    a model spec, optional model id override, optional `max_new_tokens`, and
    optional injected `processor`/`model` fakes for tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question


PROMPT_TEMPLATE_VERSION = "m3-qwen3vl-v1"
FROZEN_PROMPT_TEMPLATE = """You are answering a question about a document.
Use only the provided document evidence. If the evidence does not contain the answer, answer exactly: Not answerable.
Keep the answer concise.

Question:
{question}

Document evidence:
{context}

Answer:"""


MODEL_IDS: dict[str, str] = {
    "qwen3vl-2b-local": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b-local": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b-local": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-32b-local": "Qwen/Qwen3-VL-32B-Instruct",
}


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt text plus ordered image parts after applying the frozen template."""

    text: str
    image_parts: tuple[ImagePart, ...]
    template_version: str = PROMPT_TEMPLATE_VERSION


def model_id_for_spec(spec: str) -> str:
    """Return the Hugging Face model id for a local model spec."""

    try:
        return MODEL_IDS[spec]
    except KeyError as exc:
        raise ValueError(f"unsupported local VLM spec {spec!r}") from exc


def offline_mode_enabled() -> bool:
    """Return whether Hugging Face/transformers offline mode is active."""

    return any(os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def render_prompt(question: Question, model_input: ModelInput) -> RenderedPrompt:
    """Apply the frozen M3 prompt template to one question/model input pair."""

    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    return RenderedPrompt(
        text=FROZEN_PROMPT_TEMPLATE.format(question=question.question.strip(), context=context),
        image_parts=images,
    )


def _image_content(part: ImagePart) -> dict[str, str]:
    """Return a Qwen chat-template content block for one image part."""

    if part.image_path is not None:
        return {"type": "image", "image": str(part.image_path)}
    return {"type": "image", "image": part.data_uri()}


def messages_from_rendered_prompt(rendered: RenderedPrompt) -> list[dict[str, Any]]:
    """Interleave text and image blocks by replacing `<image>` placeholders."""

    pieces = rendered.text.split(IMAGE_PLACEHOLDER)
    expected_images = len(pieces) - 1
    if expected_images != len(rendered.image_parts):
        raise ValueError(
            f"prompt has {expected_images} image placeholders but {len(rendered.image_parts)} images"
        )

    content: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if index < len(rendered.image_parts):
            content.append(_image_content(rendered.image_parts[index]))
    return [{"role": "user", "content": content}]


def _value_from_batch(batch: Any, key: str) -> Any:
    """Read a value from a BatchEncoding/dict-like object."""

    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _shape_last_dim(value: Any) -> int:
    """Return the last dimension of a tensor/list-like value, or zero."""

    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(value)
    return 0


def _count_text_tokens(processor: Any, text: str) -> int:
    """Best-effort text-token count using the processor tokenizer."""

    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    except Exception:
        return max(1, len(text.split()))


def _count_visual_tokens(batch: Any) -> int:
    """Estimate visual tokens from Qwen image grid metadata."""

    grid = _value_from_batch(batch, "image_grid_thw")
    if grid is None:
        return 0
    try:
        rows = grid.tolist()
    except Exception:
        rows = grid
    if not isinstance(rows, list):
        return 0
    if rows and all(isinstance(value, (int, float)) for value in rows):
        rows = [rows]
    total = 0
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        total += int(row[0]) * int(row[1]) * int(row[2])
    return total


def _slice_generated_ids(generated_ids: Any, input_len: int) -> Any:
    """Remove prompt ids from a generated sequence tensor/list."""

    try:
        return generated_ids[:, input_len:]
    except Exception:
        pass
    if isinstance(generated_ids, list) and generated_ids:
        first = generated_ids[0]
        if isinstance(first, list):
            return [first[input_len:]]
        return generated_ids[input_len:]
    return generated_ids


def _generated_token_count(output_ids: Any, answer: str) -> int:
    """Return the number of generated tokens, with a text fallback."""

    count = _shape_last_dim(output_ids)
    return count if count > 0 else max(1, len(answer.split()))


class LocalVLMBackend(Reasoner):
    """Qwen3-VL local backend using Hugging Face `transformers` generation."""

    def __init__(
        self,
        spec: str,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        processor: Any | None = None,
        model: Any | None = None,
        process_vision_info: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.spec = spec
        self.model_id = model_id or model_id_for_spec(spec)
        self.max_new_tokens = int(max_new_tokens)
        self._processor = processor
        self._model = model
        self._process_vision_info = process_vision_info
        self.local_files_only = offline_mode_enabled() if local_files_only is None else bool(local_files_only)

    def _load_components(self) -> tuple[Any, Any]:
        """Load and cache the Qwen processor/model pair."""

        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        import transformers
        from transformers import AutoProcessor

        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText")

        common_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        processor = AutoProcessor.from_pretrained(self.model_id, **common_kwargs)
        model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            **common_kwargs,
        )
        model.eval()
        self._processor = processor
        self._model = model
        return processor, model

    def _vision_inputs(self, messages: list[dict[str, Any]]) -> tuple[Any, Any]:
        """Resolve image/video inputs for the Qwen processor."""

        if self._process_vision_info is not None:
            return self._process_vision_info(messages)
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        rendered = render_prompt(question, model_input)
        messages = messages_from_rendered_prompt(rendered)
        processor, model = self._load_components()

        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._vision_inputs(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        input_len = _shape_last_dim(_value_from_batch(inputs, "input_ids"))
        input_text_tokens = _count_text_tokens(processor, rendered.text.replace(IMAGE_PLACEHOLDER, ""))
        input_visual_tokens = _count_visual_tokens(inputs)

        import torch

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        latency_s = time.perf_counter() - start

        output_ids = _slice_generated_ids(generated_ids, input_len)
        answer = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return Prediction(
            text=answer,
            model_spec=self.spec,
            input_text_tokens=input_text_tokens,
            input_visual_tokens=input_visual_tokens,
            output_tokens=_generated_token_count(output_ids, answer),
            latency_s=latency_s,
            metadata={
                "backend": "hf-transformers",
                "model_id": self.model_id,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "max_new_tokens": self.max_new_tokens,
                "n_image_parts": len(rendered.image_parts),
                "local_files_only": self.local_files_only,
                "load_class": "Qwen3VLForConditionalGeneration",
            },
        )
