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


QUANT_SUFFIXES = ("-4bit", "-8bit")


def model_id_for_spec(spec: str) -> str:
    """Return the Hugging Face model id for a local model spec.

    A trailing quantization suffix (`-4bit`/`-8bit`) is stripped before lookup,
    so a quantized spec resolves to the same base checkpoint.
    """

    base = spec
    for suffix in QUANT_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    try:
        return MODEL_IDS[base]
    except KeyError as exc:
        raise ValueError(f"unsupported local VLM spec {spec!r}") from exc


def offline_mode_enabled() -> bool:
    """Return whether Hugging Face/transformers offline mode is active."""

    return any(os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def hf_cache_dir_from_env() -> str | None:
    """Return the configured Hub cache root used by repo prestage."""

    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _tokenizer_of(processor: Any) -> Any:
    """Return the text tokenizer of a processor (or the processor itself)."""

    return getattr(processor, "tokenizer", processor)


def _truncate_context(
    tokenizer: Any, context: str, n_images: int, max_input_tokens: int, per_image_tokens: int
) -> str:
    """Truncate context text so text + vision tokens stay under the input cap.

    The V100 has only the O(seq^2) math attention kernel (probe 1004834), so a
    ~30k-token `TL` layout dump OOMs. This keeps every image placeholder (so the
    image/placeholder counts still match) and trims the free text to fit the
    budget left after reserving for the images and the prompt template. Truncated
    cells put the images first, then the trimmed text.
    """

    reserve = 256  # prompt template + question + answer cue + generation headroom
    text_budget = max(256, max_input_tokens - n_images * per_image_tokens - reserve)
    text_only = context.replace(IMAGE_PLACEHOLDER, "")
    try:
        ids = tokenizer(text_only, add_special_tokens=False)["input_ids"]
    except Exception:
        ids = tokenizer(text_only)["input_ids"] if hasattr(tokenizer, "__call__") else []
    if not ids or len(ids) <= text_budget:
        return context
    truncated = tokenizer.decode(ids[:text_budget], skip_special_tokens=True)
    return (IMAGE_PLACEHOLDER * n_images) + truncated


def render_prompt(
    question: Question,
    model_input: ModelInput,
    *,
    tokenizer: Any | None = None,
    max_input_tokens: int | None = None,
    per_image_tokens: int = 800,
) -> RenderedPrompt:
    """Apply the frozen M3 prompt template to one question/model input pair.

    When a tokenizer and `max_input_tokens` are given, the context text is
    truncated so the attention sequence stays within the cap (see
    `_truncate_context`); image cells are unaffected unless their text is large.
    """

    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    if tokenizer is not None and max_input_tokens:
        context = _truncate_context(
            tokenizer, context, len(images), max_input_tokens, per_image_tokens
        )
    return RenderedPrompt(
        text=FROZEN_PROMPT_TEMPLATE.format(question=question.question.strip(), context=context),
        image_parts=images,
    )


def _image_content(part: ImagePart, max_pixels: int | None = None) -> dict[str, Any]:
    """Return a Qwen chat-template content block for one image part.

    When `max_pixels` is set it is attached to the block; `qwen_vl_utils`
    downscales the page to that budget before tokenizing, which caps the vision
    tokens per page and keeps the attention sequence from blowing up VRAM.
    """

    block: dict[str, Any] = {"type": "image"}
    block["image"] = str(part.image_path) if part.image_path is not None else part.data_uri()
    if max_pixels is not None:
        block["max_pixels"] = int(max_pixels)
    return block


def messages_from_rendered_prompt(
    rendered: RenderedPrompt, max_pixels: int | None = None
) -> list[dict[str, Any]]:
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
            content.append(_image_content(rendered.image_parts[index], max_pixels))
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
        max_pixels: int | None = None,
        max_input_tokens: int | None = None,
        quantization: str | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        process_vision_info: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.spec = spec
        self.model_id = model_id or model_id_for_spec(spec)
        self.max_new_tokens = int(max_new_tokens)
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self.max_input_tokens = int(max_input_tokens) if max_input_tokens is not None else None
        self.quantization = quantization
        self._processor = processor
        self._model = model
        self._process_vision_info = process_vision_info
        self.local_files_only = offline_mode_enabled() if local_files_only is None else bool(local_files_only)
        self.cache_dir = hf_cache_dir_from_env()

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
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir
        processor = AutoProcessor.from_pretrained(self.model_id, **common_kwargs)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": "auto",
            **common_kwargs,
        }
        quant_config = self._quantization_config()
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        max_memory = self._max_memory_map()
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        model = model_cls.from_pretrained(self.model_id, **load_kwargs)
        model.eval()
        self._processor = processor
        self._model = model
        return processor, model

    @staticmethod
    def _max_memory_map() -> dict[int, str] | None:
        """Reserve per-GPU headroom when sharding across multiple GPUs.

        `device_map="auto"` otherwise fills each GPU with weights and leaves
        almost no room for activations, so a longer-sequence cell's attention
        tips one GPU into OOM (bf16 8B on 2xV100 died at 13GiB-in-use + a 2.9GiB
        alloc). Capping each GPU below its physical size forces the weights to
        leave ~5GiB/GPU free for the activation/KV/attention peak. Single-GPU
        loads (e.g. 4-bit on one V100) return None and use the GPU fully.
        """

        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return None
        reserve_gib = 5
        mapping: dict[int, str] = {}
        for index in range(torch.cuda.device_count()):
            total_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            mapping[index] = f"{max(4, int(total_gib - reserve_gib))}GiB"
        return mapping

    def _quantization_config(self) -> Any | None:
        """Build a bitsandbytes config for a 4-bit/8-bit load, or None for bf16.

        Lets the 8B fit a single 16GB V100 (bf16 weights are ~16GB and OOM one
        GPU; 4-bit is ~7GB, 8-bit ~10GB). Quantized weights are an approximation
        of the pre-registered bf16 model, so this is for single-GPU iteration and
        the appendix quant-sensitivity row, not the main tables. See
        `SINGLE_GPU_8B_FEASIBILITY.md`.
        """

        if self.quantization is None:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"unsupported quantization {self.quantization!r}; expected 4bit or 8bit")

    def free(self) -> None:
        """Drop the loaded model/processor so they release GPU memory.

        Called by the generate loop between experiments; the next `answer` reloads
        lazily. Injected test fakes are cleared too, which is fine for teardown.
        """

        self._model = None
        self._processor = None

    def _vision_inputs(self, messages: list[dict[str, Any]]) -> tuple[Any, Any]:
        """Resolve image/video inputs for the Qwen processor."""

        if self._process_vision_info is not None:
            return self._process_vision_info(messages)
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)

    @staticmethod
    def _sdpa_context() -> Any:
        """Prefer the memory-efficient SDPA kernel over the math kernel.

        The V100 (sm_70) has no FlashAttention-2, so PyTorch's default attention
        can fall back to the math kernel, which materializes the full
        [heads, seq, seq] score matrix. A long multi-page visual sequence then
        tries to allocate tens of GiB and OOMs even after the weights are
        quantized (the score matrix is an activation, unaffected by 4-bit
        weights). The memory-efficient (cutlass) kernel computes attention in
        tiles at O(seq) memory and runs on Volta, so forcing it keeps long-cell
        attention inside 16GB. Falls back cleanly on older torch.
        """

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except Exception:
            from contextlib import nullcontext

            return nullcontext()
        # Priority order: efficient (works on V100) first, math only as a last
        # resort. Flash is listed but is a no-op on Volta.
        return sdpa_kernel(
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        )

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        processor, model = self._load_components()
        per_image_tokens = (self.max_pixels // (28 * 28)) if self.max_pixels else 800
        rendered = render_prompt(
            question,
            model_input,
            tokenizer=_tokenizer_of(processor) if self.max_input_tokens else None,
            max_input_tokens=self.max_input_tokens,
            per_image_tokens=per_image_tokens,
        )
        messages = messages_from_rendered_prompt(rendered, self.max_pixels)

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
        with self._sdpa_context(), torch.inference_mode():
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
                "max_pixels": self.max_pixels,
                "max_input_tokens": self.max_input_tokens,
                "quantization": self.quantization,
                "n_image_parts": len(rendered.image_parts),
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
                "load_class": "Qwen3VLForConditionalGeneration",
            },
        )
