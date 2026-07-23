"""Qwen3-VL reasoner backend on HF transformers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question


from config import DEFAULT_PROMPT_MODE, PROMPT_MODES

PROMPT_TEMPLATE_VERSION = "qwen3vl-v1"
PROMPT_HEADER = "You are answering a question about a document."
PROMPT_BODY = "Question:\n{question}\n\nDocument evidence:\n{context}\n\nAnswer:"
# The instruction used when a caller does not set one (the answerable-cell default).
DEFAULT_INSTRUCTION = PROMPT_MODES[DEFAULT_PROMPT_MODE]


MODEL_IDS: dict[str, str] = {
    "qwen3vl-2b-local": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b-local": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b-local": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-32b-local": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3vl-8b-thinking-local": "Qwen/Qwen3-VL-8B-Thinking",
}

QUANT_SUFFIXES = ("-4bit", "-8bit")


@dataclass(frozen=True)
class RenderedPrompt:
    """Prompt text plus ordered image parts after applying the frozen template."""

    text: str
    image_parts: tuple[ImagePart, ...]
    template_version: str = PROMPT_TEMPLATE_VERSION


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
    """Return the configured Hub cache root used by prestage."""

    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def render_prompt(question: Question, model_input: ModelInput, instruction: str | None = None) -> RenderedPrompt:
    """Apply the prompt template with a chosen instruction preamble.

    The full context is fed: there is no input-token cap, so nothing is trimmed.
    `instruction` is the abstention/guidance preamble; None uses the default
    (abstention-targeted) instruction, and an empty string gives no guidance.
    """

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    header = PROMPT_HEADER if not instruction.strip() else f"{PROMPT_HEADER}\n{instruction.strip()}"
    text = f"{header}\n\n" + PROMPT_BODY.format(question=question.question.strip(), context=context)
    return RenderedPrompt(text=text, image_parts=images)


def _image_content(part: ImagePart, max_pixels: int | None = None) -> dict[str, Any]:
    """Return a Qwen chat-template content block for one image part.

    When `max_pixels` is set it is attached to the block; `qwen_vl_utils`
    downscales the page to that budget before tokenizing, which caps the vision
    tokens per page.
    """

    block: dict[str, Any] = {"type": "image"}
    block["image"] = str(part.image_path) if part.image_path is not None else part.data_uri()
    if max_pixels is not None:
        block["max_pixels"] = int(max_pixels)
    return block


def messages_from_rendered_prompt(rendered: RenderedPrompt, max_pixels: int | None = None) -> list[dict[str, Any]]:
    """Interleave text and image blocks by replacing `<image>` placeholders."""

    pieces = rendered.text.split(IMAGE_PLACEHOLDER)
    expected_images = len(pieces) - 1
    if expected_images != len(rendered.image_parts):
        raise ValueError(f"prompt has {expected_images} image placeholders but {len(rendered.image_parts)} images")

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


class _FirstTokenTimer:
    """Streamer that records when generation emits its first decoded token.

    `generate` calls `put` once with the prompt ids, then once per new token, so
    the second call marks the first generated token. Time-to-first-token is a
    close proxy for prefill (ingesting the representation) and lets one `generate`
    yield both the prefill and decode split, with no second prefill forward.
    """

    def __init__(self) -> None:
        self.first_token_time: float | None = None
        self._calls = 0

    def put(self, _value: Any) -> None:
        self._calls += 1
        if self._calls == 2 and self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        return None


class Qwen3VLBackend(Reasoner):
    """Qwen3-VL local backend using Hugging Face `transformers` generation."""

    #: Recorded per cell; a subclass with its own prompt assembly overrides both
    #: `render` and this id so its cells are distinguishable in metadata.
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION

    def __init__(
        self,
        spec: str,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        max_pixels: int | None = None,
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

        common_kwargs = {"trust_remote_code": True, "local_files_only": self.local_files_only}
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir
        processor = AutoProcessor.from_pretrained(self.model_id, **common_kwargs)
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto", **common_kwargs}
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
        almost no room for activations, so a longer-sequence cell's attention tips
        one GPU into OOM. Capping each GPU below its physical size forces the
        weights to leave headroom for the activation/KV/attention peak. Single-GPU
        loads return None and use the GPU fully.
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

        Lets the 8B fit a single 16GB GPU (bf16 weights are ~16GB; 4-bit ~7GB,
        8-bit ~10GB). Quantized weights approximate the bf16 model, so this is for
        single-GPU iteration and the quantization sweep.
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

        Called by the run loop between specs; the next `answer` reloads lazily.
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

        A GPU without FlashAttention-2 can fall back to the math kernel, which
        materialises the full score matrix; a long visual sequence then OOMs. The
        memory-efficient (cutlass) kernel computes attention in tiles at O(seq)
        memory, so forcing it keeps long-cell attention bounded. Falls back
        cleanly on older torch.
        """

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except Exception:
            from contextlib import nullcontext

            return nullcontext()
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])

    def render(self, question: Question, model_input: ModelInput) -> RenderedPrompt:
        """Assemble this backend's prompt; the per-model override point."""

        return render_prompt(question, model_input, self.prompt_instruction)

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        processor, model = self._load_components()
        rendered = self.render(question, model_input)
        messages = messages_from_rendered_prompt(rendered, self.max_pixels)

        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._vision_inputs(messages)
        inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        input_len = _shape_last_dim(_value_from_batch(inputs, "input_ids"))
        total_text_tokens = _count_text_tokens(processor, rendered.text.replace(IMAGE_PLACEHOLDER, ""))
        total_visual_tokens = _count_visual_tokens(inputs)

        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        with self._sdpa_context(), torch.inference_mode():
            # One generate for the whole cell. A streamer marks the first decoded
            # token, so time-to-first-token approximates prefill (ingesting the
            # representation) and the remainder is decode, with no second prefill
            # forward paid on every cell.
            timer = _FirstTokenTimer()
            gen_start = time.perf_counter()
            generated_ids = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=timer
            )
            if cuda:
                torch.cuda.synchronize()
            latency_s = time.perf_counter() - gen_start

        peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if cuda else 0
        prefill_latency_s = (timer.first_token_time - gen_start) if timer.first_token_time is not None else latency_s
        decode_latency_s = max(0.0, latency_s - prefill_latency_s)

        output_ids = _slice_generated_ids(generated_ids, input_len)
        answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        output_tokens = _generated_token_count(output_ids, answer)
        # Output-side truncation canary: greedy decoding stops early only on EOS,
        # so hitting the budget means the answer was cut, not finished. The
        # input-side tokens_dropped/truncation_occurred fields cannot see this.
        output_truncated = output_tokens >= self.max_new_tokens

        return Prediction(
            text=answer,
            model_spec=self.spec,
            total_text_tokens=total_text_tokens,
            total_visual_tokens=total_visual_tokens,
            text_tokens_fed=total_text_tokens,  # no cap: everything is fed
            output_tokens=output_tokens,
            latency_s=latency_s,
            prefill_latency_s=prefill_latency_s,
            decode_latency_s=decode_latency_s,
            peak_vram_bytes=peak_vram_bytes,
            metadata={
                "backend": "hf-transformers",
                "model_id": self.model_id,
                "prompt_template_version": self.prompt_template_version,
                "max_new_tokens": self.max_new_tokens,
                "output_truncated": output_truncated,
                "max_pixels": self.max_pixels,
                "quantization": self.quantization,
                "n_image_parts": len(rendered.image_parts),
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
                "load_class": "Qwen3VLForConditionalGeneration",
            },
        )
