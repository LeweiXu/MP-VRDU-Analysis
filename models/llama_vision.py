"""Llama-3.2-Vision reasoner backend: cross-attention image conditioning, the
architectural contrast to the in-sequence vision tokens of Qwen3-VL/InternVL."""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Any

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question

MODEL_IDS: dict[str, str] = {
    # Gated repo: prestage needs HF_TOKEN.
    "llama3.2-11b-vision-local": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

from config import DEFAULT_PROMPT_MODE, PROMPT_MODES

PROMPT_TEMPLATE_VERSION = "llama3.2-vision-v1"
PROMPT_HEADER = "You are answering a question about a document."
PROMPT_BODY = "Question:\n{question}\n\nDocument evidence:\n{context}\n\nAnswer:"
DEFAULT_INSTRUCTION = PROMPT_MODES[DEFAULT_PROMPT_MODE]

# Mllama tiles each image into up to 4 fixed 560px tiles regardless of input
# size; one tile is (560 / 14) ** 2 = 1600 patches plus a separator token. The
# resolution presets therefore act as a FIDELITY ladder (the page is downscaled
# to the pixel budget before the processor), not a token-cost ladder: compare
# cost across families via the recorded `total_visual_tokens`, never the preset
# name. And because conditioning is cross-attention, these vision tokens never
# enter the text sequence at all, so prefill scaling is structurally different
# from the in-sequence families; that contrast is the reason this backend exists.
LLAMA_TILE_SIZE = 560
LLAMA_MAX_TILES = 4
LLAMA_TOKENS_PER_TILE = (LLAMA_TILE_SIZE // 14) ** 2 + 1


def model_id_for_spec(spec: str) -> str:
    """Return the Hugging Face model id for a Llama-Vision model spec."""

    try:
        return MODEL_IDS[spec]
    except KeyError as exc:
        raise ValueError(f"unsupported Llama-Vision spec {spec!r}") from exc


def hf_cache_dir_from_env() -> str | None:
    """Return the configured Hugging Face cache directory."""

    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def render_prompt(
    question: Question, model_input: ModelInput, instruction: str | None = None
) -> tuple[str, tuple[ImagePart, ...]]:
    """Render the prompt with a chosen instruction preamble (full context)."""

    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    header = PROMPT_HEADER if not instruction.strip() else f"{PROMPT_HEADER}\n{instruction.strip()}"
    text = f"{header}\n\n" + PROMPT_BODY.format(question=question.question.strip(), context=context)
    return text, images


def messages_from_prompt(prompt: str, image_count: int) -> list[dict[str, Any]]:
    """One user message whose content interleaves text and image blocks at the
    `<image>` placeholder positions (the processor pairs blocks with the PIL
    images in order)."""

    pieces = prompt.split(IMAGE_PLACEHOLDER)
    if len(pieces) - 1 != image_count:
        raise ValueError(f"prompt has {len(pieces) - 1} image placeholders but {image_count} images")
    content: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if index < image_count:
            content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def resize_for_budget(image, max_pixels: int | None):
    """Downscale a PIL page image to the resolution preset's pixel budget.

    Mllama's own tiling ignores `max_pixels`, so the budget is applied to the
    input image instead: the model sees a genuinely lossier page at a lower
    preset, which is what the resolution axis means everywhere else.
    """

    if not max_pixels:
        return image
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def _estimated_tiles(image) -> int:
    """The tile count Mllama will spend on an image (estimate, capped at 4)."""

    width, height = image.size
    across = -(-width // LLAMA_TILE_SIZE) * -(-height // LLAMA_TILE_SIZE)
    return max(1, min(LLAMA_MAX_TILES, across))


class LlamaVisionBackend(Reasoner):
    """Llama-3.2-Vision local backend via `MllamaForConditionalGeneration`."""

    prompt_template_version: str = PROMPT_TEMPLATE_VERSION

    def __init__(
        self,
        spec: str,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        max_pixels: int | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.spec = spec
        self.model_id = model_id or model_id_for_spec(spec)
        self.max_new_tokens = int(max_new_tokens)
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self._processor = processor
        self._model = model
        if local_files_only is None:
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "") == "1"
        self.local_files_only = bool(local_files_only)
        self.cache_dir = hf_cache_dir_from_env()

    def _load_components(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        common = {"local_files_only": self.local_files_only}
        if self.cache_dir:
            common["cache_dir"] = self.cache_dir
        self._processor = AutoProcessor.from_pretrained(self.model_id, **common)
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto", **common
        )
        return self._processor, self._model

    def free(self) -> None:
        """Drop the model/processor references so the driver can free the GPU."""

        self._model = None
        self._processor = None

    def render(self, question: Question, model_input: ModelInput) -> tuple[str, tuple[ImagePart, ...]]:
        """Assemble this backend's prompt; the per-model override point."""

        return render_prompt(question, model_input, self.prompt_instruction)

    def _pil_images(self, parts: tuple[ImagePart, ...]) -> list[Any]:
        from PIL import Image

        images = []
        for part in parts:
            source = part.image_path if part.image_path else BytesIO(part.read_bytes())
            image = Image.open(source).convert("RGB")
            images.append(resize_for_budget(image, self.max_pixels))
        return images

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        import torch

        processor, model = self._load_components()
        prompt, image_parts = self.render(question, model_input)
        images = self._pil_images(image_parts)
        messages = messages_from_prompt(prompt, len(images))

        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images or None, chat_text, add_special_tokens=False, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        input_len = int(inputs["input_ids"].shape[-1])

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # Same first-token streamer as the Qwen backend: HF generate exposes the
        # boundary here (unlike InternVL's chat()), so the prefill/decode split
        # is real, which matters because cross-attention prefill is the number
        # this family is run to measure.
        from models.qwen3vl import _FirstTokenTimer

        timer = _FirstTokenTimer()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=timer
            )
        if cuda:
            torch.cuda.synchronize()
        latency_s = time.perf_counter() - start
        peak_vram_bytes = int(torch.cuda.max_memory_allocated()) if cuda else 0
        prefill_latency_s = (timer.first_token_time - start) if timer.first_token_time is not None else latency_s
        decode_latency_s = max(0.0, latency_s - prefill_latency_s)

        output_ids = generated[:, input_len:]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output_tokens = int(output_ids.shape[-1])
        output_truncated = output_tokens >= self.max_new_tokens

        total_visual_tokens = sum(_estimated_tiles(img) for img in images) * LLAMA_TOKENS_PER_TILE
        total_text_tokens = max(0, input_len)
        return Prediction(
            text=answer,
            model_spec=self.spec,
            total_text_tokens=total_text_tokens,
            # Estimate; cross-attention tokens never join the text sequence, so
            # this is a cost figure, not a context-length figure.
            total_visual_tokens=total_visual_tokens,
            text_tokens_fed=total_text_tokens,  # no cap: everything is fed
            output_tokens=output_tokens,
            latency_s=latency_s,
            prefill_latency_s=prefill_latency_s,
            decode_latency_s=decode_latency_s,
            peak_vram_bytes=peak_vram_bytes,
            metadata={
                "backend": "hf-transformers-mllama",
                "model_id": self.model_id,
                "prompt_template_version": self.prompt_template_version,
                "max_new_tokens": self.max_new_tokens,
                "output_truncated": output_truncated,
                "max_pixels": self.max_pixels,
                "n_image_parts": len(image_parts),
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
                "load_class": "MllamaForConditionalGeneration",
            },
        )
