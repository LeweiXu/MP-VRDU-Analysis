"""Real VLM generators (Stage 2 interface; run on Kaya / local stand-in).

These are thin wrappers around Qwen2.5-VL. They lazy-import torch/transformers
so the light local env can import this module without the GPU stack. The 7B
generator does NOT fit comfortably on the 12GB dev box (Operating Principle 2):
- LocalSmallVLM: Qwen2.5-VL-3B (or 4-bit 7B) — local CODE-PATH testing only.
- KayaVLM:       full Qwen2.5-VL-7B (later 32B) — the real runs.

Both build the chat input identically from (images, text); only the checkpoint
and device differ. NEVER report numbers from LocalSmallVLM.
"""

from __future__ import annotations

from typing import Optional

from ..logging_utils import get_logger
from .base import Generator

log = get_logger("generate")

PROMPT_PREFIX = (
    "You are answering a question about a document. Use ONLY the provided "
    "evidence. If the evidence does not contain the answer, reply exactly "
    "'Not answerable'. Give the shortest exact answer.\n\n"
)


class _QwenVLBase(Generator):
    def __init__(self, model_id: str, max_new_tokens: int = 256,
                 temperature: float = 0.0, device_map: str = "auto",
                 load_in_4bit: bool = False, max_pixels: Optional[int] = None,
                 min_pixels: Optional[int] = None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        log.info("loading VLM %s (4bit=%s)", self.model_id, self.load_in_4bit)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs)
        # min/max_pixels bound vision tokens per image — critical for fitting
        # many-page inputs (e.g. no-retrieval first-N pages) on a 12GB box.
        proc_kwargs = {}
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = self.max_pixels
        if self.min_pixels is not None:
            proc_kwargs["min_pixels"] = self.min_pixels
        self._processor = AutoProcessor.from_pretrained(self.model_id, **proc_kwargs)

    def _build_messages(self, question: str, images, text):
        content = []
        for img in (images or []):
            content.append({"type": "image", "image": img})
        body = PROMPT_PREFIX
        if text:
            body += f"Evidence text:\n{text}\n\n"
        body += f"Question: {question}"
        content.append({"type": "text", "text": body})
        return [{"role": "user", "content": content}]

    def answer(self, question: str, images: Optional[list[str]] = None,
               text: Optional[str] = None) -> str:
        self._ensure_loaded()
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(question, images, text)
        chat = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[chat], images=image_inputs,
                                 videos=video_inputs, padding=True,
                                 return_tensors="pt").to(self._model.device)
        gen = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-6))
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return out.strip()


class LocalSmallVLM(_QwenVLBase):
    """Small/quantised stand-in for local code-path testing ONLY."""

    name = "local_small_vlm"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kw):
        super().__init__(model_id=model_id, **kw)


class KayaVLM(_QwenVLBase):
    """Full Qwen2.5-VL-7B (later 32B) for real runs on Kaya."""

    name = "kaya_vlm"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", **kw):
        super().__init__(model_id=model_id, **kw)
