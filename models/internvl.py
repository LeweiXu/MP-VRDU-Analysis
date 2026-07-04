"""Run local InternVL checkpoints behind the `Reasoner` interface.

Purpose:
    Implements the Section-F4 model-family replication backend for
    `internvl3-8b-local`. It loads InternVL through Hugging Face
    `trust_remote_code`, formats the same document-question contract used by the
    Qwen backend, binds page images into the model's `chat()` API, and returns
    the common `Prediction` cost surface.

Pipeline role:
    `models.get_reasoner()` dispatches InternVL specs here. The orchestrator and
    experiments still see only the frozen `Reasoner.answer(question,
    model_input)` interface, so Table 3 can compare Qwen and InternVL rows
    without pipeline changes.

Arguments:
    None at the command line. Import callers instantiate `LocalInternVLBackend`
    with a model spec and optional injected tokenizer/model fakes for tests.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Any

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from pipeline.reasoner import Reasoner
from schema import ImagePart, Prediction, Question


MODEL_IDS: dict[str, str] = {
    "internvl3-8b-local": "OpenGVLab/InternVL3-8B",
}

PROMPT_TEMPLATE_VERSION = "f4-internvl3-v1"
FROZEN_PROMPT_TEMPLATE = """You are answering a question about a document.
Use only the provided document evidence. If the evidence does not contain the answer, answer exactly: Not answerable.
Keep the answer concise.

Question:
{question}

Document evidence:
{context}

Answer:"""


def model_id_for_spec(spec: str) -> str:
    """Return the Hugging Face model id for an InternVL model spec."""

    try:
        return MODEL_IDS[spec]
    except KeyError as exc:
        raise ValueError(f"unsupported InternVL spec {spec!r}") from exc


def hf_cache_dir_from_env() -> str | None:
    """Return the configured Hugging Face cache directory."""

    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def render_prompt(question: Question, model_input: ModelInput) -> tuple[str, tuple[ImagePart, ...]]:
    """Render the frozen prompt and return ordered image parts."""

    context, images = model_input.to_local_prompt()
    context = context.strip() or "(no document evidence was provided)"
    return FROZEN_PROMPT_TEMPLATE.format(question=question.question.strip(), context=context), images


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort tokenizer count with whitespace fallback."""

    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
    except Exception:
        return max(1, len(text.split()))


def _load_image_tensor(part: ImagePart, *, image_size: int) -> Any:
    """Load one image as an InternVL-compatible tensor."""

    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = (
        Image.open(part.image_path).convert("RGB")
        if part.image_path
        else Image.open(BytesIO(part.read_bytes())).convert("RGB")
    )
    return transform(image).unsqueeze(0).to(torch.bfloat16)


def _image_tensors(parts: tuple[ImagePart, ...], *, image_size: int, device: Any) -> Any:
    """Return concatenated image tensors or None for text-only prompts."""

    if not parts:
        return None
    import torch

    tensors = [_load_image_tensor(part, image_size=image_size) for part in parts]
    pixel_values = torch.cat(tensors, dim=0)
    return pixel_values.to(device) if device is not None else pixel_values


class LocalInternVLBackend(Reasoner):
    """InternVL local backend using the model's `chat()` helper."""

    def __init__(
        self,
        spec: str,
        *,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        image_size: int = 448,
        tokenizer: Any | None = None,
        model: Any | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.spec = spec
        self.model_id = model_id or model_id_for_spec(spec)
        self.max_new_tokens = int(max_new_tokens)
        self.image_size = int(image_size)
        self._tokenizer = tokenizer
        self._model = model
        self.local_files_only = any(os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")) if local_files_only is None else bool(local_files_only)
        self.cache_dir = hf_cache_dir_from_env()

    def _load_components(self) -> tuple[Any, Any]:
        """Load and cache the tokenizer/model pair."""

        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModel, AutoTokenizer

        common_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False, **common_kwargs)
        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            **common_kwargs,
        ).eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        prompt, images = render_prompt(question, model_input)
        tokenizer, model = self._load_components()
        device = getattr(model, "device", None)
        pixel_values = _image_tensors(images, image_size=self.image_size, device=device)
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        text_prompt = prompt.replace(IMAGE_PLACEHOLDER, "<image>")
        num_patches_list = [1 for _ in images] or None

        start = time.perf_counter()
        try:
            answer = model.chat(
                tokenizer,
                pixel_values,
                text_prompt,
                generation_config,
                num_patches_list=num_patches_list,
                return_history=False,
            )
        except TypeError:
            answer = model.chat(tokenizer, pixel_values, text_prompt, generation_config)
        latency_s = time.perf_counter() - start

        answer_text = str(answer[0] if isinstance(answer, tuple) else answer).strip()
        return Prediction(
            text=answer_text,
            model_spec=self.spec,
            input_text_tokens=_count_text_tokens(tokenizer, prompt.replace(IMAGE_PLACEHOLDER, "")),
            input_visual_tokens=len(images) * (self.image_size // 14) ** 2,
            output_tokens=max(1, len(answer_text.split())),
            latency_s=latency_s,
            metadata={
                "backend": "hf-transformers-internvl-chat",
                "model_id": self.model_id,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "max_new_tokens": self.max_new_tokens,
                "n_image_parts": len(images),
                "local_files_only": self.local_files_only,
                "cache_dir": self.cache_dir,
            },
        )
