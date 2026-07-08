"""Test the Stage-M3 local Qwen3-VL reasoner boundary.

Purpose:
    Verifies the real local-backend code path without loading model weights:
    fake processor/model objects exercise text-only and text+image generation,
    prompt-template versioning, token accounting, latency recording, and registry
    dispatch for the smoke model spec.

Test role:
    Protects the critical M3 contract that downstream stages depend on: every
    representation rung reaches the same frozen prompt template and receives a
    populated `Prediction` through the normal `Reasoner` ABC.

Arguments:
    None. Run with `python -m pytest tests/test_reasoner.py`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from models import ModelSpec, get_reasoner
from models.internvl import LocalInternVLBackend
from models.local_vlm import LocalVLMBackend, PROMPT_TEMPLATE_VERSION, hf_cache_dir_from_env, render_prompt
from models.payload import ModelInput
from schema import ImagePart, Question, TextPart


class FakeBatch(dict):
    """Small dict-like tensor batch with the `.to()` method processors expose."""

    def to(self, device):
        self["moved_to"] = str(device)
        return self


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": list(range(max(1, len(text.split()))))}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.messages: list[list[dict[str, Any]]] = []
        self.decode_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages.append(messages)
        text_parts = [
            item["text"]
            for message in messages
            for item in message["content"]
            if item.get("type") == "text"
        ]
        return "\n".join(text_parts) + "\n<|assistant|>"

    def __call__(self, *, text, images=None, videos=None, padding=True, return_tensors="pt"):
        batch = FakeBatch({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})
        if images:
            batch["image_grid_thw"] = torch.tensor([[1, 2, 3]])
        return batch

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.decode_calls += 1
        return ["mock answer"]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5, 101, 102, 103]])


def question() -> Question:
    return Question(
        id="q1",
        doc_id="doc.pdf",
        question="What is the answer?",
        gold_answer="mock answer",
        answer_format="String",
        doc_type="Academic paper",
        evidence_pages=(0,),
        evidence_sources=("Text",),
        hop="single",
        is_unanswerable=False,
    )


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def backend(processor: FakeProcessor | None = None, model: FakeModel | None = None) -> LocalVLMBackend:
    return LocalVLMBackend(
        "qwen3vl-2b-local",
        processor=processor or FakeProcessor(),
        model=model or FakeModel(),
        process_vision_info=lambda messages: (["fake-image"] if any(
            item.get("type") == "image"
            for message in messages
            for item in message["content"]
        ) else None, None),
        local_files_only=True,
    )


def test_local_vlm_answers_text_only_and_records_costs() -> None:
    processor = FakeProcessor()
    model = FakeModel()
    reasoner = backend(processor, model)
    model_input = ModelInput((TextPart("document text evidence"),))

    prediction = reasoner.answer(question(), model_input)

    assert prediction.text == "mock answer"
    assert prediction.model_spec == "qwen3vl-2b-local"
    assert prediction.input_text_tokens > 0
    assert prediction.input_visual_tokens == 0
    assert prediction.output_tokens == 3
    assert prediction.latency_s >= 0
    assert prediction.metadata["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert model.generate_kwargs["max_new_tokens"] == 64


def test_local_vlm_answers_image_inputs_and_counts_visual_tokens() -> None:
    processor = FakeProcessor()
    reasoner = backend(processor)
    model_input = ModelInput((TextPart("see page"), ImagePart(data=png_bytes())))

    prediction = reasoner.answer(question(), model_input)

    assert prediction.text
    assert prediction.input_text_tokens > 0
    assert prediction.input_visual_tokens == 6
    content = processor.messages[-1][0]["content"]
    assert any(item.get("type") == "image" for item in content)


def test_same_frozen_prompt_template_is_used_across_representations() -> None:
    model_inputs = {
        "T": ModelInput((TextPart("[text]\nalpha"),)),
        "TL": ModelInput((TextPart("[text]\nalpha"), TextPart("[layout]\n{}"))),
        "TLV": ModelInput((TextPart("[text]\nalpha"), ImagePart(data=png_bytes()))),
        "V": ModelInput((ImagePart(data=png_bytes()),)),
    }

    versions = {name: render_prompt(question(), value).template_version for name, value in model_inputs.items()}

    assert set(versions.values()) == {PROMPT_TEMPLATE_VERSION}


def test_registry_dispatches_smoke_spec_to_local_backend() -> None:
    reasoner = get_reasoner("qwen3vl-2b-local")

    assert isinstance(reasoner, LocalVLMBackend)
    assert reasoner.model_id == "Qwen/Qwen3-VL-2B-Instruct"


def test_registry_dispatches_full_qwen_specs_to_local_backend() -> None:
    specs = {
        "qwen3vl-8b-local": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3vl-32b-local": "Qwen/Qwen3-VL-32B-Instruct",
    }

    for spec, model_id in specs.items():
        reasoner = get_reasoner(spec)
        assert isinstance(reasoner, LocalVLMBackend)
        assert reasoner.model_id == model_id


def test_render_prompt_truncates_long_context_and_keeps_image_placeholders() -> None:
    from models.local_vlm import IMAGE_PLACEHOLDER, _truncate_context

    class WordTokenizer:
        """Toy whitespace tokenizer: one token per word."""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    tok = WordTokenizer()
    long_text = " ".join(f"w{i}" for i in range(5000))
    # No images: text trimmed to the budget (cap - reserve).
    out = _truncate_context(tok, long_text, 0, max_input_tokens=1000, per_image_tokens=800)
    assert len(out.split()) <= 1000 - 256
    # Two images: every placeholder is preserved so the image count still matches.
    context = f"{IMAGE_PLACEHOLDER}{long_text}{IMAGE_PLACEHOLDER}"
    out2 = _truncate_context(tok, context, 2, max_input_tokens=4096, per_image_tokens=800)
    assert out2.count(IMAGE_PLACEHOLDER) == 2
    # Short context is returned unchanged.
    assert _truncate_context(tok, "short text", 0, 4096, 800) == "short text"


def test_internvl_render_prompt_truncates_long_context() -> None:
    from models.internvl import _truncate_context, render_prompt
    from models.payload import IMAGE_PLACEHOLDER
    from schema import TextPart

    class WordTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    tok = WordTokenizer()
    long_text = " ".join(f"w{i}" for i in range(5000))
    out = _truncate_context(tok, long_text, 0, max_input_tokens=1000, per_image_tokens=1024)
    assert len(out.split()) <= 1000 - 256
    # A long TL/text cell is trimmed instead of feeding the full context (the G2 OOM).
    prompt, _ = render_prompt(
        question(), ModelInput((TextPart(long_text),)), tokenizer=tok, max_input_tokens=1000
    )
    assert len(prompt.split()) < 5000
    # Without a cap the full context is used (unchanged behaviour for tests).
    prompt_full, _ = render_prompt(question(), ModelInput((TextPart(long_text),)))
    assert long_text in prompt_full
    assert IMAGE_PLACEHOLDER  # placeholder constant is importable for parity with Qwen


def test_registry_forwards_max_input_tokens_to_internvl() -> None:
    reasoner = get_reasoner("internvl3-8b-local", max_input_tokens=4096)
    assert isinstance(reasoner, LocalInternVLBackend)
    assert reasoner.max_input_tokens == 4096


def test_model_spec_parses_quantization_suffix() -> None:
    plain = ModelSpec.parse("qwen3vl-8b-local")
    assert plain.quantization is None
    assert plain.size == "8b"
    assert plain.base_name == "qwen3vl-8b-local"

    quant = ModelSpec.parse("qwen3vl-8b-local-4bit")
    # The full name is the cache identity; base_name resolves the checkpoint.
    assert quant.name == "qwen3vl-8b-local-4bit"
    assert quant.quantization == "4bit"
    assert quant.size == "8b"          # per-size pixel cap still resolves
    assert quant.base_name == "qwen3vl-8b-local"
    assert ModelSpec.parse("qwen3vl-2b-local-8bit").quantization == "8bit"


def test_registry_dispatches_quantized_spec_with_base_model_id() -> None:
    reasoner = get_reasoner("qwen3vl-8b-local-4bit")

    assert isinstance(reasoner, LocalVLMBackend)
    assert reasoner.spec == "qwen3vl-8b-local-4bit"      # distinct cache identity
    assert reasoner.model_id == "Qwen/Qwen3-VL-8B-Instruct"  # base checkpoint
    assert reasoner.quantization == "4bit"


def test_quantization_config_builds_bnb_4bit_and_8bit() -> None:
    import pytest

    pytest.importorskip("bitsandbytes")  # installed on Kaya, not the local env
    four = LocalVLMBackend("qwen3vl-8b-local-4bit", quantization="4bit")._quantization_config()
    assert four is not None and four.load_in_4bit
    eight = LocalVLMBackend("qwen3vl-8b-local-8bit", quantization="8bit")._quantization_config()
    assert eight is not None and eight.load_in_8bit
    assert LocalVLMBackend("qwen3vl-8b-local", quantization=None)._quantization_config() is None


def test_registry_dispatches_internvl_family_spec() -> None:
    reasoner = get_reasoner("internvl3-8b-local")

    assert isinstance(reasoner, LocalInternVLBackend)
    assert reasoner.model_id == "OpenGVLab/InternVL3-8B"


def test_local_vlm_uses_repo_hub_cache_env(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / ".cache"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "transformers"))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))

    reasoner = backend()

    assert hf_cache_dir_from_env() == str(cache_dir)
    assert reasoner.cache_dir == str(cache_dir)


def test_image_path_prompt_binding(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(png_bytes())
    rendered = render_prompt(question(), ModelInput((ImagePart(image_path=image),)))
    message = rendered.text

    assert "<image>" in message
