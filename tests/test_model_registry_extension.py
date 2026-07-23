"""Phase C guards: the new model registrations, the render override point, and
the removal of the silent StubReasoner fall-through."""

import pytest

from models import ModelSpec, get_reasoner


def test_parse_new_specs():
    thinking = ModelSpec.parse("qwen3vl-8b-thinking-local")
    assert (thinking.family, thinking.size, thinking.backend) == ("qwen3vl", "8b-thinking", "local")
    llama = ModelSpec.parse("llama3.2-11b-vision-local")
    assert (llama.family, llama.size, llama.backend) == ("llama3.2", "11b-vision", "local")
    quant = ModelSpec.parse("qwen3vl-32b-local-4bit")
    assert quant.quantization == "4bit" and quant.base_name == "qwen3vl-32b-local"


def test_registry_returns_new_backends():
    # Constructors are lazy: no GPU, no network, no model load.
    from models.llama_vision import LlamaVisionBackend
    from models.qwen3vl import Qwen3VLBackend
    from models.qwen3vl_thinking import Qwen3VLThinkingBackend

    thinking = get_reasoner("qwen3vl-8b-thinking-local", max_new_tokens=2048)
    assert isinstance(thinking, Qwen3VLThinkingBackend)
    assert isinstance(thinking, Qwen3VLBackend)  # inherits load/generate/telemetry
    assert thinking.max_new_tokens == 2048
    assert thinking.model_id == "Qwen/Qwen3-VL-8B-Thinking"

    llama = get_reasoner("llama3.2-11b-vision-local")
    assert isinstance(llama, LlamaVisionBackend)
    assert llama.model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct"

    assert get_reasoner("qwen3vl-32b-local-4bit").quantization == "4bit"


def test_unknown_spec_raises_instead_of_stubbing():
    # The old fall-through returned StubReasoner for any unmatched spec, so a
    # typo produced stub answers at scale. Now only "stub" builds a stub.
    with pytest.raises(ValueError):
        get_reasoner("qwen3vl-9b-local")
    with pytest.raises(ValueError):
        get_reasoner("internvl3-8b-locale")
    from pipeline.reasoner import StubReasoner

    assert isinstance(get_reasoner("stub"), StubReasoner)


def _model_input():
    from models.payload import ModelInput
    from schema import Payload, TextPart

    return ModelInput.from_payload(Payload(modality="T", parts=(TextPart(text="[text]\nsome page text"),)))


def _question():
    from schema import Question

    return Question(id="q", doc_id="d", question="what?", gold_answer="a",
                    answer_format="str", doc_type="t", evidence_pages=(0,),
                    evidence_sources=(), hop="single", is_unanswerable=False)


def test_thinking_render_adds_final_line_contract_under_every_mode():
    from config import PROMPT_MODES
    from models.qwen3vl_thinking import THINKING_FINAL, Qwen3VLThinkingBackend

    backend = Qwen3VLThinkingBackend("qwen3vl-8b-thinking-local")
    for mode, instruction in PROMPT_MODES.items():
        backend.prompt_instruction = instruction
        text = backend.render(_question(), _model_input()).text
        assert THINKING_FINAL in text, mode
        if instruction.strip():
            # Mode semantics first, then the output contract.
            assert text.index(instruction.strip()[:30]) < text.index("After your reasoning")


def test_base_render_is_unchanged_by_the_hook():
    from models.qwen3vl import Qwen3VLBackend, render_prompt

    backend = Qwen3VLBackend("qwen3vl-8b-local")
    backend.prompt_instruction = ""
    assert backend.render(_question(), _model_input()).text == \
        render_prompt(_question(), _model_input(), "").text
    assert backend.prompt_template_version == "qwen3vl-v1"


def test_llama_pure_helpers():
    from models.llama_vision import messages_from_prompt, resize_for_budget

    msgs = messages_from_prompt("before <image> after", 1)
    kinds = [b["type"] for b in msgs[0]["content"]]
    assert kinds == ["text", "image", "text"]
    with pytest.raises(ValueError):
        messages_from_prompt("no placeholder", 1)

    class FakeImage:
        def __init__(self, w, h):
            self.size = (w, h)

        def resize(self, size):
            return FakeImage(*size)

    big = FakeImage(2000, 2000)
    small = resize_for_budget(big, 501_760)  # the med preset budget
    assert small.size[0] * small.size[1] <= 501_760
    assert resize_for_budget(big, None) is big
