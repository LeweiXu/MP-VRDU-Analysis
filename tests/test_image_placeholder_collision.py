"""Document text that literally contains the `<image>` sentinel must not be
miscounted as an image slot (regression for the g3 failure on 2306.05425v1.pdf,
a VLM paper whose text contains `<image>`)."""

from __future__ import annotations

from models.payload import IMAGE_PLACEHOLDER, ModelInput
from models.qwen3vl import RenderedPrompt, messages_from_rendered_prompt
from schema import ImagePart, TextPart


def _image() -> ImagePart:
    return ImagePart(data=b"\x89PNG\r\n\x1a\n", mime="image/png")


def test_literal_image_token_in_text_is_neutralised_text_only() -> None:
    # T rung: no real images, but the document text carries a literal "<image>".
    mi = ModelInput((TextPart(f"the model prepends a {IMAGE_PLACEHOLDER} token"),))
    text, images = mi.to_local_prompt()
    assert images == ()
    assert IMAGE_PLACEHOLDER not in text
    assert "[image]" in text
    # The backend's placeholder-vs-image check must now pass (0 == 0).
    msgs = messages_from_rendered_prompt(RenderedPrompt(text=text, image_parts=images))
    assert sum(1 for c in msgs[0]["content"] if c["type"] == "image") == 0


def test_real_image_survives_alongside_literal_token() -> None:
    # TLV-like: one real image plus document text that also mentions "<image>".
    mi = ModelInput((TextPart(f"caption with a {IMAGE_PLACEHOLDER} mention"), _image()))
    text, images = mi.to_local_prompt()
    assert len(images) == 1
    # Exactly one sentinel remains: the one inserted for the real image part.
    assert text.count(IMAGE_PLACEHOLDER) == 1
    msgs = messages_from_rendered_prompt(RenderedPrompt(text=text, image_parts=images))
    assert sum(1 for c in msgs[0]["content"] if c["type"] == "image") == 1
