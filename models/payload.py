"""Backend-neutral prompt and image container (ModelInput) passed to reasoners."""

from __future__ import annotations

from dataclasses import dataclass

from schema import ImagePart, Part, Payload, TextPart


IMAGE_PLACEHOLDER = "<image>"


@dataclass(frozen=True)
class ModelInput:
    """Ordered text and image parts handed to a reasoner backend."""

    parts: tuple[Part, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))

    @classmethod
    def from_payload(cls, payload: Payload) -> "ModelInput":
        """Map a representation `Payload` to a backend-agnostic model input."""

        return cls(payload.parts)

    @property
    def text_parts(self) -> tuple[TextPart, ...]:
        return tuple(p for p in self.parts if isinstance(p, TextPart))

    @property
    def image_parts(self) -> tuple[ImagePart, ...]:
        return tuple(p for p in self.parts if isinstance(p, ImagePart))

    def with_parts(self, *extra: Part) -> "ModelInput":
        """Return a copy with extra parts appended."""

        return ModelInput(self.parts + tuple(extra))

    # -- adapters ---------------------------------------------------------

    def to_chat_messages(self, role: str = "user") -> list[dict]:
        """Render as a chat `messages` array with base64 image parts."""

        content: list[dict] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                content.append({"type": "image_url", "image_url": {"url": part.data_uri()}})
        return [{"role": role, "content": content}]

    def to_local_prompt(self) -> tuple[str, tuple[ImagePart, ...]]:
        """Render as a prompt string with image placeholders plus the images.

        Images are returned in order so the local backend can bind each
        `<image>` placeholder to the right pixels via the model's processor.

        The `<image>` sentinel is inserted here only for real image parts, so any
        literal `<image>` inside document text (some PDFs and parser markdown
        contain it, e.g. VLM papers) is neutralised to `[image]` first. Otherwise
        it would be miscounted as an image slot and the backends' placeholder-vs-image
        check would reject the prompt.
        """

        pieces: list[str] = []
        images: list[ImagePart] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                pieces.append(part.text.replace(IMAGE_PLACEHOLDER, "[image]"))
            else:
                pieces.append(IMAGE_PLACEHOLDER)
                images.append(part)
        return "\n".join(pieces), tuple(images)

    @classmethod
    def from_chat_messages(cls, messages: list[dict]) -> "ModelInput":
        """Reconstruct a `ModelInput` from a chat `messages` array.

        Image parts come back carrying inline bytes decoded from their data URI,
        so the original image content survives even though the on-disk path is
        not recoverable.
        """

        import base64

        parts: list[Part] = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "text":
                    parts.append(TextPart(item["text"]))
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    header, _, encoded = url.partition(",")
                    mime = header.split(";")[0].removeprefix("data:") or "image/png"
                    parts.append(ImagePart(data=base64.b64decode(encoded), mime=mime))
        return cls(tuple(parts))
