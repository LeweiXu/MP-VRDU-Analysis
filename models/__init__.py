"""Reasoner registry: maps a model spec to its backend via get_reasoner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.reasoner import Reasoner


QUANTIZATIONS = ("4bit", "8bit")


@dataclass(frozen=True)
class ModelSpec:
    """A parsed model spec: family, size, backend, and optional quantization.

    An optional trailing quantization token (`-4bit`/`-8bit`) selects a
    bitsandbytes-quantized load of the same checkpoint, e.g.
    `qwen3vl-8b-local-4bit`. It stays part of `name` so quantized runs get their
    own cache rows, and `size` still resolves to the base size.
    """

    name: str
    family: str
    size: str
    backend: str
    quantization: str | None = None

    @property
    def base_name(self) -> str:
        """The spec without the quantization suffix (for model-id lookup)."""

        return f"{self.family}-{self.size}-{self.backend}"

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        """Parse a `family-size-backend[-quant]` (or `stub`) spec string."""

        raw = spec.strip()
        if raw == "stub":
            return cls(name="stub", family="stub", size="", backend="stub")
        parts = raw.split("-")
        quantization: str | None = None
        if parts and parts[-1] in QUANTIZATIONS:
            quantization = parts[-1]
            parts = parts[:-1]
        if len(parts) < 2:
            raise ValueError(f"model spec {spec!r} must be 'family-size-backend[-quant]' or 'stub'")
        backend = parts[-1]
        if backend in ("local", "api"):
            family = parts[0]
            size = "-".join(parts[1:-1])
        else:
            family = parts[0]
            size = "-".join(parts[1:])
            backend = "local"
        return cls(name=raw, family=family, size=size, backend=backend, quantization=quantization)


def get_reasoner(
    spec: str,
    *,
    max_new_tokens: int | None = None,
    max_pixels: int | None = None,
) -> "Reasoner":
    """Return a `Reasoner` for a model spec (the model-family swap point).

    `max_new_tokens` / `max_pixels` are optional generation / per-page vision
    caps for the local backends; when omitted each backend keeps its own default.
    The vision cap comes from the run's resolution preset.
    """

    parsed = ModelSpec.parse(spec)
    if parsed.backend == "stub":
        from pipeline.reasoner import StubReasoner

        return StubReasoner(spec="stub")
    if parsed.family == "qwen3vl" and parsed.size in {"2b", "4b", "8b", "32b"} and parsed.backend == "local":
        from models.qwen3vl import Qwen3VLBackend

        kwargs: dict[str, object] = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        if parsed.quantization is not None:
            kwargs["quantization"] = parsed.quantization
        return Qwen3VLBackend(parsed.name, **kwargs)
    if parsed.family == "qwen3vl" and parsed.size == "8b-thinking" and parsed.backend == "local":
        from models.qwen3vl_thinking import Qwen3VLThinkingBackend

        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        if parsed.quantization is not None:
            kwargs["quantization"] = parsed.quantization
        return Qwen3VLThinkingBackend(parsed.name, **kwargs)
    if parsed.family == "internvl3" and parsed.size == "8b" and parsed.backend == "local":
        from models.internvl import InternVLBackend

        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        return InternVLBackend(parsed.name, **kwargs)
    if parsed.family == "llama3.2" and parsed.size == "11b-vision" and parsed.backend == "local":
        from models.llama_vision import LlamaVisionBackend

        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        return LlamaVisionBackend(parsed.name, **kwargs)
    # No silent StubReasoner fall-through: a typo'd spec used to produce stub
    # answers at scale. Only the explicit "stub" spec builds a stub.
    raise ValueError(f"no reasoner registered for spec {spec!r}; use 'stub' for the stub backend")
