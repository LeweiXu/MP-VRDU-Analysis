"""Resolve experiment model specs to reasoner backend instances.

Purpose:
    Implements the model-family swap point named in the plan. `get_reasoner()`
    parses a spec string, selects a backend, and returns a `Reasoner`; pipeline
    code never imports concrete local/API backend classes directly.

Pipeline role:
    The orchestrator asks this registry for the configured reasoner. Stage M3
    dispatches the smoke Qwen3-VL local spec to `LocalVLMBackend`; unsupported
    specs still resolve to the stub until their stages wire them deliberately.

Spec grammar: ``<family>-<size>-<backend>`` (e.g. ``qwen3vl-8b-local``,
``gpt4o-api``), or the literal ``stub``. M3 supports ``qwen3vl-2b-local`` as the
critical-path smoke backend; additional local sizes and API backends remain
behind this same function.

Arguments:
    None. This module is import-only; callers pass a spec string to
    `ModelSpec.parse()` or `get_reasoner()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.reasoner import Reasoner


@dataclass(frozen=True)
class ModelSpec:
    """A parsed model spec: family, size, and backend, plus the raw name."""

    name: str
    family: str
    size: str
    backend: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        """Parse a ``family-size-backend`` (or ``stub``) spec string."""

        raw = spec.strip()
        if raw == "stub":
            return cls(name="stub", family="stub", size="", backend="stub")
        parts = raw.split("-")
        if len(parts) < 2:
            raise ValueError(
                f"model spec {spec!r} must be 'family-size-backend' or 'stub'"
            )
        backend = parts[-1]
        if backend in ("local", "api"):
            family = parts[0]
            size = "-".join(parts[1:-1])
        else:
            # No explicit backend suffix; treat the whole thing as family-size.
            family = parts[0]
            size = "-".join(parts[1:])
            backend = "local"
        return cls(name=raw, family=family, size=size, backend=backend)


def get_reasoner(spec: str) -> Reasoner:
    """Return a `Reasoner` for a model spec (the family swap point)."""

    parsed = ModelSpec.parse(spec)
    if parsed.backend == "stub":
        from pipeline.reasoner import StubReasoner

        return StubReasoner(spec="stub")
    if parsed.family == "qwen3vl" and parsed.size == "2b" and parsed.backend == "local":
        from models.local_vlm import LocalVLMBackend

        return LocalVLMBackend(parsed.name)
    # Later stages dispatch the remaining local sizes and API backends here.
    from pipeline.reasoner import StubReasoner

    return StubReasoner(spec=parsed.name)
